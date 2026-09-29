/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SPMEmitter.hpp - RedMulE schedules on SPM buffers --------===//
//
// Shared by the AISLE -> AISMEM RedMulE lowerings (transformer blocks,
// MatMul, Add): operand classification (SPM-resident or data memory) and a
// thin emitter for aismem.SPMAlloc buffers and the scheduled RedMulE / SPM
// operations on one tile.  Every launch is followed by its wait.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/AISLEToAISMEM/Math/SPMValue.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "llvm/ADT/Twine.h"

#include <optional>
#include <string>

namespace spade {
namespace spm {

using namespace mlir;

// One native RedMulE launch: Y[M x K] (+)= X[M x N] . W[N x K].
constexpr int32_t kM = 12; // rows of X and Y (ARRAY_HEIGHT * PIPE_REGS)
constexpr int32_t kN = 16; // reduction length, rows of W
constexpr int32_t kK = 16; // columns of W and Y (one SPM row)
constexpr int32_t kTile = 0; // single-tile chain

// A host-side operand: statically shaped f16 memref viewed row-major as
// [rows x cols], or a [rows x cols] window of it at (rowOff, colOff) (an
// aisle.Window lowered to memref.subview; `memref` is then the source).
struct Matrix {
  Value memref;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t rowOff = 0;
  int64_t colOff = 0;
};

// An operand of a block: in SPM already (produced by a previous block), or in
// data memory.
struct Operand {
  std::optional<SPMValue> spm;
  std::optional<Matrix> host;
};

inline FailureOr<Matrix> asMatrix(Operation *op, Value memref, StringRef name) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1)
    return op->emitError() << name << " must be a statically shaped memref";
  if (!type.getElementType().isF16())
    return op->emitError() << name
                           << ": the RedMulE lowering requires f16 tensors";
  Matrix m;
  m.memref = memref;
  m.cols = type.getShape().back();
  m.rows = type.getNumElements() / m.cols;
  return m;
}

inline LogicalResult expectShape(Operation *op, int64_t gotRows, int64_t gotCols,
    StringRef name, int64_t rows, int64_t cols) {
  if (gotRows == rows && gotCols == cols)
    return success();
  return op->emitError() << name << " is " << gotRows << "x" << gotCols
                         << ", the RedMulE lowering needs " << rows << "x"
                         << cols;
}

// Classify a converted operand: SPM-resident block result or host memref.
inline FailureOr<Operand> classify(Operation *op, Value value, StringRef name,
    int64_t rows, int64_t cols) {
  Operand o;
  if (std::optional<SPMValue> spm = getSPMValue(value)) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    int64_t gotCols = tensor ? tensor.getShape().back() : cols;
    int64_t gotRows = tensor ? tensor.getNumElements() / gotCols : rows;
    if (failed(expectShape(op, gotRows, gotCols, name, rows, cols)))
      return failure();
    o.spm = *spm;
    return o;
  }
  if (getSPMTiles(value))
    return op->emitError() << name
                           << " is a tiled SPM result; only a Window of one of "
                              "its tiles can be a RedMulE operand";
  Value memref = value;
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>())
    if (cast.getInputs().size() == 1)
      memref = cast.getInputs().front();
  // A window (aisle.Window -> memref.subview) of a data-memory operand.
  if (auto view = memref.getDefiningOp<memref::SubViewOp>()) {
    auto sourceType = cast<MemRefType>(view.getSource().getType());
    ArrayRef<int64_t> offsets = view.getStaticOffsets();
    ArrayRef<int64_t> sizes = view.getStaticSizes();
    const int64_t rank = sourceType.getRank();
    bool plain = rank >= 2 && sourceType.hasStaticShape() &&
                 !ShapedType::isDynamicShape(offsets) &&
                 !ShapedType::isDynamicShape(sizes) &&
                 llvm::all_of(view.getStaticStrides(),
                     [](int64_t s) { return s == 1; });
    for (int64_t d = 0; plain && d + 2 < rank; ++d)
      plain = sourceType.getDimSize(d) == 1 && offsets[d] == 0;
    if (!plain)
      return op->emitError() << name << ": unsupported window of "
                             << sourceType;
    FailureOr<Matrix> m = asMatrix(op, view.getSource(), name);
    if (failed(m))
      return failure();
    m->rows = sizes[rank - 2];
    m->cols = sizes[rank - 1];
    m->rowOff = offsets[rank - 2];
    m->colOff = offsets[rank - 1];
    if (failed(expectShape(op, m->rows, m->cols, name, rows, cols)))
      return failure();
    o.host = *m;
    return o;
  }
  FailureOr<Matrix> m = asMatrix(op, memref, name);
  if (failed(m) || failed(expectShape(op, m->rows, m->cols, name, rows, cols)))
    return failure();
  o.host = *m;
  return o;
}

// Is `source` a compile-time constant (a krnl.global)?  Such weights can stay
// resident in SPM and be uploaded once by the preload function.
inline bool isConstant(Value source) {
  Operation *def = source.getDefiningOp();
  return def && def->getName().getStringRef() == "krnl.global";
}

// Thin emitter for SPM buffers and the scheduled RedMulE operations on one
// tile.  Every launch is followed by its wait (single-tile chain).
class Emitter {
public:
  Emitter(ConversionPatternRewriter &rewriter, Location loc, StringRef block)
      : rewriter(rewriter), loc(loc), block(block.str()) {}

  Value alloc(int64_t rows, const Twine &name, bool resident = false) {
    SmallVector<NamedAttribute> attrs{attr("tile", kTile), attr("rows", rows),
        rewriter.getNamedAttr("resident", rewriter.getBoolAttr(resident)),
        rewriter.getNamedAttr(
            "name", rewriter.getStringAttr(block + "." + name.str()))};
    return rewriter
        .create<AISMEMSPMAllocOp>(
            loc, TypeRange{rewriter.getI32Type()}, ValueRange{}, attrs)
        .getAddress();
  }

  // host window -> fresh SPM buffer of dstRows rows (zero padded).
  // (rowOff, colOff) are relative to the matrix, which may itself be a
  // window of its memref.
  SPMValue upload(const Matrix &m, int64_t rowOff, int64_t colOff,
      int64_t rows, int64_t cols, int64_t dstRows, const Twine &name,
      bool resident, ValueRange deps, bool transpose = false) {
    rowOff += m.rowOff;
    colOff += m.colOff;
    Value source = m.memref;
    // A constant is uploaded from its exact SPM image, built here at compile
    // time (window, transpose, zero padding): a plain whole-tile copy.
    if (!isPlainUpload(source, rowOff, colOff, rows, cols, dstRows, transpose))
      if (Value image = constantImage(
              source, rowOff, colOff, rows, cols, dstRows, transpose)) {
        source = image;
        rowOff = colOff = 0;
        rows = dstRows;
        cols = kK;
        transpose = false;
      }
    Value address = alloc(dstRows, name, resident);
    SmallVector<Value> operands{source, address};
    operands.append(deps.begin(), deps.end());
    SmallVector<NamedAttribute> attrs{attr("tile", kTile),
        attr("row_offset", rowOff), attr("col_offset", colOff),
        attr("rows", rows), attr("cols", cols), attr("dst_rows", dstRows),
        attr("dst_cols", kK),
        rewriter.getNamedAttr("transpose", rewriter.getBoolAttr(transpose)),
        rewriter.getNamedAttr("relu", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("negate", rewriter.getBoolAttr(false))};
    auto op = rewriter.create<AISMEMRedMulEUploadTileOp>(loc,
        TypeRange{rewriter.getI32Type(), rewriter.getNoneType()}, operands,
        attrs);
    return {address, op.getNoneVal(), kTile};
  }

  // A constant 16x16 weight tile: resident, uploaded once by the preload
  // function (its token is never consumed, so the upload can be hoisted).
  Value weight(const Matrix &m, int64_t rowOff, int64_t colOff,
      const Twine &name) {
    bool resident = isConstant(m.memref);
    SPMValue w = upload(m, rowOff, colOff, kN, kK, kN, name, resident,
        ValueRange{});
    return w.address;
  }

  // A W operand from data memory: a rows x 16 window at (rowOff, colOff),
  // zero padded to 16 rows; resident when the source is a constant.
  // With `transpose`, the window is read as rows x 16 of the source and
  // stored transposed (a W tile of B^T for Gemm transB = 1).
  Value weightWindow(const Matrix &m, int64_t rowOff, int64_t colOff,
      int64_t rows, const Twine &name, bool transpose = false) {
    return upload(m, rowOff, colOff, rows, kK, kN, name, isConstant(m.memref),
        ValueRange{}, transpose)
        .address;
  }

  // A 12-row SPM value as a W operand: copied into a zero-filled 16-row
  // buffer (rows 12..15 must be zero).
  SPMValue padded(const SPMValue &v, const Twine &name, ValueRange deps) {
    Value w = alloc(kN, name);
    SmallVector<Value> all(deps.begin(), deps.end());
    all.push_back(v.token);
    Value z = zero(w, kN, all);
    return {w, copy(v.address, w, kM, z), kTile};
  }

  // The 12x16 identity (ones on the diagonal) as a resident X operand, so
  // that Y = I . W + Y adds W's first 12 rows to Y.  One per function.
  Value identity() {
    constexpr llvm::StringLiteral bufferName = "aismem.identity12x16";
    Operation *parent = rewriter.getInsertionBlock()->getParentOp();
    AISMEMSPMAllocOp found;
    parent->walk([&](AISMEMSPMAllocOp a) {
      if (!found && a.getName() && *a.getName() == bufferName)
        found = a;
    });
    if (found)
      return found.getAddress();

    Type f16 = rewriter.getF16Type();
    auto tensorType = RankedTensorType::get({kM, kN}, f16);
    SmallVector<APFloat> values;
    for (int64_t i = 0; i < kM; ++i)
      for (int64_t j = 0; j < kN; ++j) {
        APFloat v(i == j ? 1.0f : 0.0f);
        bool lost;
        v.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &lost);
        values.push_back(v);
      }
    OperationState state(loc, "krnl.global");
    state.addAttribute("shape", rewriter.getI64ArrayAttr({kM, kN}));
    state.addAttribute("name", rewriter.getStringAttr("aismem_identity_12x16"));
    state.addAttribute("value", DenseElementsAttr::get(tensorType, values));
    state.addAttribute("alignment", rewriter.getI64IntegerAttr(16));
    state.addTypes(MemRefType::get({kM, kN}, f16));
    Value global = rewriter.create(state)->getResult(0);

    SmallVector<NamedAttribute> attrs{attr("tile", kTile), attr("rows", kM),
        rewriter.getNamedAttr("resident", rewriter.getBoolAttr(true)),
        rewriter.getNamedAttr("name", rewriter.getStringAttr(bufferName))};
    Value address = rewriter
                        .create<AISMEMSPMAllocOp>(loc,
                            TypeRange{rewriter.getI32Type()}, ValueRange{},
                            attrs)
                        .getAddress();
    Matrix m{global, kM, kN};
    SmallVector<Value> operands{m.memref, address};
    SmallVector<NamedAttribute> uattrs{attr("tile", kTile),
        attr("row_offset", 0), attr("col_offset", 0), attr("rows", kM),
        attr("cols", kN), attr("dst_rows", kM), attr("dst_cols", kK),
        rewriter.getNamedAttr("transpose", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("relu", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("negate", rewriter.getBoolAttr(false))};
    rewriter.create<AISMEMRedMulEUploadTileOp>(loc,
        TypeRange{rewriter.getI32Type(), rewriter.getNoneType()}, operands,
        uattrs);
    return address;
  }

  // Block operand as an SPM buffer: as is, or uploaded from data memory.
  SPMValue activation(const Operand &o, const Twine &name) {
    if (o.spm)
      return *o.spm;
    return upload(*o.host, 0, 0, kM, kN, kM, name, /*resident=*/false,
        ValueRange{});
  }

  Value zero(Value address, int64_t rows, ValueRange deps) {
    SmallVector<Value> operands{address};
    operands.append(deps.begin(), deps.end());
    return rewriter
        .create<AISMEMRedMulEZeroOp>(loc, TypeRange{rewriter.getNoneType()},
            operands,
            ArrayRef<NamedAttribute>{
                attr("tile", kTile), attr("elements", rows * kK)})
        .getNoneVal();
  }

  // Launch Y (+)= X . W and wait for it.
  Value gemm(Value x, Value w, Value y, ValueRange deps) {
    SmallVector<Value> operands{x, w, y};
    operands.append(deps.begin(), deps.end());
    Value launched =
        rewriter
            .create<AISMEMRedMulEGEMMOp>(loc,
                TypeRange{rewriter.getNoneType()}, operands,
                ArrayRef<NamedAttribute>{attr("tile", kTile), attr("k", kK),
                    attr("m", kM), attr("n", kN)})
            .getNoneVal();
    return rewriter
        .create<AISMEMRedMulEWaitOp>(loc, TypeRange{rewriter.getNoneType()},
            ValueRange{launched},
            ArrayRef<NamedAttribute>{attr("mask", 1 << kTile)})
        .getNoneVal();
  }

  Value relu(Value address, int64_t rows, ValueRange deps) {
    SmallVector<Value> operands{address};
    operands.append(deps.begin(), deps.end());
    return rewriter
        .create<AISMEMSPMReluOp>(loc, TypeRange{rewriter.getNoneType()},
            operands,
            ArrayRef<NamedAttribute>{attr("tile", kTile), attr("rows", rows)})
        .getNoneVal();
  }

  Value transpose(Value source, Value destination, int64_t rows,
      int64_t dstRows, ValueRange deps) {
    SmallVector<Value> operands{source, destination};
    operands.append(deps.begin(), deps.end());
    return rewriter
        .create<AISMEMSPMTransposeOp>(loc, TypeRange{rewriter.getNoneType()},
            operands,
            ArrayRef<NamedAttribute>{attr("tile", kTile), attr("rows", rows),
                attr("dst_rows", dstRows)})
        .getNoneVal();
  }

  Value copy(Value source, Value destination, int64_t rows, ValueRange deps) {
    SmallVector<Value> operands{source, destination};
    operands.append(deps.begin(), deps.end());
    return rewriter
        .create<AISMEMSPMCopyOp>(loc, TypeRange{rewriter.getNoneType()},
            operands,
            ArrayRef<NamedAttribute>{attr("tile", kTile), attr("rows", rows)})
        .getNoneVal();
  }

  // The accumulator Y for `Y = ... + C`: C's own buffer when it may be
  // overwritten, else a fresh copy; zeros without C.
  SPMValue accumulator(const std::optional<Operand> &c, bool cIsDead,
      ValueRange deps) {
    if (!c) {
      Value y = alloc(kM, "Y");
      return {y, zero(y, kM, deps), kTile};
    }
    if (c->host) // upload() already makes a private copy
      return activation(*c, "Y");
    if (cIsDead) // residual in place
      return *c->spm;
    Value y = alloc(kM, "Y");
    SmallVector<Value> all(deps.begin(), deps.end());
    all.push_back(c->spm->token);
    return {y, copy(c->spm->address, y, kM, all), kTile};
  }

  // Does this upload copy whole 16-element rows of a row-major [.. x 16]
  // source unchanged (the runtime's plain-copy path)?
  static bool isPlainUpload(Value source, int64_t rowOff, int64_t colOff,
      int64_t rows, int64_t cols, int64_t dstRows, bool transpose) {
    auto type = cast<MemRefType>(source.getType());
    return rowOff == 0 && colOff == 0 && !transpose && cols == kK &&
           rows == dstRows && type.getShape().back() == kK &&
           type.getNumElements() == dstRows * kK;
  }

  // For a constant (krnl.global) source: a krnl.global holding exactly what
  // the upload would put in SPM, [dstRows x 16] -- the rows x cols window at
  // (rowOff, colOff), transposed on request, zero padded.  One per distinct
  // image and function (found again by name).  Null for other sources.
  Value constantImage(Value source, int64_t rowOff, int64_t colOff,
      int64_t rows, int64_t cols, int64_t dstRows, bool transpose) {
    Operation *global = source.getDefiningOp();
    if (!isConstant(source))
      return {};
    auto dense = global->getAttrOfType<DenseElementsAttr>("value");
    auto name = global->getAttrOfType<StringAttr>("name");
    if (!dense || !name || !dense.getElementType().isF16())
      return {};
    const std::string imageName =
        (name.getValue() + "_spm_r" + Twine(rowOff) + "c" + Twine(colOff) +
            "_" + Twine(rows) + "x" + Twine(cols) + "_" + Twine(dstRows) +
            (transpose ? "t" : ""))
            .str();
    Operation *parent = rewriter.getInsertionBlock()->getParentOp();
    Value found;
    parent->walk([&](Operation *op) {
      if (!found && op->getName().getStringRef() == "krnl.global")
        if (auto n = op->getAttrOfType<StringAttr>("name"))
          if (n.getValue() == imageName)
            found = op->getResult(0);
    });
    if (found)
      return found;

    const int64_t ld = cast<MemRefType>(source.getType()).getShape().back();
    SmallVector<APFloat> all(dense.getValues<APFloat>());
    APFloat zero = APFloat::getZero(APFloat::IEEEhalf());
    SmallVector<APFloat> image(dstRows * kK, zero);
    const int64_t outRows = transpose ? cols : rows;
    const int64_t outCols = transpose ? rows : cols;
    for (int64_t i = 0; i < outRows; ++i)
      for (int64_t j = 0; j < outCols; ++j) {
        const int64_t si = transpose ? j : i, sj = transpose ? i : j;
        image[i * kK + j] = all[(rowOff + si) * ld + colOff + sj];
      }
    Type f16 = rewriter.getF16Type();
    auto tensorType = RankedTensorType::get({dstRows, kK}, f16);
    OperationState state(loc, "krnl.global");
    state.addAttribute("shape", rewriter.getI64ArrayAttr({dstRows, kK}));
    state.addAttribute("name", rewriter.getStringAttr(imageName));
    state.addAttribute("value", DenseElementsAttr::get(tensorType, image));
    state.addAttribute("alignment", rewriter.getI64IntegerAttr(16));
    state.addTypes(MemRefType::get({dstRows, kK}, f16));
    return rewriter.create(state)->getResult(0);
  }

  NamedAttribute attr(StringRef name, int64_t value) {
    return rewriter.getNamedAttr(
        name, rewriter.getI32IntegerAttr(static_cast<int32_t>(value)));
  }

private:
  ConversionPatternRewriter &rewriter;
  Location loc;
  std::string block;
};

// All users of `value` are `op` itself: the value may be overwritten by op.
inline bool onlyUsedBy(Value value, Operation *op) {
  return llvm::all_of(
      value.getUsers(), [&](Operation *user) { return user == op; });
}

inline std::string blockName(Operation *op, StringRef fallback) {
  if (auto name = op->getAttrOfType<StringAttr>("onnx_node_name"))
    return name.str();
  if (auto loc = dyn_cast<NameLoc>(op->getLoc()))
    return loc.getName().str();
  return fallback.str();
}


} // namespace spm
} // namespace spade
