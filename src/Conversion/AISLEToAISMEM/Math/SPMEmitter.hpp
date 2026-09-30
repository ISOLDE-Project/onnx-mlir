/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SPMEmitter.hpp - RedMulE schedules on SPM buffers --------===//
//
// Shared by the AISLE -> AISMEM RedMulE lowerings (transformer blocks,
// MatMul, Add): operand classification (SPM-resident or data memory) and a
// thin emitter for aismem.SPMAlloc buffers and the scheduled RedMulE / SPM
// operations of a tile (see Emitter).
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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Twine.h"

#include <cassert>
#include <optional>
#include <string>

namespace spade {
namespace spm {

using namespace mlir;

// One native RedMulE launch: Y[M x K] (+)= X[M x N] . W[N x K].
constexpr int32_t kM = 12; // rows of X and Y (ARRAY_HEIGHT * PIPE_REGS)
constexpr int32_t kN = 16; // reduction length, rows of W
constexpr int32_t kK = 16; // columns of W and Y (one SPM row)
constexpr int32_t kTile = 0; // default RedMulE instance

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
// tile (RedMulE instance `tile()`, kTile by default).  Every buffer it
// allocates and every operation it emits carries that tile; `on(t)` gives an
// emitter for another tile of the same block.  A RedMulE instance only sees
// its own SPM: X, W and Y of a launch must be on the launching tile, and
// `move()` brings an SPM value over from another tile.
//
// gemm() launches and waits at once (the single-tile chain); launch() and
// wait() let independent launches on different tiles run concurrently.
class Emitter {
public:
  Emitter(ConversionPatternRewriter &rewriter, Location loc, StringRef block,
      int32_t tile = kTile)
      : rewriter(rewriter), loc(loc), block(block.str()), tile_(tile) {}

  // The same block on RedMulE instance `t`.
  Emitter on(int32_t t) const { return Emitter(rewriter, loc, block, t); }
  int32_t tile() const { return tile_; }

  Value alloc(int64_t rows, const Twine &name, bool resident = false) {
    SmallVector<NamedAttribute> attrs{attr("tile", tile_), attr("rows", rows),
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
    SmallVector<NamedAttribute> attrs{attr("tile", tile_),
        attr("row_offset", rowOff), attr("col_offset", colOff),
        attr("rows", rows), attr("cols", cols), attr("dst_rows", dstRows),
        attr("dst_cols", kK),
        rewriter.getNamedAttr("transpose", rewriter.getBoolAttr(transpose)),
        rewriter.getNamedAttr("relu", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("negate", rewriter.getBoolAttr(false))};
    auto op = rewriter.create<AISMEMRedMulEUploadTileOp>(loc,
        TypeRange{rewriter.getI32Type(), rewriter.getNoneType()}, operands,
        attrs);
    return {address, op.getNoneVal(), tile_};
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
  // `cols` < 16 (not transposed): a W narrower than 16 columns, zero padded.
  Value weightWindow(const Matrix &m, int64_t rowOff, int64_t colOff,
      int64_t rows, const Twine &name, bool transpose = false,
      int64_t cols = kK) {
    return upload(m, rowOff, colOff, rows, cols, kN, name,
        isConstant(m.memref), ValueRange{}, transpose)
        .address;
  }

  // A 12-row SPM value as a W operand: copied into a zero-filled 16-row
  // buffer (rows 12..15 must be zero).
  SPMValue padded(const SPMValue &v, const Twine &name, ValueRange deps) {
    if (v.tile != tile_) // the move pads to 16 rows itself
      return move(v, kM, kN, name, /*transpose=*/false, deps);
    Value w = alloc(kN, name);
    SmallVector<Value> all(deps.begin(), deps.end());
    all.push_back(v.token);
    Value z = zero(w, kN, all);
    return {w, copy(v.address, w, kM, z), tile_};
  }

  // A constant 12 x 16 X operand, resident: `value(i, j)` for row i, column
  // j.  One buffer per name and function (per tile: the name carries it).
  Value residentMatrix(StringRef baseName, StringRef globalBase,
      llvm::function_ref<float(int64_t, int64_t)> value) {
    const std::string bufferName =
        tile_ == 0 ? baseName.str() : (baseName + ".t" + Twine(tile_)).str();
    const std::string globalName =
        tile_ == 0 ? globalBase.str()
                   : (globalBase + "_t" + Twine(tile_)).str();
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
        APFloat v(value(i, j));
        bool lost;
        v.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &lost);
        values.push_back(v);
      }
    OperationState state(loc, "krnl.global");
    state.addAttribute("shape", rewriter.getI64ArrayAttr({kM, kN}));
    state.addAttribute("name", rewriter.getStringAttr(globalName));
    state.addAttribute("value", DenseElementsAttr::get(tensorType, values));
    state.addAttribute("alignment", rewriter.getI64IntegerAttr(16));
    state.addTypes(MemRefType::get({kM, kN}, f16));
    Value global = rewriter.create(state)->getResult(0);

    SmallVector<NamedAttribute> attrs{attr("tile", tile_), attr("rows", kM),
        rewriter.getNamedAttr("resident", rewriter.getBoolAttr(true)),
        rewriter.getNamedAttr("name", rewriter.getStringAttr(bufferName))};
    Value address = rewriter
                        .create<AISMEMSPMAllocOp>(loc,
                            TypeRange{rewriter.getI32Type()}, ValueRange{},
                            attrs)
                        .getAddress();
    SmallVector<NamedAttribute> uattrs{attr("tile", tile_),
        attr("row_offset", 0), attr("col_offset", 0), attr("rows", kM),
        attr("cols", kN), attr("dst_rows", kM), attr("dst_cols", kK),
        rewriter.getNamedAttr("transpose", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("relu", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("negate", rewriter.getBoolAttr(false))};
    rewriter.create<AISMEMRedMulEUploadTileOp>(loc,
        TypeRange{rewriter.getI32Type(), rewriter.getNoneType()},
        ValueRange{global, address}, uattrs);
    return address;
  }

  // The 12x16 identity (ones on the diagonal) as a resident X operand, so
  // that Y = I . W + Y adds W's first 12 rows to Y.  One per function.
  Value identity() {
    return residentMatrix("aismem.identity12x16", "aismem_identity_12x16",
        [](int64_t i, int64_t j) { return i == j ? 1.0f : 0.0f; });
  }

  // The mean over L rows as a resident X operand: fp16(1/L) in columns
  // 0..L-1 of every row, so that Y = POOL . pad16(H) puts the mean of H's
  // L rows into every row of Y (radar_attention's tf_pool).
  Value meanPool(int64_t l) {
    const float inv = 1.0f / static_cast<float>(l);
    return residentMatrix(("aismem.meanpool" + Twine(l)).str(),
        ("aismem_meanpool_" + Twine(l)).str(),
        [&](int64_t, int64_t j) { return j < l ? inv : 0.0f; });
  }

  // Block operand as an SPM buffer on this tile: as is, moved over from
  // another tile, or uploaded from data memory.
  SPMValue activation(const Operand &o, const Twine &name) {
    if (o.spm)
      return o.spm->tile == tile_ ? *o.spm : move(*o.spm, kM, kM, name);
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
                attr("tile", tile_), attr("elements", rows * kK)})
        .getNoneVal();
  }

  // Launch Y (+)= X . W on this tile; the result is valid after a wait()
  // on the returned launch token.
  Value launch(Value x, Value w, Value y, ValueRange deps) {
    SmallVector<Value> operands{x, w, y};
    operands.append(deps.begin(), deps.end());
    return rewriter
        .create<AISMEMRedMulEGEMMOp>(loc, TypeRange{rewriter.getNoneType()},
            operands,
            ArrayRef<NamedAttribute>{attr("tile", tile_), attr("k", kK),
                attr("m", kM), attr("n", kN)})
        .getNoneVal();
  }

  // One barrier for launches on any tiles (mask = their tiles).
  Value wait(ValueRange launched) {
    int64_t mask = 0;
    for (Value t : launched)
      if (auto gemm = t.getDefiningOp<AISMEMRedMulEGEMMOp>())
        mask |= int64_t(1) << gemm.getTile();
    assert(mask != 0 && "wait() needs RedMulEGEMM launch tokens");
    return rewriter
        .create<AISMEMRedMulEWaitOp>(loc, TypeRange{rewriter.getNoneType()},
            launched, ArrayRef<NamedAttribute>{attr("mask", mask)})
        .getNoneVal();
  }

  // Launch Y (+)= X . W and wait for it.
  Value gemm(Value x, Value w, Value y, ValueRange deps) {
    return wait(ValueRange{launch(x, w, y, deps)});
  }

  // An SPM value of another tile, rows x 16, brought to a fresh buffer of
  // dstRows rows on this tile (zero padded; transposed on request, then the
  // result has 16 rows of `rows` columns).  Goes through data memory: the
  // tiles have no SPM-to-SPM path.  The source tile must be idle.
  SPMValue move(const SPMValue &v, int64_t rows, int64_t dstRows,
      const Twine &name, bool transpose = false, ValueRange deps = {}) {
    Value dst = alloc(dstRows, name);
    SmallVector<Value> operands{v.address, dst};
    operands.append(deps.begin(), deps.end());
    operands.push_back(v.token);
    Value t = rewriter
                  .create<AISMEMSPMMoveTileOp>(loc,
                      TypeRange{rewriter.getNoneType()}, operands,
                      ArrayRef<NamedAttribute>{attr("src_tile", v.tile),
                          attr("dst_tile", tile_), attr("rows", rows),
                          attr("dst_rows", dstRows),
                          rewriter.getNamedAttr(
                              "transpose", rewriter.getBoolAttr(transpose))})
                  .getNoneVal();
    return {dst, t, tile_};
  }

  Value relu(Value address, int64_t rows, ValueRange deps) {
    SmallVector<Value> operands{address};
    operands.append(deps.begin(), deps.end());
    return rewriter
        .create<AISMEMSPMReluOp>(loc, TypeRange{rewriter.getNoneType()},
            operands,
            ArrayRef<NamedAttribute>{attr("tile", tile_), attr("rows", rows)})
        .getNoneVal();
  }

  Value transpose(Value source, Value destination, int64_t rows,
      int64_t dstRows, ValueRange deps) {
    SmallVector<Value> operands{source, destination};
    operands.append(deps.begin(), deps.end());
    return rewriter
        .create<AISMEMSPMTransposeOp>(loc, TypeRange{rewriter.getNoneType()},
            operands,
            ArrayRef<NamedAttribute>{attr("tile", tile_), attr("rows", rows),
                attr("dst_rows", dstRows)})
        .getNoneVal();
  }

  Value copy(Value source, Value destination, int64_t rows, ValueRange deps) {
    SmallVector<Value> operands{source, destination};
    operands.append(deps.begin(), deps.end());
    return rewriter
        .create<AISMEMSPMCopyOp>(loc, TypeRange{rewriter.getNoneType()},
            operands,
            ArrayRef<NamedAttribute>{attr("tile", tile_), attr("rows", rows)})
        .getNoneVal();
  }

  // The accumulator Y for `Y = ... + C`: C's own buffer when it may be
  // overwritten, else a fresh copy; zeros without C.
  SPMValue accumulator(const std::optional<Operand> &c, bool cIsDead,
      ValueRange deps) {
    if (!c) {
      Value y = alloc(kM, "Y");
      return {y, zero(y, kM, deps), tile_};
    }
    if (c->host || c->spm->tile != tile_) // uploaded / moved: private copy
      return activation(*c, "Y");
    if (cIsDead) // residual in place
      return *c->spm;
    Value y = alloc(kM, "Y");
    SmallVector<Value> all(deps.begin(), deps.end());
    all.push_back(c->spm->token);
    return {y, copy(c->spm->address, y, kM, all), tile_};
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
  int32_t tile_;
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
