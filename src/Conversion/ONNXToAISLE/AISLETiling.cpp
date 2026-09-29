/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AISLETiling.cpp - Split AISLE products into RedMulE tiles ----===//
//
// aisle-tile rewrites every f16 aisle.MatMul / aisle.GEMM that is larger than
// one RedMulE launch into native launches (Y[12x16] (+)= X[12x16] . W[16x16])
// on aisle.Window views of its operands:
//
//   Y[12 x 16N] = A[12 x 16K] . B[16K x 16N] (+ C)
//
//   for n < N:                       one output tile, independent
//     acc = Window(C, [0, 16n])      (GEMM)  |  none (MatMul)
//     for k < K:                     chained: every launch accumulates
//       acc = GEMM(Window(A, [0, 16k]), Window(B, [16k, 16n]), acc)
//             (MatMul for the first launch without C: Y = 0)
//   Y = Concat(acc_0 .. acc_N-1, axis = last)
//
// transB = 1 reads Window(B, [16n, 16k]) and keeps transB on the tile.  A
// Window of a Concat that selects one tile folds to that tile, so a product
// that consumes a tiled product reads its SPM tiles directly.
//
// Constant operands are split at compile time instead: a window of an
// onnx.Constant whose every user is a product being tiled becomes its own
// onnx.Constant holding just that tile (for transB, already transposed, so
// the tile needs no transposing upload).  Every constant tile is then a
// contiguous 16x16 (W) or 12x16 (C) global: the uploads take the runtime's
// plain-copy path, and the original constant, now unused, is dropped.
//
// All tiles run on RedMulE instance 0 for now; the tile GEMMs carry
// `onnx_node_name = "<op>[n<n>,k<k>]"`, which also names their SPM buffers.
// M is not tiled (A must have 12 rows).  Anything else is left untouched:
// the legacy AISLE GEMM lowering or Krnl handle it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialect/AISLE/AISLEDialect.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Twine.h"

#include <optional>
#include <string>

#define DEBUG_TYPE "aisle-tile"

using namespace mlir;

namespace spade {

namespace {

constexpr int64_t kRows = 12; // M of one RedMulE launch
constexpr int64_t kTile = 16; // K and N of one RedMulE launch

// f16, static, rank >= 2 with leading dims of 1: the last two dims.
bool f16Matrix(Value value, int64_t &rows, int64_t &cols) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 2 ||
      !type.getElementType().isF16())
    return false;
  for (int64_t d = 0; d + 2 < type.getRank(); ++d)
    if (type.getDimSize(d) != 1)
      return false;
  rows = type.getDimSize(type.getRank() - 2);
  cols = type.getDimSize(type.getRank() - 1);
  return true;
}

struct Product {
  Operation *op = nullptr;
  Value a, b, c;
  bool transB = false;
  int64_t k = 0, n = 0;
};

// A MatMul/GEMM that fits the RedMulE tiling but is not one launch already.
std::optional<Product> tileable(Operation *op) {
  Product p;
  p.op = op;
  if (auto mm = dyn_cast<AISLEMatMulOp>(op)) {
    p.a = mm.getA();
    p.b = mm.getB();
  } else if (auto gemm = dyn_cast<AISLEGEMMOp>(op)) {
    if (gemm.getTransA() != 0)
      return std::nullopt;
    p.a = gemm.getA();
    p.b = gemm.getB();
    p.c = gemm.getC();
    p.transB = gemm.getTransB() != 0;
  } else {
    return std::nullopt;
  }
  int64_t m, k, br, bc, ym, yn;
  if (!f16Matrix(p.a, m, k) || !f16Matrix(p.b, br, bc) ||
      !f16Matrix(op->getResult(0), ym, yn) ||
      cast<RankedTensorType>(p.b.getType()).getRank() != 2)
    return std::nullopt;
  const int64_t kb = p.transB ? bc : br, n = p.transB ? br : bc;
  if (m != kRows || kb != k || k % kTile != 0 || n % kTile != 0 || k == 0 ||
      n == 0 || ym != kRows || yn != n)
    return std::nullopt;
  if (p.c) {
    int64_t cr, cc;
    if (!f16Matrix(p.c, cr, cc) || cr != kRows || cc != n ||
        p.c.getType() != op->getResult(0).getType())
      return std::nullopt; // broadcast C: not handled here
  }
  if (k == kTile && n == kTile)
    return std::nullopt; // already one launch
  p.k = k / kTile;
  p.n = n / kTile;
  return p;
}

// Constants that may be split into tiles: dense onnx.Constant ops used only
// by products being tiled (so the whole constant dies after tiling).
using SplitSet = llvm::DenseSet<Operation *>;

// The dense f16/f32/f64 value of a splittable constant.
DenseElementsAttr splittable(Value value, const SplitSet &split) {
  auto constant = value.getDefiningOp<ONNXConstantOp>();
  if (!constant || !split.contains(constant))
    return {};
  auto dense = dyn_cast_or_null<DenseElementsAttr>(constant.getValueAttr());
  if (!dense || !isa<FloatType>(dense.getElementType()))
    return {};
  return dense;
}

// rows x cols block at (r, c) of the last two dims of `dense` (leading dims
// are 1), optionally transposed, as a new onnx.Constant.
Value constantTile(OpBuilder &b, Location loc, DenseElementsAttr dense,
    int64_t r, int64_t c, int64_t rows, int64_t cols, bool transpose) {
  auto type = cast<RankedTensorType>(dense.getType());
  const int64_t ld = type.getShape().back();
  SmallVector<APFloat> all(dense.getValues<APFloat>());
  SmallVector<APFloat> tile;
  const int64_t outRows = transpose ? cols : rows;
  const int64_t outCols = transpose ? rows : cols;
  for (int64_t i = 0; i < outRows; ++i)
    for (int64_t j = 0; j < outCols; ++j) {
      const int64_t si = transpose ? j : i, sj = transpose ? i : j;
      tile.push_back(all[(r + si) * ld + c + sj]);
    }
  SmallVector<int64_t> shape(type.getShape());
  shape[shape.size() - 2] = outRows;
  shape.back() = outCols;
  auto tileType = RankedTensorType::get(shape, type.getElementType());
  return b.create<ONNXConstantOp>(loc, TypeRange{tileType}, ValueRange{},
      ArrayRef<NamedAttribute>{b.getNamedAttr(
          "value", DenseElementsAttr::get(tileType, tile))});
}

// `value` restricted to rows [r, r + rows) and columns [c, c + cols) of its
// last two dims (leading dims are 1).  No op when that is all of it; a new
// constant when `value` is a splittable constant (transposed on request),
// else an aisle.Window.  `transposed` tells whether the result was.
Value window(OpBuilder &b, Location loc, Value value, int64_t r, int64_t c,
    int64_t rows, int64_t cols, const SplitSet &split,
    bool transpose = false, bool *transposed = nullptr) {
  if (transposed)
    *transposed = false;
  if (DenseElementsAttr dense = splittable(value, split)) {
    auto type = cast<RankedTensorType>(value.getType());
    const bool whole = type.getShape().back() == cols &&
                       type.getShape()[type.getRank() - 2] == rows;
    if (!whole || transpose) {
      if (transposed)
        *transposed = transpose;
      return constantTile(b, loc, dense, r, c, rows, cols, transpose);
    }
  }
  auto type = cast<RankedTensorType>(value.getType());
  const int64_t rank = type.getRank();
  SmallVector<int64_t> shape(type.getShape());
  SmallVector<int64_t> offsets(rank, 0);
  shape[rank - 2] = rows;
  shape[rank - 1] = cols;
  offsets[rank - 2] = r;
  offsets[rank - 1] = c;
  auto windowType = RankedTensorType::get(shape, type.getElementType());
  if (windowType == type)
    return value;
  return b.createOrFold<AISLEWindowOp>(
      loc, windowType, value, b.getDenseI64ArrayAttr(offsets));
}

// The rank-4 [1, 4] i32 shape operand of the AISLE ops.
Value shapeOperand(OpBuilder &b, Location loc, const char *name, Value of) {
  auto type = cast<RankedTensorType>(of.getType());
  SmallVector<int32_t> dims(4 - type.getRank(), 1);
  for (int64_t d : type.getShape())
    dims.push_back(static_cast<int32_t>(d));
  auto attrType = RankedTensorType::get({1, 4}, b.getI32Type());
  return b.create<AISLEQConstantOp>(loc, name,
      DenseElementsAttr::get(attrType, ArrayRef<int32_t>(dims)));
}

std::string baseName(Operation *op) {
  if (auto name = op->getAttrOfType<StringAttr>("onnx_node_name"))
    return name.str();
  return isa<AISLEGEMMOp>(op) ? "gemm" : "matmul";
}

void tile(const Product &p, const SplitSet &split) {
  Operation *op = p.op;
  Location loc = op->getLoc();
  OpBuilder b(op);
  auto yType = cast<RankedTensorType>(op->getResult(0).getType());
  SmallVector<int64_t> tileShape(yType.getShape());
  tileShape.back() = kTile;
  auto tileType = RankedTensorType::get(tileShape, yType.getElementType());
  const std::string base = baseName(op);
  auto si64 = [&](int64_t v) {
    return b.getIntegerAttr(
        IntegerType::get(b.getContext(), 64, IntegerType::Signed), v);
  };

  SmallVector<Value> tiles;
  for (int64_t n = 0; n < p.n; ++n) {
    Value acc =
        p.c ? window(b, loc, p.c, 0, n * kTile, kRows, kTile, split) : Value();
    for (int64_t k = 0; k < p.k; ++k) {
      Value a = window(b, loc, p.a, 0, k * kTile, kRows, kTile, split);
      // transB: a constant B is transposed here, at compile time.
      bool preTransposed = false;
      Value w = p.transB ? window(b, loc, p.b, n * kTile, k * kTile, kTile,
                               kTile, split, /*transpose=*/true, &preTransposed)
                         : window(b, loc, p.b, k * kTile, n * kTile, kTile,
                               kTile, split);
      const bool tileTransB = p.transB && !preTransposed;
      Value aShape = shapeOperand(b, loc, "A_shape", a);
      Value wShape = shapeOperand(b, loc, "B_shape", w);
      Operation *launch;
      if (acc) {
        launch = b.create<AISLEGEMMOp>(loc, TypeRange{tileType},
            ValueRange{a, aShape, w, wShape, acc},
            ArrayRef<NamedAttribute>{b.getNamedAttr("transA", si64(0)),
                b.getNamedAttr("transB", si64(tileTransB ? 1 : 0))});
      } else {
        assert(!tileTransB && "aisle-tile: MatMul has no transB");
        launch = b.create<AISLEMatMulOp>(loc, TypeRange{tileType},
            ValueRange{a, aShape, w, wShape}, ArrayRef<NamedAttribute>{});
      }
      launch->setAttr("onnx_node_name",
          b.getStringAttr(base + "[n" + std::to_string(n) + ",k" +
                          std::to_string(k) + "]"));
      acc = launch->getResult(0);
    }
    tiles.push_back(acc);
  }
  Value result = tiles.front();
  if (tiles.size() > 1)
    result = b.create<AISLEConcatOp>(loc, TypeRange{yType}, ValueRange(tiles),
                  ArrayRef<NamedAttribute>{b.getNamedAttr(
                      "axis", b.getI64IntegerAttr(yType.getRank() - 1))})
                 .getResult();
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
}

struct AISLETilingPass
    : public PassWrapper<AISLETilingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AISLETilingPass)

  StringRef getArgument() const override { return "aisle-tile"; }
  StringRef getDescription() const override {
    return "Split f16 aisle.MatMul / aisle.GEMM into native RedMulE launches "
           "(12x16x16) on aisle.Window views, chained over K, concatenated "
           "over N.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spade::AISLEDialect>();
  }

  void runOnOperation() final {
    // Program order: a producer is tiled before its consumers, so their
    // Windows see its Concat and fold to its tiles.
    SmallVector<Product> products;
    llvm::DenseSet<Operation *> tiled;
    getOperation().walk([&](Operation *op) {
      if (std::optional<Product> p = tileable(op)) {
        products.push_back(*p);
        tiled.insert(op);
      }
    });
    // Constants read only by products being tiled are split into tiles.
    SplitSet split;
    getOperation().walk([&](ONNXConstantOp constant) {
      if (!constant->use_empty() &&
          llvm::all_of(constant->getUsers(),
              [&](Operation *user) { return tiled.contains(user); }))
        split.insert(constant);
    });
    for (const Product &p : products) {
      // Operands may have been replaced by an earlier rewrite.
      std::optional<Product> now = tileable(p.op);
      if (now)
        tile(*now, split);
    }
    // The split constants are unused now.
    for (Operation *constant : split)
      if (constant->use_empty())
        constant->erase();
  }
};

} // namespace

std::unique_ptr<Pass> createAISLETilingPass() {
  return std::make_unique<AISLETilingPass>();
}

} // namespace spade
