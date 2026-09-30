/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MatMulAdd.cpp - Lowering ONNX MatMul / Add to AISLE ---------===//
//
// onnx.MatMul -> aisle.MatMul   for f16  [.., 12, 16K] x [16K, 16N]
// onnx.Add    -> aisle.Add      for f16  [.., 12, 16]  +  [.., 12, 16]
// onnx.Gemm   -> aisle.GEMM     for f16  [12, 16K] x [16K, 16N] (+ [12, 16N]),
//                               alpha = beta = 1, transA = 0, any transB
// (MatMul and Gemm also take M < 12 rows and one N < 16 column tile)
// onnx.ReduceMean -> aisle.ReduceMean  f16 [.., L <= 12, D <= 16] over L
//
// onnx-mlir canonicalizes a rank-2 MatMul followed by an Add into onnx.Gemm,
// so Gemm is how the radar input projection + positional encoding usually
// arrives.  Products are kept whole here; the aisle-tile pass splits them
// into native RedMulE launches on aisle.Window views (C becomes the Y preload
// of the first K-tile).  Slicing at the ONNX level would need onnx.Slice,
// which has no RedMulE lowering and is not legal inside this conversion.
//
// ("..": leading dimensions of size 1.)  All are executed on RedMulE by
// AISLEToAISMEM (Y = X.W + Y; MatMul with Y = 0, Add with X = identity).
// Anything else is left to the Krnl lowering.  Residual Adds that follow a
// transformer block were already folded into that block by
// fuseTransformerBlocks() and never reach this pattern.
//
//===----------------------------------------------------------------------===//

#include "../helper.hpp"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#define DEBUG_TYPE "ONNXToAISLE_MatMulAdd"

using namespace mlir;

namespace spade {

namespace {

constexpr int64_t kRows = 12; // one RedMulE tile: Y[12 x 16]
constexpr int64_t kCols = 16;

// M rows of a product: at most one RedMulE tile (12); fewer rows run as a
// zero-padded tile.  N columns: whole 16-wide tiles, or one narrower tile.
bool fitsRows(int64_t m) { return m >= 1 && m <= kRows; }
bool fitsCols(int64_t n) { return n >= 1 && (n % kCols == 0 || n < kCols); }

// Keep the ONNX node name: it names the tiles and their SPM buffers.
void copyName(Operation *from, Operation *to) {
  if (Attribute name = from->getAttr("onnx_node_name"))
    to->setAttr("onnx_node_name", name);
}

// f16, static, rank >= 2, leading dims 1; returns the last two dims.
bool asF16Matrix(Value value, int64_t &rows, int64_t &cols) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 2 ||
      !type.getElementType().isF16())
    return false;
  for (int64_t i = 0; i + 2 < type.getRank(); ++i)
    if (type.getDimSize(i) != 1)
      return false;
  rows = type.getShape()[type.getRank() - 2];
  cols = type.getShape().back();
  return true;
}

} // namespace

struct ONNXMatMulToAISLE : public ConversionPattern {
  ONNXMatMulToAISLE(MLIRContext *ctx)
      : ConversionPattern(ONNXMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto oldOp = cast<ONNXMatMulOp>(op);
    int64_t m, k, kb, n, ym, yn;
    if (!asF16Matrix(oldOp.getA(), m, k) || !asF16Matrix(oldOp.getB(), kb, n) ||
        !asF16Matrix(oldOp.getY(), ym, yn))
      return rewriter.notifyMatchFailure(op, "needs static f16 matrices");
    auto bType = cast<RankedTensorType>(oldOp.getB().getType());
    if (bType.getRank() != 2 || !fitsRows(m) || !fitsCols(n) || kb != k ||
        k % kCols != 0 || k == 0)
      return rewriter.notifyMatchFailure(
          op, "RedMulE MatMul needs [M <= 12, 16K] x [16K, 16N or N < 16]");

    ONNXMatMulOpAdaptor adaptor(operands);
    SmallVector<Value> newOperands{adaptor.getA(),
        onnx_to_aisle::create<ONNXMatMulOpAdaptor>(
            rewriter, oldOp, "A_shape", &ONNXMatMulOpAdaptor::getA),
        adaptor.getB(),
        onnx_to_aisle::create<ONNXMatMulOpAdaptor>(
            rewriter, oldOp, "B_shape", &ONNXMatMulOpAdaptor::getB)};
    auto newOp = rewriter.create<spade::AISLEMatMulOp>(op->getLoc(),
        TypeRange{oldOp.getY().getType()}, newOperands,
        ArrayRef<NamedAttribute>{});
    copyName(op, newOp);
    rewriter.replaceOp(op, newOp.getY());
    return success();
  }
};

struct ONNXAddToAISLE : public ConversionPattern {
  ONNXAddToAISLE(MLIRContext *ctx)
      : ConversionPattern(ONNXAddOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto oldOp = cast<ONNXAddOp>(op);
    int64_t ar, ac, br, bc, cr, cc;
    if (!asF16Matrix(oldOp.getA(), ar, ac) ||
        !asF16Matrix(oldOp.getB(), br, bc) ||
        !asF16Matrix(oldOp.getC(), cr, cc))
      return rewriter.notifyMatchFailure(op, "needs static f16 matrices");
    if (ar != kRows || br != kRows || cr != kRows || ac != kCols ||
        bc != kCols || cc != kCols)
      return rewriter.notifyMatchFailure(
          op, "RedMulE Add needs [12, 16] + [12, 16]");

    ONNXAddOpAdaptor adaptor(operands);
    SmallVector<Value> newOperands{adaptor.getA(),
        onnx_to_aisle::create<ONNXAddOpAdaptor>(
            rewriter, oldOp, "A_shape", &ONNXAddOpAdaptor::getA),
        adaptor.getB(),
        onnx_to_aisle::create<ONNXAddOpAdaptor>(
            rewriter, oldOp, "B_shape", &ONNXAddOpAdaptor::getB)};
    auto newOp = rewriter.create<spade::AISLEAddOp>(op->getLoc(),
        TypeRange{oldOp.getC().getType()}, newOperands,
        ArrayRef<NamedAttribute>{});
    copyName(op, newOp);
    rewriter.replaceOp(op, newOp.getC());
    return success();
  }
};

// Benefit 2: tried before the generic onnx.Gemm pattern of GEMM.cpp.
struct ONNXGemmToAISLE : public ConversionPattern {
  ONNXGemmToAISLE(MLIRContext *ctx)
      : ConversionPattern(ONNXGemmOp::getOperationName(), 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto oldOp = cast<ONNXGemmOp>(op);
    if (oldOp.getAlpha().convertToDouble() != 1.0 ||
        oldOp.getBeta().convertToDouble() != 1.0 || oldOp.getTransA() != 0)
      return rewriter.notifyMatchFailure(
          op, "RedMulE Gemm needs alpha = beta = 1, transA = 0");
    const bool transB = oldOp.getTransB() != 0;
    int64_t m, k, rb, cb, ym, yn;
    auto aType = dyn_cast<RankedTensorType>(oldOp.getA().getType());
    auto bType = dyn_cast<RankedTensorType>(oldOp.getB().getType());
    if (!aType || !bType || aType.getRank() != 2 || bType.getRank() != 2 ||
        !asF16Matrix(oldOp.getA(), m, k) || !asF16Matrix(oldOp.getB(), rb, cb) ||
        !asF16Matrix(oldOp.getY(), ym, yn))
      return rewriter.notifyMatchFailure(op, "needs static rank-2 f16");
    const int64_t kb = transB ? cb : rb, n = transB ? rb : cb;
    if (!fitsRows(m) || !fitsCols(n) || kb != k || k % kCols != 0 || k == 0)
      return rewriter.notifyMatchFailure(
          op, "RedMulE Gemm needs [M <= 12, 16K] x [16K, 16N or N < 16]");
    const bool hasC = !isa<NoneType>(oldOp.getC().getType());
    if (hasC) {
      int64_t cr, cc;
      auto cType = dyn_cast<RankedTensorType>(oldOp.getC().getType());
      if (!cType || cType.getRank() != 2 || !asF16Matrix(oldOp.getC(), cr, cc) ||
          cr != m || cc != n)
        return rewriter.notifyMatchFailure(
            op, "RedMulE Gemm needs C of shape [M, N] (no broadcast)");
    } else if (transB) {
      return rewriter.notifyMatchFailure(op, "transB without C");
    }

    ONNXGemmOpAdaptor adaptor(operands);
    Value aShape = onnx_to_aisle::create<ONNXGemmOpAdaptor>(
        rewriter, oldOp, "A_shape", &ONNXGemmOpAdaptor::getA);
    Value bShape = onnx_to_aisle::create<ONNXGemmOpAdaptor>(
        rewriter, oldOp, "B_shape", &ONNXGemmOpAdaptor::getB);
    Operation *newOp;
    if (hasC) {
      SmallVector<Value> newOperands{adaptor.getA(), aShape, adaptor.getB(),
          bShape, adaptor.getC()};
      SmallVector<NamedAttribute> attrs{
          rewriter.getNamedAttr("transA", oldOp.getTransAAttr()),
          rewriter.getNamedAttr("transB", oldOp.getTransBAttr())};
      newOp = rewriter.create<spade::AISLEGEMMOp>(op->getLoc(),
          TypeRange{oldOp.getY().getType()}, newOperands, attrs);
    } else {
      SmallVector<Value> newOperands{adaptor.getA(), aShape, adaptor.getB(),
          bShape};
      newOp = rewriter.create<spade::AISLEMatMulOp>(op->getLoc(),
          TypeRange{oldOp.getY().getType()}, newOperands,
          ArrayRef<NamedAttribute>{});
    }
    copyName(op, newOp);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

// onnx.ReduceMean over the frames axis of an f16 [.., L, D] tensor
// (L <= 12, D <= 16, leading dims 1): -> aisle.ReduceMean, executed on
// RedMulE as POOL . pad16(X) with POOL = fp16(1/L) (see AISLEToAISMEM).
struct ONNXReduceMeanToAISLE : public ConversionPattern {
  ONNXReduceMeanToAISLE(MLIRContext *ctx)
      : ConversionPattern(ONNXReduceMeanOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto oldOp = cast<ONNXReduceMeanOp>(op);
    int64_t l, d;
    auto type = dyn_cast<RankedTensorType>(oldOp.getData().getType());
    if (!asF16Matrix(oldOp.getData(), l, d) || !fitsRows(l) || d > kCols)
      return rewriter.notifyMatchFailure(
          op, "RedMulE ReduceMean needs f16 [.., L <= 12, D <= 16]");
    // The reduced axis must be the frames axis (rank - 2), given as a
    // constant.
    auto axesOp = oldOp.getAxes().getDefiningOp<ONNXConstantOp>();
    auto axes = axesOp ? dyn_cast_or_null<DenseElementsAttr>(
                             axesOp.getValueAttr())
                       : DenseElementsAttr();
    const int64_t rank = type.getRank();
    if (!axes || axes.getNumElements() != 1)
      return rewriter.notifyMatchFailure(op, "needs one constant axis");
    int64_t axis = (*axes.getValues<APInt>().begin()).getSExtValue();
    if (axis < 0)
      axis += rank;
    if (axis != rank - 2)
      return rewriter.notifyMatchFailure(op, "only the frames axis (-2)");
    auto outType = dyn_cast<RankedTensorType>(oldOp.getReduced().getType());
    if (!outType || !outType.hasStaticShape() || outType.getShape().back() != d)
      return rewriter.notifyMatchFailure(op, "needs a static result");

    ONNXReduceMeanOpAdaptor adaptor(operands);
    Value xShape = onnx_to_aisle::create<ONNXReduceMeanOpAdaptor>(
        rewriter, oldOp, "X_shape", &ONNXReduceMeanOpAdaptor::getData);
    auto axesType = RankedTensorType::get({1, 1}, rewriter.getI32Type());
    Value axesValue = rewriter.create<spade::AISLEQConstantOp>(op->getLoc(),
        "Axes", DenseElementsAttr::get(axesType,
                    ArrayRef<int32_t>{static_cast<int32_t>(axis)}));
    auto newOp = rewriter.create<spade::AISLEReduceMeanOp>(op->getLoc(),
        TypeRange{outType}, ValueRange{adaptor.getData(), xShape, axesValue},
        ArrayRef<NamedAttribute>{});
    copyName(op, newOp);
    rewriter.replaceOp(op, newOp.getY());
    return success();
  }
};

void populateLoweringONNXToAISLEMatMulAddOpPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<ONNXMatMulToAISLE, ONNXAddToAISLE, ONNXGemmToAISLE,
      ONNXReduceMeanToAISLE>(ctx);
}

} // namespace spade
