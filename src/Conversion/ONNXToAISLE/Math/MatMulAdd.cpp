/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MatMulAdd.cpp - Lowering ONNX MatMul / Add to AISLE ---------===//
//
// onnx.MatMul -> aisle.MatMul   for f16  [.., 12, 16K] x [16K, 16N]
// onnx.Add    -> aisle.Add      for f16  [.., 12, 16]  +  [.., 12, 16]
// onnx.Gemm   -> aisle.GEMM     for f16  [12, 16K] x [16K, 16N] (+ [12, 16N]),
//                               alpha = beta = 1, transA = 0, any transB
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
    if (bType.getRank() != 2 || m != kRows || n % kCols != 0 || n == 0 ||
        kb != k || k % kCols != 0 || k == 0)
      return rewriter.notifyMatchFailure(
          op, "RedMulE MatMul needs [12, 16K] x [16K, 16N]");

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
    if (m != kRows || n % kCols != 0 || n == 0 || kb != k ||
        k % kCols != 0 || k == 0)
      return rewriter.notifyMatchFailure(
          op, "RedMulE Gemm needs [12, 16K] x [16K, 16N]");
    const bool hasC = !isa<NoneType>(oldOp.getC().getType());
    if (hasC) {
      int64_t cr, cc;
      auto cType = dyn_cast<RankedTensorType>(oldOp.getC().getType());
      if (!cType || cType.getRank() != 2 || !asF16Matrix(oldOp.getC(), cr, cc) ||
          cr != kRows || cc != n)
        return rewriter.notifyMatchFailure(
            op, "RedMulE Gemm needs C of shape [12, N] (no broadcast)");
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

void populateLoweringONNXToAISLEMatMulAddOpPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<ONNXMatMulToAISLE, ONNXAddToAISLE, ONNXGemmToAISLE>(ctx);
}

} // namespace spade
