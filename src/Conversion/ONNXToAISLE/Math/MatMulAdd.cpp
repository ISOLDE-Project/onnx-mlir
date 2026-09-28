/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MatMulAdd.cpp - Lowering ONNX MatMul / Add to AISLE ---------===//
//
// onnx.MatMul -> aisle.MatMul   for f16  [.., 12, 16K] x [16K, 16]
// onnx.Add    -> aisle.Add      for f16  [.., 12, 16]  +  [.., 12, 16]
//
// ("..": leading dimensions of size 1.)  Both are executed on RedMulE by
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
    if (bType.getRank() != 2 || m != kRows || n != kCols || kb != k ||
        k % kCols != 0)
      return rewriter.notifyMatchFailure(
          op, "RedMulE MatMul needs [12, 16K] x [16K, 16]");

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
    rewriter.replaceOp(op, newOp.getC());
    return success();
  }
};

void populateLoweringONNXToAISLEMatMulAddOpPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<ONNXMatMulToAISLE, ONNXAddToAISLE>(ctx);
}

} // namespace spade
