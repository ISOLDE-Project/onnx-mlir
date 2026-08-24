/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ComplexGEMM.cpp - Lowering RedMulE Complex GEMM --------===//
//
// Lower the ISOLDE ONNX extension onnx.RedMulEComplexGemm to the AISLE
// split-complex GEMM operation. This pass intentionally does not decompose
// the operation into real GEMMs; RedMulE scheduling belongs in AISLEToAISMEM.
//
//===----------------------------------------------------------------------===//

#include "../helper.hpp"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#define DEBUG_TYPE "ONNXToAISLE_ComplexGEMM"

using namespace mlir;

namespace spade {

struct ONNXRedMulEComplexGEMMOpLowering : public ConversionPattern {
  using theOperation = mlir::ONNXRedMulEComplexGemmOp;
  using theAdaptor = mlir::ONNXRedMulEComplexGemmOpAdaptor;
  using theNewOp = spade::AISLEComplexGEMMOp;

  ONNXRedMulEComplexGEMMOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    theOperation oldOp = llvm::dyn_cast<theOperation>(op);
    if (!oldOp)
      return failure();

    theAdaptor operandAdaptor(operands);

    Value ar = operandAdaptor.getAr();
    Value ai = operandAdaptor.getAi();
    Value br = operandAdaptor.getBr();
    Value bi = operandAdaptor.getBi();

    // AISLE currently carries explicit shape tensors alongside data tensors.
    Value arShape = onnx_to_aisle::create<theAdaptor>(
        rewriter, oldOp, "Ar_shape", &theAdaptor::getAr);
    Value aiShape = onnx_to_aisle::create<theAdaptor>(
        rewriter, oldOp, "Ai_shape", &theAdaptor::getAi);
    Value brShape = onnx_to_aisle::create<theAdaptor>(
        rewriter, oldOp, "Br_shape", &theAdaptor::getBr);
    Value biShape = onnx_to_aisle::create<theAdaptor>(
        rewriter, oldOp, "Bi_shape", &theAdaptor::getBi);

    SmallVector<Value> newOperands{
        ar, arShape, ai, aiShape, br, brShape, bi, biShape};

    // Preserve the frontend result types exactly. In particular, an f16
    // ONNX model stays f16 through AISLE. A target-driven f32 -> f16 decision
    // belongs in a later RedMulE-specific lowering.
    SmallVector<Type> resultTypes{
        oldOp.getCr().getType(), oldOp.getCi().getType()};

    SmallVector<NamedAttribute> newAttrs;
    auto newOp =
        rewriter.create<theNewOp>(loc, resultTypes, newOperands, newAttrs);
    rewriter.replaceOp(op, {newOp.getCr(), newOp.getCi()});
    return success();
  }
};

void populateLoweringONNXToAISLEComplexGEMMOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<ONNXRedMulEComplexGEMMOpLowering>(ctx);
}

} // namespace spade
