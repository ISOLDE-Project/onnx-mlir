/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ Transformer.cpp - Lowering ISOLDE transformer blocks ------===//
//
// onnx.MultiHeadAttention      -> aisle.MultiHeadAttention
// onnx.PositionwiseFeedForward -> aisle.PositionwiseFeedForward
//
// Before the dialect conversion, fuseTransformerBlocks() rewrites the ONNX IR
// in place so that the AISLE ops carry what the RedMulE schedule needs:
//
//   * constant attention scales are folded into the weights
//       scale      -> Wq  (Q K^T * s == (X (Wq s)) K^T)
//       post_scale -> Wv  (relu only: (A * p) V == A (V p))
//     exactly what tformer.py does at export time;
//
//   * a residual Add is folded into the accumulator C
//       Add(h, Block(h, ...)) -> Block(h, ..., C = h)
//     which AISLEToAISMEM implements as a Y preload (free on RedMulE).
//
// The block ops themselves are kept intact, as ComplexGEMM is: tiling and
// tile assignment belong to AISLEToAISMEM.
//
//===----------------------------------------------------------------------===//

#include "../helper.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ONNXToAISLE_Transformer"

using namespace mlir;
using namespace onnx_mlir;

namespace spade {

//===----------------------------------------------------------------------===//
// Pre-conversion rewrites on the ONNX dialect
//===----------------------------------------------------------------------===//

namespace {

// Returns `weights * factor` as a new onnx.Constant, or a null Value when
// `weights` is not a compile-time constant.
Value scaleConstant(OpBuilder &builder, Location loc, Value weights,
    double factor) {
  ElementsAttr elements = getElementAttributeFromONNXValue(weights);
  if (!elements)
    return Value();
  OnnxElementsAttrBuilder elementsBuilder(builder.getContext());
  FloatType f64 = builder.getF64Type();
  ElementsAttr wide = elementsBuilder.castToFPElementType(elements, f64);
  ElementsAttr scaled =
      elementsBuilder.transform(wide, f64, [factor](WideNum n) {
        return WideNum::widen<BType::DOUBLE>(
            factor * n.narrow<BType::DOUBLE>());
      });
  ElementsAttr narrowed =
      elementsBuilder.castElementType(scaled, elements.getElementType());
  return OnnxBuilder(builder, loc).constant(narrowed);
}

// scale -> Wq and (relu) post_scale -> Wv, when the weights are constants.
void foldAttentionScales(ONNXMultiHeadAttentionOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();

  auto wqType = dyn_cast<RankedTensorType>(op.getWq().getType());
  bool scaleKnown = op.getScale().has_value() ||
                    (wqType && !wqType.isDynamicDim(1));
  if (scaleKnown) {
    float scale = op.getScaleOrDefault();
    if (scale == 1.0f) {
      op.setScaleAttr(builder.getF32FloatAttr(1.0f));
    } else if (Value wq = scaleConstant(builder, loc, op.getWq(), scale)) {
      op.getWqMutable().assign(wq);
      op.setScaleAttr(builder.getF32FloatAttr(1.0f));
      LLVM_DEBUG(llvm::dbgs() << "folded scale " << scale << " into Wq\n");
    }
  }

  float post = op.getPostScale().convertToFloat();
  if (op.getNormalization() == "relu" && post != 1.0f) {
    if (Value wv = scaleConstant(builder, loc, op.getWv(), post)) {
      op.getWvMutable().assign(wv);
      op.setPostScaleAttr(builder.getF32FloatAttr(1.0f));
      LLVM_DEBUG(llvm::dbgs() << "folded post_scale " << post << " into Wv\n");
    }
  }
}

// If `blockResult` is produced by a MultiHeadAttention or
// PositionwiseFeedForward op that has no accumulator yet and no other user,
// returns that op.
Operation *fusableBlock(Value blockResult) {
  Operation *def = blockResult.getDefiningOp();
  if (!def || !blockResult.hasOneUse())
    return nullptr;
  if (auto mha = dyn_cast<ONNXMultiHeadAttentionOp>(def))
    return isa<NoneType>(mha.getC().getType()) ? def : nullptr;
  if (auto ffn = dyn_cast<ONNXPositionwiseFeedForwardOp>(def))
    return isa<NoneType>(ffn.getC().getType()) ? def : nullptr;
  return nullptr;
}

// Add(residual, Block(...)) -> Block(..., C = residual).  Only when no
// broadcasting is involved, i.e. both Add operands and the result have one
// and the same static type.
bool foldResidualAdd(ONNXAddOp add) {
  auto resultType = dyn_cast<RankedTensorType>(add.getC().getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;
  for (int side = 0; side < 2; ++side) {
    Value blockValue = add.getOperand(side);
    Value residual = add.getOperand(1 - side);
    Operation *block = fusableBlock(blockValue);
    if (!block || residual.getType() != resultType ||
        blockValue.getType() != resultType)
      continue;
    // The residual must dominate the block op: it becomes one of its
    // operands.  Values defined after the block (other than the block's own
    // result) cannot be used there.
    if (Operation *resDef = residual.getDefiningOp())
      if (resDef->getBlock() == block->getBlock() &&
          block->isBeforeInBlock(resDef))
        continue;

    OpBuilder builder(add);
    Operation *fused = nullptr;
    if (auto mha = dyn_cast<ONNXMultiHeadAttentionOp>(block)) {
      fused = builder.create<ONNXMultiHeadAttentionOp>(mha.getLoc(),
          resultType, mha.getXq(), mha.getXkv(), mha.getWq(), mha.getWk(),
          mha.getWv(), mha.getWo(), residual, mha.getNumHeadsAttr(),
          mha.getScaleAttr(), mha.getPostScaleAttr(),
          mha.getNormalizationAttr());
    } else {
      auto ffn = cast<ONNXPositionwiseFeedForwardOp>(block);
      fused = builder.create<ONNXPositionwiseFeedForwardOp>(ffn.getLoc(),
          resultType, ffn.getX(), ffn.getW1(), ffn.getW2(), residual,
          ffn.getActivationAttr());
    }
    add.getC().replaceAllUsesWith(fused->getResult(0));
    add.erase();
    block->erase();
    LLVM_DEBUG(llvm::dbgs() << "folded residual Add into "
                            << fused->getName() << "\n");
    return true;
  }
  return false;
}

} // namespace

void fuseTransformerBlocks(ModuleOp module) {
  module.walk([](ONNXMultiHeadAttentionOp op) { foldAttentionScales(op); });

  SmallVector<ONNXAddOp> adds;
  module.walk([&](ONNXAddOp add) { adds.push_back(add); });
  for (ONNXAddOp add : adds)
    (void)foldResidualAdd(add);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

struct ONNXMultiHeadAttentionOpLowering : public ConversionPattern {
  using theOperation = mlir::ONNXMultiHeadAttentionOp;
  using theAdaptor = mlir::ONNXMultiHeadAttentionOpAdaptor;
  using theNewOp = spade::AISLEMultiHeadAttentionOp;

  ONNXMultiHeadAttentionOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto oldOp = llvm::dyn_cast<theOperation>(op);
    if (!oldOp)
      return failure();
    theAdaptor operandAdaptor(operands);

    auto wqType = dyn_cast<RankedTensorType>(oldOp.getWq().getType());
    if (!oldOp.getScale().has_value() && (!wqType || wqType.isDynamicDim(1)))
      return rewriter.notifyMatchFailure(
          op, "default scale 1/sqrt(d_k) needs a static Wq width");
    float scale = oldOp.getScaleOrDefault();

    SmallVector<Value> newOperands{
        operandAdaptor.getXq(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "Xq_shape", &theAdaptor::getXq),
        operandAdaptor.getXkv(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "Xkv_shape", &theAdaptor::getXkv),
        operandAdaptor.getWq(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "Wq_shape", &theAdaptor::getWq),
        operandAdaptor.getWk(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "Wk_shape", &theAdaptor::getWk),
        operandAdaptor.getWv(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "Wv_shape", &theAdaptor::getWv),
        operandAdaptor.getWo(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "Wo_shape", &theAdaptor::getWo)};
    Value c = operandAdaptor.getC();
    if (!isa<NoneType>(c.getType()))
      newOperands.push_back(c);

    SmallVector<NamedAttribute> newAttrs{
        rewriter.getNamedAttr("num_heads", oldOp.getNumHeadsAttr()),
        rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)),
        rewriter.getNamedAttr("post_scale", oldOp.getPostScaleAttr()),
        rewriter.getNamedAttr("normalization", oldOp.getNormalizationAttr())};

    // Preserve the frontend result type exactly (f16 stays f16).
    auto newOp = rewriter.create<theNewOp>(
        loc, TypeRange{oldOp.getY().getType()}, newOperands, newAttrs);
    rewriter.replaceOp(op, newOp.getY());
    return success();
  }
};

struct ONNXPositionwiseFeedForwardOpLowering : public ConversionPattern {
  using theOperation = mlir::ONNXPositionwiseFeedForwardOp;
  using theAdaptor = mlir::ONNXPositionwiseFeedForwardOpAdaptor;
  using theNewOp = spade::AISLEPositionwiseFeedForwardOp;

  ONNXPositionwiseFeedForwardOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto oldOp = llvm::dyn_cast<theOperation>(op);
    if (!oldOp)
      return failure();
    theAdaptor operandAdaptor(operands);

    SmallVector<Value> newOperands{operandAdaptor.getX(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "X_shape", &theAdaptor::getX),
        operandAdaptor.getW1(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "W1_shape", &theAdaptor::getW1),
        operandAdaptor.getW2(),
        onnx_to_aisle::create<theAdaptor>(
            rewriter, oldOp, "W2_shape", &theAdaptor::getW2)};
    Value c = operandAdaptor.getC();
    if (!isa<NoneType>(c.getType()))
      newOperands.push_back(c);

    SmallVector<NamedAttribute> newAttrs{
        rewriter.getNamedAttr("activation", oldOp.getActivationAttr())};
    auto newOp = rewriter.create<theNewOp>(
        loc, TypeRange{oldOp.getY().getType()}, newOperands, newAttrs);
    rewriter.replaceOp(op, newOp.getY());
    return success();
  }
};

void populateLoweringONNXToAISLETransformerOpPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<ONNXMultiHeadAttentionOpLowering,
      ONNXPositionwiseFeedForwardOpLowering>(ctx);
}

} // namespace spade
