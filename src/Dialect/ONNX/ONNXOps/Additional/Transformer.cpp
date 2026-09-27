/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Transformer.cpp - ISOLDE transformer ONNX ops ------------===//
//
// Verifiers and shape inference for the com.isolde transformer building
// blocks: onnx.MultiHeadAttention and onnx.PositionwiseFeedForward.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include <cmath>

using namespace mlir;
using namespace onnx_mlir;

namespace {

constexpr int64_t kDyn = ShapedType::kDynamic;

RankedTensorType rankedType(Value v) {
  return dyn_cast<RankedTensorType>(v.getType());
}

// Dimension counted from the back (fromBack = 1 is the last dimension), or
// kDyn when unknown.
int64_t dimFromBack(Value v, int64_t fromBack) {
  RankedTensorType t = rankedType(v);
  if (!t || t.getRank() < fromBack)
    return kDyn;
  return t.getDimSize(t.getRank() - fromBack);
}

LogicalResult checkRank(
    Operation *op, Value v, StringRef name, int64_t lo, int64_t hi) {
  RankedTensorType t = rankedType(v);
  if (!t)
    return success();
  if (t.getRank() < lo || t.getRank() > hi)
    return op->emitOpError() << name << " must have rank in [" << lo << ", "
                             << hi << "], got " << t.getRank();
  return success();
}

LogicalResult checkSameDim(
    Operation *op, int64_t a, int64_t b, StringRef what) {
  if (a == kDyn || b == kDyn || a == b)
    return success();
  return op->emitOpError() << what << " mismatch (" << a << " vs " << b << ")";
}

LogicalResult checkSameElementType(Operation *op, ValueRange values) {
  Type first;
  for (Value v : values) {
    if (isa<NoneType>(v.getType()))
      continue;
    Type t = getElementType(v.getType());
    if (!first)
      first = t;
    else if (t != first)
      return op->emitOpError("all tensor operands must share one element type");
  }
  return success();
}

// Output shape = X shape with the last dimension replaced by `lastDim`.
LogicalResult inferRowWise(Operation *op, Value x, int64_t lastDim) {
  RankedTensorType xType = rankedType(x);
  if (!xType)
    return success();
  SmallVector<int64_t, 3> shape(xType.getShape());
  shape.back() = lastDim;
  updateType(op, op->getResult(0), shape, xType.getElementType());
  return success();
}

// C, when present, must have the shape of the result.
LogicalResult checkAccumulator(Operation *op, Value c, Value like,
    int64_t lastDim) {
  if (isa<NoneType>(c.getType()))
    return success();
  RankedTensorType cType = rankedType(c);
  RankedTensorType likeType = rankedType(like);
  if (!cType || !likeType)
    return success();
  if (cType.getRank() != likeType.getRank())
    return op->emitOpError("C must have the rank of the result");
  for (int64_t i = 0; i + 1 < cType.getRank(); ++i)
    if (failed(checkSameDim(op, cType.getDimSize(i), likeType.getDimSize(i),
            "C / result dimension")))
      return failure();
  return checkSameDim(op, cType.getShape().back(), lastDim, "C / result width");
}

} // namespace

//===----------------------------------------------------------------------===//
// MultiHeadAttention
//===----------------------------------------------------------------------===//

float ONNXMultiHeadAttentionOp::getScaleOrDefault() {
  if (std::optional<APFloat> scale = getScale())
    return scale->convertToFloat();
  int64_t heads = getNumHeads();
  int64_t width = dimFromBack(getWq(), 1);
  assert(width != kDyn && heads > 0 && "default scale needs a static d_k");
  return 1.0f / std::sqrt(static_cast<float>(width / heads));
}

LogicalResult ONNXMultiHeadAttentionOp::verify() {
  Operation *op = getOperation();
  StringRef norm = getNormalization();
  if (norm != "softmax" && norm != "relu")
    return emitOpError("normalization must be \"softmax\" or \"relu\", got \"")
           << norm << "\"";
  if (norm == "softmax" && getPostScale().convertToFloat() != 1.0f)
    return emitOpError("post_scale must be 1.0 with softmax normalization");
  int64_t heads = getNumHeads();
  if (heads < 1)
    return emitOpError("num_heads must be >= 1");

  if (failed(checkSameElementType(op,
          {getXq(), getXkv(), getWq(), getWk(), getWv(), getWo(), getC()})))
    return failure();
  if (failed(checkRank(op, getXq(), "Xq", 2, 3)) ||
      failed(checkRank(op, getXkv(), "Xkv", 2, 3)) ||
      failed(checkRank(op, getWq(), "Wq", 2, 2)) ||
      failed(checkRank(op, getWk(), "Wk", 2, 2)) ||
      failed(checkRank(op, getWv(), "Wv", 2, 2)) ||
      failed(checkRank(op, getWo(), "Wo", 2, 2)))
    return failure();

  RankedTensorType xq = rankedType(getXq()), xkv = rankedType(getXkv());
  if (xq && xkv && xq.getRank() != xkv.getRank())
    return emitOpError("Xq and Xkv must have the same rank");
  if (xq && xkv && xq.getRank() == 3 &&
      failed(checkSameDim(op, xq.getDimSize(0), xkv.getDimSize(0), "batch")))
    return failure();

  int64_t dq = dimFromBack(getXq(), 1), dkv = dimFromBack(getXkv(), 1);
  int64_t wqIn = dimFromBack(getWq(), 2), wqOut = dimFromBack(getWq(), 1);
  int64_t wkIn = dimFromBack(getWk(), 2), wkOut = dimFromBack(getWk(), 1);
  int64_t wvIn = dimFromBack(getWv(), 2), wvOut = dimFromBack(getWv(), 1);
  int64_t woIn = dimFromBack(getWo(), 2), woOut = dimFromBack(getWo(), 1);
  if (failed(checkSameDim(op, dq, wqIn, "Xq / Wq")) ||
      failed(checkSameDim(op, dkv, wkIn, "Xkv / Wk")) ||
      failed(checkSameDim(op, dkv, wvIn, "Xkv / Wv")) ||
      failed(checkSameDim(op, wqOut, wkOut, "Wq / Wk width")) ||
      failed(checkSameDim(op, wvOut, woIn, "Wv / Wo")))
    return failure();
  if (wqOut != kDyn && wqOut % heads != 0)
    return emitOpError("Wq width must be divisible by num_heads");
  if (wvOut != kDyn && wvOut % heads != 0)
    return emitOpError("Wv width must be divisible by num_heads");
  return checkAccumulator(op, getC(), getXq(), woOut);
}

LogicalResult ONNXMultiHeadAttentionOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getXq()) || !hasShapeAndRank(getWo()))
    return success();
  return inferRowWise(getOperation(), getXq(), dimFromBack(getWo(), 1));
}

//===----------------------------------------------------------------------===//
// PositionwiseFeedForward
//===----------------------------------------------------------------------===//

LogicalResult ONNXPositionwiseFeedForwardOp::verify() {
  Operation *op = getOperation();
  if (getActivation() != "relu")
    return emitOpError("activation must be \"relu\", got \"")
           << getActivation() << "\"";
  if (failed(checkSameElementType(op, {getX(), getW1(), getW2(), getC()})))
    return failure();
  if (failed(checkRank(op, getX(), "X", 2, 3)) ||
      failed(checkRank(op, getW1(), "W1", 2, 2)) ||
      failed(checkRank(op, getW2(), "W2", 2, 2)))
    return failure();
  if (failed(checkSameDim(op, dimFromBack(getX(), 1), dimFromBack(getW1(), 2),
          "X / W1")) ||
      failed(checkSameDim(op, dimFromBack(getW1(), 1), dimFromBack(getW2(), 2),
          "W1 / W2")))
    return failure();
  return checkAccumulator(op, getC(), getX(), dimFromBack(getW2(), 1));
}

LogicalResult ONNXPositionwiseFeedForwardOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW2()))
    return success();
  return inferRowWise(getOperation(), getX(), dimFromBack(getW2(), 1));
}
