/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ AISLEOps.cpp - ONNX Operations ---------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file provides definition of AISLE dialect operations.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISLE/AISLEOps.hpp"

#include "mlir/Dialect/Traits.h"
#include "llvm/ADT/ArrayRef.h"
//===----------------------------------------------------------------------===//
// Unsupported Operations
//===---------------------------------------------------------------------===//

// Operations for which shape inference has not been implemented yet
// If you add the implementation for one op, move it out of this section
// Also please add test case in test/mlir/onnx/onnx_shape_inference.mlir
// Followed by the implementation of lowering to Krnl and
// Enable the corresponding node test in check-onnx-backend

#define NOT_IMPLEMENTED_INFER_SHAPES(T)                                        \
  mlir::LogicalResult mlir::T::inferShapes(                                    \
      std::function<void(mlir::Region &)> doShapeInference) {                  \
    return emitOpError(                                                        \
        "op is not supported at this time. Please open an issue on "           \
        "https://github.com/onnx/onnx-mlir and/or consider contributing "      \
        "code. "                                                               \
        "Error encountered in shape inference.");                              \
  }

// Listed alphabetically.
//NOT_IMPLEMENTED_INFER_SHAPES(ONNXAdagradOp)
//NOT_IMPLEMENTED_INFER_SHAPES(ONNXAdamOp)

//===----------------------------------------------------------------------===//
// Window / Concat
//===----------------------------------------------------------------------===//

namespace spade {

using namespace mlir;

LogicalResult AISLEWindowOp::verify() {
  auto in = dyn_cast<RankedTensorType>(getInput().getType());
  auto out = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!in || !out || !in.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("needs statically shaped input and result");
  if (in.getElementType() != out.getElementType())
    return emitOpError("input and result element types differ");
  ArrayRef<int64_t> offsets = getOffsets();
  if (in.getRank() != out.getRank() ||
      static_cast<int64_t>(offsets.size()) != in.getRank())
    return emitOpError("input, result and offsets must have the same rank");
  for (int64_t d = 0; d < in.getRank(); ++d)
    if (offsets[d] < 0 || offsets[d] + out.getDimSize(d) > in.getDimSize(d))
      return emitOpError() << "window [" << offsets[d] << ", "
                           << offsets[d] + out.getDimSize(d)
                           << ") is outside dimension " << d << " of size "
                           << in.getDimSize(d);
  return success();
}

OpFoldResult AISLEWindowOp::fold(FoldAdaptor) {
  auto in = cast<RankedTensorType>(getInput().getType());
  auto out = cast<RankedTensorType>(getOutput().getType());
  ArrayRef<int64_t> offsets = getOffsets();
  // The whole input.
  if (in == out && llvm::all_of(offsets, [](int64_t o) { return o == 0; }))
    return getInput();
  // A window of a window: one window on the original input.
  if (auto inner = getInput().getDefiningOp<AISLEWindowOp>()) {
    SmallVector<int64_t> sum(offsets);
    for (auto [s, o] : llvm::zip(sum, inner.getOffsets()))
      s += o;
    getInputMutable().assign(inner.getInput());
    setOffsets(sum);
    return getOutput();
  }
  // Exactly one piece of a Concat.
  if (auto concat = getInput().getDefiningOp<AISLEConcatOp>()) {
    const int64_t axis = concat.getAxis();
    int64_t start = 0;
    for (Value piece : concat.getInputs()) {
      if (piece.getType() == out && offsets[axis] == start &&
          llvm::all_of(llvm::enumerate(offsets), [&](auto e) {
            return static_cast<int64_t>(e.index()) == axis || e.value() == 0;
          }))
        return piece;
      start += cast<RankedTensorType>(piece.getType()).getDimSize(axis);
    }
  }
  return {};
}

LogicalResult AISLEConcatOp::verify() {
  if (getInputs().empty())
    return emitOpError("needs at least one input");
  auto out = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!out || !out.hasStaticShape())
    return emitOpError("needs a statically shaped result");
  const int64_t axis = getAxis();
  if (axis < 0 || axis >= out.getRank())
    return emitOpError() << "axis " << axis << " is out of range";
  int64_t total = 0;
  for (Value v : getInputs()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || !t.hasStaticShape() || t.getRank() != out.getRank() ||
        t.getElementType() != out.getElementType())
      return emitOpError("inputs must be static tensors like the result");
    for (int64_t d = 0; d < out.getRank(); ++d)
      if (d != axis && t.getDimSize(d) != out.getDimSize(d))
        return emitOpError() << "input dimension " << d
                             << " differs from the result";
    total += t.getDimSize(axis);
  }
  if (total != out.getDimSize(axis))
    return emitOpError() << "inputs add up to " << total << " along axis "
                         << axis << ", the result has "
                         << out.getDimSize(axis);
  return success();
}

OpFoldResult AISLEConcatOp::fold(FoldAdaptor) {
  if (getInputs().size() == 1)
    return getInputs().front();
  return {};
}

} // namespace spade

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/AISLE/AISLEOps.cpp.inc"

