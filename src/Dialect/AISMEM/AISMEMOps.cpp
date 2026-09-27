/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXOps.cpp - ONNX Operations ---------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file provides definition of AISMEM dialect operations.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISMEM/AISMEMOps.hpp"

#include "mlir/Dialect/Traits.h"

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
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/AISMEM/AISMEMOps.cpp.inc"


//===----------------------------------------------------------------------===//
// RedMulEUploadTile
//===----------------------------------------------------------------------===//

mlir::LogicalResult spade::AISMEMRedMulEUploadTileOp::verify() {
  auto type = mlir::dyn_cast<mlir::MemRefType>(getSource().getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1)
    return emitOpError("source must be a statically shaped memref");
  if (!type.getElementType().isF16())
    return emitOpError("source must hold f16 elements");

  const int64_t ld = type.getShape().back();
  const int64_t srcRows = type.getNumElements() / ld;
  const int64_t rowOffset = getRowOffset(), colOffset = getColOffset();
  const int64_t rows = getRows(), cols = getCols();
  const int64_t dstRows = getDstRows(), dstCols = getDstCols();

  if (rowOffset < 0 || colOffset < 0 || rows <= 0 || cols <= 0)
    return emitOpError("window offsets must be >= 0 and extents > 0");
  if (rowOffset + rows > srcRows || colOffset + cols > ld)
    return emitOpError("window [")
           << rowOffset << ":" << rowOffset + rows << ", " << colOffset << ":"
           << colOffset + cols << "] exceeds the " << srcRows << "x" << ld
           << " source";
  if (dstCols != 16)
    return emitOpError("dst_cols must be one RedMulE row (16), got ")
           << dstCols;
  if (dstRows <= 0)
    return emitOpError("dst_rows must be > 0");
  const int64_t outRows = getTranspose() ? cols : rows;
  const int64_t outCols = getTranspose() ? rows : cols;
  if (outRows > dstRows || outCols > dstCols)
    return emitOpError("the (transposed) window does not fit in the ")
           << dstRows << "x" << dstCols << " destination";
  return mlir::success();
}
