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

