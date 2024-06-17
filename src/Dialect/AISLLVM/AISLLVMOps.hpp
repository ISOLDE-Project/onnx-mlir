/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- AISLLVMOps.hpp - AISLLVM Operations -------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file defines AISLLVM operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/AISLLVM/AISLLVMAttributes.hpp"
#include "src/Dialect/AISLLVM/AISLLVMDialect.hpp"
#include "src/Dialect/AISLLVM/AISLLVMTypes.hpp"
//#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
//#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
//#include "src/Interface/ShapeInferenceOpInterface.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

//namespace mlir {
// OpSet level supported by onnx-mlir
//static constexpr int CURRENT_ONNX_OPSET = 17;
//} // end namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/AISLLVM/AISLLVMOps.hpp.inc"

