/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- AISMEMOps.hpp - AISMEM Operations -------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file defines AISMEM operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/AISMEM/AISMEMAttributes.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMTypes.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"
//#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
//#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
//#include "src/Interface/ShapeInferenceOpInterface.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

//namespace mlir {
// OpSet level supported by onnx-mlir
//static constexpr int CURRENT_ONNX_OPSET = 17;
//} // end namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/AISMEM/AISMEMOps.hpp.inc"

