/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- AISLEOps.hpp - AISLE Operations -------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file defines AISLE operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/AISLE/AISLEAttributes.hpp"
#include "src/Dialect/AISLE/AISLEDialect.hpp"
#include "src/Dialect/AISLE/AISLETypes.hpp"
#include "llvm/ADT/ArrayRef.h"


#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


#define GET_OP_CLASSES
#include "src/Dialect/AISLE/AISLEOps.hpp.inc"

