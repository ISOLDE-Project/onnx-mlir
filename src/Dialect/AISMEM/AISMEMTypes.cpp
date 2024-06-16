/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- AISMEMTypes.cpp --------------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file provides definition of AISMEM types.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISMEM/AISMEMTypes.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AISMEM Types: TableGen generated implementation
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "src/Dialect/AISMEM/AISMEMTypes.cpp.inc"

// See explanation in AISMEMDialect::initialize() in AISMEMDialect.cpp.
void spade::AISMEMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Dialect/AISMEM/AISMEMTypes.cpp.inc"
      >();
}
