/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- AISLETypes.cpp --------------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file provides definition of AISLE types.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISLE/AISLETypes.hpp"
#include "src/Dialect/AISLE/AISLEDialect.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AISLE Types: TableGen generated implementation
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "src/Dialect/AISLE/AISLETypes.cpp.inc"

// See explanation in AISLEDialect::initialize() in AISLEDialect.cpp.
void spade::AISLEDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Dialect/AISLE/AISLETypes.cpp.inc"
      >();
}
