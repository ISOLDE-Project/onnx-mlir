/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- AISLEDialect.cpp ---------------------------===//
//
//
// =============================================================================
//
// This file provides definition of AISLE dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISLE/AISLEDialect.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AISLE Dialect: TableGen generated implementation
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void spade::AISLEDialect::initialize() {
  // Types and attributes are added in these private methods which are
  // implemented in AISLETypes.cpp and AISLEAttributes.cpp where they have
  // the necessary access to the underlying storage classes from
  // TableGen generated code in AISLETypes.cpp.inc and AISLEAttributes.cpp.inc.
  // (This emulates the approach in the mlir builtin dialect.)
  registerTypes();
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "src/Dialect/AISLE/AISLEOps.cpp.inc"
      >();
}

// Code for AISLE_Dialect class
#include "src/Dialect/AISLE/AISLEDialect.cpp.inc"
