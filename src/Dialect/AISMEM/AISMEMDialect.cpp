/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- AISMEMDialect.cpp ---------------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file provides definition of AISMEM dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AISMEM Dialect: TableGen generated implementation
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void spade::AISMEMDialect::initialize() {
  // Types and attributes are added in these private methods which are
  // implemented in AISMEMTypes.cpp and AISMEMAttributes.cpp where they have
  // the necessary access to the underlying storage classes from
  // TableGen generated code in AISMEMTypes.cpp.inc and AISMEMAttributes.cpp.inc.
  // (This emulates the approach in the mlir builtin dialect.)
  registerTypes();
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "src/Dialect/AISMEM/AISMEMOps.cpp.inc"
      >();
}

// Code for AISMEM_Dialect class
#include "src/Dialect/AISMEM/AISMEMDialect.cpp.inc"
