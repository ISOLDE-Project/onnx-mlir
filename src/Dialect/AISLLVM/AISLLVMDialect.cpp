/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- AISLLVMDialect.cpp ---------------------------===//
//
// Copyright 2023
//
// =============================================================================
//
// This file provides definition of AISLLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISLLVM/AISLLVMDialect.hpp"
#include "src/Dialect/AISLLVM/AISLLVMOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AISLLVM Dialect: TableGen generated implementation
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void spade::AISLLVMDialect::initialize() {
  // Types and attributes are added in these private methods which are
  // implemented in AISLLVMTypes.cpp and AISLLVMAttributes.cpp where they have
  // the necessary access to the underlying storage classes from
  // TableGen generated code in AISLLVMTypes.cpp.inc and AISLLVMAttributes.cpp.inc.
  // (This emulates the approach in the mlir builtin dialect.)
  registerTypes();
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "src/Dialect/AISLLVM/AISLLVMOps.cpp.inc"
      >();
}

// Code for AISLLVM_Dialect class
#include "src/Dialect/AISLLVM/AISLLVMDialect.cpp.inc"
