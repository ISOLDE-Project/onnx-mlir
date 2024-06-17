/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- AISLLVMTypes.cpp --------------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file provides definition of AISLLVM types.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISLLVM/AISLLVMTypes.hpp"
#include "src/Dialect/AISLLVM/AISLLVMDialect.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AISLLVM Types: TableGen generated implementation
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "src/Dialect/AISLLVM/AISLLVMTypes.cpp.inc"

// See explanation in AISLLVMDialect::initialize() in AISLLVMDialect.cpp.
void spade::AISLLVMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Dialect/AISLLVM/AISLLVMTypes.cpp.inc"
      >();
}
