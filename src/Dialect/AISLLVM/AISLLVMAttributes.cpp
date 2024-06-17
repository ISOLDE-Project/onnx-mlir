/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- AISLLVMAttributes.cpp -----------------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file defines attributes in the AISLLVM Dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISLLVM/AISLLVMAttributes.hpp"
#include "src/Dialect/AISLLVM/AISLLVMDialect.hpp"
//#include "src/Dialect/AISLLVM/AISLLVMOps/OpHelper.hpp"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
//using namespace onnx_mlir;


//===----------------------------------------------------------------------===//
// AISLLVM Attributes: TableGen generated implementation
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/AISLLVM/AISLLVMAttributes.cpp.inc"

// See explanation in AISLLVMDialect::initialize() in AISLLVMDialect.cpp.
void spade::AISLLVMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/AISLLVM/AISLLVMAttributes.cpp.inc"
      >();
}
