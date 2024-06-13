/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- AISLEAttributes.cpp -----------------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file defines attributes in the AISLE Dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISLE/AISLEAttributes.hpp"

#include "src/Dialect/AISLE/AISLEDialect.hpp"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
//using namespace onnx_mlir;


//===----------------------------------------------------------------------===//
// AISLE Attributes: TableGen generated implementation
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/AISLE/AISLEAttributes.cpp.inc"

// See explanation in AISLEDialect::initialize() in AISLEDialect.cpp.
void spade::AISLEDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/AISLE/AISLEAttributes.cpp.inc"
      >();
}
