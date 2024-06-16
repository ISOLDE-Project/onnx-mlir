/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- AISMEMAttributes.cpp -----------------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file defines attributes in the AISMEM Dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/AISMEM/AISMEMAttributes.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"


#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
//using namespace onnx_mlir;


//===----------------------------------------------------------------------===//
// AISMEM Attributes: TableGen generated implementation
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/AISMEM/AISMEMAttributes.cpp.inc"


void spade::AISMEMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/AISMEM/AISMEMAttributes.cpp.inc"
      >();
}
