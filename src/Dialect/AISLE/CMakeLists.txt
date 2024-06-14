# SPDX-License-Identifier: Apache-2.0



add_onnx_mlir_dialect(AISLE aisle)
add_onnx_mlir_dialect_doc(aisle AISLE.td)

add_onnx_mlir_library(OMAISLEOps
  # Top files for AISLE dialect
  AISLEAttributes.cpp
  AISLEDialect.cpp
  AISLEOps.cpp
  AISLETypes.cpp 

  DEPENDS
  OMAISLEIncGen
  OMResultTypeInferenceOpInterfaceIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMMlirDialects
  OMMlirUtilities
  )

