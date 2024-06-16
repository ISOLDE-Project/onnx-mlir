# SPDX-License-Identifier: Apache-2.0


add_onnx_mlir_dialect(AISMEM aismem)
add_onnx_mlir_dialect_doc(aismem AISMEM.td)


add_onnx_mlir_library(OMAISMEMOps
  # Top files for AISMEM dialect
  AISMEMAttributes.cpp
  AISMEMDialect.cpp
  AISMEMOps.cpp
  AISMEMTypes.cpp 

  DEPENDS
  OMAISMEMIncGen
  OMResultTypeInferenceOpInterfaceIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMMlirDialects
  OMMlirUtilities
  )



