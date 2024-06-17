# SPDX-License-Identifier: Apache-2.0


add_onnx_mlir_dialect(AISLLVM aisllvm)
add_onnx_mlir_dialect_doc(aisllvm AISLLVM.td)


add_onnx_mlir_library(OMAISLLVMOps
  # Top files for AISLLVM dialect
  AISLLVMAttributes.cpp
  AISLLVMDialect.cpp
  AISLLVMOps.cpp
  AISLLVMTypes.cpp 

  DEPENDS
  OMAISMEMIncGen
  OMAISLLVMIncGen
  OMResultTypeInferenceOpInterfaceIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMMlirDialects
  OMMlirUtilities
  )



