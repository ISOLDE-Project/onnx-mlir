# SPDX-License-Identifier: Apache-2.0


# Please keep in alphabetical order.
add_onnx_mlir_library(OMONNXToAISLE
  ConvertONNXToAISLE.cpp
  Math/GEMM.cpp
  

  LINK_LIBS PUBLIC
  OMAISLEOps
  OMONNXOps
  OMSupport
  MLIRFuncDialect
  MLIRFuncTransforms
  )
