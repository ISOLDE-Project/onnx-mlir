# SPDX-License-Identifier: Apache-2.0




# Please keep in alphabetical order.
add_onnx_mlir_library(OMAISLEToAISMEM
  ConvertAISLEToAISMEM.cpp
  Constant/QConstant.cpp
  DMA/hstack.cpp
  Math/GEMM.cpp
  
  LINK_LIBS PUBLIC

  OMAISLEOps
  OMAISMEMOps
  OMSupport
  MLIRFuncDialect
  MLIRFuncTransforms
  )
