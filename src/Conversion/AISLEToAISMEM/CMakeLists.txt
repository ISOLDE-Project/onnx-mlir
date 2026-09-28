# SPDX-License-Identifier: Apache-2.0




# Please keep in alphabetical order.
add_onnx_mlir_library(OMAISLEToAISMEM
  ConvertAISLEToAISMEM.cpp
  Constant/QConstant.cpp
  DMA/hstack.cpp
  Math/ComplexGEMM.cpp
  Math/GEMM.cpp
  Math/MatMulAdd.cpp
  SPMAllocation.cpp
  Math/Transformer.cpp
  
  LINK_LIBS PUBLIC

  OMAISLEOps
  OMAISMEMOps
  OMSupport
  MLIRFuncDialect
  MLIRFuncTransforms
  )
