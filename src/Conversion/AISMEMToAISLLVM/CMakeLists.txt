# SPDX-License-Identifier: Apache-2.0


# Please keep in alphabetical order.
add_onnx_mlir_library(OMAISMEMToAISLLVM
  Constant/QConstant.cpp
  Krnl/KrnlEntryPoint.cpp
  Math/GEMM.cpp
  Memref/AllocOp.cpp
  ConvertAISMEMToAISLLVM.cpp
  ConvertAISMEMToAISLLVM2.cpp
  
  LINK_LIBS PUBLIC
  
  OMAISLLVMOps
  OMAISMEMOps
  OMSupport
  MLIRFuncDialect
  MLIRFuncTransforms
  )

 
