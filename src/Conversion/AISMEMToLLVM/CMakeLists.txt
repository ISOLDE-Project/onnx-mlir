# SPDX-License-Identifier: Apache-2.0


# Please keep in alphabetical order.
add_onnx_mlir_library(OMAISMEMToLLVM
  Constant/QConstant.cpp
  DMA/DmaStartOp.cpp
  DMA/DmaWaitOp.cpp
  Func/FuncToLLVM.cpp
  Krnl/KrnlEntryPoint.cpp
  Krnl/ConvertKrnlToLLVM.cpp
  Krnl/KrnlGlobal.cpp
  Math/GEMM.cpp
  Memref/AllocOp.cpp
  MLIR/UnrealizedConversionCast.cpp
  ConvertAISMEMToLLVM.cpp
  
  LINK_LIBS PUBLIC
  
  OMAISLLVMOps
  OMAISMEMOps
  OMSupport
  MLIRFuncDialect
  MLIRFuncTransforms
  )

 
