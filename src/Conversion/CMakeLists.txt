# SPDX-License-Identifier: Apache-2.0

add_subdirectory(AISLEToAISMEM)
add_subdirectory(AISMEMToLLVM)
add_subdirectory(KrnlSeqToMemref)
add_subdirectory(KrnlToAffine)
add_subdirectory(KrnlToLLVM)
add_subdirectory(ONNXConversionCommon)
add_subdirectory(ONNXToAISLE)
add_subdirectory(ONNXToKrnl)

if (ONNX_MLIR_ENABLE_STABLEHLO)
  add_subdirectory(ONNXToStablehlo)
endif()

add_subdirectory(ONNXToTOSA)
