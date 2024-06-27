
# 1 Build library

If [CMakeLists.txt](./CMakeLists.txt) was modified, run:  
```
make ONNX_MLIR_CMAKE_TARGET=OMAISMEMToLLVM toolchain-onnx-mlir
```
otherwise:  
```
make ONNX_MLIR_CMAKE_TARGET=OMAISMEMToLLVM compiler
```
Expected result:
```
[100%] Building CXX object src/Conversion/ONNXToAISLE/CMakeFiles/OMONNXToAISLE.dir/ConvertONNXToAISLE.cpp.o
[100%] Linking CXX static library ../../../Debug/lib/libOMONNXToAISLE.a
make[4]: Leaving directory '/home/uic52463/hdd2/task5.2/toolchain/onnx-mlir/build'
[100%] Built target OMONNXToAISLE
```
# FAILURE : no matched legalization pattern

To see (more) debug information, the compiler have too be run with --debug flag, e.g.
```
make  ONNX_MODEL=memref.alloc.mlir  ONNX_MLIR_FLAGS=--debug   test
```
To investigate the failure, you have to check the complete pipeline. 
Example:
Snippet from [SpadeCompilerPasses.cpp](../../Compiler/SpadeCompilerPasses.cpp)
```
  pm.addPass(spade::krnl::createConvertKrnlToLLVMPass(
      onnx_mlir::verifyInputTensors,
      /*useLRODATA=*/(onnx_mlir::modelSize == onnx_mlir::ModelSize::large),
      /*storeConstantsToFile=*/onnx_mlir::storeConstantsToFile,
      onnx_mlir::constantsToFileSingleThreshold,
      onnx_mlir::constantsToFileTotalThreshold, outputNameNoExt,
      onnx_mlir::enableParallel));
  pm.addPass(spade::createLowerToLLVMIRPass());
```
if legalization pattern is added in **createLowerToLLVMIRPass()**, make sure that operation/dialect(e.g. memref.alloc) is declared legal for pass **spade::krnl::createConvertKrnlToLLVMPass()**.
Otherwise the outpu will be similar to:  
```
//===-------------------------------------------===//
Legalizing operation : 'memref.alloc'(0xcc55e30) {
  %15 = "memref.alloc"() <{alignment = 16 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//
```
Solution:
```
//only memref::AllocOp, memref::DeallocOp are legall
  target.addLegalOp<memref::AllocOp>();
  target.addLegalOp<memref::DeallocOp>();

```