
# 1 Build library

If [CMakeLists.txt](./CMakeLists.txt) was modified, run:  
```
make ONNX_MLIR_CMAKE_TARGET=OMAISLEToAISMEM toolchain-onnx-mlir
```
otherwise:  
```
make ONNX_MLIR_CMAKE_TARGET=OMAISLEToAISMEM compiler
```
Expected result:
```
[100%] Building CXX object src/Conversion/ONNXToAISLE/CMakeFiles/OMONNXToAISLE.dir/ConvertONNXToAISLE.cpp.o
[100%] Linking CXX static library ../../../Debug/lib/libOMONNXToAISLE.a
make[4]: Leaving directory '/home/uic52463/hdd2/task5.2/toolchain/onnx-mlir/build'
[100%] Built target OMONNXToAISLE
```