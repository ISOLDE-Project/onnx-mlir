# AISLLVM	**A**utomot**I**ve Demon**S**trator **LLVM** dialect

After you edit [AISLLVM.td](AISLLVM.td) make sure you run:  
```
make ONNX_MLIR_CMAKE_TARGET=OMAISLLVMIncGen toolchain-onnx-mlir
```
or  
```
 make ONNX_MLIR_CMAKE_TARGET=OMAISLLVMIncGen compiler
 ```
Output should be similar to:
```
...
[  0%] Building AISLLVMAttributes.cpp.inc...
[  0%] Building AISLLVMAttributes.hpp.inc...
[  0%] Building AISLLVMDialect.cpp.inc...
[100%] Building AISLLVMDialect.hpp.inc...
[100%] Building AISLLVMOps.cpp.inc...
[100%] Building AISLLVMOps.hpp.inc...
[100%] Building AISLLVMTypes.cpp.inc...
[100%] Building AISLLVMTypes.hpp.inc...
make[4]: Leaving directory '/home/uic52463/hdd2/task5.2/toolchain/onnx-mlir/build'
```

Try to see if the library gets build:

```
make ONNX_MLIR_CMAKE_TARGET=OMAISLLVMOps compiler
```
Output should be similar to: 
```

```