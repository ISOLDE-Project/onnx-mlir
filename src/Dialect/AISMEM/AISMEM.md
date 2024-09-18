# AISMEM	**A**utomot**I**ve Demon**S**trator **MEM**ref dialect

Dialect has to be registered in [CompilerDialects.cpp](../../Compiler/CompilerDialects.cpp).  
After you edit [AISMEM.td](AISMEM.td) make sure you run:  
```
make ONNX_MLIR_CMAKE_TARGET=OMAISMEMIncGen toolchain-onnx-mlir
```
or  
```
 make ONNX_MLIR_CMAKE_TARGET=OMAISMEMIncGen compiler
 ```
Output should be similar to:
```
...
[100%] Built target OMAISMEMIncGen
```

TRy to see if the library gets build:

```
make ONNX_MLIR_CMAKE_TARGET=OMAISMEMOps compiler
```
Output should be similar to: 
```
[100%] Built target OMAISMEMOps
```