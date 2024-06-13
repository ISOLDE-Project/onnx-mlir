# AISLE - **A**utomot**I**ve demon**S**trator m**L**ir dial**E**ct

After you edit [AISLE.td](AISLE.td) make sure you run:  
```
make ONNX_MLIR_CMAKE_TARGET=OMAISLEIncGen toolchain-onnx-mlir
```
Output should be similar to:
```
[  0%] Building AISLEAttributes.cpp.inc...
[  0%] Building AISLEAttributes.hpp.inc...
[  0%] Building AISLEDialect.cpp.inc...
[  0%] Building AISLEDialect.hpp.inc...
[100%] Building AISLEOps.cpp.inc...
[100%] Building AISLEOps.hpp.inc...
[100%] Building AISLETypes.cpp.inc...
[100%] Building AISLETypes.hpp.inc...
```

TRy to see if the library gets build:

```
make ONNX_MLIR_CMAKE_TARGET=OMAISLEOps toolchain-onnx-mlir
```
Outpur should be similar to: 
```
[ 88%] Building CXX object src/Dialect/AISLE/CMakeFiles/OMAISLEOps.dir/AISLEAttributes.cpp.o
[100%] Linking CXX static library ../../../Debug/lib/libOMAISLEOps.a
```