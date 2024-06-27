# Testing  
## Testing using a specific input
```
make  ONNX_MODEL=gemm.mlir test
```
Default value for ONNX_MODEL is given in [Makefile](./Makefile) therefore the following is legal too:
```
make  test
```
## Testing with selective debug info
```
make test ONNX_MLIR_FLAGS=--debug-only=AISLEToAISMEM_hstack
```
or multiple modules separated by comma
```
 make  ONNX_MODEL=qconstant.mlir  ONNX_MLIR_FLAGS=--debug-only=AISMEMToAISLLVM_KrnlEntryPoint,AISMEMToAISLLVM_QConstant  test
```
## Testing with full debug info
```
make test ONNX_MLIR_FLAGS=--debug
```