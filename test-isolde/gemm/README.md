# Testing  
## Testing usign a specific input
```
make  ONNX_MODEL=gemm.mlir test
```
## Testing with selective debug info
```
make test ONNX_MLIR_FLAGS=--debug-only=AISLEToAISMEM_hstack
```
## Testing with full debug info
```
make test ONNX_MLIR_FLAGS=--debug
```