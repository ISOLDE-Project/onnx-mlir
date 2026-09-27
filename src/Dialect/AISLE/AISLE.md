# AISLE - **A**utomot**I**ve demon**S**trator m**L**ir dial**E**ct

Dialect has to be registered in [CompilerDialects.cpp](../../Compiler/CompilerDialects.cpp).  
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
Output should be similar to: 
```
[ 88%] Building CXX object src/Dialect/AISLE/CMakeFiles/OMAISLEOps.dir/AISLEAttributes.cpp.o
[100%] Linking CXX static library ../../../Debug/lib/libOMAISLEOps.a
```
# Transformer building blocks (com.isolde)

The encoder of *Attention Is All You Need* (Vaswani et al. 2017,
arXiv:1706.03762) is described in ONNX with the custom domain `com.isolde`;
see `test-isolde/transformer/models/generate_transformer.py` and
`ibex/isolde/sw/radar_attention/tformer_onnx.py`.  Every `com.isolde` node is
shipped with a model-local `FunctionProto` (standard ONNX ops), so the model
runs unchanged in onnxruntime.

| Paper | ONNX file | onnx-mlir import | AISLE | AISLE -> AISMEM |
|---|---|---|---|---|
| Encoder layer (Fig. 1) | `com.isolde.EncoderLayer` | inlined (function body) | – | – |
| Multi-head attention (3.2.2) | `com.isolde.MultiHeadAttention` | `onnx.MultiHeadAttention` | `aisle.MultiHeadAttention` | RedMulE schedule, 3 tiles |
| Scaled dot-product attention (3.2.1) | `com.isolde.ScaledDotProductAttention` | inlined only if MHA is decomposed | – | – |
| Position-wise FFN (3.3) | `com.isolde.PositionwiseFeedForward` | `onnx.PositionwiseFeedForward` | `aisle.PositionwiseFeedForward` | RedMulE schedule, d_ff/16 tiles |
| Add & Norm | `Add` (Norm = identity) | `onnx.Add` | folded into `C` | Y preload |

* `--functions-to-decompose=MultiHeadAttention` (and/or
  `--functions-to-decompose=PositionwiseFeedForward`; repeat the flag, it is
  not comma separated) makes the importer inline the function bodies instead,
  e.g. for a CPU target; a node whose domain is not `com.isolde` always takes
  that path.
* `convert-onnx-to-aisle` first folds constant `scale` / `post_scale` into
  `Wq` / `Wv` and a following residual `Add` into the accumulator `C`.
* `convert-aisle-to-aismem` currently covers one head, L = 12, d_model = 16,
  d_ff = 16k, f16, `normalization = "relu"` (every product is one native
  `Y[12x16] += X[12x16] W[16x16]` launch); anything else is diagnosed.
  Host-side transforms (K^T, zero padding, W1/W2 tiles, ReLU) are done by
  `aismem.RedMulEUploadTile`, i.e. `omrm_upload_tile_f16` in
  `isolde/system/bsp/onnx_redmule_runtime.c`, on raw FP16 bits.

Test: `test-isolde/transformer` (`make golden check BLOCK=layer|mha|ffn`) and
`test/mlir/isolde/*.mlir`.
