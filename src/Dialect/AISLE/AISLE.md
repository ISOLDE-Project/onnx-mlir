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

## MatMul and Add on RedMulE

`convert-onnx-to-aisle` also lowers f16 `onnx.MatMul` `[.., 12, 16K] x [16K,
16]` (for now the input projection `[12, 32] x [32, 16]`) and f16 `onnx.Add`
`[.., 12, 16] + [.., 12, 16]` to `aisle.MatMul` / `aisle.Add`; other shapes
and types stay on the Krnl path.  Both become the native launch
`Y = X . W + Y`:

* MatMul: `Y = 0`, then one launch per 16-wide K-tile
  (`X = A[:, 16k:16k+16]`, `W = B[16k:16k+16, :]`, resident), Y accumulates;
* Add: `X = I` (12x16 identity, resident, one per function), `W = pad16(A)`
  (resident when constant), `Y = B`, in place when B may be overwritten, so
  `Y = I . W + Y = A + B` in one launch.

For the radar encoder this puts the input projection and the positional
encoding on RedMulE (3 launches).  Note: the firmware instead preloads the
positional encoding into the projection's Y (2 launches), which rounds
differently; the compiled projection differs from `tf_golden_proj` by at most
1.2e-3 and the logits keep the same class.

## SPM-resident activations and SPM row management

The RedMulE schedules keep every intermediate in the tile-private SPM
(single-tile chain on tile 0): a GEMM's Y buffer is the X or W operand of the
next one, V is written into a zero-filled 16-row buffer (already `pad16(V)`),
the residual accumulates into the block input's own rows (`h = O Wo + h`),
and ReLU / `K^T` run in place in SPM (`aismem.SPMRelu`,
`aismem.SPMTranspose`, `aismem.SPMCopy`; `omrm_spm_*_f16` on the core
today, loader modes later).  Data memory is touched only where a block's
input or result meets something that is not a RedMulE block (download is
materialized by `convert-aisle-to-aismem`).

Buffers are `aismem.SPMAlloc` ops (`tile`, `rows`, `name`, `resident`).  The
`aismem-spm-allocate` pass (in the spade pipeline right after
`convert-aisle-to-aismem`) assigns their rows:

* resident buffers (constant weights) at the bottom of the SPM, for the whole
  program; their uploads move into a generated `<entry>_preload()` function
  (`void main_graph_preload(void)`), which firmware calls once after reset;
* all other buffers by a first-fit linear scan over their lifetimes in
  program order (a buffer read by a launch lives until the wait that retires
  it), so dead rows are reused;
* options: `rows-per-tile` (512 = the 32 KiB narrow window of `demo_3`),
  `resident-weights` (default true), `print-map`.  Overflow is an error that
  lists every buffer.  The function gets `aismem.spm_rows_used`.

Per encoder layer (radar model) this moves 0 values between data memory and
SPM, against 7,680 in the download/upload schedule; the core touches 1,216
values inside SPM (K^T, ReLU S, ReLU U).  Two layers use 388 of 512 rows of
tile 0 including the resident weights.

Test: `test-isolde/transformer` (`make golden check BLOCK=layer|mha|ffn`) and
`test/mlir/isolde/*.mlir`.
