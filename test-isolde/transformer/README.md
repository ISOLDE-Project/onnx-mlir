# transformer: ISOLDE encoder blocks on RedMulE

Compiles the `com.isolde` transformer building blocks (Vaswani et al. 2017)
down to the explicit RedMulE/SPM schedule, and checks that schedule
bit-exactly on the host.

```
make golden BLOCK=layer      # models/transformer_layer.onnx + .npz
make check  BLOCK=layer      # onnx-mlir --EmitSPADEMLIR + check_aismem.py
make graph  BLOCK=mha        # all SPADE emission levels (common.mk)
```

`BLOCK` is one of

| BLOCK | graph | RedMulE launches (waits) | DMEM<->SPM per call | resident |
|---|---|---|---|---|
| `mha`   | `h + MultiHeadAttention(h, h)` | 6 (4) | 1,088 up, 576 down | 1,024 |
| `ffn`   | `h + PositionwiseFeedForward(h)` (d_ff = `D_FF`, default 48) | 6 (6) | 192 up, 192 down | 1,536 |
| `layer` | `EncoderLayer` = both, residuals included | 12 (10) | 1,088 up, 576 down | 2,560 |
| `proj`  | `MatMul([1,12,32], [32,16]) + P` (input embedding + positional encoding) | 3 (3) | 384 up, 192 down | 960 |
| `proj_layer` | `proj` followed by `EncoderLayer` | 15 (13) | 1,280 up, 960 down | 3,520 |

(fp16 values.)  Attention uses the three RedMulE tiles: Q, K and V are
launched on tiles 0, 1 and 2 together and share one wait; the block input is
uploaded to (or, when it is already in SPM, moved to) each of them, and K^T
(transposed on the way) and V are moved to tile 0 through data memory
(`aismem.SPMMoveTile`, `omrm_spm_move_f16`), where S, O and Y are computed.
Everything else stays in the SPM of tile 0: Y buffers feed the next GEMM
directly, the residual Adds accumulate in place, ReLU runs in SPM.  Constant
weights are resident (Wk in tile 1, Wv in tile 2): `main_graph_preload()`
uploads them once.  Outside attention, every launch is followed by its wait.

MatMul and Add use the same launch, `Y = X . W + Y`: the `aisle-tile` pass
splits the MatMul into two chained K-tile launches on `aisle.Window` views
(the first zeroes Y, the second accumulates); Add sets X to the 12x16 identity
(resident), W to one operand zero padded to 16 rows, and Y to the other
operand (in place when it may be overwritten).

A rank-2 MatMul followed by an Add is canonicalized by onnx-mlir into
`onnx.Gemm` (see `test-isolde/projection`).  It becomes one `aisle.GEMM`,
which `aisle-tile` splits into K-tiles (and N-tiles when wider than 16): the
Add costs no extra launch, C is uploaded into Y of the first K-tile
(firmware `launch_bias`), the next K-tile accumulates (`launch_accumulate`).
`transB = 1` uploads the W windows transposed.  Windows are `aisle.Window`
views folded into the upload offsets, never `onnx.Slice` (no RedMulE
lowering, and not legal inside ONNXToAISLE); see `src/Dialect/AISLE/AISLE.md`.

* `models/generate_transformer.py` writes the model (every `com.isolde` op
  carries a FunctionProto body, so onnxruntime runs it as-is) and an `.npz`
  with the input, the bit-exact FP16 RedMulE reference `y_redmule` and a
  float64 reference `y_float`.  Scales are left unfolded on purpose.
* `check_aismem.py` runs `main_graph_preload` and then `main_graph` twice on
  a model of the tile-private SPMs (`aismem.RedMulE*`, `aismem.SPM*`),
  compares both results bit-exactly with `y_redmule`, tracks which buffer
  owns every SPM row (a read of rows another live buffer overwrote is an
  error), reports the data moved, and fails if host arithmetic is left over.

Current limits of the RedMulE lowering (diagnosed, not miscompiled): one head,
L = 12, d_model = 16, d_ff multiple of 16, f16, `normalization = "relu"`,
scales foldable into constant weights.  The runtime needs
`omrm_upload_tile_f16`, `omrm_spm_move_f16` and
`omrm_spm_{relu,transpose,copy}_f16` in
`isolde/system/bsp/onnx_redmule_runtime.c`; firmware must call
`main_graph_preload()` once before the first inference.

## Example
```sh
make BLOCK=mha DEBUG_DIALECT_CONVERSION=yes ONNX_IR_DUMP=after ONNX_DEBUG_LOG_DIR=debug-mha  graph
```