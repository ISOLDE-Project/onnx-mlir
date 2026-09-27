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

| BLOCK | graph | RedMulE launches / barriers |
|---|---|---|
| `mha`   | `h + MultiHeadAttention(h, h)` | 6 / 4 |
| `ffn`   | `h + PositionwiseFeedForward(h)` (d_ff = `D_FF`, default 48) | 6 / 4 |
| `layer` | `EncoderLayer` = both, residuals included | 12 / 8 |

The counts match `ibex/isolde/sw/radar_attention` (per layer: 3 + 1 + 1 + 1
launches for attention, 3 + 3 for the FFN; the residual Adds are free, they
are preloaded into RedMulE's Y).

* `models/generate_transformer.py` writes the model (every `com.isolde` op
  carries a FunctionProto body, so onnxruntime runs it as-is) and an `.npz`
  with the input, the bit-exact FP16 RedMulE reference `y_redmule` and a
  float64 reference `y_float`.  Scales are left unfolded on purpose.
* `check_aismem.py` interprets the entry function of `graph.spade.mlir`
  (`aismem.RedMulE*` ops on a model of the tile-private SPMs) and compares the
  result with `y_redmule`; it fails if any host arithmetic is left over.

Current limits of the RedMulE lowering (diagnosed, not miscompiled): one head,
L = 12, d_model = 16, d_ff multiple of 16, f16, `normalization = "relu"`,
scales foldable into constant weights.  The runtime needs
`omrm_upload_tile_f16` in `isolde/system/bsp/onnx_redmule_runtime.c` and a
platform with three RedMulE tiles (`demo_3`).
