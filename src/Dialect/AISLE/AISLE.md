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

## MatMul, Gemm and Add on RedMulE

`convert-onnx-to-aisle` also lowers f16 `onnx.MatMul` `[.., M, 16K] x [16K,
N]`, f16 `onnx.Gemm` `[M, 16K] x [16K, N] (+ [M, N])` (alpha = beta = 1,
transA = 0, any transB; onnx-mlir fuses a rank-2 MatMul + Add into it), with
M <= 12 and N a multiple of 16 or N < 16, f16 `onnx.Add` `[.., 12, 16] + [..,
12, 16]` and f16 `onnx.ReduceMean` over the frames axis of `[.., L <= 12, D <=
16]` to `aisle.MatMul` / `aisle.GEMM` / `aisle.Add` / `aisle.ReduceMean`;
other shapes and types stay on their previous paths.  Products are kept whole
at this point.

### Tiling: `aisle-tile`, `aisle.Window`, `aisle.Concat`

The `aisle-tile` pass (right after `convert-onnx-to-aisle`) first turns
`Add(MatMul(A, B), C)` with an `[M, N]` C (no broadcast beyond leading 1s,
the MatMul has no other user) into `GEMM(A, B, C)`: onnx-mlir only does this
for rank 2, and the rank-3 `[1, 12, 32]` input projection + positional
encoding of the encoder needs it to accumulate onto C like the firmware.  It
then splits every such
product that is larger than one RedMulE launch into native launches
`Y[12x16] (+)= X[12x16] . W[16x16]` on views of its operands:

```
for n < N:                              independent output tiles
  acc = Window(C, [0, 16n])  | none     (Gemm | MatMul)
  for k < K:                            chained, accumulating
    acc = GEMM(Window(A, [0, 16k]), Window(B, [16k, 16n]), acc)
          (MatMul for the first launch without C: Y = 0)
Y = Concat(acc_0 .. acc_N-1, axis = last)
```

* `aisle.Window` is a static rectangular view (offsets; the result type gives
  the sizes).  It never copies: in AISLE -> AISMEM a Window of data memory
  becomes a `memref.subview` that the upload reading it folds into its
  `row_offset` / `col_offset`, and a Window of a tiled SPM result selects a
  tile.  A Window of a Window folds into one; a Window that selects a whole
  piece of a Concat folds to that piece, so a product that consumes a tiled
  product reads its tiles straight from SPM.
* `aisle.Concat` assembles the N-tiles.  In AISMEM the tiles stay in their
  SPM buffers; only a non-RedMulE user makes them go back to data memory, one
  strided `aismem.RedMulEDownload` (`dst_offset`, `dst_ld`) per tile.
* Tiles are named `<onnx node>[n<n>,k<k>]`, which also names their SPM
  buffers.  All tiles run on RedMulE instance 0 for now; M is not tiled.
* Partial tiles: M < 12 rows and one N-tile narrower than 16 columns run the
  same launch on zero-padded operands (rows of Y depend only on the same rows
  of X, columns only on the same columns of W); only the M x N corner is
  downloaded (`aismem.RedMulEDownload` `cols`, runtime
  `omrm_download_tile_f16(tile, spm, dst, dst_ld, rows, cols)`, also used
  for the strided N-tile downloads).
* Constant operands are split at compile time instead of windowed: an
  `onnx.Constant` read only by products being tiled becomes one constant per
  tile (a transB tile already transposed, so it carries transB = 0), and the
  original constant disappears.  Each W / C tile is then a contiguous 16x16 /
  12x16 global, uploaded with the runtime's plain-copy path (no offsets, no
  transpose); identical tiles are merged by CSE, their uploads by
  `aismem-spm-allocate`.

Each tile then becomes one launch `Y = X . W + Y`:

* MatMul: `Y = 0`;
* GEMM: `Y = C`: a C from data memory is uploaded into Y (firmware
  `launch_bias`); the previous K-tile's result is accumulated in place
  (`launch_accumulate`).  transB uploads the W window transposed;
* Add: `X = I` (12x16 identity, resident, one per function), `W = pad16(A)`
  (resident when constant), `Y = B`, in place when B may be overwritten, so
  `Y = I . W + Y = A + B` in one launch.
* ReduceMean (radar_attention `tf_pool`): `X = POOL` (fp16(1/L) in columns
  0..L-1 of every row, resident), `W = pad16(H)`, `Y = 0`: every row of Y is
  the mean over H's L rows.  The result stays in SPM; a following product
  uses the whole replicated buffer as X (the classifier head), and a download
  copies row 0.

Every upload of a constant is prepared at compile time: AISLE -> AISMEM
stores the exact SPM image the upload would produce (window, transpose, zero
padding to 16 rows, e.g. the Add's `pad16(P)`) as its own `[rows x 16]`
`krnl.global` (`<constant>_spm_r<row>c<col>_<rows>x<cols>_<dst rows>[t]`) and
uploads that as a plain copy.  A constant read only through such uploads is
then dropped, so the only extra data is the padding rows.

Constant W windows are resident (uploaded once by `<entry>_preload`), and
`aismem-spm-allocate` merges repeated uploads of the same data-memory window
(e.g. the X tile shared by all N-tiles).  The radar encoder's input
projection + positional encoding (rank 2 as Gemm, rank 3 as MatMul + Add) is
2 launches and bit-exact with the firmware's `tf_golden_proj`.

The whole 2-layer radar encoder (`test-isolde/attention`, `make check`) then
compiles without any Krnl code: 28 launches (as the firmware; 24 waits
against the firmware's 20 barriers), logits bit-exact with
`tf_logits_golden`, and a 4-element download of row 0 of the head's Y.

## SPM-resident activations and SPM row management

The RedMulE schedules keep every intermediate in the tile-private SPMs: a
GEMM's Y buffer is the X or W operand of the next one on the same tile, the
residual accumulates into the block input's own rows (`h = O Wo + h`), and
ReLU runs in place in SPM (`aismem.SPMRelu`, `aismem.SPMCopy`;
`omrm_spm_*_f16` on the core today, loader modes later).  Data memory is
touched only where a block's input or result meets something that is not a
RedMulE block (download is materialized by `convert-aisle-to-aismem`), and
where a value moves between tiles.

Multi-tile schedules.  The emitter (`SPMEmitter.hpp`) is bound to one RedMulE
instance and tags every buffer and operation with it; `emit.on(t)` targets
another tile.  `launch()` starts a GEMM without waiting and `wait()` retires
any set of launches with one `aismem.RedMulEWait` (mask = their tiles), so
independent launches on different tiles overlap; `gemm()` is launch + wait.
A RedMulE instance only reads its own SPM, and the tiles have no SPM-to-SPM
path: `emit.move()` emits `aismem.SPMMoveTile` (runtime `omrm_spm_move_f16`:
download from the source tile, upload to the destination, optionally
transposed, zero padded).  MultiHeadAttention uses it: Q, K and V run on
tiles 0, 1 and 2 behind one wait (mask 0x7), then K^T (transposed on the way,
replacing `SPMTranspose`) and V (padded to 16 rows) move to tile 0 for
S = ReLU(Q K^T), O = S V and Y = O Wo + h.

Buffers are `aismem.SPMAlloc` ops (`tile`, `rows`, `name`, `resident`).  The
`aismem-spm-allocate` pass (in the spade pipeline right after
`convert-aisle-to-aismem`) assigns their rows:

* resident buffers (constant weights) at the bottom of the SPM, for the whole
  program; their uploads move into a generated `<entry>_preload()` function
  (`void main_graph_preload(void)`), which firmware calls once after reset;
* all other buffers by a first-fit linear scan over their lifetimes in
  program order (a buffer read by a launch lives until the wait that retires
  it), so dead rows are reused;
* capacity: `rows-per-tile`, onnx-mlir `--redmule-spm-rows` (default 256).
  The narrow window of `demo_3` is 32 KiB (512 rows), but the tmp/cluster
  RTL addresses each tile's bank memories with 8 row bits
  (`isolde_log_interconnect`: `addr[TCDM_AW-1:2] = mems_add`, TCDM_AW = 10),
  so row r + 256 is row r.  A 512-row schedule for the 2-layer radar encoder
  (384 rows on tile 0) ran on RTL and produced NaN logits.  Use 512 only
  on an RTL that addresses the whole window;
* if a tile does not fit, resident weights are demoted to per-call uploads,
  one at a time (the one giving the smallest peak), with a warning on
  stderr: the 2-layer encoder keeps 4,032 fp16 resident and uploads 8
  weights (2,048 fp16) per call in 256 rows.  Overflow even without residents
  is an error that lists every buffer;
* other options: `resident-weights` (`--redmule-resident-weights`, default
  true), `print-map` (`--redmule-print-spm-map`).  The function gets
  `aismem.spm_rows_used`.  `check_aismem.py` flags rows beyond `SPM_ROWS`
  (environment, default 256).

Per encoder layer (radar model), with attention on three tiles, this moves
1,088 values into SPM and 576 out (the layer input to three tiles, K and V
through data memory, the result), against 7,680 in the download/upload
schedule; the core touches 768 values inside SPM (ReLU S, ReLU U).

Test: `test-isolde/transformer` (`make golden check BLOCK=layer|mha|ffn`) and
`test/mlir/isolde/*.mlir`.
