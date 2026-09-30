// RUN: onnx-mlir-opt --aismem-spm-allocate="print-map=true" %s | FileCheck %s --check-prefix=FIT
// RUN: onnx-mlir-opt --aismem-spm-allocate="print-map=true rows-per-tile=48" %s 2>&1 | FileCheck %s --check-prefix=DEMOTE
// RUN: not onnx-mlir-opt --aismem-spm-allocate="rows-per-tile=36" %s 2>&1 | FileCheck %s --check-prefix=OVERFLOW

// Two chained GEMMs, each with its own constant weight.  Both weights
// resident need 56 rows; with fewer, weights are demoted to per-call uploads
// (live only around their GEMM) until the tile fits.
func.func @two(%arg0: memref<12x16xf16>) -> memref<12x16xf16> {
  %w1 = "krnl.global"() {name = "w1", shape = [16, 16], value = dense<1.000000e+00> : tensor<16x16xf16>} : () -> memref<16x16xf16>
  %w1a = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 16 : i32, resident = true, name = "W1"}> : () -> i32
  %w1n, %w1t = "aismem.RedMulEUploadTile"(%w1, %w1a) <{tile = 0 : i32, row_offset = 0 : i32, col_offset = 0 : i32, rows = 16 : i32, cols = 16 : i32, dst_rows = 16 : i32}> : (memref<16x16xf16>, i32) -> (i32, none)
  %x = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 12 : i32, name = "X"}> : () -> i32
  %xn, %xt = "aismem.RedMulEUploadTile"(%arg0, %x) <{tile = 0 : i32, row_offset = 0 : i32, col_offset = 0 : i32, rows = 12 : i32, cols = 16 : i32, dst_rows = 12 : i32}> : (memref<12x16xf16>, i32) -> (i32, none)
  %a = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 12 : i32, name = "A"}> : () -> i32
  %az = "aismem.RedMulEZero"(%a, %xt) <{tile = 0 : i32, elements = 192 : i32}> : (i32, none) -> none
  %ag = "aismem.RedMulEGEMM"(%x, %w1a, %a, %az) <{tile = 0 : i32, k = 16 : i32, m = 12 : i32, n = 16 : i32}> : (i32, i32, i32, none) -> none
  %aw = "aismem.RedMulEWait"(%ag) <{mask = 1 : i32}> : (none) -> none
  %w2 = "krnl.global"() {name = "w2", shape = [16, 16], value = dense<2.000000e+00> : tensor<16x16xf16>} : () -> memref<16x16xf16>
  %w2a = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 16 : i32, resident = true, name = "W2"}> : () -> i32
  %w2n, %w2t = "aismem.RedMulEUploadTile"(%w2, %w2a) <{tile = 0 : i32, row_offset = 0 : i32, col_offset = 0 : i32, rows = 16 : i32, cols = 16 : i32, dst_rows = 16 : i32}> : (memref<16x16xf16>, i32) -> (i32, none)
  %c = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 12 : i32, name = "C"}> : () -> i32
  %cz = "aismem.RedMulEZero"(%c, %aw) <{tile = 0 : i32, elements = 192 : i32}> : (i32, none) -> none
  %cg = "aismem.RedMulEGEMM"(%a, %w2a, %c, %cz) <{tile = 0 : i32, k = 16 : i32, m = 12 : i32, n = 16 : i32}> : (i32, i32, i32, none) -> none
  %cw = "aismem.RedMulEWait"(%cg) <{mask = 1 : i32}> : (none) -> none
  %out = memref.alloc() {alignment = 16 : i64} : memref<12x16xf16>
  %d = "aismem.RedMulEDownload"(%c, %out, %cw) <{tile = 0 : i32, elements = 192 : i32}> : (i32, memref<12x16xf16>, none) -> none
  return %out : memref<12x16xf16>
}

// Everything fits in the default 256 rows: both weights resident.
// FIT:       SPM map of @two (256 rows per tile, 64 B per row)
// FIT-NEXT:    tile 0: 56 rows used
// FIT-NEXT:      rows [0, 16)  W1  resident
// FIT-NEXT:      rows [16, 32)  W2  resident
// FIT-LABEL: func.func @two_preload()
// FIT-COUNT-2: "aismem.RedMulEUploadTile"

// 48 rows: both weights become per-call uploads; W2 reuses W1's rows.
// DEMOTE:      warning: @two: SPM of tile 0 (48 rows): 2 constant buffer(s), 32 rows, are uploaded on every call instead of once by the preload function
// DEMOTE:      SPM map of @two (48 rows per tile, 64 B per row)
// DEMOTE-NEXT:   tile 0: 40 rows used
// DEMOTE-NEXT:     rows [0, 16)  W1  live {{.*}}  (constant, uploaded per call)
// DEMOTE-NEXT:     rows [0, 16)  W2  live {{.*}}  (constant, uploaded per call)
// DEMOTE-LABEL: func.func @two
// DEMOTE:        "aismem.SPMAlloc"() <{name = "W1", resident = false, row = 0 : i32
// DEMOTE-NEXT:   "aismem.RedMulEUploadTile"
// DEMOTE:        "aismem.SPMAlloc"() <{name = "W2", resident = false, row = 0 : i32
// DEMOTE-NEXT:   "aismem.RedMulEUploadTile"
// DEMOTE-NOT:  func.func @two_preload

// OVERFLOW: error: SPM of tile 0 overflows: 40 rows needed, 36 available (0 resident)
