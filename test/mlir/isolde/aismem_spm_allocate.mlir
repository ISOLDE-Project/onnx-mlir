// RUN: onnx-mlir-opt --aismem-spm-allocate="print-map=true" %s | FileCheck %s
// RUN: not onnx-mlir-opt --aismem-spm-allocate="rows-per-tile=36" %s 2>&1 | FileCheck %s --check-prefix=OVERFLOW
// RUN: onnx-mlir-opt --aismem-spm-allocate="resident-weights=false" %s | FileCheck %s --check-prefix=STREAM

// Three chained GEMMs on one tile, W resident.  A buffer read by a launch is
// live until the wait that retires it; dead rows are reused (first fit).
func.func @chain(%arg0: memref<12x16xf16>) -> memref<12x16xf16> {
  %w = "krnl.global"() {name = "w", shape = [16, 16], value = dense<1.000000e+00> : tensor<16x16xf16>} : () -> memref<16x16xf16>
  %wa = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 16 : i32, resident = true, name = "W"}> : () -> i32
  %wn, %wt = "aismem.RedMulEUploadTile"(%w, %wa) <{tile = 0 : i32, row_offset = 0 : i32, col_offset = 0 : i32, rows = 16 : i32, cols = 16 : i32, dst_rows = 16 : i32}> : (memref<16x16xf16>, i32) -> (i32, none)
  %x = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 12 : i32, name = "X"}> : () -> i32
  %xn, %xt = "aismem.RedMulEUploadTile"(%arg0, %x) <{tile = 0 : i32, row_offset = 0 : i32, col_offset = 0 : i32, rows = 12 : i32, cols = 16 : i32, dst_rows = 12 : i32}> : (memref<12x16xf16>, i32) -> (i32, none)
  %a = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 12 : i32, name = "A"}> : () -> i32
  %az = "aismem.RedMulEZero"(%a, %xt) <{tile = 0 : i32, elements = 192 : i32}> : (i32, none) -> none
  %ag = "aismem.RedMulEGEMM"(%x, %wa, %a, %az) <{tile = 0 : i32, k = 16 : i32, m = 12 : i32, n = 16 : i32}> : (i32, i32, i32, none) -> none
  %aw = "aismem.RedMulEWait"(%ag) <{mask = 1 : i32}> : (none) -> none
  %b = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 12 : i32, name = "B"}> : () -> i32
  %bz = "aismem.RedMulEZero"(%b, %aw) <{tile = 0 : i32, elements = 192 : i32}> : (i32, none) -> none
  %bg = "aismem.RedMulEGEMM"(%a, %wa, %b, %bz) <{tile = 0 : i32, k = 16 : i32, m = 12 : i32, n = 16 : i32}> : (i32, i32, i32, none) -> none
  %bw = "aismem.RedMulEWait"(%bg) <{mask = 1 : i32}> : (none) -> none
  %c = "aismem.SPMAlloc"() <{tile = 0 : i32, rows = 12 : i32, name = "C"}> : () -> i32
  %cz = "aismem.RedMulEZero"(%c, %bw) <{tile = 0 : i32, elements = 192 : i32}> : (i32, none) -> none
  %cg = "aismem.RedMulEGEMM"(%b, %wa, %c, %cz) <{tile = 0 : i32, k = 16 : i32, m = 12 : i32, n = 16 : i32}> : (i32, i32, i32, none) -> none
  %cw = "aismem.RedMulEWait"(%cg) <{mask = 1 : i32}> : (none) -> none
  %out = memref.alloc() {alignment = 16 : i64} : memref<12x16xf16>
  %d = "aismem.RedMulEDownload"(%c, %out, %cw) <{tile = 0 : i32, elements = 192 : i32}> : (i32, memref<12x16xf16>, none) -> none
  return %out : memref<12x16xf16>
}

// CHECK:       SPM map of @chain (256 rows per tile, 64 B per row)
// CHECK-NEXT:    tile 0: 40 rows used
// CHECK-NEXT:      rows [0, 16)  W  resident
// CHECK-NEXT:      rows [16, 28)  X  live
// CHECK-NEXT:      rows [16, 28)  B  live
// CHECK-NEXT:      rows [28, 40)  A  live
// CHECK-NEXT:      rows [28, 40)  C  live

// CHECK-LABEL: func.func @chain
// CHECK-SAME:    aismem.spm_rows_used = {tile0 = 40 : i64}
// CHECK:         "aismem.SPMAlloc"() <{name = "W", resident = true, row = 0 : i32
// The weight upload has moved to the preload function.
// CHECK-NOT:     "aismem.RedMulEUploadTile"(%{{.*}}krnl
// CHECK:         "aismem.SPMAlloc"() <{name = "X", resident = false, row = 16 : i32
// CHECK:         "aismem.SPMAlloc"() <{name = "A", resident = false, row = 28 : i32
// CHECK:         "aismem.SPMAlloc"() <{name = "B", resident = false, row = 16 : i32
// CHECK:         "aismem.SPMAlloc"() <{name = "C", resident = false, row = 28 : i32
// CHECK:         return

// CHECK-LABEL: func.func @chain_preload()
// CHECK-SAME:    aismem.preload
// CHECK:         [[G:%.+]] = "krnl.global"() {name = "w"
// CHECK:         [[A:%.+]] = "aismem.SPMAlloc"() <{name = "W", resident = true, row = 0 : i32
// CHECK:         "aismem.RedMulEUploadTile"([[G]], [[A]])
// CHECK:         return

// W is needed by all three GEMMs: demoting it does not help.
// OVERFLOW: error: SPM of tile 0 overflows: 40 rows needed, 36 available (0 resident)

// Without residency W is an ordinary buffer: uploaded per call, no preload.
// STREAM-LABEL: func.func @chain
// STREAM:       "aismem.SPMAlloc"() <{name = "W", resident = true, row = 0 : i32
// STREAM:       "aismem.RedMulEUploadTile"
// STREAM-NOT:   func.func @chain_preload
