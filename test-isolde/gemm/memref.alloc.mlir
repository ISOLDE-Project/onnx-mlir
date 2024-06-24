module attributes {llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128", llvm.target_triple = "riscv32-unknown-elf", "onnx-mlir.symbol-postfix" = "graph"} {
  func.func @main_graph(%arg0: memref<1x8xf32> {onnx.name = "A"}) -> (memref<1x10xf32> {onnx.name = "Y"}) attributes {llvm.emit_c_interface} {
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
    return %alloc_1 : memref<1x10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8] , \22name\22 : \22A\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22Y\22 }\0A\0A]\00"} : () -> ()
}
