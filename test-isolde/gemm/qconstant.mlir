module attributes {llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128", llvm.target_triple = "riscv32-unknown-elf", "onnx-mlir.symbol-postfix" = "graph"} {
  func.func @main_graph(%arg0: tensor<1x8xf32> {onnx.name = "A"}) -> (tensor<1x4xi32> {onnx.name = "Y"}) {
    %0 = "aisle.qconstant"() <{name = "A_shape", shape = [1, 4], value = dense<[[1, 1, 1, 8]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
    return %0 : tensor<1x4xi32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
