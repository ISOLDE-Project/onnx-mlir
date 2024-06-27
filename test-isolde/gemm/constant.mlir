module attributes {llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128", llvm.target_triple = "riscv32-unknown-elf", "onnx-mlir.symbol-postfix" = "graph"} {
  func.func @main_graph(%arg0: tensor<1x8xf32> {onnx.name = "A"}) -> (tensor<1x10xf32> {onnx.name = "Y"}) {
    %0 = onnx.Constant dense<[[0.462124646, 0.466326267, 0.3663297, 0.0929221734, 0.466550738, 0.568839073, 0.669290781, 0.877571046, 0.820650458, 0.289381236]]> : tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
