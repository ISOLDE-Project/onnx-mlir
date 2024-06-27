module attributes {llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128", llvm.target_triple = "riscv32-unknown-elf", "onnx-mlir.symbol-postfix" = "graph"} {
  func.func @main_graph(%arg0: tensor<1x8xf32> {onnx.name = "A"}) -> (tensor<1x10xf32> {onnx.name = "Y"}) {
    %0 = onnx.Constant dense<[[0.731334209, 0.358675748, 0.00849229749, 0.480855376, 0.90700668, 0.599353552, 0.485235959, 0.336357206], [0.320891291, 0.676933109, 0.558484852, 0.973365604, 0.520273268, 0.268208086, 0.593142331, 0.517368436], [9.927560e-01, 0.164088205, 0.226776063, 0.416885704, 0.288721681, 0.446609169, 0.761447787, 0.164485082], [0.6006881, 0.609213829, 0.651609361, 0.94865638, 0.445503294, 0.162826315, 0.249741212, 0.44077161], [0.478716791, 0.342522115, 2.685040e-01, 0.659137964, 6.91016553E-4, 0.904493033, 0.331368297, 0.55342716], [0.0217391588, 0.709012032, 0.602195084, 0.243997544, 0.196936011, 0.104713961, 0.849715888, 0.708038151], [0.377636462, 0.0219174828, 0.338509649, 0.389775902, 0.252068877, 0.218442991, 0.0997830256, 0.283064663], [0.442542136, 0.621983409, 0.247031316, 0.731752157, 0.231725186, 0.60557264, 0.297909588, 0.159199685], [0.203066885, 0.52630657, 0.00111013884, 0.198113248, 0.311970681, 0.0666151717, 0.196403354, 0.576940596], [0.311937451, 0.337957829, 0.593948185, 0.49952358, 0.447188675, 0.54532367, 0.161518514, 0.441531122]]> : tensor<10x8xf32>
    %1 = onnx.Constant dense<[[0.462124646, 0.466326267, 0.3663297, 0.0929221734, 0.466550738, 0.568839073, 0.669290781, 0.877571046, 0.820650458, 0.289381236]]> : tensor<1x10xf32>
    %2 = "onnx.Gemm"(%arg0, %0, %1) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "onnx.Gemm_0", transA = 0 : si64, transB = 1 : si64} : (tensor<1x8xf32>, tensor<10x8xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %2 : tensor<1x10xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
