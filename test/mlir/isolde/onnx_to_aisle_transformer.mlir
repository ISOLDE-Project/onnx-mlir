// RUN: onnx-mlir-opt --convert-onnx-to-aisle --split-input-file %s | FileCheck %s

// Encoder layer: constant scales are folded into Wq / Wv, the two residual
// Adds become the accumulator C of the AISLE ops.
func.func @encoder_layer(%h: tensor<1x12x16xf16>) -> tensor<1x12x16xf16> {
  %wq = onnx.Constant dense<1.000000e+00> : tensor<16x16xf16>
  %wk = onnx.Constant dense<2.000000e+00> : tensor<16x16xf16>
  %wv = onnx.Constant dense<3.000000e+00> : tensor<16x16xf16>
  %wo = onnx.Constant dense<4.000000e+00> : tensor<16x16xf16>
  %w1 = onnx.Constant dense<5.000000e-01> : tensor<16x48xf16>
  %w2 = onnx.Constant dense<2.500000e-01> : tensor<48x16xf16>
  %none = "onnx.NoValue"() {value} : () -> none
  %attn = "onnx.MultiHeadAttention"(%h, %h, %wq, %wk, %wv, %wo, %none) {normalization = "relu", num_heads = 1 : si64, post_scale = 5.000000e-01 : f32, scale = 2.500000e-01 : f32} : (tensor<1x12x16xf16>, tensor<1x12x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, none) -> tensor<1x12x16xf16>
  %h1 = "onnx.Add"(%h, %attn) : (tensor<1x12x16xf16>, tensor<1x12x16xf16>) -> tensor<1x12x16xf16>
  %ffn = "onnx.PositionwiseFeedForward"(%h1, %w1, %w2, %none) {activation = "relu"} : (tensor<1x12x16xf16>, tensor<16x48xf16>, tensor<48x16xf16>, none) -> tensor<1x12x16xf16>
  %y = "onnx.Add"(%ffn, %h1) : (tensor<1x12x16xf16>, tensor<1x12x16xf16>) -> tensor<1x12x16xf16>
  return %y : tensor<1x12x16xf16>
}
// CHECK-LABEL: func.func @encoder_layer
// CHECK-SAME:    ([[H:%.+]]: tensor<1x12x16xf16>)
// CHECK-DAG:     [[WQ:%.+]] = onnx.Constant dense<2.500000e-01> : tensor<16x16xf16>
// CHECK-DAG:     [[WV:%.+]] = onnx.Constant dense<1.500000e+00> : tensor<16x16xf16>
// CHECK:         [[ATTN:%.+]] = "aisle.MultiHeadAttention"([[H]], {{%.+}}, [[H]], {{%.+}}, [[WQ]], {{%.+}}, {{%.+}}, {{%.+}}, [[WV]], {{%.+}}, {{%.+}}, {{%.+}}, [[H]])
// CHECK-SAME:      normalization = "relu"
// CHECK-SAME:      post_scale = 1.000000e+00 : f32, scale = 1.000000e+00 : f32
// CHECK:         [[Y:%.+]] = "aisle.PositionwiseFeedForward"([[ATTN]], {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, [[ATTN]])
// CHECK-NOT:     onnx.Add
// CHECK:         return [[Y]]

// -----

// Without scale the default 1/sqrt(d_k) = 0.25 is used; a broadcasting Add
// is not a residual and stays an onnx.Add.
func.func @default_scale_no_fusion(%h: tensor<1x12x16xf16>, %b: tensor<16xf16>) -> tensor<1x12x16xf16> {
  %w = onnx.Constant dense<1.000000e+00> : tensor<16x16xf16>
  %none = "onnx.NoValue"() {value} : () -> none
  %attn = "onnx.MultiHeadAttention"(%h, %h, %w, %w, %w, %w, %none) {normalization = "relu"} : (tensor<1x12x16xf16>, tensor<1x12x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, none) -> tensor<1x12x16xf16>
  %y = "onnx.Add"(%attn, %b) : (tensor<1x12x16xf16>, tensor<16xf16>) -> tensor<1x12x16xf16>
  return %y : tensor<1x12x16xf16>
}
// CHECK-LABEL: func.func @default_scale_no_fusion
// CHECK:         onnx.Constant dense<2.500000e-01> : tensor<16x16xf16>
// CHECK:         [[ATTN:%.+]] = "aisle.MultiHeadAttention"
// CHECK-SAME:      scale = 1.000000e+00 : f32
// CHECK:         "onnx.Add"([[ATTN]], {{%.+}})
