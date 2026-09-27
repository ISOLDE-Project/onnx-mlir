// RUN: onnx-mlir-opt --split-input-file --verify-diagnostics %s

func.func @bad_normalization(%x: tensor<12x16xf32>, %w: tensor<16x16xf32>) -> tensor<12x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{normalization must be "softmax" or "relu"}}
  %y = "onnx.MultiHeadAttention"(%x, %x, %w, %w, %w, %w, %none) {normalization = "sigmoid"} : (tensor<12x16xf32>, tensor<12x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, none) -> tensor<12x16xf32>
  return %y : tensor<12x16xf32>
}

// -----

func.func @softmax_post_scale(%x: tensor<12x16xf32>, %w: tensor<16x16xf32>) -> tensor<12x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{post_scale must be 1.0 with softmax normalization}}
  %y = "onnx.MultiHeadAttention"(%x, %x, %w, %w, %w, %w, %none) {post_scale = 2.0 : f32} : (tensor<12x16xf32>, tensor<12x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, none) -> tensor<12x16xf32>
  return %y : tensor<12x16xf32>
}

// -----

func.func @heads_do_not_divide(%x: tensor<12x16xf32>, %w: tensor<16x16xf32>) -> tensor<12x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{Wq width must be divisible by num_heads}}
  %y = "onnx.MultiHeadAttention"(%x, %x, %w, %w, %w, %w, %none) {num_heads = 3 : si64} : (tensor<12x16xf32>, tensor<12x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, none) -> tensor<12x16xf32>
  return %y : tensor<12x16xf32>
}

// -----

func.func @ffn_shape_mismatch(%x: tensor<12x16xf32>, %w1: tensor<16x48xf32>, %w2: tensor<32x16xf32>) -> tensor<12x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{W1 / W2 mismatch (48 vs 32)}}
  %y = "onnx.PositionwiseFeedForward"(%x, %w1, %w2, %none) : (tensor<12x16xf32>, tensor<16x48xf32>, tensor<32x16xf32>, none) -> tensor<12x16xf32>
  return %y : tensor<12x16xf32>
}

// -----

func.func @ffn_activation(%x: tensor<12x16xf32>, %w1: tensor<16x48xf32>, %w2: tensor<48x16xf32>) -> tensor<12x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{activation must be "relu"}}
  %y = "onnx.PositionwiseFeedForward"(%x, %w1, %w2, %none) {activation = "gelu"} : (tensor<12x16xf32>, tensor<16x48xf32>, tensor<48x16xf32>, none) -> tensor<12x16xf32>
  return %y : tensor<12x16xf32>
}

// -----

func.func @upload_tile_window(%src: memref<12x16xf16>, %addr: i32) -> i32 {
  // expected-error @+1 {{exceeds the 12x16 source}}
  %next, %tok = "aismem.RedMulEUploadTile"(%src, %addr) <{tile = 0 : i32, row_offset = 4 : i32, col_offset = 0 : i32, rows = 12 : i32, cols = 16 : i32, dst_rows = 16 : i32}> : (memref<12x16xf16>, i32) -> (i32, none)
  return %next : i32
}
