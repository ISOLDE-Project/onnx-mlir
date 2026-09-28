/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace spade {

void populateLoweringONNXToAISLEComplexGEMMOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx);
  
void populateLoweringONNXToAISLEGEMMOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel);
    
            
// ISOLDE transformer blocks: onnx.MultiHeadAttention and
// onnx.PositionwiseFeedForward.
void populateLoweringONNXToAISLETransformerOpPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx);

// In-place ONNX rewrites run before the conversion: fold constant attention
// scales into Wq/Wv and residual Adds into the blocks' accumulator C.
void fuseTransformerBlocks(ModuleOp module);

// f16 MatMul [12, 16K] x [16K, 16] and Add [12, 16] + [12, 16] on RedMulE.
void populateLoweringONNXToAISLEMatMulAddOpPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx);

//insert new pattern above this line
} //namespace spade

