/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace spade {

 void populateLoweringAISLEQConstantOpPattern(RewritePatternSet &patterns,
      TypeConverter &typeConverter, MLIRContext *ctx);

void populateLoweringAISLEComplexGEMMOpPattern(RewritePatternSet &patterns,
        TypeConverter &typeConverter, MLIRContext *ctx);

void populateLoweringAISLEGEMMOpPattern(RewritePatternSet &patterns,
        TypeConverter &typeConverter, MLIRContext *ctx); 

void populateLoweringAISLEhstackOpPattern(RewritePatternSet &patterns,
        TypeConverter &typeConverter, MLIRContext *ctx);
                        
// ISOLDE transformer blocks -> explicit RedMulE/SPM schedules.
void populateLoweringAISLETransformerOpPatterns(RewritePatternSet &patterns,
        TypeConverter &typeConverter, MLIRContext *ctx);

//insert new pattern above this line
} // namespace spade
