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

void populateLoweringAISLEGEMMOpPattern(RewritePatternSet &patterns,
        TypeConverter &typeConverter, MLIRContext *ctx); 

void populateLoweringAISLEhstackOpPattern(RewritePatternSet &patterns,
        TypeConverter &typeConverter, MLIRContext *ctx);
                        
//insert new pattern above this line
} // namespace spade