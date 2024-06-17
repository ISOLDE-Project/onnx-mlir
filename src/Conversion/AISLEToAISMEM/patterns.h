/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace spade {


void populateLoweringAISLEGEMMOpPattern(RewritePatternSet &patterns,
        TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel); 
                        
//insert new pattern above this line
} // namespace spade