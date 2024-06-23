/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {
void populateLoweringKrnlGlobalOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);
}
}

namespace spade {

void populateAISMEMQConstantOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateMemrefAllocOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringAISMEMGEMMOpPattern(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx);

// insert new pattern above this line
} // namespace spade