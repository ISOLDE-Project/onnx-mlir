/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlEntryPoint.cpp - Lower KrnlEntryPointOp -------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file lowers the KrnlEntryPointOp operator.
//
//===----------------------------------------------------------------------===//

#include <errno.h>

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/JSON.h"

#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "AISMEMToAISLLVM_KrnlEntryPoint"

using namespace mlir;

namespace spade {

extern uint64_t KRNL_ENTRY_POINT_ID;

class KrnlEntryPointOpLowering : public OpRewritePattern<KrnlEntryPointOp> {
public:
  using OpRewritePattern<KrnlEntryPointOp>::OpRewritePattern;

  KrnlEntryPointOpLowering(LLVMTypeConverter &typeConverter, MLIRContext *ctx)
      : OpRewritePattern<KrnlEntryPointOp>(ctx) {}

  LogicalResult matchAndRewrite(
      KrnlEntryPointOp op, PatternRewriter &rewriter) const override {
    // Location loc = op.getLoc();

    rewriter.eraseOp(op);

    return success();
  }
};

void populateLoweringKrnlEntryPointOpPattern(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<KrnlEntryPointOpLowering>(typeConverter, ctx);
}

} // namespace spade
