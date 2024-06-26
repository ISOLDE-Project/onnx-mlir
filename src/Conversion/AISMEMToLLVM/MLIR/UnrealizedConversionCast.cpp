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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Conversion/AISMEMToLLVM/AISMEMToLLVMCommon.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/AISLLVM/AISLLVMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Support/SpadeSupport.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#include "mlir/Support/TypeID.h"
#define DEBUG_TYPE "AISMEMToLLVM_UnrealizedConversionCastOp"

using namespace mlir;

namespace {

struct UnrealizedConversionCastPattern : public ConvertToLLVMPattern {

  using theOperation = mlir::UnrealizedConversionCastOp;
  using theAdaptor = mlir::UnrealizedConversionCastOpAdaptor;


  explicit UnrealizedConversionCastPattern(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            theOperation::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op_, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {

    theOperation castOp = llvm::dyn_cast<theOperation>(op_);
    theAdaptor operandAdaptor(castOp);
    Location loc = castOp.getLoc();
    ModuleOp module = castOp->getParentOfType<ModuleOp>();
    MLIRContext *context = module.getContext();

    Type resultType = castOp->getResultTypes()[0];

    // Create the element type (i32 in this case)
    Type elementType = rewriter.getIntegerType(32);
    // Create the vector type with shape [4]
    VectorType vectorType = VectorType::get({4}, elementType);
    Value firstOperand = operandAdaptor.getOperands()[0];
    if (firstOperand.getType() == vectorType &&
        resultType.isa<LLVM::LLVMPointerType>()) {
      auto ptrType = LLVM::LLVMPointerType::get(context);
      Value size = rewriter.create<LLVM::ConstantOp>(loc, elementType, 64);
      auto allocatedPtr =
          rewriter.create<LLVM::AllocaOp>(loc, ptrType, vectorType, size, 16);
      rewriter.create<LLVM::StoreOp>(loc, firstOperand, allocatedPtr);
      rewriter.replaceOp(castOp, allocatedPtr);
      castOp.replaceAllUsesWith(allocatedPtr);
      LLVM_DEBUG({
        llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
        spade::dumpBlock(castOp);
        llvm::errs() << "---\n";
      });
      return success();
    } else {
      LLVM_DEBUG({
        llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
        ::llvm::outs()
            << "UnrealizedConversionCastOp() unsuported operaration\n";
        firstOperand.dump();
        resultType.dump();
        castOp.dump();
        ::llvm::outs() << "\n";
      });
    }
    return failure();
  }
};

struct UnrealizedConversionCastPass
    : public PassWrapper<UnrealizedConversionCastPass,
          OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnrealizedConversionCastPass)

  StringRef getArgument() const override { return "AISMEMToLLVM_UnrealizedConversionCastOp"; }

  StringRef getDescription() const override {
    return "Implement (some) UnrealizedConversionCastOp";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    LowerToLLVMOptions options(ctx);
    options.allocLowering = LowerToLLVMOptions::AllocLowering::None;
    options.useBarePtrCallConv = true;

    spade::AISMEMTypeConverter typeConverter(ctx, options);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<spade::AISMEMDialect>();
    target.addLegalDialect<spade::AISLLVMDialect>();
    // illegal stuff
    target.addIllegalOp<UnrealizedConversionCastOp>();

    //
    RewritePatternSet patterns(ctx);
    patterns.add<UnrealizedConversionCastPattern>(typeConverter, ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace spade {
std::unique_ptr<Pass> createUnrealizedConversionCastPass() {
  return std::make_unique<UnrealizedConversionCastPass>();
}

} // namespace spade
// Register the pass using a lambda function
namespace {
static PassRegistration<UnrealizedConversionCastPass> pass(
    []() -> std::unique_ptr<Pass> {
      return spade::createUnrealizedConversionCastPass();
    });
} // namespace