/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ConvertAISMEMToAISLLVM.cpp - Lowering to AISLLVM patterns
//-------------------===//
//  Following dialects are lowered:
//   -  Krnl
//   -  MemRef
// Following operations are lowered
//   -  AISMEMQConstantOp
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "patterns.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/AISLLVM/AISLLVMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include <optional>
#include <tuple>

#include "src/Conversion/AISMEMToAISLLVM/AISMEMToAISLLVMCommon.hpp"

using namespace mlir;
using std::nullopt;
using std::optional;

namespace spade {

void populateAISMEMToAISLLVMConversionPattern(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx) {

 
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  ///
  onnx_mlir::krnl::populateLoweringKrnlGlobalOpPattern(
      typeConverter, patterns, ctx);

  populateLoweringKrnlEntryPointOpPattern(patterns, typeConverter, ctx);
  // AllocOp
  populateMemrefAllocOpPattern(typeConverter, patterns, ctx);
  // QConstant
  populateAISMEMQConstantOpPattern(typeConverter, patterns, ctx);

  populateLoweringAISMEMGEMMOpPattern(patterns, typeConverter, ctx);
}

//===----------------------------------------------------------------------===//
// AISMEM to AISMEM Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
struct AISMEMToAISLLVMLoweringPass
    : public PassWrapper<AISMEMToAISLLVMLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AISMEMToAISLLVMLoweringPass)

  StringRef getArgument() const override { return "convert-aismem-to-aisllvm"; }

  StringRef getDescription() const override {
    return "Lower (some)AISMEM ops to AISLLVM dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  AISMEMToAISLLVMLoweringPass() = default;

  void runOnOperation() final;

public:
};

void AISMEMToAISLLVMLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  //target.addLegalDialect<spade::AISLLVMDialect>();
  //target.addLegalOp<ModuleOp>();
  //target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  RewritePatternSet patterns(&getContext());

  LowerToLLVMOptions options(&getContext());
  options.allocLowering = LowerToLLVMOptions::AllocLowering::None;
  options.useBarePtrCallConv = true;
  spade::AISMEMTypeConverter typeConverter(&getContext(), options);

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    // CallOp is legal only if types have been converted to Std types.
    return typeConverter.isLegal(op);
  });

  // Operations that are legal only if types are not tensors.
  target.addDynamicallyLegalOp<mlir::func::ReturnOp>([&](Operation *op) {
    return llvm::none_of(op->getOperandTypes(),
        [](Type type) { return type.isa<MemRefType>(); });
  });

  // Define patterns.
  populateAISMEMToAISLLVMConversionPattern(
      patterns, typeConverter, &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToLLVMIRPass() {
  return std::make_unique<AISMEMToAISLLVMLoweringPass>();
}

} // namespace spade