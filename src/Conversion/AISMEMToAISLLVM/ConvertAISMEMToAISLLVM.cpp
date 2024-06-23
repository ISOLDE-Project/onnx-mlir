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
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
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
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using std::nullopt;
using std::optional;

namespace spade {

void populateAISMEMToAISLLVMConversionPattern(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx) {
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

void populateAffineToAISLLVMConversionPattern(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx) {

  /*
   * copy from onnx_mlir::krnl::populateAffineAndKrnlToLLVMConversion()
   * TODO: clean-up/re-evaluate later
   */

#ifdef SPADE_VECTOR_FEAT
  // TODO: look at what is done in
  // mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.cpp in function
  // LowerVectorToLLVMPass::runOnOperation() and see what we should do about it.
  // They run it in two steps, and add additional lowerings.

  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(
      patterns, vector::VectorTransformsOptions());
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransformsOptions());
#endif

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

#ifdef SPADE_VECTOR_FEAT
  populateShapeToStandardConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  memref::populateExpandOpsPatterns(patterns);
  // Use polynomial approximation for math.{tanh, sin, cos and exp} for better
  // performance.
  populateMathPolynomialApproximationPatterns(patterns);
  arith::populateArithExpandOpsPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
#endif
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
#ifdef SPADE_OPENMP_FEAT
  // Enable OpenMP-to-LLVM pass when enable parallelism
  if (enableParallel) {
    populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);
  }
#endif
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  populateReconcileUnrealizedCastsPatterns(patterns);
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
  /*
   * copy from ConvertKrnlToLLVMPass::runOnOperation()
   * TODO: clean-up/re-evaluate later
   */
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  // OpBuilder builder(ctx);
  // const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  // LowerToLLVMOptions options(ctx, dataLayoutAnalysis.getAtOrAbove(module));

  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  LowerToLLVMOptions options(ctx);
  //options.allocLowering = LowerToLLVMOptions::AllocLowering::None;
  //options.useBarePtrCallConv = true;

  spade::AISMEMTypeConverter typeConverter(&getContext(), options);

  // target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  //   // FuncOp is legal only if types have been converted to Std types.
  //   return typeConverter.isSignatureLegal(op.getFunctionType());
  // });

  // target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
  //   // CallOp is legal only if types have been converted to Std types.
  //   return typeConverter.isLegal(op);
  // });

  // // Operations that are legal only if types are not tensors.
  // target.addDynamicallyLegalOp<mlir::func::ReturnOp>([&](Operation *op) {
  //   return llvm::none_of(op->getOperandTypes(),
  //       [](Type type) { return type.isa<MemRefType>(); });
  // });

  RewritePatternSet patterns(ctx);
  // Define patterns.

  populateAffineToAISLLVMConversionPattern(
      patterns, typeConverter, &getContext());

  populateAISMEMToAISLLVMConversionPattern(
      patterns, typeConverter, &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  llvm::outs()<<"-----------\n";
  module->dump();
  llvm::outs()<<"-----------\n";
  auto passResult= applyPartialConversion(module, target, std::move(patterns));
  llvm::outs()<<"-----------\n";
  module->dump();
  llvm::outs()<<"-----------\n";

  if (failed(passResult)) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToLLVMIRPass() {
  return std::make_unique<AISMEMToAISLLVMLoweringPass>();
}

} // namespace spade