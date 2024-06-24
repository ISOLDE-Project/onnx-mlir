/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ConvertAISMEMToAISLLVM.cpp - Lowering to AISLLVM patterns --------===//
//  Following dialects are lowered:
//   -  Krnl
//   -  MemRef
// Following operations are lowered
//   -  AISMEMQConstantOp
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
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

#include "src/Conversion/AISMEMToLLVM/AISMEMToLLVMCommon.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using std::nullopt;
using std::optional;

namespace spade {

void populateAISMEMToLLVMConversionPattern(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx) {
  
  // QConstant
  populateAISMEMQConstantOpPattern(typeConverter, patterns, ctx);

}

//===----------------------------------------------------------------------===//
// AISMEM to LLVMIR Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
struct AISMEMToLLVMLoweringPass
    : public PassWrapper<AISMEMToLLVMLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AISMEMToLLVMLoweringPass)

  StringRef getArgument() const override { return "convert-aismem-to-llvmir"; }

  StringRef getDescription() const override {
    return "Lower (some)AISMEM ops to LLVM dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  AISMEMToLLVMLoweringPass() = default;

  void runOnOperation() final;

public:
};

void AISMEMToLLVMLoweringPass::runOnOperation() {
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
  options.allocLowering = LowerToLLVMOptions::AllocLowering::None;
  options.useBarePtrCallConv = true;

  spade::AISMEMTypeConverter typeConverter(&getContext(), options);

  RewritePatternSet patterns(ctx);

  // Define patterns.
  populateAISMEMToLLVMConversionPattern(patterns, typeConverter, &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  llvm::errs() << "-- spade::AISMEMToLLVMLoweringPass::runOnOperation() ---------\n";
  module->dump();
  llvm::errs() << "-----------\n";
  LogicalResult passResult = applyPartialConversion(module, target, std::move(patterns));
  llvm::errs() << "++ spade::AISMEMToLLVMLoweringPass::runOnOperation(): "<< succeeded(passResult) <<"---------\n";
  module->dump();
  llvm::outs() << "-----------\n";

  if (failed(passResult)) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToLLVMIRPass() {
  return std::make_unique<AISMEMToLLVMLoweringPass>();
}

} // namespace spade