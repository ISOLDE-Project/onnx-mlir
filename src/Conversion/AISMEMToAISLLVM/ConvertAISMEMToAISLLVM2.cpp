/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
//  ConvertAISMEMToAISLLVM2.cpp - Lowering func::FuncDialect patterns
//  Following dialects are lowered:
//   - func::FuncDialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

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
#include "src/Conversion/AISMEMToAISLLVM/AISMEMToLLVMCommon.hpp"
#include "src/Dialect/AISLLVM/AISLLVMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include <tuple>

using namespace mlir;

namespace spade_2 {

void populateAISMEMToAISLLVMConversionPattern(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx) {

  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
}

//===----------------------------------------------------------------------===//
// AISMEM to AISMEM Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
struct AISMEMToAISLLVMLoweringPass
    : public PassWrapper<AISMEMToAISLLVMLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AISMEMToAISLLVMLoweringPass)

  StringRef getArgument() const override {
    return "convert-aismem-to-aisllvm-2";
  }

  StringRef getDescription() const override {
    return "Lower (some)AISMEM ops to AISLLVM dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  AISMEMToAISLLVMLoweringPass() = default;

  void runOnOperation() final;

public:
  // Some ops (RNN ops for example) are lowered to other ONNX ops such as
  // ONNXMatMulOp, ONNXSplitOp, ONNXTransposeOp, etc. These ONNX ops are then
  // lowered into krnl ops in this pass.
  //
  // To write LIT tests for operations that are lowered to other ONNX
  // operations, we do not need to check the final generated krnl code (which is
  // lengthy). It is more convenient to check the intermediate generated code
  // including ONNX ops. We trust the lowering of the other ONNX ops.
  //
  // This flag is used in LIT tests to stop the lowering of the other ONNX ops.
  // Usage: onnx-mlir-opt --convert-onnx-to-krnl='emit-intermediate-ir'
};

void AISMEMToAISLLVMLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target
      .addLegalDialect<KrnlDialect, arith::ArithDialect, linalg::LinalgDialect,
          math::MathDialect, memref::MemRefDialect, shape::ShapeDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<spade::AISLLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  // Needed to support unsigned int computations. To be removed if we use a
  // scheme that does not rely on the UnrealizedConversionCastOp.
  // target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
  // Make ONNXNoneOp legal so that other ONNX ops can use it during the
  // lowering. ONNXNoneOp will be dangling and removed by calling
  // canonicalization after the lowering.
  // target.addLegalOp<::mlir::ONNXNoneOp>();
  // target.addIllegalOp<::spade::AISMEMQConstantOp>();
  // target.addIllegalOp<::spade::AISMEMConvOp>();

  /*
    if (emitIntermediateIR) {
      // Only used for writing LIT tests for ONNX operations that are lowered to
      // other ONNX operations. The following operations are prevented from
    being
      // lowered further. See the comment in the declaration of
      // 'emitIntermediateIR' for more details.
      target.addLegalOp<ONNXMatMulOp>();
      target.addLegalOp<ONNXReshapeOp>();
      target.addLegalOp<ONNXSplitV11Op>();
      target.addLegalOp<ONNXSqueezeV11Op>();
      target.addLegalOp<ONNXTransposeOp>();
    }
  */
  // Conversion target for accelerators.
  // for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
  //  accel->conversionTargetONNXToKrnl(target);

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  LowerToLLVMOptions options(&getContext());
  options.allocLowering = LowerToLLVMOptions::AllocLowering::None;
  options.useBarePtrCallConv = true;
  spade::AISMEMTypeConverter typeConverter(&getContext(), options);

  // Define patterns.
  populateAISMEMToAISLLVMConversionPattern(
      patterns, typeConverter, &getContext());

  // Rewrite patterns for accelerators.
  // for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
  //  accel->rewritePatternONNXToKrnl(patterns, krnlTypeConverter,
  //  &getContext());

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

} // namespace spade_2