/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ConvertONNXToAISLE.cpp - Lowering ONNX Dialect -------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "patterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/AISLE/AISLEDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace spade {


void populateONNXToAISLEConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling,
    bool enableParallel) {
// GEMM
    populateLoweringONNXToAISLEGEMMOpPattern(patterns,typeConverter,ctx,enableParallel);


    }



//===----------------------------------------------------------------------===//
// ONNX to AISLE Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
struct ONNXToAISLELoweringPass
    : public PassWrapper<ONNXToAISLELoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToAISLELoweringPass)

  StringRef getArgument() const override { return "convert-onnx-to-contirv"; }

  StringRef getDescription() const override {
    return "Lower (some)ONNX ops to AISLE dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ONNXToAISLELoweringPass() = default;
  ONNXToAISLELoweringPass(const ONNXToAISLELoweringPass &pass)
      : PassWrapper<ONNXToAISLELoweringPass, OperationPass<ModuleOp>>() {}
  ONNXToAISLELoweringPass(
      bool emitDealloc, bool enableTiling, bool enableParallel) {
    // Below, need explicit assignment to enable implicit conversion of bool to
    // Option<bool>.
    this->emitDealloc = emitDealloc;
    this->enableTiling = enableTiling;
    this->enableParallel = enableParallel;
  }
  ONNXToAISLELoweringPass(int optLevel, bool enableParallel)
      : ONNXToAISLELoweringPass(
            /*emitDealloc=*/false, /*enableTiling=*/optLevel >= 3,
            enableParallel) {}

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
  Option<bool> emitIntermediateIR{*this, "emit-intermediate-ir",
      llvm::cl::desc(
          "Emit intermediate IR rather than lowering to the krnl dialect."),
      llvm::cl::init(false)};
  Option<bool> emitDealloc{*this, "emit-dealloc",
      llvm::cl::desc("Emit dealloc for allocated memrefs or not."),
      llvm::cl::init(false)};
  Option<bool> enableTiling{*this, "enable-tiling",
      llvm::cl::desc("Enable loop tiling and unrolling optimizations"),
      llvm::cl::init(false)};
  Option<bool> enableParallel{*this, "enable-parallel",
      llvm::cl::desc("Enable parallelization"), llvm::cl::init(false)};
};

void ONNXToAISLELoweringPass::runOnOperation() {
  ModuleOp module = getOperation();


  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<spade::AISLEDialect>();
  // Needed to support unsigned int computations. To be removed if we use a
  // scheme that does not rely on the UnrealizedConversionCastOp.
  //target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
  // Make ONNXNoneOp legal so that other ONNX ops can use it during the
  // lowering. ONNXNoneOp will be dangling and removed by calling
  // canonicalization after the lowering.
  target.addLegalOp<::mlir::ONNXNoneOp>();




/*
  if (emitIntermediateIR) {
    // Only used for writing LIT tests for ONNX operations that are lowered to
    // other ONNX operations. The following operations are prevented from being
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
  //for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
  //  accel->conversionTargetONNXToKrnl(target);

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  TypeConverter aTypeConverter;

  // Define patterns.
  populateONNXToAISLEConversionPattern(
      patterns, aTypeConverter, &getContext(), enableTiling, enableParallel);

  // Rewrite patterns for accelerators.
  //for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
  //  accel->rewritePatternONNXToKrnl(patterns, krnlTypeConverter, &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToAISLEPass() {
  return std::make_unique<ONNXToAISLELoweringPass>();
}

std::unique_ptr<Pass> createLowerToAISLEPass(int optLevel, bool enableParallel) {
  return std::make_unique<ONNXToAISLELoweringPass>(optLevel, enableParallel);
}

std::unique_ptr<Pass> createLowerToAISLEPass(
    bool emitDealloc, bool enableTiling, bool enableParallel) {
  return std::make_unique<ONNXToAISLELoweringPass>(
      emitDealloc, enableTiling, enableParallel);
}


} // namespace spade