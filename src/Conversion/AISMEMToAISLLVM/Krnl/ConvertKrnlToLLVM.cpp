/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToLLVM.cpp - Krnl Dialect Lowering  ---------------===//
//
// Copyleft
//
// =============================================================================
//
// This file implements the lowering of Krnl operations to a combination of
// other dialects (affine, std, LLVM).
//
//===----------------------------------------------------------------------===//

#include <fstream>

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"

#include "onnx/onnx_pb.h"

#include "src/Conversion/AISMEMToAISLLVM/Krnl/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/AISMEMToAISLLVM/AISMEMToLLVMCommon.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/AISLLVM/AISLLVMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

using namespace mlir;

#define DEBUG_TYPE "spade_krnl_to_llvm"

namespace spade {
namespace krnl {

//std::string EXTERNAL_CONSTANT_PREFIX = "om_external_constant_ex";

//uint64_t KRNL_ENTRY_POINT_ID = 0;

void populateAffineAndKrnlToLLVMConversion(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, SmallVector<MemRefType, 4>> &inputMemRefTypes,
    std::map<std::string, SmallVector<MemRefType, 4>> &outputMemRefTypes,
    bool verifyInputTensors, bool enableParallel) {


  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  populateReconcileUnrealizedCastsPatterns(patterns);
  krnl::populateKrnlToLLVMConversion(typeConverter, patterns, ctx,
      constantOutputs, singleEntryPoint, entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps, inputMemRefTypes, outputMemRefTypes, verifyInputTensors);
}



//===----------------------------------------------------------------------===//
// Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

struct ConvertKrnlToLLVMPass
    : public PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertKrnlToLLVMPass)

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ConvertKrnlToLLVMPass() = default;
  ConvertKrnlToLLVMPass(const ConvertKrnlToLLVMPass &pass)
      : PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>>() {}
  ConvertKrnlToLLVMPass(bool verifyInputTensors, bool useLRODATA,
      bool storeConstantsToFile, uint64_t constantsToFileSingleThreshold,
      uint64_t constantsToFileTotalThreshold, std::string outputNameNoExt,
      bool enableParallel) {
    this->verifyInputTensors = verifyInputTensors;
    // Exclusive options. no option or only one option can be True.
    this->useLRODATA = useLRODATA;
    this->storeConstantsToFile = storeConstantsToFile;
    this->constantsToFileSingleThreshold = constantsToFileSingleThreshold;
    this->constantsToFileTotalThreshold = constantsToFileTotalThreshold;
    this->outputNameNoExt = outputNameNoExt;
    this->enableParallel = enableParallel;
  }

  StringRef getArgument() const override { return "convert-krnl-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the Krnl Affine and Std dialects to LLVM.";
  }

  void runOnOperation() final;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect>();
  }

  Option<bool> verifyInputTensors{*this, "verify-input-tensors",
      llvm::cl::desc(
          "Verify input tensors whenever the entry point function is called.\n"
          "Data type and shape are verified. Enable this may introduce "
          "overhead in inferencing."),
      llvm::cl::init(false)};

  Option<bool> useLRODATA{*this, "use-lrodata-section",
      llvm::cl::desc("Put global constants into the large read-only data "
                     "section. This is for linking large object files"),
      llvm::cl::init(false)};

  Option<bool> storeConstantsToFile{*this, "store-constants-to-file",
      llvm::cl::desc("Put global constants to a file."), llvm::cl::init(false)};

  Option<float> constantsToFileTotalThreshold{*this,
      "constants-to-file-total-threshold",
      llvm::cl::desc(
          "Put global constants to a file if the total size in "
          "bytes of constants is greater than this threshold. "
          "store-constants-to-file must be enabled for this to be effective. "
          "Only count contants whose size is greater than "
          "constants-to-file-single-threshold. Value is in GB."),
      llvm::cl::init(2.0)};

  Option<float> constantsToFileSingleThreshold{*this,
      "constants-to-file-single-threshold",
      llvm::cl::desc(
          "Put global constants to a file if a single constant's size in "
          "bytes is greater than this threshold. "
          "store-constants-to-file must be enabled for this to be effective. "
          "Total sizes in bytes of satisfied constants must be greater than "
          "constants-to-file-total-threshold. Value is in KB."),
      llvm::cl::init(1.0)};

  Option<bool> enableParallel{*this, "enable-parallel",
      llvm::cl::desc("Enable parallelization"), llvm::cl::init(false)};

private:
  std::string outputNameNoExt = "./model";
};

void ConvertKrnlToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  //const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(ctx/*, dataLayoutAnalysis.getAtOrAbove(module)*/);
  

  options.allocLowering = LowerToLLVMOptions::AllocLowering::None;
  options.useBarePtrCallConv = true;


  // Global Op for entry point names and their input/output JSON signatures,
  // those will generated when lowering KrnlEntryPoint.
  // This info is used to generate global signature functions.
  SmallVector<LLVM::GlobalOp, 1> entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps;

  // Keep original MemRefTypes for inputs and outputs. These information will be
  // used for constructing OMTensors for inputs and outputs.
  // We have to record this information at this point before they are
  // disappeared during the lowering to LLVM. For example, unsigned types do
  // not exist at LLVM level, typed pointers becomes opaque if opaque point is
  // enabled.
  std::map<std::string, SmallVector<MemRefType, 4>> inputMemRefTypes;
  std::map<std::string, SmallVector<MemRefType, 4>> outputMemRefTypes;
  onnx_mlir::krnl::recordInputOutputMemRefTypes(module, inputMemRefTypes, outputMemRefTypes);

  // Determine whether the module has a single entry point or not.
  bool singleEntryPoint = onnx_mlir::krnl::hasSingleEntryPoint(module);

  // Determine whether an output OMTensor should own the underlying buffer or
  // not.
  SmallVector<bool, 4> outputOMTensorOwnerships;
  onnx_mlir::krnl::determineOwnershipForOutputOMTensors(module, outputOMTensorOwnerships);

  // If storeConstantsToFile, copy constants from GlobalOp and write to a single
  // file.
  // A single constant's size must be greater than singleThreshold.
  // The total size of contants must be greater than totalThreshold.
  std::string fname = outputNameNoExt + ".constants.bin";
  if (storeConstantsToFile) {
    storeConstantsToFile = onnx_mlir::krnl::extractConstantsToFile(module, fname,
        (uint64_t)constantsToFileSingleThreshold * 1024,
        (uint64_t)constantsToFileTotalThreshold * 1024 * 1024 * 1024);
  }

#if 0
  // Request C wrapper emission via attribute.
  for (auto func : module.getOps<func::FuncOp>()) {
    func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
        UnitAttr::get(&getContext()));
  }
#else
  for (auto funcOp : module.getOps<func::FuncOp>()) {
       if (funcOp->hasAttr("llvm.emit_c_interface")) {
        funcOp->removeAttr("llvm.emit_c_interface");
      }
  }
#endif  

  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<spade::AISMEMDialect>();
  target.addLegalDialect<spade::AISLLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Convert types to legal types for the LLVM dialect.
  LLVMTypeConverter typeConverter(ctx, options);
  onnx_mlir::krnl::customizeTypeConverter(typeConverter);
  //spade::AISMEMTypeConverter typeConverter(&getContext(), options);

  // omp::ParallelOp can only be legalized when its region is legal
  target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
      [&](Operation *op) { return typeConverter.isLegal(&op->getRegion(0)); });
  // Currently, only minimum required OpenMP Ops are marked as legal, in the
  // future integration of OpenMP, probably more OpenMP Ops are required to be
  // marked as legal. Please refer the Conversion/OpenMPToLLVM/OpenMPtoLLVM.cpp
  // in MLIR repo to see see how to legalize them.
  target.addLegalOp<omp::TerminatorOp, omp::YieldOp>();
  // We have a combination of `krnl`, `affine`, `vector`, and `std` operations.
  // We lower in stages until all the code is in the LLVM dialect.
  RewritePatternSet patterns(ctx);

  populateAffineAndKrnlToLLVMConversion(patterns, typeConverter, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, inputMemRefTypes, outputMemRefTypes,
      verifyInputTensors, enableParallel);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  // Generate signature functions.
  // if (entryGlobalOps.size() >= 1)
  //   onnx_mlir::krnl::genSignatureFunction(
  //       module, entryGlobalOps, inSigGlobalOps, outSigGlobalOps);

  // If globals are stored on external files. Emit helper functions to load
  // constants from files.
  if (storeConstantsToFile) {
    // Register runtime function calls, e.g. omXXX functions.
    const RuntimeAPIRegistry &apiRegistry =
        RuntimeAPIRegistry(module, builder, typeConverter);

    // Emit a function, omLoadConstantsFromFile, that loads contants from files
    // to memory.
    onnx_mlir::krnl::loadConstantsFromFile(module, apiRegistry, entryGlobalOps);
  }

  // Annotate global constants with `.lrodata` section if required.
  // Make sure this is always called at the end of this pass.
  if (useLRODATA) {
    module->walk([&](LLVM::GlobalOp gop) -> WalkResult {
      // Put all global constants into `.lrodata` instead of `.rodata` because
      // AI workloads often have a large amount of constants, especially large
      // language models.
      gop.getOperation()->setAttr("section", StringAttr::get(ctx, ".lrodata"));
      return WalkResult::advance();
    });
  }
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to LLVM.
std::unique_ptr<Pass> createConvertKrnlToLLVMPass() {
  return std::make_unique<ConvertKrnlToLLVMPass>();
}
std::unique_ptr<Pass> createConvertKrnlToLLVMPass(bool verifyInputTensors,
    bool useLRODATA, bool storeConstantsToFile,
    float constantsToFileSingleThreshold, float constantsToFileTotalThreshold,
    std::string outputNameNoExt, bool enableParallel) {
  return std::make_unique<ConvertKrnlToLLVMPass>(verifyInputTensors, useLRODATA,
      storeConstantsToFile, constantsToFileSingleThreshold,
      constantsToFileTotalThreshold, outputNameNoExt, enableParallel);
}

void populateKrnlToLLVMConversion(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, SmallVector<MemRefType, 4>> &inputMemRefTypes,
    std::map<std::string, SmallVector<MemRefType, 4>> &outputMemRefTypes,
    bool verifyInputTensors) {
  populateLoweringKrnlEntryPointOpPattern(typeConverter, patterns, ctx);
  onnx_mlir::krnl::populateLoweringKrnlGlobalOpPattern(
      typeConverter, patterns, ctx);
}

} // namespace krnl
} // namespace spade
