/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- SpadeCompilerPasses.cpp
//-------------------------===//
//
// Copyleft
//
// =============================================================================
//
// Functions for configuring and adding passes.
//
// REQUEST: To the extent possible, passes here should not sample global
// optimization parameters specified in CompilerOptions.hpp. The passes should
// use parameters that are set by these global options where these passes are
// called. The idea is to keep our code as free of "rogue" global options used
// in random places in the code.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/DisposableGarbageCollector.hpp"
#include "src/Compiler/SpadeCompilerPasses.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Dialect/AISLE/AISLEDialect.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir_spade {

void configurePasses() {
  // Set global vector machine support.
  onnx_mlir::VectorMachineSupport::setGlobalVectorMachineSupport(
      onnx_mlir::march, onnx_mlir::mcpu, "");
  onnx_mlir::configureConstPropONNXToONNXPass(
      onnx_mlir::onnxConstPropRoundFPToInt,
      onnx_mlir::onnxConstPropExpansionBound,
      onnx_mlir::onnxConstPropDisablePatterns, onnx_mlir::disableConstantProp);
  onnx_mlir::configureOnnxToKrnlLoweringPass(
      onnx_mlir::optReport == onnx_mlir::OptReport::Parallel,
      onnx_mlir::enableParallel, onnx_mlir::parallelizeOps,
      onnx_mlir::optReport == onnx_mlir::OptReport::Simd,
      !onnx_mlir::disableSimdOption);
}

void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel, bool enableCSE,
    bool enableInstrumentONNXSignature, std::string ONNXOpsStatFormat) {
  if (enableCSE)
    // Eliminate common sub-expressions before lowering to Krnl.
    // TODO: enable this by default when we make sure it works flawlessly.
    pm.addPass(mlir::createCSEPass());
  // Verify ONNX ops before lowering to Krnl.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXPreKrnlVerifyPass());
  // Print statistics about ONNX ops if enabled.
  if (ONNXOpsStatFormat.length() > 0) {
    transform(ONNXOpsStatFormat.begin(), ONNXOpsStatFormat.end(),
        ONNXOpsStatFormat.begin(), ::toupper);
    bool printAsJSON = ONNXOpsStatFormat.compare("JSON") == 0;
    bool printAsTXT = ONNXOpsStatFormat.compare("TXT") == 0;
    if (printAsJSON || printAsTXT) {
      // TODO: we should write the output of this pass in a file but I was not
      // able to use raw_fd_ostream of a file without it crashing.
      pm.addNestedPass<func::FuncOp>(
          mlir::createPrintOpStatsPass(llvm::outs(), printAsJSON));
    } else {
      llvm::errs() << "Skip onnx-ops-stats: expected JSON or TXT format, got \""
                   << ONNXOpsStatFormat << "\"\n";
    }
  }

  // Print Signatures of each op at runtime if enabled. Should not run
  // signature and instrument passes at the same time.
  if (enableInstrumentONNXSignature)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createInstrumentONNXSignaturePass());
  pm.addPass(onnx_mlir::createLowerToKrnlPass(/*enableTiling*/ optLevel >= 3,
      /*enableSIMD*/ optLevel >= 3 && !onnx_mlir::disableSimdOption,
      onnx_mlir::enableParallel,
      /*opsToCall*/ onnx_mlir::opsForCall));
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // opportunities.
  pm.addPass(mlir::createCanonicalizerPass());
}

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::krnl::createConvertKrnlToAffinePass());
}

void addKrnlToLLVMPasses(
    mlir::OpPassManager &pm, std::string outputNameNoExt, bool enableCSE) {
  if (enableCSE)
    // Eliminate common sub-expressions before lowering to Krnl.
    // TODO: enable this by default when we make sure it works flawlessly.
    pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<func::FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Early introduction of omp causes problems with bufferization, delay for
  // now. May revise this decision later.

  // After affine is lowered, KrnlRegion for affine scope can be removed.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::krnl::createLowerKrnlRegionPass());

  if (onnx_mlir::enableParallel) {
    // Pass to ensure that memory allocated by parallel loops stay inside the
    // parallel region (privatization of memory). Otherwise, all threads would
    // end up sharing the same temporary data. This pass works on affine
    // parallel operations, and must be executed (in presence of OMP
    // parallelism) before bufferization. In practical terms, this pass add
    // memref.alloca_scope inside each parallel for.
    pm.addPass(onnx_mlir::createProcessScfParallelPrivatePass());
    // No canonicalize passes are allowed between that pass above and the buffer
    // management passes.
  }

  // Hoist allocations out of loop nests to avoid stack overflow.
  pm.addPass(bufferization::createBufferLoopHoistingPass());

  // Use MLIR buffer deallocation pass to emit buffer deallocs.
  // Currently this has to be done *after* lowering the affine dialect because
  // operations in that dialect do not conform to the requirements explained
  // in https://mlir.llvm.org/docs/BufferDeallocationInternals.
  if (onnx_mlir::useOldBufferization) {
    pm.addNestedPass<func::FuncOp>(
        mlir::bufferization::createBufferDeallocationPass());
  } else {
    bufferization::BufferDeallocationPipelineOptions bufferDeallocOptions;
    mlir::bufferization::buildBufferDeallocationPipeline(
        pm, bufferDeallocOptions);
    pm.addPass(mlir::createBufferizationToMemRefPass());
  }

  // Late introduction of OpenMP, after bufferization.
  if (onnx_mlir::enableParallel) {
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    //  The alloca_scope ops are somewhat fragile; canonicalize remove them when
    //  redundant, which helps reliability of the compilation of these ops.
    pm.addPass(mlir::createCanonicalizerPass());
  }

  // The pass below is needed for subview and collapseShape.. Unfortunately,
  // MLIR supports only collapse for scalar loaded by scalar memory at this
  // time. Uncomment if subview/collapse are used.
  // pm.addNestedPass<func::FuncOp>(krnl::createConvertSeqToMemrefPass());

  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  // pm.addPass(onnx_mlir::krnl::createConvertKrnlToLLVMPass(
  //     onnx_mlir::verifyInputTensors,
  //     /*useLRODATA=*/(onnx_mlir::modelSize == onnx_mlir::ModelSize::large),
  //     /*storeConstantsToFile=*/onnx_mlir::storeConstantsToFile,
  //     onnx_mlir::constantsToFileSingleThreshold,
  //     onnx_mlir::constantsToFileTotalThreshold, outputNameNoExt,
  //     onnx_mlir::enableParallel));
  pm.addPass(spade::createLowerToLLVMIRPass());
  pm.addPass(spade_2::createLowerToLLVMIRPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

onnx_mlir::InputIRLevelType determineInputIRLevel(
    mlir::OwningOpRef<ModuleOp> &module) {
  Operation *moduleOp = module->getOperation();

  // Collect dialect namespaces.
  llvm::SmallDenseSet<StringRef> dialectNamespace;
  moduleOp->walk([&](mlir::Operation *op) {
    dialectNamespace.insert(op->getDialect()->getNamespace());
  });

  // If there are ONNX or AISLE ops, the input level is ONNX.
  bool hasONNXOps = llvm::any_of(dialectNamespace,
      [&](StringRef ns) { return (ns == ONNXDialect::getDialectNamespace()); });
  bool hasAISLEOps = llvm::any_of(dialectNamespace, [&](StringRef ns) {
    return (ns == spade::AISLEDialect::getDialectNamespace());
  });
  if (hasONNXOps || hasAISLEOps)
    return onnx_mlir::ONNXLevel;

  // If there are Krnl ops, the input level is MLIR.
  bool hasKrnlOps = llvm::any_of(dialectNamespace,
      [&](StringRef ns) { return (ns == KrnlDialect::getDialectNamespace()); });
  if (hasKrnlOps)
    return onnx_mlir::MLIRLevel;

  // Otherwise, set to the lowest level, LLVMLevel.
  return onnx_mlir::LLVMLevel;
}

void addPasses(mlir::OwningOpRef<ModuleOp> &module, mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType emissionTarget, std::string outputNameNoExt) {
  onnx_mlir::InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  if (inputIRLevel <= onnx_mlir::ONNXLevel &&
      emissionTarget >= onnx_mlir::EmitSPADEIR) {
    onnx_mlir::addONNXToMLIRPasses(
        pm, /*target CPU*/ onnx_mlir::maccel.empty());
    pm.addPass(spade::createLowerToAISLEPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (emissionTarget >= onnx_mlir::EmitSPADEMLIR) {
    if (inputIRLevel <= onnx_mlir::ONNXLevel)
      onnx_mlir_spade::addONNXToKrnlPasses(pm, onnx_mlir::OptimizationLevel,
          /*enableCSE*/ true, onnx_mlir::instrumentONNXSignature,
          onnx_mlir::ONNXOpStats);
    if (inputIRLevel <= onnx_mlir::MLIRLevel) {
      onnx_mlir_spade::addKrnlToAffinePasses(pm);
    }
    pm.addPass(spade::createLowerToAISMEMPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (inputIRLevel <= onnx_mlir::LLVMLevel &&
      emissionTarget >= onnx_mlir::EmitSPADELLVMIR) {
    onnx_mlir_spade::addKrnlToLLVMPasses(
        pm, outputNameNoExt, /*enableCSE=*/true);
  }
}

} // namespace onnx_mlir_spade
