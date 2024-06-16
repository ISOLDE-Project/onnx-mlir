/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- CompilerPasses.hpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for configuring and adding passes.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/Pass/PassManager.h"

namespace onnx_mlir_spade {
// Configures passes up front based on command line options.
void configurePasses();

void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel, bool enableCSE,
    bool enableInstrumentONNXSignature, std::string ONNXOpsStatFilename);
void addKrnlToAffinePasses(mlir::PassManager &pm);
void addKrnlToLLVMPasses(
    mlir::OpPassManager &pm, std::string outputNameNoExt, bool enableCSE);
onnx_mlir::InputIRLevelType determineInputIRLevel(
    mlir::OwningOpRef<mlir::ModuleOp> &module);
void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType emissionTarget, std::string outputNameNoExt);

} // namespace onnx_mlir_spade
