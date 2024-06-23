/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToLLVM.hpp - Krnl Dialect Lowering  ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Lowering of Krnl operations to a combination of other dialects.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

const std::string DEFAULT_DYN_ENTRY_POINT = "run_main_graph_ex";

namespace spade {
namespace krnl {

void populateAffineAndKrnlToLLVMConversion(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx,
    llvm::ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &inputMemRefTypes,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &outputMemRefTypes,
    bool verifyInputTensors, bool enableParallel);

void populateKrnlToLLVMConversion(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
    llvm::ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &inputMemRefTypes,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &outputMemRefTypes,
    bool verifyInputTensors);

void determineOwnershipForOutputOMTensors(mlir::ModuleOp &module,
    llvm::SmallVectorImpl<bool> &outputOMTensorOwnerships);

void recordEntryPointSignatures(mlir::ModuleOp &module,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps);

void genSignatureFunction(mlir::ModuleOp &module,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps);
} // namespace krnl

void populateLoweringKrnlEntryPointOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

} // namespace spade

namespace onnx_mlir {
namespace krnl {

void populateLoweringKrnlCallOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlEntryPointOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx, llvm::ArrayRef<bool> constantOutputs,
    bool singleEntryPoint,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &inputMemRefTypes,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &outputMemRefTypes,
    bool verifyInputTensors);

void populateLoweringKrnlFindIndexOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlGlobalOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlInstrumentOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlMemcpyOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlPrintOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlPrintTensorOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlRandomNormalOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlStrlenOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlStrncmpOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlUnaryMathOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlVectorTypeCastOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlNoneOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

/// This function emits three functions: omQueryEntryPoints, omInputSignature
/// and omOutputSignature.
/// - omQueryEntryPoints has type of `**i8 (*i64)` to query an array of entry
/// point names.
/// - omInputSignature and omOutputSignature have type of type `*i8 (*i8)` to
/// return input and output signatures of the given entry point.
void genSignatureFunction(mlir::ModuleOp &module,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps);

/// Extract then pack constant arrays and store to a file.
/// Return true if there are constants that are OK to store on files.
/// A single constant's size must be greater than singleThreshold.
/// The total size of contants must be greater than totalThreshold.
bool extractConstantsToFile(mlir::ModuleOp &module, std::string filepath,
    uint64_t singleThreshold, uint64_t totalThreshold);

/// Emit a function "omLoadConstantsFromFile" in the IR to load constants from
/// external files into global operations.
void loadConstantsFromFile(mlir::ModuleOp &module,
    const RuntimeAPIRegistry &apiRegistry,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    bool calledByEntryPoint = true);

void determineOwnershipForOutputOMTensors(mlir::ModuleOp &module,
    llvm::SmallVectorImpl<bool> &outputOMTensorOwnerships);

void PostfixEntrypointNames(mlir::ModuleOp &module);

/// Keep original MemRefTypes for inputs and outputs. These information will be
/// used for constructing OMTensors for inputs and outputs. We have to record
/// this information at this point before they are disappeared during the
/// lowering to LLVM. For example, unsigned types do not exist at LLVM level,
/// typed pointers becomes opaque if opaque point is enabled.
void recordInputOutputMemRefTypes(mlir::ModuleOp &module,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &inputMemRefTypes,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &outputMemRefTypes);

bool hasSingleEntryPoint(mlir::ModuleOp &module);
} // namespace krnl
} // namespace onnx_mlir