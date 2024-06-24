/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ AllocOp.cpp - Lower KrnlGlobalOp ---------------------------===//
//
// Copyright 2023
//
// =============================================================================
//
// This file lowers the memref.alloc operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <string>

#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Support/SpadeSupport.hpp"

#include "src/Support/logger.hpp"

#define DEBUG_TYPE "AISMEMToLLVM_AllocOP"

using namespace mlir;

namespace spade {

DECLARE_LOGGER(MemManager, MemManager.yaml)

using AllocOp = memref::AllocOp;

class AllocOpLowering : public ConvertToLLVMPattern {
public:
  explicit AllocOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            AllocOp::getOperationName(), context, typeConverter) {}

  LLVM::LLVMFuncOp getSymbol(ModuleOp module,
      ConversionPatternRewriter &rewriter, AllocOp &allocOp, Type retType,
      StringRef funcName) const {
    Location loc = allocOp.getLoc();

    auto memFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!memFunc) {
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      //
      auto arg0Type = rewriter.getI32Type();

      auto funcType = LLVM::LLVMFunctionType::get(
          retType, ::llvm::ArrayRef<Type>{arg0Type}, false);
      memFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
    }
    return memFunc;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto allocOp = llvm::dyn_cast<AllocOp>(op);
    Location loc = allocOp.getLoc();
    mlir::MLIRContext *ctx = op->getContext();
    {
      std::ostringstream asc;
      asc << "---\n";
      MemManager::getInstance().header(asc);
    }
    LLVM::LLVMFuncOp memFunc = nullptr;
    {
      ModuleOp module = op->getParentOfType<ModuleOp>();
      // auto *context = module.getContext();
      auto funcName = StringRef(reserveFuncName);
      // const Type type = allocOp->getResult(0).getType();
      // const MemRefType memRefTy = type.cast<mlir::MemRefType>();
      // const Type elementType =
      //    typeConverter->convertType(memRefTy.getElementType());
      auto retType = LLVM::LLVMPointerType::get(ctx);
      memFunc = getSymbol(module, rewriter, allocOp, retType, funcName);
    }

    if (!memFunc)
      return failure();

    auto arg0Type = rewriter.getI32Type();
    std::int32_t arg0Val = ++hashValue;
    Value arg0 = rewriter.create<LLVM::ConstantOp>(loc, arg0Type, arg0Val);

    ValueRange args = {arg0};
    auto newCallOp = rewriter.create<LLVM::CallOp>(loc, memFunc, args);

    mlir::UnrealizedConversionCastOp castOp;
    LLVM::ExtractValueOp extractValueOp;
    LLVM::ReturnOp returnOp;

    for (auto *user : op->getUsers()) {
      castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
      if (castOp)
        break;
    }

    if (castOp) {
      for (auto *user : castOp->getUsers()) {
        extractValueOp = llvm::dyn_cast<LLVM::ExtractValueOp>(user);
        if (extractValueOp)
          break;
      }
    }

    if (extractValueOp) {
      for (auto *user : extractValueOp->getUsers()) {
        returnOp = llvm::dyn_cast<LLVM::ReturnOp>(user);
        if (returnOp)
          break;
      }
    }

    LLVM_DEBUG({
      llvm::errs()
          << "-- Before update in AllocOpLowering::matchAndRewrite() --\n";
      spade::dumpBlock(op);
      llvm::errs() << "---\n";
    });
    SmallVector<Value, 4> newValues(
        newCallOp->getResults().begin(), newCallOp->getResults().end());

    // rewriter.replaceOp(op, {newCallOp.getODSResults(0)});
    rewriter.replaceOp(op, newValues);
    if (returnOp) {

      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp, newValues);

      // Erase intermediate operations
      rewriter.eraseOp(extractValueOp);
      rewriter.eraseOp(castOp);
    }
    {
      std::ostringstream asc;
      const Type type = allocOp->getResult(0).getType();
      const MemRefType memRefTy = type.cast<mlir::MemRefType>();
      auto array = memRefTy.getShape().data();
      auto len = memRefTy.getShape().size();
      asc << reserveFuncName << "_" << arg0Val << ":\n";
      asc << "   - " << arg0Val << "\n";
      for (size_t i = 0; i < len; ++i)
        asc << "   - " << array[i] << "\n";
      MemManager::getInstance().info(asc);
    }

    for (auto user : allocOp->getUsers()) {
      auto deallocOp = llvm::dyn_cast<memref::DeallocOp>(user);
      if (deallocOp) {
        auto retType = LLVM::LLVMVoidType::get(
            rewriter
                .getContext()); // typeConverter->convertType(rewriter.getNoneType());
        memFunc = getSymbol(deallocOp->getParentOfType<ModuleOp>(), rewriter,
            allocOp, retType, StringRef(releaseFuncName));
        if (!memFunc)
          return failure();
        ValueRange args = {arg0};
        auto newCallOp = rewriter.create<LLVM::CallOp>(loc, memFunc, args);
        newCallOp->moveBefore(deallocOp);
        // rewriter.replaceOp(deallocOp, {newCallOp.getODSResults(0)});
        rewriter.eraseOp(deallocOp);
      }
    }
    LLVM_DEBUG({
      llvm::errs()
          << "-- After update in AllocOpLowering::matchAndRewrite() --\n";
      spade::dumpBlock(op);
      llvm::errs() << "---\n";
    });

    return success();
  }

private:
  inline static int32_t hashValue = 0;
  inline static std::string reserveFuncName = "_reserveMemory";
  inline static std::string releaseFuncName = "_releaseMemory";
};

// int32_t AllocOpLowering::hashValue = 21;

void populateMemrefAllocOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<AllocOpLowering>(typeConverter, ctx);
}

} // namespace spade
