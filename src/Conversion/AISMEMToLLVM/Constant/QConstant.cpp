/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ QConstant.cpp - Lower KrnlGlobalOp ---------------------------===//
//
// Copyleft
// =============================================================================
//
// This file lowers the QConstant operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Support/SpadeSupport.hpp"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <string>

#define DEBUG_TYPE "AISMEMToLLVM_QConstant"

using namespace mlir;

namespace spade {

class AISMEMQConstantOpLowering : public ConvertToLLVMPattern {
  using theOperation = spade::AISMEMQConstantOp;
  using theAdaptor = spade::AISMEMQConstantOpAdaptor;

public:
  explicit AISMEMQConstantOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            theOperation::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto qConstant = llvm::dyn_cast<theOperation>(op);
    Location loc = qConstant.getLoc();

    LLVM_DEBUG({
      llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
      spade::dumpBlock(op);
      llvm::outs() << "-----------\n";
    });
    // The element type of the array.
    const Type type = op->getResult(0).getType();
    const MemRefType memRefTy = type.cast<mlir::MemRefType>();
    const Type constantElementType =
        typeConverter->convertType(memRefTy.getElementType());
    Type globalType = constantElementType;

    VectorType vectorType = VectorType::get({4}, globalType);
    Value vector = rewriter.create<LLVM::UndefOp>(loc, vectorType);
    int32_t index = 0;
    Value lastValue = vector;
    auto denseElements = qConstant.getValue().cast<DenseElementsAttr>();
    auto intOrFpEltAttr = denseElements.dyn_cast<DenseIntOrFPElementsAttr>();
    if (!intOrFpEltAttr)
      return failure();
    auto densetype = intOrFpEltAttr.getType();
    auto elementType = densetype.getElementType();
    if (!elementType.isIntOrIndex())
      return failure();
    auto range = intOrFpEltAttr.getValues<llvm::APInt>();
    for (auto attr : range) {
      auto elem = *attr.getRawData();
      // auto elem= array.getValue()[i].cast<globalType>().getInt();
      Value value = rewriter.create<LLVM::ConstantOp>(loc, globalType, elem);
      Value pos = rewriter.create<LLVM::ConstantOp>(loc, globalType, index++);
      lastValue = rewriter.create<LLVM::InsertElementOp>(
          loc, vectorType, lastValue, value, pos);
    }

//
    
    rewriter.replaceOp(op, {lastValue});
  

    mlir::UnrealizedConversionCastOp castOp;
    

    LLVM::AllocaOp allocatedPtr;
    for (auto *user : op->getUsers()) {
      castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
      if (castOp) {
        Type resultType = castOp->getResults()[0].getType();
        if (resultType.isa<LLVM::LLVMPointerType>()) {
          if (!allocatedPtr) {
            auto ptrType = LLVM::LLVMPointerType::get(op->getContext());
            Value size = rewriter.create<LLVM::ConstantOp>(loc, globalType, 64);
            allocatedPtr = rewriter.create<LLVM::AllocaOp>(
                loc, ptrType, vectorType, size, 16);
            rewriter.create<LLVM::StoreOp>(loc, lastValue, allocatedPtr);
          }
          rewriter.replaceAllUsesWith(castOp->getResults()[0], {allocatedPtr});
          bool deadOp = mlir::isOpTriviallyDead(castOp);
          bool useEmpty = castOp.use_empty();
          if (deadOp && useEmpty) {
            castOp->dropAllUses();
            rewriter.eraseOp(castOp);
          }
        }
      }
    }


    LLVM_DEBUG({
      llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
      spade::dumpBlock(op);
      llvm::outs() << "-----------\n";
    });
    return success();
  }
};

void populateAISMEMQConstantOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<AISMEMQConstantOpLowering>(typeConverter, ctx);
}

} // namespace spade
