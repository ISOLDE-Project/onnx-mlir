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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Support/SpadeSupport.hpp"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <string>

#define DEBUG_TYPE "AISMEMToAISLLVM_QConstant"

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

    ///
    LLVM_DEBUG({
      auto pczs_doqa = __PRETTY_FUNCTION__;
      std::string bkcl_l;
      if (strlen(pczs_doqa) < 120)
        bkcl_l = pczs_doqa;
      else
        bkcl_l = "unknown";
      ::llvm::outs() << bkcl_l.c_str() << "\n";
      qConstant.dump();
      ::llvm::outs() << "\n";
    });

    ///

    // The element type of the array.
    const Type type = op->getResult(0).getType();
    const MemRefType memRefTy = type.cast<mlir::MemRefType>();
    const Type constantElementType =
        typeConverter->convertType(memRefTy.getElementType());
    Type globalType = constantElementType;

    LLVM_DEBUG({
      ::llvm::outs() << "constantElementType :";
      constantElementType.dump();
      ::llvm::outs() << "globalType :";
      globalType.dump();
    });

    VectorType vectorType = VectorType::get({4}, globalType);
    LLVM_DEBUG({ vectorType.dump(); });
    Value vector = rewriter.create<LLVM::UndefOp>(loc, vectorType);
    int32_t index = 0;
    Value lastValue = vector;
    auto denseElements = qConstant.getValue().cast<DenseElementsAttr>();
    LLVM_DEBUG({ denseElements.dump(); });
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
      LLVM_DEBUG({
        ::llvm::outs() << index << ": ";
        lastValue.dump();
      });
    }
     
     auto ptrType = LLVM::LLVMPointerType::get(op->getContext());
    auto allocatedPtr = rewriter.create<LLVM::AllocaOp>(loc, vectorType, elementType, lastValue, 16);
    rewriter.create<LLVM::StoreOp>(loc, lastValue, allocatedPtr);
    auto castedPtr = rewriter.create<LLVM::BitcastOp>(loc, ptrType, allocatedPtr);
    //rewriter.replaceOp(op, {lastValue});
    rewriter.replaceOp(op, static_cast<Value>(castedPtr));
    //qConstant.replaceAllUsesWith(lastValue);
    LLVM_DEBUG({
      ::llvm::outs() << "after\n";
      spade::dumpBlock(op);
    });
    return success();
  }
};

void populateAISMEMQConstantOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<AISMEMQConstantOpLowering>(typeConverter, ctx);
}

} // namespace spade
