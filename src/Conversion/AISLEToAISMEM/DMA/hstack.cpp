/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- hstack.cpp - Lowering hstack Op ---------------------===//
//
// This file lowers the AISLE hstack Operator to AISMEM dialect.
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/AISLEToAISMEM/helper.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Support/SpadeSupport.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "AISLEToAISMEM_hstack"
using namespace mlir;

namespace spade {

struct AISMEMhstackOpLowering : public ConversionPattern {

  using theOperation = spade::AISLEhstack;
  using theAdaptor = spade::AISLEhstackAdaptor;

  AISMEMhstackOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    theOperation oldOp = llvm::dyn_cast<theOperation>(op);
    theAdaptor operandAdaptor(operands);
    theAdaptor attrAdaptor(oldOp);
    IntegerAttr alignmentAttr = rewriter.getI64IntegerAttr(16);

    ::mlir::Value A = operandAdaptor.getA();
    ::mlir::Value B = operandAdaptor.getB();

    LLVM_DEBUG({ ::llvm::outs()<<"before\n";spade::dumpBlock(op); });
    LLVM_DEBUG({ ::llvm::outs()<<"A:\n";A.dump(); });
    LLVM_DEBUG({ ::llvm::outs()<<"B:\n";A.dump(); });

    bool insert_dealloc = aisle_to_aismem::shallInsertDealoc(oldOp);

    // create the result
    auto results = oldOp.getODSResults(0);
    Type ResultType = (*results.begin()).getType();
    auto theType = aisle_to_aismem::convertTensorToMemRef(ResultType);

    auto result = rewriter.create<memref::AllocOp>(loc, theType);
    result.setAlignmentAttr(alignmentAttr);
    LLVM_DEBUG({ ::llvm::outs()<<"result:\n";result.dump(); });

    //-
    if (insert_dealloc) {
      auto *parentBlock = result->getBlock();
      auto dealloc = rewriter.create<memref::DeallocOp>(loc, result);
      dealloc->moveBefore(&parentBlock->back());
    }

    // Define constants
    Value numElements = rewriter.create<arith::ConstantIndexOp>(loc, 256);
    Value idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Allocate DMA tags
    mlir::MemRefType tagType = MemRefType::get(/*shape =*/{1},
        /*elementType=*/rewriter.getI32Type(), /*map=*/
        AffineMap::get(
            1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext()),
        /*memorySpaceInd=*/4);

    Value tag0 = rewriter.create<memref::AllocOp>(loc, tagType);
    Value tag1 = rewriter.create<memref::AllocOp>(loc, tagType);

    // DMA transfer for the first half of the result
    rewriter.create<memref::DmaStartOp>(loc, A,
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0),
            rewriter.create<arith::ConstantIndexOp>(loc, 0)},
        result,
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0),
            rewriter.create<arith::ConstantIndexOp>(loc, 0)},
        numElements, tag0, idx);

    // DMA transfer for the second half of the result
    rewriter.create<memref::DmaStartOp>(loc, B,
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0),
            rewriter.create<arith::ConstantIndexOp>(loc, 0)},
        result,
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0),
            rewriter.create<arith::ConstantIndexOp>(loc, 5)},
        numElements, tag1, idx);

    // Wait for both DMA operations to complete
    rewriter.create<memref::DmaWaitOp>(loc, tag0, idx, numElements);
    rewriter.create<memref::DmaWaitOp>(loc, tag1, idx, numElements);

    //
    ::llvm::SmallVector<::mlir::Value> tblgen_repl_values;
    for (auto v :
        ::llvm::SmallVector<::mlir::Value, 4>{result.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(oldOp, tblgen_repl_values);
    oldOp->replaceAllUsesWith(result);

    LLVM_DEBUG({ ::llvm::outs()<<"after\n";spade::dumpBlock(result); });
    return ::mlir::success();
  }
};

void populateLoweringAISLEhstackOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<AISMEMhstackOpLowering>(ctx);
}

} // namespace spade
