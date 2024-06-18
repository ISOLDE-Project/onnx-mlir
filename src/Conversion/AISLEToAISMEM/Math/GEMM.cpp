/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- GEMM.cpp - Lowering GEMM Op -------------------===//
//
// This file lowers the AISMEM GEMM Operator to AISMEM dialect.
//
//===----------------------------------------------------------------------===//

// #include "src/Compiler/CompilerOptions.hpp"
#include "helper.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "set"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include"src/Support/SpadeSupport.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "AISLE_TO_AISMEM_GEMM"
using namespace mlir;

namespace spade {


struct AISMEMGEMMOpLowering : public ConversionPattern {

    /**!!
    */  
    using theOperation  = spade::AISLEGEMMOp;
    using theAdaptor    = spade::AISLEGEMMOpAdaptor;
    using theNewOp      = spade::AISMEMGEMMOp;

    AISMEMGEMMOpLowering(MLIRContext *ctx)
        : ConversionPattern(theOperation::getOperationName(), 1, ctx) {

    }

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const final {



        Location loc = op->getLoc();
        theOperation oldOp = llvm::dyn_cast<theOperation>(op);
        theAdaptor operandAdaptor(operands);
        theAdaptor attrAdaptor(oldOp);

        ::mlir::Value A = operandAdaptor.getA();
        ::mlir::Value A_Shape = operandAdaptor.getAShape();
        ::mlir::Value B = operandAdaptor.getB();
        ::mlir::Value B_Shape = operandAdaptor.getBShape();
        ::mlir::Value C = operandAdaptor.getC();


        LLVM_DEBUG({ spade::dumpBlock(op); });
        //*
        //* transform parameter
        //*

        std::set<Operation *> obsoleteOps;
        aisle_to_aismem::getConversionCastOperand(A, obsoleteOps);
        aisle_to_aismem::getConversionCastOperand(A_Shape, obsoleteOps);
        aisle_to_aismem::getConversionCastOperand(B, obsoleteOps);
        aisle_to_aismem::getConversionCastOperand(B_Shape, obsoleteOps);
        aisle_to_aismem::getConversionCastOperand(C, obsoleteOps);

        bool insert_dealloc = aisle_to_aismem::shallInsertDealoc(oldOp);

        auto results = oldOp.getODSResults(0);
        Type convOpResType = (*results.begin()).getType();
        auto theType = aisle_to_aismem::convertTensorToMemRef(convOpResType);
        /*
        * in case of dynamic tensors, use the DimOp of the arg%0
        */
        auto theDim = aisle_to_aismem::inferDim(op);

        auto newAlloc =
            aisle_to_aismem::insertAlloc(rewriter, loc, theDim, theType);
    //-
        if (insert_dealloc) {
            auto *parentBlock = newAlloc->getBlock();
            auto dealloc = rewriter.create<memref::DeallocOp>(loc, newAlloc);
            dealloc->moveBefore(&parentBlock->back());
        }
        //**
        ::llvm::SmallVector<::mlir::Value> tblgen_values;
        ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
        tblgen_values.push_back(A);
        tblgen_values.push_back(A_Shape);
        tblgen_values.push_back(B);
        tblgen_values.push_back(B_Shape);
        tblgen_values.push_back(C);
        tblgen_values.push_back(newAlloc);
        
        tblgen_attrs.push_back(::mlir::NamedAttribute(oldOp.getTransAAttrName(),oldOp.getTransAAttr()));
        tblgen_attrs.push_back(::mlir::NamedAttribute(oldOp.getTransBAttrName(),oldOp.getTransBAttr()));

        ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
        tblgen_types.push_back(rewriter.getNoneType());
    

        ::llvm::SmallVector<::mlir::Value> tblgen_repl_values;
        for (auto v :
            ::llvm::SmallVector<::mlir::Value, 4>{newAlloc.getODSResults(0)}) {
        tblgen_repl_values.push_back(v);
        }

        theNewOp tblgen_newOperation_0;
        tblgen_newOperation_0 =
            rewriter.create<theNewOp>(loc, tblgen_types, tblgen_values, tblgen_attrs);
        tblgen_newOperation_0->moveAfter(newAlloc);

        // lower down operation
        rewriter.replaceOp(oldOp, tblgen_repl_values);
        oldOp->replaceAllUsesWith(newAlloc);

        aisle_to_aismem::eraseOp(rewriter, obsoleteOps);

        LLVM_DEBUG({ spade::dumpBlock(tblgen_newOperation_0); });
        return ::mlir::success();
    }
};

void populateLoweringAISLEGEMMOpPattern(RewritePatternSet &patterns,
        TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
    patterns.insert<AISMEMGEMMOpLowering>(ctx);
}

} // namespace spade
