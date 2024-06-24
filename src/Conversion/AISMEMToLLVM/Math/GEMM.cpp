/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- GEMM.cpp - Lowering GEMM Op -------------------===//
//
// This file lowers the AISMEM GEMM Operator to AISLLVM dialect.
//
//===----------------------------------------------------------------------===//

// #include "src/Compiler/CompilerOptions.hpp"
#include "helper.hpp"

#define DEBUG_TYPE "MemToLLVM_GEMM"
using namespace mlir;

namespace spade {

struct AISMEMGEMMOpLowering : public ConvertToLLVMPattern {

  using theOperation = spade::AISMEMGEMMOp;
  using theAdaptor = spade::AISMEMGEMMOpAdaptor;
  using theNewOp = spade::AISLLVMGEMMOp;

  AISMEMGEMMOpLowering(LLVMTypeConverter &typeConverter, MLIRContext *ctx)
      : ConvertToLLVMPattern(
            theOperation::getOperationName(), ctx, typeConverter) {
    // ctx->getOrLoadDialect<spade::AISMEMDialect>();
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    theOperation oldOp = llvm::dyn_cast<theOperation>(op);
    theAdaptor operandAdaptor(operands);
    theAdaptor attrAdaptor(oldOp);

    ::mlir::Value Y = operandAdaptor.getY();
    ::mlir::Value A = operandAdaptor.getA();
    ::mlir::Value A_Shape = operandAdaptor.getAShape();
    ::mlir::Value B = operandAdaptor.getB();
    ::mlir::Value B_Shape = operandAdaptor.getBShape();
    ::mlir::Value C = operandAdaptor.getC();

    int32_t transA = attrAdaptor.getTransA();
    int32_t transB = attrAdaptor.getTransB();

    Value valueTransA =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), transA);
    Value valueTransB =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), transB);

    LLVM_DEBUG({ spade::dumpBlock(op); });
    //*
    //* transform parameter
    //*

    //**
    ::llvm::SmallVector<::mlir::Value> tblgen_values;
    tblgen_values.push_back(Y);
    tblgen_values.push_back(A);
    tblgen_values.push_back(A_Shape);
    tblgen_values.push_back(B);
    tblgen_values.push_back(B_Shape);
    tblgen_values.push_back(C);
    tblgen_values.push_back(valueTransA);
    tblgen_values.push_back(valueTransB);

    VectorType resultType = VectorType::get({4}, rewriter.getI32Type());
    ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
    tblgen_types.push_back(resultType);

    theNewOp tblgen_newOperation_0;
    tblgen_newOperation_0 =
        rewriter.create<theNewOp>(loc, tblgen_types, tblgen_values);

    ::llvm::SmallVector<::mlir::Value> tblgen_repl_values;
    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_newOperation_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    // lower down operation
    rewriter.replaceOp(oldOp, tblgen_repl_values);

    LLVM_DEBUG({ spade::dumpBlock(tblgen_newOperation_0); });
    return ::mlir::success();
  }
};

void populateLoweringAISMEMGEMMOpPattern(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<AISMEMGEMMOpLowering>(typeConverter, ctx);
}

} // namespace spade
