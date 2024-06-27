/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- GEMM.cpp - Lowering GEMM Op -------------------===//
//
// This file lowers the AISMEM GEMM Operator to LLVM dialect.
//
//===----------------------------------------------------------------------===//
//test cmd:
//make  ONNX_MODEL=gemm.mlir  ONNX_MLIR_FLAGS=--debug-only=AISMEMToLLVM_GEMM  test
// #include "src/Compiler/CompilerOptions.hpp"
#include "helper.hpp"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "AISMEMToLLVM_GEMM"
using namespace mlir;

namespace {

struct AISMEMGEMMOpLowering : public ConvertToLLVMPattern {

  using theOperation = spade::AISMEMGEMMOp;
  using theAdaptor = spade::AISMEMGEMMOpAdaptor;
  using theNewOp = spade::AISLLVMGEMMOp;
  using setCastOps = ::std::set<UnrealizedConversionCastOp>;

  AISMEMGEMMOpLowering(LLVMTypeConverter &typeConverter, MLIRContext *ctx)
      : ConvertToLLVMPattern(
            theOperation::getOperationName(), ctx, typeConverter) {
    // ctx->getOrLoadDialect<spade::AISMEMDialect>();
  }

  inline void getConversionCastOperand(
      Value &val, setCastOps &obsoleteOps) const {
        auto op =val.getDefiningOp();
        if(!op) return;
    UnrealizedConversionCastOp cast_0 =
        llvm::dyn_cast<UnrealizedConversionCastOp>(op);
    if (cast_0) {
      obsoleteOps.insert(cast_0);
      val = *cast_0.getInputs().begin();
      getConversionCastOperand(val, obsoleteOps);
    }
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final{

    Location loc = op->getLoc();
    theOperation oldOp = llvm::dyn_cast<theOperation>(op);
    theAdaptor operandAdaptor(operands);
    // auto ctx = oldOp->getContext();
    // const LLVMTypeConverter *typeConverter =
    //     static_cast<const LLVMTypeConverter *>(getTypeConverter());
    theAdaptor attrAdaptor(oldOp);

    LLVM_DEBUG({
      llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
       spade::dumpBlock(op);
      llvm::errs() << "-----------\n";
    });

    setCastOps obsoleteOps;
    ::mlir::Value Y = operandAdaptor.getY();
    //Y.dump();
    getConversionCastOperand(Y, obsoleteOps);

    ::mlir::Value A = operandAdaptor.getA();
    //A.dump();
    getConversionCastOperand(A, obsoleteOps);

    ::mlir::Value A_Shape = operandAdaptor.getAShape();
    //A_Shape.dump();
    getConversionCastOperand(A_Shape, obsoleteOps);

    ::mlir::Value B = operandAdaptor.getB();
    //B.dump();
    getConversionCastOperand(B, obsoleteOps);

    ::mlir::Value B_Shape = operandAdaptor.getBShape();
    //B_Shape.dump();
    getConversionCastOperand(B_Shape, obsoleteOps);

    ::mlir::Value C = operandAdaptor.getC();
   // C.dump();
    getConversionCastOperand(C, obsoleteOps);

    int32_t transA = attrAdaptor.getTransA();
    int32_t transB = attrAdaptor.getTransB();

    Value valueTransA =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), transA);
    Value valueTransB =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), transB);

    ::llvm::SmallVector<::mlir::Value> tblgen_values;
    tblgen_values.push_back(Y);
    tblgen_values.push_back(A);
    tblgen_values.push_back(A_Shape);
    tblgen_values.push_back(B);
    tblgen_values.push_back(B_Shape);
    tblgen_values.push_back(C);
    tblgen_values.push_back(valueTransA);
    tblgen_values.push_back(valueTransB);

    LLVM_DEBUG({
      llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
      llvm::errs() << "Operands\n";
      for (auto val : tblgen_values)
        val.dump();
      llvm::errs() << "Block\n";
      spade::dumpBlock(op);
      llvm::errs() << "-----------\n";
    });

    // Type resultType = LLVMIntegerType::get(builder.getContext(), 1);
    /**26.06.2024
    ** runtime error,error: invalid vector element type, I could't find a proper
    solution to it Type I32Type = rewriter.getIntegerType(32); auto llvmI32Type
    =typeConverter->convertType(I32Type); auto llvmVector4I32Type =
    LLVM::LLVMFixedVectorType::get(llvmI32Type, 4);
     ::llvm::SmallVector<::mlir::Type, 4> tblgen_types{llvmVector4I32Type};
    **/
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
 /*  
//not a good ideea to erase castOps,
//runtime error
//onnx-mlir: /home/uic52463/hdd2/task5.2/toolchain/riscv-llvm/mlir/lib/IR/Operation.cpp:514: 
void llvm::ilist_traits<mlir::Operation>::removeNodeFromList(llvm::ilist_traits<mlir::Operation>::Operation*): 
Assertion `op->block && "not already in an operation block!"' failed.
  */
    //  for(auto castOp: obsoleteOps){
    //       bool deadOp = mlir::isOpTriviallyDead(castOp);
    // bool useEmpty = castOp.use_empty();
    // if (deadOp && useEmpty) {
    //   llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
    //   LLVM_DEBUG({ llvm::errs() << "erasing: ";castOp.print(llvm::errs()); llvm::errs()<<  "\n";});
    //   castOp->dropAllUses();
    //   rewriter.eraseOp(castOp);
    // }
    //  }
    

    LLVM_DEBUG({
      llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
      spade::dumpBlock(tblgen_newOperation_0);
      llvm::outs() << "-----------\n";
    });
    return ::mlir::success();
  }
};

} // namespace
namespace spade {
void populateLoweringAISMEMGEMMOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<AISMEMGEMMOpLowering>(typeConverter, ctx);
}

} // namespace spade
