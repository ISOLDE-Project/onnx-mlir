/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- GEMM.cpp - Lowering GEMM Op -------------------===//
//
// This file lowers the ONNX Reluolution Operators to AISLE dialect.
//
//===----------------------------------------------------------------------===//

#include "../helper.hpp"


using namespace mlir;

namespace spade {

struct ONNXGEMMOpLowering : public ConversionPattern {

    
    using theOperation  = mlir::ONNXGemmOp;
    using theAdaptor    = mlir::ONNXGemmOpAdaptor;
    using theNewOp      = spade::AISLEGEMMOp;

  ONNXGEMMOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {
         //ctx->getOrLoadDialect<spade::AISLEDialect>();
      }
  

  


  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {



    Location loc = op->getLoc();
    theOperation oldOp = llvm::dyn_cast<theOperation>(op);
    theAdaptor operandAdaptor(operands);
    theAdaptor attrAdaptor(oldOp);

    ::mlir::Value A = operandAdaptor.getA();
    ::mlir::Value A_Shape = onnx_to_aisle::create< theAdaptor>(rewriter, oldOp, "A_shape",&theAdaptor::getA);;
    ::mlir::Value B = operandAdaptor.getB();
    ::mlir::Value B_Shape = onnx_to_aisle::create< theAdaptor>(rewriter, oldOp, "B_shape",&theAdaptor::getB);;
    ::mlir::Value C = operandAdaptor.getC();
    

    double  alpha = attrAdaptor.getAlpha().convertToDouble();
    double  beta = attrAdaptor.getBeta().convertToDouble();
    if(alpha != 1.0 || beta != 1.0){
      assert(false && " Scalar multiplier(s) not supported, shall be set to 1.0 ");

    }
        
    ::llvm::SmallVector<::mlir::Value> tblgen_repl_values;
    theNewOp tblgen_newOperation_0;
    {
      ::llvm::SmallVector<::mlir::Value> tblgen_values; 
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
        tblgen_values.push_back(A);
        tblgen_values.push_back(A_Shape);
        tblgen_values.push_back(B);
        tblgen_values.push_back(B_Shape);
        tblgen_values.push_back(C);
        
        tblgen_attrs.push_back(::mlir::NamedAttribute(oldOp.getTransAAttrName(),attrAdaptor.getTransAAttr()));
        tblgen_attrs.push_back(::mlir::NamedAttribute(oldOp.getTransBAttrName(),attrAdaptor.getTransBAttr()));
  
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: oldOp.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_newOperation_0 = rewriter.create<theNewOp>(loc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_newOperation_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op, tblgen_repl_values);
    return ::mlir::success();
  }
};

     
void populateLoweringONNXToAISLEGEMMOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXGEMMOpLowering>( ctx);
}

} // namespace spade
