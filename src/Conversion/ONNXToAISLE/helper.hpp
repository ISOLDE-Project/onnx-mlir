/*AISLE
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include <functional>
#include <netinet/in.h>

 using namespace mlir;
 namespace onnx_to_aisle{

  inline void getVectorShape(Value value, SmallVector<int>& Vec)  {
    TensorType t_type = value.getType().cast<TensorType>();
    if(!t_type)
      return;
    auto x_shape=t_type.getShape();
    auto rank=x_shape.size();
    /*
    * check hardware constrain 
    **********
    *   rank greater than 4 not supported in hardware
    **********
    */    
    if(rank>4)
      assert(false && " rank greater than 4 not supported in hardware "); 
    for(size_t i=0;i<4-rank;++i)
       Vec.push_back(1);
    for(size_t i=0;i<rank;++i)
      Vec.push_back(x_shape[i]);    
  }

  inline void getAttrValues(::mlir::ArrayAttr array, SmallVector<int>& Vec)  {
    //array.dump();
    for(size_t i=0;i<array.size();++i){
        mlir::Attribute attr = array.getValue()[i];
        if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
            Vec.push_back(intAttr.getInt());
        }
      //  attr.dump();

    }
  }


    template< typename _Class,typename _Func, typename _Operation>
    spade::AISLEQConstantOp create(ConversionPatternRewriter &rewriter,  _Operation& op, const char* name, _Func mem_func){
        auto elementType= rewriter.getI32Type();
        auto paramShape = mlir::RankedTensorType::get(llvm::ArrayRef<std::int64_t>{1,4},elementType);
        _Class adaptor(op);
        auto get_val = std::mem_fn(mem_func);
        Value X = get_val(adaptor);
        SmallVector<int> Vec;
        onnx_to_aisle::getVectorShape(X,Vec);
        auto input_shape_dense = DenseElementsAttr::get(paramShape, ArrayRef<int>(Vec));
        auto iShapeParam= rewriter.create<spade::AISLEQConstantOp>(op.getLoc()
                                                                                    ,name
                                                                                    ,input_shape_dense);
        return iShapeParam;
    }
    

     template<  typename _Operation>
     spade::AISLEQConstantOp create(ConversionPatternRewriter &rewriter,  _Operation& op, const char* name,  SmallVector<int>& Vec){
        auto elementType= rewriter.getI32Type();
        auto paramShape = mlir::RankedTensorType::get(llvm::ArrayRef<std::int64_t>{1,4},elementType);
        
        auto input_shape_dense = DenseElementsAttr::get(paramShape, ArrayRef<int>(Vec));
        auto iShapeParam= rewriter.create<spade::AISLEQConstantOp>(op.getLoc()
                                                                                    ,name
                                                                                    ,input_shape_dense);
        return iShapeParam;
    }

  
 }