/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- GEMM.cpp - Lowering GEMM Op -------------------===//
//
// This file lowers the ONNX Reluolution Operators to AISLE dialect.
//
//===----------------------------------------------------------------------===//

#include "../helper.hpp"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Support/SpadeSupport.hpp"
#include <bits/stdint-intn.h>
#include <cstddef>

#define DEBUG_TYPE "ONNXToAISLE_GEMM"

using namespace mlir;

namespace spade {

struct ONNXGEMMOpLowering : public ConversionPattern {

  using theOperation = mlir::ONNXGemmOp;
  using theAdaptor = mlir::ONNXGemmOpAdaptor;
  using theNewOp = spade::AISLEGEMMOp;

  ONNXGEMMOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    theOperation oldOp = llvm::dyn_cast<theOperation>(op);
    theAdaptor operandAdaptor(operands);
    theAdaptor attrAdaptor(oldOp);

    //# Define the split point
    ::mlir::Value C = operandAdaptor.getC();
    TensorType typeC = C.getType().cast<TensorType>();
    assert(typeC);
    auto C_shape = typeC.getShape();
    int64_t splitPoint = C_shape[1] / 2;

    int64_t transB = attrAdaptor.getTransB();

    if (transB != 1) {
      assert(false && " transB != 0 is not implemented, 2024.06.20");
    }

    // get A
    ::mlir::Value A = operandAdaptor.getA();
    ::mlir::Value A_Shape = onnx_to_aisle::create<theAdaptor>(
        rewriter, oldOp, "A_shape", &theAdaptor::getA);
    // get shape
    TensorType typeA = A.getType().cast<TensorType>();
    assert(typeA);
    auto AShape = typeA.getShape();

    // split B
    std::pair<int64_t, int64_t> row;
    std::pair<int64_t, int64_t> col;

    mlir::Value B = operandAdaptor.getB();
    TensorRawData rawDataB(B);

    TensorRawData rawDataB1;
    row = {0, splitPoint};
    col = {0, rawDataB.shape[1]};
    rawDataB.splitMatrix(rawDataB1, row, col);
    auto B1 = rawDataB1.createONNXConstantOp(rewriter, loc);
    ::mlir::Value B1_Shape =
        onnx_to_aisle::create(rewriter, B1, "B1_shape", rawDataB1.shape);
    //
    llvm::SmallVector<int64_t> gemm1_shape;
    gemm1_shape.push_back(AShape[0]);
    gemm1_shape.push_back(rawDataB1.shape[0]);

    TensorRawData rawDataB2;
    row = {splitPoint, rawDataB.shape[0]};
    col = {0, rawDataB.shape[1]};
    rawDataB.splitMatrix(rawDataB2, row, col);
    auto B2 = rawDataB2.createONNXConstantOp(rewriter, loc);
    ::mlir::Value B2_Shape =
        onnx_to_aisle::create(rewriter, B2, "B2_shape", rawDataB1.shape);
    //
    llvm::SmallVector<int64_t> gemm2_shape;
    gemm2_shape.push_back(AShape[0]);
    gemm2_shape.push_back(rawDataB2.shape[0]);

    // split C
    TensorRawData rawDataC(C);

    TensorRawData rawDataC1;
    row = {0, rawDataC.shape[0]};
    col = {0, splitPoint};
    rawDataC.splitMatrix(rawDataC1, row, col);
    auto C1 = rawDataC1.createONNXConstantOp(rewriter, loc);

    TensorRawData rawDataC2;
    row = {0, rawDataC.shape[0]};
    col = {splitPoint, rawDataC.shape[1]};
    rawDataC.splitMatrix(rawDataC2, row, col);
    auto C2 = rawDataC2.createONNXConstantOp(rewriter, loc);

    double alpha = attrAdaptor.getAlpha().convertToDouble();
    double beta = attrAdaptor.getBeta().convertToDouble();
    if (alpha != 1.0 || beta != 1.0) {
      assert(
          false && " Scalar multiplier(s) not supported, shall be set to 1.0 ");
    }

    ::llvm::SmallVector<::mlir::Value> tblgen_repl_values;
    theNewOp tblgen_newOperation_0;
    {
      ::llvm::SmallVector<::mlir::Value> tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      tblgen_values.push_back(A);
      tblgen_values.push_back(A_Shape);
      tblgen_values.push_back(B1);
      tblgen_values.push_back(B1_Shape);
      tblgen_values.push_back(C1);

      tblgen_attrs.push_back(::mlir::NamedAttribute(
          oldOp.getTransAAttrName(), attrAdaptor.getTransAAttr()));
      tblgen_attrs.push_back(::mlir::NamedAttribute(
          oldOp.getTransBAttrName(), attrAdaptor.getTransBAttr()));
      // return type
      auto elementType = rewriter.getF32Type();
      auto retType = mlir::RankedTensorType::get(gemm1_shape, elementType);

      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      tblgen_types.push_back(retType);

      tblgen_newOperation_0 = rewriter.create<theNewOp>(
          loc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    theNewOp tblgen_newOperation_1;
    {
      ::llvm::SmallVector<::mlir::Value> tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      tblgen_values.push_back(A);
      tblgen_values.push_back(A_Shape);
      tblgen_values.push_back(B2);
      tblgen_values.push_back(B2_Shape);
      tblgen_values.push_back(C2);

      tblgen_attrs.push_back(::mlir::NamedAttribute(
          oldOp.getTransAAttrName(), attrAdaptor.getTransAAttr()));
      tblgen_attrs.push_back(::mlir::NamedAttribute(
          oldOp.getTransBAttrName(), attrAdaptor.getTransBAttr()));
      // return type
      auto elementType = rewriter.getF32Type();
      auto retType = mlir::RankedTensorType::get(gemm2_shape, elementType);

      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      tblgen_types.push_back(retType);

      tblgen_newOperation_1 = rewriter.create<theNewOp>(
          loc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    spade::AISLEhstack tblgen_newOperation_2;
    {
      ::llvm::SmallVector<::mlir::Value> tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      tblgen_values.push_back(tblgen_newOperation_0);
      tblgen_values.push_back(tblgen_newOperation_1);

      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : oldOp.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_newOperation_2 = rewriter.create<spade::AISLEhstack>(
          loc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_newOperation_2.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }
    assert(tblgen_repl_values.size());

    rewriter.replaceOp(op, tblgen_repl_values);

    LLVM_DEBUG({ spade::dumpUsers(op); });

    LLVM_DEBUG({ spade::dumpBlock(tblgen_newOperation_2); });

    return ::mlir::success();
  }
};

void populateLoweringONNXToAISLEGEMMOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXGEMMOpLowering>(ctx);
}

} // namespace spade
