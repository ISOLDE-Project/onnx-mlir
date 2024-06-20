/*AISLE
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <bits/stdint-intn.h>
#include <functional>
#include <netinet/in.h>

using namespace mlir;
namespace onnx_to_aisle {

inline void getVectorShape(Value value, llvm::SmallVector<int32_t>&Vec) {
  TensorType t_type = value.getType().cast<TensorType>();
  if (!t_type)
    return;
  auto x_shape = t_type.getShape();
  auto rank = x_shape.size();
  /*
   * check hardware constrain
   **********
   *   rank greater than 4 not supported in hardware
   **********
   */
  if (rank > 4)
    assert(false && " rank greater than 4 not supported in hardware ");
  for (size_t i = 0; i < 4 - rank; ++i)
    Vec.push_back(1);
  for (size_t i = 0; i < rank; ++i)
    Vec.push_back(x_shape[i]);
}

inline void getAttrValues(::mlir::ArrayAttr array, SmallVector<int> &Vec) {
  // array.dump();
  for (size_t i = 0; i < array.size(); ++i) {
    mlir::Attribute attr = array.getValue()[i];
    if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
      Vec.push_back(intAttr.getInt());
    }
    //  attr.dump();
  }
}

template <typename _Class, typename _Func, typename _Operation>
spade::AISLEQConstantOp create(ConversionPatternRewriter &rewriter,
    _Operation &op, const char *name, _Func mem_func) {
  auto elementType = rewriter.getI32Type();
  auto paramShape = mlir::RankedTensorType::get(
      llvm::ArrayRef<int64_t>{1, 4}, elementType);
  _Class adaptor(op);
  auto get_val = std::mem_fn(mem_func);
  Value X = get_val(adaptor);
  SmallVector<int32_t> Vec;
  onnx_to_aisle::getVectorShape(X, Vec);
  auto input_shape_dense =
      DenseElementsAttr::get(paramShape, ArrayRef<int32_t>(Vec));
  auto iShapeParam = rewriter.create<spade::AISLEQConstantOp>(
      op.getLoc(), name, input_shape_dense);
  return iShapeParam;
}

template <typename _Operation>
spade::AISLEQConstantOp create(ConversionPatternRewriter &rewriter,
    _Operation &op, const char *name, const SmallVector<int64_t> &shape) {
  auto elementType = rewriter.getI32Type();
  auto paramShape = mlir::RankedTensorType::get(
      llvm::ArrayRef<int64_t>{1, 4}, elementType);
  SmallVector<int32_t> Vec;
  //make it rank 4
  auto rank = shape.size();
  /*
   * check hardware constrain
   **********
   *   rank greater than 4 not supported in hardware
   **********
   */
  if (rank > 4)
    assert(false && " rank greater than 4 not supported in hardware ");
  for (size_t i = 0; i < 4 - rank; ++i)
    Vec.push_back(1);
  for (size_t i = 0; i < rank; ++i)
    Vec.push_back(shape[i]);
  auto input_shape_dense =
      DenseElementsAttr::get(paramShape, ArrayRef<int32_t>(Vec));
  auto iShapeParam = rewriter.create<spade::AISLEQConstantOp>(
      op.getLoc(), name, input_shape_dense);
  return iShapeParam;
}

} // namespace onnx_to_aisle
//
namespace spade {

struct TensorRawData {
  TensorRawData(mlir::Value &value) { copyFrom(value); }
  TensorRawData() {}
  std::vector<float> rawData;
  llvm::SmallVector<int64_t> shape;
  void copyFrom(mlir::Value &value) {
    ONNXConstantOp onnxConst = value.getDefiningOp<mlir::ONNXConstantOp>();
    if (!onnxConst)
      return;
    mlir::DenseElementsAttr denseAttr =
        onnxConst.getValueAttr().dyn_cast<mlir::DenseElementsAttr>();
    if (!denseAttr)
      return;
    auto tensorType = denseAttr.getType().dyn_cast<mlir::RankedTensorType>();
    if (tensorType && tensorType.getElementType().isF32()) {
      rawData.clear();
      rawData.resize(tensorType.getNumElements());
      size_t i = 0;
      for (auto value : denseAttr.getValues<mlir::APFloat>()) {
        rawData[i++] = value.convertToFloat();
      }
      auto srcShape=tensorType.getShape();
      shape.assign(srcShape.begin(),srcShape.end());

    }
  }
  //
  mlir::ONNXConstantOp createONNXConstantOp(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) {
    // Create an element type (f32 in this case)
    auto elementType = rewriter.getF32Type();

    // Create a RankedTensorType from the shape and element type
    auto tensorType = mlir::RankedTensorType::get(shape, elementType);

    // Create a DenseElementsAttr from the values and tensor type
    auto denseAttr =
        mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(rawData));

    // Create the ONNXConstantOp using the builder
    //(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute sparse_value, Attribute value)
    mlir::ONNXConstantOp constantOp = rewriter.create<mlir::ONNXConstantOp>(
        loc,  /*sparse_value=*/nullptr, denseAttr);
    assert(!constantOp.getSparseValue().has_value()); //sparse type unsupported by onnx framework
    return constantOp;
  }
  size_t rank() { return shape.size(); }
  int64_t tensor_index_to_offset(llvm::ArrayRef<int64_t> index) {
    assert(rank() == index.size());
    int64_t offset = 0;
    for (size_t i = 0; i < rank(); i++) {
      // REQUIRE(index[i] >= 0 && index[i] < shape[i]);
      offset = offset * shape[i] + index[i];
    }
    return offset;
  }
  void splitMatrix(TensorRawData &dst, std::pair<int64_t, int64_t> &row,
      std::pair<int64_t, int64_t> &col) {
    int64_t o = 0;
    int64_t offset = 0;
    int64_t row_range = row.second - row.first;
    int64_t col_range = col.second - col.first;
    assert(row_range && col_range);
    assert(rank() == 2);
    dst.rawData.clear();
    dst.rawData.resize(row_range * col_range);
    for (int64_t i = row.first; i < row.second; ++i)
      for (int64_t j = col.first; j < col.second; ++j) {
        llvm::ArrayRef<int64_t> index = {i, j};
        offset = tensor_index_to_offset(index);
        dst.rawData[o++] = rawData[offset];
      }
    dst.shape = llvm::SmallVector<int64_t>{row_range, col_range};
  }

  void splitVector(TensorRawData &dst, std::pair<int64_t, int64_t> &idx) {
    int64_t o = 0;
    int64_t idx_range = idx.second - idx.first;
    assert(idx_range);
    assert(rank() == 1);
    dst.rawData.clear();
    dst.rawData.resize(idx_range);
    for (int64_t offset = idx.first; offset < idx.second; ++offset)
      dst.rawData[o++] = rawData[offset];
    dst.shape[0]= idx_range;
  }
};
} // namespace spade