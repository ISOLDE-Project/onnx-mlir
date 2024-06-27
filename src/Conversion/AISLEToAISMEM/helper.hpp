/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <set>

using namespace mlir;
namespace aisle_to_aismem {

inline MemRefType convertTensorToMemRef(Type operand) {
  TensorType type = operand.cast<TensorType>();
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

inline void getConversionCastOperand(
    Value &op, ::std::set<Operation *> &obsoleteOps) {
  Operation *prevOp = op.getDefiningOp();
  UnrealizedConversionCastOp cast_0 =
      llvm::dyn_cast<UnrealizedConversionCastOp>(prevOp);
  if (cast_0) {
    obsoleteOps.insert(cast_0);
    // prevOp->dump();
    // prev=prevOp->getOperand(0);
    op = *cast_0.getInputs().begin();
  }
}

inline void getConversionCastOperand(mlir::Value &val) {
  Operation *Op = val.getDefiningOp();
  UnrealizedConversionCastOp cast_0 =
      llvm::dyn_cast<UnrealizedConversionCastOp>(Op);
  if (cast_0) {
    val = *cast_0.getInputs().begin();
  }
}

template <typename castop>
inline mlir::Value insertConversionCast(
    ConversionPatternRewriter &rewriter, Location loc, mlir::Value X) {
  ::mlir::Value x_cast;
  if (X.getType().isa<TensorType>())
    x_cast =
        *rewriter.create<castop>(loc, convertTensorToMemRef(X.getType()), X)
             .getODSResults(0)
             .begin();
  else
    x_cast = X;
}
template <typename operation>
inline bool shallInsertDealoc(operation &oldOp) {
  bool insert_dealloc = true;
  auto results = oldOp.getODSResults(0);
  auto convOpResult = *results.begin();
  Type convOpResType = convOpResult.getType();

  auto begin = results.user_begin();
  auto user_0 = *begin;

  // auto theType = convertTensorToMemRef(convOpResType);

  for (auto v : user_0->getUsers()) {
    func::ReturnOp retOp = llvm::dyn_cast<mlir::func::ReturnOp>(v);
    if (retOp) {
      insert_dealloc = false;
      break;
    }
  }
  return insert_dealloc;
}

inline memref::DimOp inferDim(Operation *op) {
  /*
   * in case of dynamic tensors, use the DimOp of the arg%0
   */
  memref::DimOp result;
  {
    Block *block = op->getBlock();
    for (Block::iterator it = block->begin(); it != block->end(); ++it) {
      result = llvm::dyn_cast<memref::DimOp>(*it);
      if (result) {
        break;
      }
    }
  }
  return result;
}
inline memref::AllocOp insertAlloc(ConversionPatternRewriter &rewriter,
    const Location &loc, const memref::DimOp &theDim,
    const MemRefType &theType) {
  memref::AllocOp newAlloc;
  IntegerAttr alignmentAttr = rewriter.getI64IntegerAttr(16);
  if (theDim) {
    memref::DimOp dim = theDim;
    newAlloc = rewriter.create<memref::AllocOp>(loc, theType, ValueRange{dim});
    DenseI32ArrayAttr segment =
        rewriter.getDenseI32ArrayAttr(ArrayRef<int32_t>{1, 0});
    newAlloc->setAttr(::llvm::StringRef("operand_segment_sizes"), segment);
  } else {
    newAlloc = rewriter.create<memref::AllocOp>(loc, theType);
  }
  newAlloc.setAlignmentAttr(alignmentAttr);
  return newAlloc;
}

inline void eraseOp(
    ConversionPatternRewriter &rewriter, std::set<Operation *> &obsoleteOps) {
  for (auto o : obsoleteOps) {
    UnrealizedConversionCastOp castOp =
        llvm::dyn_cast<UnrealizedConversionCastOp>(o);
    rewriter.eraseOp(castOp);
  }
}

} // namespace aisle_to_aismem