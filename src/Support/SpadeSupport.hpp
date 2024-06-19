#pragma once
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>
#include <set>

namespace spade{

template<typename OperantionType>
struct ObsoleteOperations : public ::std::set<mlir::Operation *>{
void eraseOps(mlir::ConversionPatternRewriter& rewriter){
    for (auto o : *this) {
      OperantionType castOp = llvm::dyn_cast<OperantionType>(o);
      assert(castOp);
      rewriter.eraseOp(castOp);
    }
}
};
template<typename OperantionType>
inline void getOpOperand(
    mlir::Value& op, ObsoleteOperations<OperantionType>& obsoleteOps) {
  mlir::Operation *prevOp = op.getDefiningOp();
  auto cast_0 =
      llvm::dyn_cast<OperantionType>(prevOp);
  if (cast_0) {
    obsoleteOps.insert(cast_0);
    // prevOp->dump();
    // prev=prevOp->getOperand(0);
    op = *cast_0.getInputs().begin();
  }
}

inline void dumpBlock(mlir::Operation *op){
    using Block = mlir::Block;
    Block *block = op->getBlock();
    for (Block::iterator it = block->begin(); it != block->end(); ++it) {
      it->dump();
    }
}
template<typename OperantionType>
inline void dumpUsers(OperantionType op){
    ::llvm::outs()<<">>Users of\n";
    op->dump();
    ::llvm::outs()<<"\n";
    for (auto v : op->getUsers()) {
      v->dump();
    }
    ::llvm::outs()<<"<<End of users list\n";
}


}//namespace spade