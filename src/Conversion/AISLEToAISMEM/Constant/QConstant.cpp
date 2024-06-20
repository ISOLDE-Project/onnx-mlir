/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ QConstant.cpp - Lowering QConstant Op ------------------===//
//
// Copyleft
//
// =============================================================================
//
// This file lowers the AISLE QConstant Operators to AISMEM dialect.
//
//===----------------------------------------------------------------------===//

// #include "src/Compiler/CompilerOptions.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "set"
#include "src/Dialect/AISLE/AISLEDialect.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace spade {

template <typename Oty>
Operation *getUser_0(Oty &Op) {
  auto results = Op.getODSResults(0);
  auto begin = results.user_begin();
  auto user_0 = *begin;
  return user_0;
}

struct AISLEQConstantOpLowering : public ConversionPattern {
  AISLEQConstantOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            spade::AISLEQConstantOp::getOperationName(), 1, ctx) {
    // ctx->getOrLoadDialect<spade::AISLEDialect>();
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    using AISLEQConstantOp = spade::AISLEQConstantOp;

    Location loc = op->getLoc();
    AISLEQConstantOp qConst = llvm::dyn_cast<AISLEQConstantOp>(op);

    //**
    auto newQConst = rewriter.create<spade::AISMEMQConstantOp>(loc, qConst);

    // lower down convOp
    rewriter.replaceOp(qConst, static_cast<mlir::Value>(newQConst));
    // qConst->replaceAllUsesWith(newQConst);

    return ::mlir::success();
  };
};
void populateLoweringAISLEQConstantOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<AISLEQConstantOpLowering>(typeConverter, ctx);
}
} // namespace spade
