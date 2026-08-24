/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ComplexGEMM.cpp - Lowering ComplexGEMM Op --------===//
//
// This file lowers the AISLE split-complex GEMM operator to AISMEM.
//
// Milestone A intentionally performs only tensor-to-memref conversion:
//   * preserve one ComplexGEMM operation,
//   * allocate explicit Cr/Ci output buffers,
//   * do not yet expose RedMulE tiles, SPM placement, or phase scheduling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/AISLEToAISMEM/helper.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Support/SpadeSupport.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "AISLEToAISMEM_ComplexGEMM"

using namespace mlir;

namespace spade {

struct AISLEComplexGEMMOpLowering : public ConversionPattern {
  using theOperation = spade::AISLEComplexGEMMOp;
  using theAdaptor = spade::AISLEComplexGEMMOpAdaptor;
  using theNewOp = spade::AISMEMComplexGEMMOp;

  // Deliberately do not attach the pass TypeConverter here.
  //
  // At this point the function ABI has already been bufferized to memrefs,
  // while aisle.ComplexGEMM still has tensor operands/results surrounded by
  // unrealized_conversion_cast bridges.  This pattern consumes those bridges
  // explicitly, exactly where the tensor/memref boundary is visible.
  AISLEComplexGEMMOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto oldOp = llvm::dyn_cast<theOperation>(op);
    if (!oldOp)
      return failure();

    theAdaptor operandAdaptor(operands);

    Value ar = operandAdaptor.getAr();
    Value arShape = operandAdaptor.getArShape();
    Value ai = operandAdaptor.getAi();
    Value aiShape = operandAdaptor.getAiShape();
    Value br = operandAdaptor.getBr();
    Value brShape = operandAdaptor.getBrShape();
    Value bi = operandAdaptor.getBi();
    Value biShape = operandAdaptor.getBiShape();

    LLVM_DEBUG({ spade::dumpBlock(op); });

    // The surrounding bufferization has produced:
    //
    //   memref -> unrealized_conversion_cast -> tensor -> aisle.ComplexGEMM
    //
    // AISMEM consumes memrefs, so unwrap the input-side bridge casts.
    aisle_to_aismem::getConversionCastOperand(ar);
    aisle_to_aismem::getConversionCastOperand(arShape);
    aisle_to_aismem::getConversionCastOperand(ai);
    aisle_to_aismem::getConversionCastOperand(aiShape);
    aisle_to_aismem::getConversionCastOperand(br);
    aisle_to_aismem::getConversionCastOperand(brShape);
    aisle_to_aismem::getConversionCastOperand(bi);
    aisle_to_aismem::getConversionCastOperand(biShape);

    MemRefType crType =
        aisle_to_aismem::convertTensorToMemRef(oldOp.getCr().getType());
    MemRefType ciType =
        aisle_to_aismem::convertTensorToMemRef(oldOp.getCi().getType());

    // Allocate explicit AISMEM result buffers.
    auto theDim = aisle_to_aismem::inferDim(op);
    auto crAlloc =
        aisle_to_aismem::insertAlloc(rewriter, loc, theDim, crType);
    auto ciAlloc =
        aisle_to_aismem::insertAlloc(rewriter, loc, theDim, ciType);

    // Cr/Ci escape through the converted function return, so do not insert
    // deallocations here.
    SmallVector<Value> newOperands{ar, arShape, ai, aiShape, br, brShape,
        bi, biShape, crAlloc.getResult(), ciAlloc.getResult()};
    SmallVector<Type> resultTypes{rewriter.getNoneType()};
    SmallVector<NamedAttribute> attributes;

    auto newOp = rewriter.create<theNewOp>(
        loc, resultTypes, newOperands, attributes);
    newOp->moveAfter(ciAlloc);

    // AISLE still has tensor SSA results while AISMEM writes explicit memrefs.
    // Bridge the newly allocated result buffers back to the *same tensor types*
    // as the old AISLE results.  This is deliberately explicit: it keeps
    // replaceOp type-preserving, so DialectConversion does not need to invent a
    // tensor<->memref materialization while legalizing the illegal AISLE op.
    Value crTensor =
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, oldOp.getCr().getType(), ValueRange{crAlloc.getResult()})
            .getResult(0);
    Value ciTensor =
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, oldOp.getCi().getType(), ValueRange{ciAlloc.getResult()})
            .getResult(0);

    // Existing users (currently tensor->memref ABI bridge casts) are rewritten
    // to consume these tensor values when the pattern commits.  The inverse
    // memref->tensor->memref cast pairs can be reconciled by the normal
    // unrealized-cast cleanup later in the pipeline.
    rewriter.replaceOp(oldOp, ValueRange{crTensor, ciTensor});

    LLVM_DEBUG({ spade::dumpBlock(newOp); });
    return success();
  }
};

void populateLoweringAISLEComplexGEMMOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<AISLEComplexGEMMOpLowering>(ctx);
}

} // namespace spade