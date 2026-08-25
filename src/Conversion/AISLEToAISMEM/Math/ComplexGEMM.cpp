/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ComplexGEMM.cpp - Lowering ComplexGEMM Op --------===//
//
// This file lowers the AISLE split-complex GEMM operator to an explicit
// RedMulE schedule in AISMEM.
//
// The generated schedule mirrors the C programming model in
// isolde/sw/complex_gemm/complex_gemm.c:
//
//   Phase 1  (accumulate = false: establish SPM layout, zero Y)
//     RM0: Yr  = Ar * Br
//     RM1: Yi  = Ar * Bi
//     wait RM0 + RM1
//
//   Phase 2  (accumulate = true: rewrite X/W only, keep Y)
//     RM0: Yr += Ai * (-Bi)   (negateB negates Bi during upload)
//     RM1: Yi += Ai * Br
//     wait RM0 + RM1
//
//   Download
//     RM0.Yr -> Cr
//     RM1.Yi -> Ci
//
// Each tile owns a persistent Y accumulator that lives in tile-local SPM
// between the two phases. It is modelled here as an explicit memref operand
// (yReal / yImag) so that:
//   * every RedMulEGEMM carries a memory write effect and is therefore not
//     eliminated by canonicalization DCE, and
//   * the phase1 -> phase2 -> download ordering is a real memory dependency
//     rather than relying on textual position alone.
//
// The yReal/yImag buffers are provisional at this level: the later
// AISMEM->LLVM lowering realizes them as tile-local SPM (spm_write / spm_read),
// not as host allocations -- exactly like Cr/Ci are produced by a download.
//
// tile and mask are i32 to match the rv32im RISC-V core the RTL instantiates.
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "AISLEToAISMEM_ComplexGEMM"

using namespace mlir;

namespace spade {

namespace {

constexpr int32_t RMReal = 0;
constexpr int32_t RMImag = 1;
constexpr int32_t RMMask = (1 << RMReal) | (1 << RMImag);

// One asynchronous GEMM launch on `tile`, accumulating into the persistent
// tile-local accumulator `y`.
static spade::AISMEMRedMulEGEMMOp createRedMulEGEMM(
    ConversionPatternRewriter &rewriter, Location loc, Value a, Value aShape,
    Value b, Value bShape, Value y, int32_t tile, bool accumulate,
    bool negateB) {
  SmallVector<Value> operands{a, aShape, b, bShape, y};
  SmallVector<Type> resultTypes{rewriter.getNoneType()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("tile", rewriter.getI32IntegerAttr(tile)),
      rewriter.getNamedAttr("accumulate", rewriter.getBoolAttr(accumulate)),
      rewriter.getNamedAttr("negateB", rewriter.getBoolAttr(negateB))};

  return rewriter.create<spade::AISMEMRedMulEGEMMOp>(
      loc, resultTypes, operands, attributes);
}

// Barrier for the RedMulE tiles selected by `mask`.
static spade::AISMEMRedMulEWaitOp createRedMulEWait(
    ConversionPatternRewriter &rewriter, Location loc, int32_t mask) {
  SmallVector<Type> resultTypes{rewriter.getNoneType()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("mask", rewriter.getI32IntegerAttr(mask))};

  return rewriter.create<spade::AISMEMRedMulEWaitOp>(
      loc, resultTypes, ValueRange{}, attributes);
}

// Copy the persistent tile-local Y accumulator (`source`) into the
// host-visible output buffer (`destination`).
static spade::AISMEMRedMulEDownloadOp createRedMulEDownload(
    ConversionPatternRewriter &rewriter, Location loc, Value source,
    Value destination, int32_t tile) {
  SmallVector<Value> operands{source, destination};
  SmallVector<Type> resultTypes{rewriter.getNoneType()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("tile", rewriter.getI32IntegerAttr(tile))};

  return rewriter.create<spade::AISMEMRedMulEDownloadOp>(
      loc, resultTypes, operands, attributes);
}

} // namespace

struct AISLEComplexGEMMOpLowering : public ConversionPattern {
  using theOperation = spade::AISLEComplexGEMMOp;
  using theAdaptor = spade::AISLEComplexGEMMOpAdaptor;

  // Deliberately do not attach the pass TypeConverter here.
  //
  // At this point the function ABI has already been bufferized to memrefs,
  // while aisle.ComplexGEMM still has tensor operands/results surrounded by
  // unrealized_conversion_cast bridges. This pattern consumes those bridges
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

    // Host-visible result buffers. These escape through the converted function
    // return, so no deallocation is inserted here.
    auto theDim = aisle_to_aismem::inferDim(op);
    auto crAlloc =
        aisle_to_aismem::insertAlloc(rewriter, loc, theDim, crType);
    auto ciAlloc =
        aisle_to_aismem::insertAlloc(rewriter, loc, theDim, ciType);

    // Persistent per-tile Y accumulators (same shape/type as Cr/Ci). These are
    // provisional AISMEM buffers that AISMEM->LLVM realizes as tile-local SPM
    // rather than host memory, so no host dealloc is emitted here either.
    auto yReal =
        aisle_to_aismem::insertAlloc(rewriter, loc, theDim, crType);
    auto yImag =
        aisle_to_aismem::insertAlloc(rewriter, loc, theDim, ciType);

    // ---------------------------------------------------------------------
    // Phase 1
    //   RM0: Yr = Ar * Br
    //   RM1: Yi = Ar * Bi
    //
    // accumulate=false tells the lower level to establish the tile-local SPM
    // X/W/Y layout and clear Y before the asynchronous launch.
    // ---------------------------------------------------------------------
    auto phase1Real = createRedMulEGEMM(rewriter, loc, ar, arShape, br,
        brShape, yReal.getResult(), RMReal, /*accumulate=*/false,
        /*negateB=*/false);
    auto phase1Imag = createRedMulEGEMM(rewriter, loc, ar, arShape, bi,
        biShape, yImag.getResult(), RMImag, /*accumulate=*/false,
        /*negateB=*/false);
    auto phase1Wait = createRedMulEWait(rewriter, loc, RMMask);

    // ---------------------------------------------------------------------
    // Phase 2
    //   RM0: Yr += Ai * (-Bi)
    //   RM1: Yi += Ai * Br
    //
    // accumulate=true means X/W are rewritten but Y must remain untouched.
    // negateB=true on RM0 requests FP16 sign-bit negation while Bi is uploaded.
    // ---------------------------------------------------------------------
    auto phase2Real = createRedMulEGEMM(rewriter, loc, ai, aiShape, bi,
        biShape, yReal.getResult(), RMReal, /*accumulate=*/true,
        /*negateB=*/true);
    auto phase2Imag = createRedMulEGEMM(rewriter, loc, ai, aiShape, br,
        brShape, yImag.getResult(), RMImag, /*accumulate=*/true,
        /*negateB=*/false);
    auto phase2Wait = createRedMulEWait(rewriter, loc, RMMask);

    // Only after the second barrier are the persistent tile-local Y buffers
    // copied to the host-visible output memrefs.
    auto downloadReal = createRedMulEDownload(rewriter, loc, yReal.getResult(),
        crAlloc.getResult(), RMReal);
    auto downloadImag = createRedMulEDownload(rewriter, loc, yImag.getResult(),
        ciAlloc.getResult(), RMImag);

    // Keep the explicit schedule contiguous and *after every allocation*, so
    // each Y use is dominated by its def. This also keeps graph.test.aismem
    // deterministic while the AISMEM->LLVM lowering is developed.
    phase1Real->moveAfter(yImag);
    phase1Imag->moveAfter(phase1Real);
    phase1Wait->moveAfter(phase1Imag);
    phase2Real->moveAfter(phase1Wait);
    phase2Imag->moveAfter(phase2Real);
    phase2Wait->moveAfter(phase2Imag);
    downloadReal->moveAfter(phase2Wait);
    downloadImag->moveAfter(downloadReal);

    // AISLE still has tensor SSA results while AISMEM writes explicit memrefs.
    // Bridge the newly allocated result buffers back to the same tensor types
    // as the old AISLE results. This keeps replaceOp type-preserving.
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

    rewriter.replaceOp(oldOp, ValueRange{crTensor, ciTensor});

    LLVM_DEBUG({ spade::dumpBlock(downloadImag); });
    return success();
  }
};

void populateLoweringAISLEComplexGEMMOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<AISLEComplexGEMMOpLowering>(ctx);
}

} // namespace spade