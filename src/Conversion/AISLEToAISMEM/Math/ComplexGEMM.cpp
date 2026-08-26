/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ComplexGEMM.cpp - Lowering ComplexGEMM Op --------===//
//
// Lower split-complex GEMM to an explicit RedMulE/SPM schedule in AISMEM.
//
// The lowering mirrors the C programming model:
//
//   Phase 1
//     RM0: upload Ar -> X, Br -> W, zero Y, GEMM
//     RM1: upload Ar -> X, Bi -> W, zero Y, GEMM
//     wait RM0 + RM1
//
//   Phase 2
//     RM0: overwrite X with Ai, overwrite W with -Bi, GEMM (Y preserved)
//     RM1: overwrite X with Ai, overwrite W with  Br, GEMM (Y preserved)
//     wait RM0 + RM1
//
//   Download
//     RM0.Y -> Cr
//     RM1.Y -> Ci
//
// SPM addresses are explicit SSA values.  spm_write returns the next SPM
// address, so the phase-1 address chain naturally establishes:
//
//   x_addr = get_addr_start(0)
//   w_addr = upload(X, x_addr).next
//   y_addr = upload(W, w_addr).next
//
// Phase 2 reuses x_addr and w_addr, while y_addr is left untouched.
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
#include <limits>

#define DEBUG_TYPE "AISLEToAISMEM_ComplexGEMM"

using namespace mlir;

namespace spade {

namespace {

constexpr int32_t RMReal = 0;
constexpr int32_t RMImag = 1;
constexpr int32_t RMMask = (1 << RMReal) | (1 << RMImag);
constexpr int32_t SPMBank = 0;

static spade::AISMEMRedMulEAddrStartOp createRedMulEAddrStart(
    ConversionPatternRewriter &rewriter, Location loc, int32_t tile) {
  SmallVector<Type> resultTypes{rewriter.getI32Type()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("tile", rewriter.getI32IntegerAttr(tile)),
      rewriter.getNamedAttr("bank", rewriter.getI32IntegerAttr(SPMBank))};

  return rewriter.create<spade::AISMEMRedMulEAddrStartOp>(
      loc, resultTypes, ValueRange{}, attributes);
}

// Copy one host memref into a selected RedMulE private SPM address.
//
// The returned i32 is the next address returned by spm_write.  negate=true
// means that FP16 sign bits are flipped while copying (used only for -Bi).
// dependencies are scheduling tokens and do not represent data payloads.
static spade::AISMEMRedMulEUploadOp createRedMulEUpload(
    ConversionPatternRewriter &rewriter, Location loc, Value source,
    Value sourceShape, Value spmAddress, ValueRange dependencies, int32_t tile,
    bool negate) {
  SmallVector<Value> operands{source, sourceShape, spmAddress};
  operands.append(dependencies.begin(), dependencies.end());

  SmallVector<Type> resultTypes{rewriter.getI32Type(), rewriter.getNoneType()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("tile", rewriter.getI32IntegerAttr(tile)),
      rewriter.getNamedAttr("negate", rewriter.getBoolAttr(negate))};

  return rewriter.create<spade::AISMEMRedMulEUploadOp>(
      loc, resultTypes, operands, attributes);
}

// Zero the output Y region in private SPM before phase 1.
static spade::AISMEMRedMulEZeroOp createRedMulEZero(
    ConversionPatternRewriter &rewriter, Location loc, Value spmAddress,
    ValueRange dependencies, int32_t tile, int32_t elements) {
  SmallVector<Value> operands{spmAddress};
  operands.append(dependencies.begin(), dependencies.end());

  SmallVector<Type> resultTypes{rewriter.getNoneType()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("tile", rewriter.getI32IntegerAttr(tile)),
      rewriter.getNamedAttr("elements", rewriter.getI32IntegerAttr(elements))};

  return rewriter.create<spade::AISMEMRedMulEZeroOp>(
      loc, resultTypes, operands, attributes);
}

// Launch redmule.gemm using already-populated tile-private X/W/Y addresses.
static spade::AISMEMRedMulEGEMMOp createRedMulEGEMM(
    ConversionPatternRewriter &rewriter, Location loc, Value xAddress,
    Value wAddress, Value yAddress, ValueRange dependencies, int32_t tile,
    int32_t k, int32_t m, int32_t n) {
  SmallVector<Value> operands{xAddress, wAddress, yAddress};
  operands.append(dependencies.begin(), dependencies.end());

  SmallVector<Type> resultTypes{rewriter.getNoneType()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("tile", rewriter.getI32IntegerAttr(tile)),
      rewriter.getNamedAttr("k", rewriter.getI32IntegerAttr(k)),
      rewriter.getNamedAttr("m", rewriter.getI32IntegerAttr(m)),
      rewriter.getNamedAttr("n", rewriter.getI32IntegerAttr(n))};

  return rewriter.create<spade::AISMEMRedMulEGEMMOp>(
      loc, resultTypes, operands, attributes);
}

static spade::AISMEMRedMulEWaitOp createRedMulEWait(
    ConversionPatternRewriter &rewriter, Location loc, ValueRange dependencies,
    int32_t mask) {
  SmallVector<Value> operands(dependencies.begin(), dependencies.end());
  SmallVector<Type> resultTypes{rewriter.getNoneType()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("mask", rewriter.getI32IntegerAttr(mask))};

  return rewriter.create<spade::AISMEMRedMulEWaitOp>(
      loc, resultTypes, operands, attributes);
}

// Copy the persistent tile-private Y region back to a host-visible memref.
static spade::AISMEMRedMulEDownloadOp createRedMulEDownload(
    ConversionPatternRewriter &rewriter, Location loc, Value spmAddress,
    Value destination, Value dependency, int32_t tile, int32_t elements) {
  SmallVector<Value> operands{spmAddress, destination, dependency};
  SmallVector<Type> resultTypes{rewriter.getNoneType()};
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("tile", rewriter.getI32IntegerAttr(tile)),
      rewriter.getNamedAttr("elements", rewriter.getI32IntegerAttr(elements))};

  return rewriter.create<spade::AISMEMRedMulEDownloadOp>(
      loc, resultTypes, operands, attributes);
}

static LogicalResult getStaticGemmDimensions(Operation *op, Value ar, Value ai,
    Value br, Value bi, int32_t &k, int32_t &m, int32_t &n,
    int32_t &yElements) {
  auto arType = dyn_cast<MemRefType>(ar.getType());
  auto aiType = dyn_cast<MemRefType>(ai.getType());
  auto brType = dyn_cast<MemRefType>(br.getType());
  auto biType = dyn_cast<MemRefType>(bi.getType());

  if (!arType || !aiType || !brType || !biType || arType.getRank() != 2 ||
      aiType.getRank() != 2 || brType.getRank() != 2 || biType.getRank() != 2)
    return op->emitError("RedMulE ComplexGEMM currently requires rank-2 memrefs");

  if (!arType.hasStaticShape() || !aiType.hasStaticShape() ||
      !brType.hasStaticShape() || !biType.hasStaticShape())
    return op->emitError(
        "RedMulE ComplexGEMM currently requires static matrix dimensions");

  const int64_t m64 = arType.getDimSize(0);
  const int64_t n64 = arType.getDimSize(1);
  const int64_t k64 = brType.getDimSize(1);

  if (aiType.getDimSize(0) != m64 || aiType.getDimSize(1) != n64 ||
      brType.getDimSize(0) != n64 || biType.getDimSize(0) != n64 ||
      biType.getDimSize(1) != k64)
    return op->emitError("inconsistent split-complex GEMM matrix dimensions");

  const int64_t maxI32 = std::numeric_limits<int32_t>::max();
  const int64_t yElems64 = m64 * k64;
  if (m64 > maxI32 || n64 > maxI32 || k64 > maxI32 || yElems64 > maxI32)
    return op->emitError("RedMulE GEMM dimensions do not fit in i32");

  // Match the redmule_gemm_async programming model exactly:
  //   redmule_gemm_async(tile, x, w, y, K, M, N)
  // For A[M x N] * B[N x K], this is K=k64, M=m64, N=n64.
  k = static_cast<int32_t>(k64);
  m = static_cast<int32_t>(m64);
  n = static_cast<int32_t>(n64);
  yElements = static_cast<int32_t>(yElems64);
  return success();
}

} // namespace

struct AISLEComplexGEMMOpLowering : public ConversionPattern {
  using theOperation = spade::AISLEComplexGEMMOp;
  using theAdaptor = spade::AISLEComplexGEMMOpAdaptor;

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

    // Unwrap the tensor -> memref bridge casts produced by the surrounding
    // bufferization pipeline.
    aisle_to_aismem::getConversionCastOperand(ar);
    aisle_to_aismem::getConversionCastOperand(arShape);
    aisle_to_aismem::getConversionCastOperand(ai);
    aisle_to_aismem::getConversionCastOperand(aiShape);
    aisle_to_aismem::getConversionCastOperand(br);
    aisle_to_aismem::getConversionCastOperand(brShape);
    aisle_to_aismem::getConversionCastOperand(bi);
    aisle_to_aismem::getConversionCastOperand(biShape);

    int32_t k = 0, m = 0, n = 0, yElements = 0;
    if (failed(getStaticGemmDimensions(
            op, ar, ai, br, bi, k, m, n, yElements)))
      return failure();

    MemRefType crType =
        aisle_to_aismem::convertTensorToMemRef(oldOp.getCr().getType());
    MemRefType ciType =
        aisle_to_aismem::convertTensorToMemRef(oldOp.getCi().getType());

    auto theDim = aisle_to_aismem::inferDim(op);
    auto crAlloc =
        aisle_to_aismem::insertAlloc(rewriter, loc, theDim, crType);
    auto ciAlloc =
        aisle_to_aismem::insertAlloc(rewriter, loc, theDim, ciType);

    // ---------------------------------------------------------------------
    // Establish each tile's private SPM base address.
    // ---------------------------------------------------------------------
    auto rm0Base = createRedMulEAddrStart(rewriter, loc, RMReal);
    auto rm1Base = createRedMulEAddrStart(rewriter, loc, RMImag);

    // ---------------------------------------------------------------------
    // Phase 1 / RM0
    //   X <- Ar
    //   W <- Br
    //   Y <- 0
    //   Y  = Ar * Br
    // ---------------------------------------------------------------------
    auto rm0P1X = createRedMulEUpload(rewriter, loc, ar, arShape,
        rm0Base.getAddress(), ValueRange{}, RMReal, /*negate=*/false);

    // rm0P1X.next_address is the W address.
    auto rm0P1W = createRedMulEUpload(rewriter, loc, br, brShape,
        rm0P1X.getNextAddress(), ValueRange{rm0P1X.getNoneVal()}, RMReal,
        /*negate=*/false);

    // rm0P1W.next_address is the persistent Y address.
    auto rm0P1Zero = createRedMulEZero(rewriter, loc,
        rm0P1W.getNextAddress(), ValueRange{rm0P1W.getNoneVal()}, RMReal,
        yElements);

    auto rm0P1Gemm = createRedMulEGEMM(rewriter, loc, rm0Base.getAddress(),
        rm0P1X.getNextAddress(), rm0P1W.getNextAddress(),
        ValueRange{rm0P1Zero.getNoneVal()}, RMReal, k, m, n);

    // ---------------------------------------------------------------------
    // Phase 1 / RM1
    //   X <- Ar
    //   W <- Bi
    //   Y <- 0
    //   Y  = Ar * Bi
    // ---------------------------------------------------------------------
    auto rm1P1X = createRedMulEUpload(rewriter, loc, ar, arShape,
        rm1Base.getAddress(), ValueRange{}, RMImag, /*negate=*/false);

    auto rm1P1W = createRedMulEUpload(rewriter, loc, bi, biShape,
        rm1P1X.getNextAddress(), ValueRange{rm1P1X.getNoneVal()}, RMImag,
        /*negate=*/false);

    auto rm1P1Zero = createRedMulEZero(rewriter, loc,
        rm1P1W.getNextAddress(), ValueRange{rm1P1W.getNoneVal()}, RMImag,
        yElements);

    auto rm1P1Gemm = createRedMulEGEMM(rewriter, loc, rm1Base.getAddress(),
        rm1P1X.getNextAddress(), rm1P1W.getNextAddress(),
        ValueRange{rm1P1Zero.getNoneVal()}, RMImag, k, m, n);

    auto phase1Wait = createRedMulEWait(rewriter, loc,
        ValueRange{rm0P1Gemm.getNoneVal(), rm1P1Gemm.getNoneVal()}, RMMask);

    // ---------------------------------------------------------------------
    // Phase 2 / RM0
    //   overwrite X at its original address with Ai
    //   overwrite W at its original address with -Bi
    //   DO NOT zero Y
    //   Y += Ai * (-Bi)
    // ---------------------------------------------------------------------
    auto rm0P2X = createRedMulEUpload(rewriter, loc, ai, aiShape,
        rm0Base.getAddress(), ValueRange{phase1Wait.getNoneVal()}, RMReal,
        /*negate=*/false);

    auto rm0P2W = createRedMulEUpload(rewriter, loc, bi, biShape,
        rm0P1X.getNextAddress(), ValueRange{rm0P2X.getNoneVal()}, RMReal,
        /*negate=*/true);

    auto rm0P2Gemm = createRedMulEGEMM(rewriter, loc, rm0Base.getAddress(),
        rm0P1X.getNextAddress(), rm0P1W.getNextAddress(),
        ValueRange{rm0P2W.getNoneVal()}, RMReal, k, m, n);

    // ---------------------------------------------------------------------
    // Phase 2 / RM1
    //   overwrite X at its original address with Ai
    //   overwrite W at its original address with Br
    //   DO NOT zero Y
    //   Y += Ai * Br
    // ---------------------------------------------------------------------
    auto rm1P2X = createRedMulEUpload(rewriter, loc, ai, aiShape,
        rm1Base.getAddress(), ValueRange{phase1Wait.getNoneVal()}, RMImag,
        /*negate=*/false);

    auto rm1P2W = createRedMulEUpload(rewriter, loc, br, brShape,
        rm1P1X.getNextAddress(), ValueRange{rm1P2X.getNoneVal()}, RMImag,
        /*negate=*/false);

    auto rm1P2Gemm = createRedMulEGEMM(rewriter, loc, rm1Base.getAddress(),
        rm1P1X.getNextAddress(), rm1P1W.getNextAddress(),
        ValueRange{rm1P2W.getNoneVal()}, RMImag, k, m, n);

    auto phase2Wait = createRedMulEWait(rewriter, loc,
        ValueRange{rm0P2Gemm.getNoneVal(), rm1P2Gemm.getNoneVal()}, RMMask);

    // ---------------------------------------------------------------------
    // Download persistent Y from each private SPM into the ordinary output
    // memrefs.  These correspond to RM0.Y -> Cr and RM1.Y -> Ci.
    // ---------------------------------------------------------------------
    auto downloadReal = createRedMulEDownload(rewriter, loc,
        rm0P1W.getNextAddress(), crAlloc.getResult(), phase2Wait.getNoneVal(),
        RMReal, yElements);

    auto downloadImag = createRedMulEDownload(rewriter, loc,
        rm1P1W.getNextAddress(), ciAlloc.getResult(), phase2Wait.getNoneVal(),
        RMImag, yElements);

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

    (void)downloadReal;
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