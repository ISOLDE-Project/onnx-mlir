/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MatMulAdd.cpp - Lowering MatMul and Add to RedMulE ---------===//
//
// Both ops become native RedMulE launches  Y = X . W + Y  on SPM buffers:
//
//   aisle.MatMul  C[12 x 16] = A[12 x 16K] . B[16K x 16]
//     Y <- 0; for k < K:  X <- A[:, 16k:16k+16], W <- B[16k:16k+16, :],
//     launch (Y accumulates).  K = 2 for the radar input projection.
//
//   aisle.Add     C[12 x 16] = A + B
//     X <- I (12x16 identity, resident), W <- pad16(one operand),
//     Y <- the other operand; one launch: Y = I . W + Y.
//     Y is the operand that can be overwritten in place (an SPM result with
//     no other user) when there is one; constants go to W (resident).
//
// Results stay in SPM (see SPMValue.hpp); operands from data memory are
// uploaded, constant weights are resident (aismem-spm-allocate).
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/AISLEToAISMEM/Math/SPMEmitter.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "AISLEToAISMEM_MatMulAdd"

using namespace mlir;

namespace spade {

using namespace spade::spm;

namespace {

// Static [.., rows, cols] shape of a ranked tensor with leading 1s, or
// failure.
LogicalResult matrixShape(Value value, int64_t &rows, int64_t &cols) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 2)
    return failure();
  for (int64_t i = 0; i + 2 < type.getRank(); ++i)
    if (type.getDimSize(i) != 1)
      return failure();
  rows = type.getShape()[type.getRank() - 2];
  cols = type.getShape().back();
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// MatMul
//===----------------------------------------------------------------------===//

struct AISLEMatMulOpLowering : public ConversionPattern {
  AISLEMatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(AISLEMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto oldOp = cast<AISLEMatMulOp>(op);
    AISLEMatMulOpAdaptor adaptor(operands);

    int64_t m = 0, k = 0, kb = 0, n = 0;
    if (failed(matrixShape(oldOp.getA(), m, k)) ||
        failed(matrixShape(oldOp.getB(), kb, n)) || m != kM || n != kK ||
        kb != k || k % kN != 0 || k == 0)
      return op->emitError("RedMulE MatMul supports [12, 16K] x [16K, 16] "
                           "(K >= 1) only");
    FailureOr<Operand> a = classify(op, adaptor.getA(), "A", kM, k);
    FailureOr<Operand> b = classify(op, adaptor.getB(), "B", k, kK);
    if (failed(a) || failed(b))
      return failure();
    if (!a->host || !b->host)
      return op->emitError(
          "RedMulE MatMul: both operands must come from data memory (a "
          "16K-wide SPM operand is not supported yet)");

    Emitter emit(rewriter, op->getLoc(), blockName(op, "matmul"));
    const int64_t tiles = k / kN;
    SmallVector<Value> w, x;
    for (int64_t t = 0; t < tiles; ++t)
      w.push_back(emit.weightWindow(
          *b->host, t * kN, 0, kN, "W[" + Twine(t * kN) + ",:]"));
    for (int64_t t = 0; t < tiles; ++t)
      x.push_back(emit.upload(*a->host, 0, t * kN, kM, kN, kM,
                          "X[:," + Twine(t * kN) + "]", /*resident=*/false,
                          ValueRange{})
                      .address);

    // Y = 0, then one launch per K-tile, accumulating in Y.
    Value y = emit.alloc(kM, "Y");
    Value t = emit.zero(y, kM, ValueRange{});
    for (int64_t i = 0; i < tiles; ++i)
      t = emit.gemm(x[i], w[i], y, ValueRange{t});

    rewriter.replaceOp(op, makeSPMTensor(rewriter, op->getLoc(),
                               oldOp.getY().getType(), {y, t, kTile}));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Add
//===----------------------------------------------------------------------===//

struct AISLEAddOpLowering : public ConversionPattern {
  AISLEAddOpLowering(MLIRContext *ctx)
      : ConversionPattern(AISLEAddOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto oldOp = cast<AISLEAddOp>(op);
    AISLEAddOpAdaptor adaptor(operands);

    FailureOr<Operand> a = classify(op, adaptor.getA(), "A", kM, kK);
    FailureOr<Operand> b = classify(op, adaptor.getB(), "B", kM, kK);
    if (failed(a) || failed(b))
      return failure();

    // Which operand becomes Y (accumulator) and which W (padded operand).
    // Preference for Y: an SPM value we may overwrite, then a non-constant
    // host tensor (it is uploaded anyway), then anything else.
    Value origA = oldOp.getA(), origB = oldOp.getB();
    auto rank = [&](const Operand &o, Value orig) {
      if (o.spm)
        return onlyUsedBy(orig, op) && origA != origB ? 0 : 3;
      return isConstant(o.host->memref) ? 2 : 1;
    };
    bool aIsY = rank(*a, origA) <= rank(*b, origB);
    const Operand &yOp = aIsY ? *a : *b;
    const Operand &wOp = aIsY ? *b : *a;
    Value yOrig = aIsY ? origA : origB;

    Emitter emit(rewriter, op->getLoc(), blockName(op, "add"));
    Value identity = emit.identity();

    // W: the other operand, zero padded to 16 rows.
    Value w;
    SmallVector<Value> deps;
    if (wOp.host) {
      w = emit.weightWindow(*wOp.host, 0, 0, kM, "W");
    } else {
      SPMValue pw = emit.padded(*wOp.spm, "W|0", ValueRange{});
      w = pw.address;
      deps.push_back(pw.token);
    }

    // Y: in place when allowed, else a private copy.
    SPMValue y;
    if (yOp.spm && onlyUsedBy(yOrig, op) && origA != origB) {
      y = *yOp.spm;
    } else if (yOp.host) {
      y = emit.activation(yOp, "Y");
    } else {
      Value copyTo = emit.alloc(kM, "Y");
      y = {copyTo, emit.copy(yOp.spm->address, copyTo, kM, yOp.spm->token),
          kTile};
    }
    deps.push_back(y.token);

    Value t = emit.gemm(identity, w, y.address, deps);
    rewriter.replaceOp(op, makeSPMTensor(rewriter, op->getLoc(),
                               oldOp.getC().getType(), {y.address, t, kTile}));
    return success();
  }
};

void populateLoweringAISLEMatMulAddOpPatterns(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<AISLEMatMulOpLowering, AISLEAddOpLowering>(ctx);
}

} // namespace spade
