/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MatMulAdd.cpp - Lowering MatMul and Add to RedMulE ---------===//
//
// Every op becomes one native RedMulE launch  Y = X . W + Y  on SPM buffers
// (aisle-tile has already split larger products into such launches on
// aisle.Window views):
//
//   aisle.MatMul  C[12 x 16] = A[12 x 16] . B[16 x 16]           Y <- 0
//   aisle.GEMM    Y[12 x 16] = A[12 x 16] . B (or B^T) + C       Y <- C
//     C is preloaded into the RedMulE accumulator (firmware launch_bias);
//     a C that is the previous K-tile's result is accumulated in place
//     (launch_accumulate).
//
//   aisle.Add     C[12 x 16] = A + B
//     X <- I (12x16 identity, resident), W <- pad16(one operand),
//     Y <- the other operand; one launch: Y = I . W + Y.
//     Y is the operand that can be overwritten in place (an SPM result with
//     no other user) when there is one; constants go to W (resident).
//
//   aisle.Window  of data memory: a memref.subview that the upload reading
//                 it folds into its offsets; of a tiled SPM result: a tile.
//   aisle.Concat  of SPM tiles: a tiled SPM value (no data movement).
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

#include <optional>

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

// May `op` overwrite the SPM buffer of `orig`?  Only if op is its sole user
// and it is not a Window (a Window's tile belongs to a larger result that
// may have other users).
bool mayOverwrite(Value orig, Operation *op) {
  return orig && onlyUsedBy(orig, op) && !orig.getDefiningOp<AISLEWindowOp>();
}

} // namespace

//===----------------------------------------------------------------------===//
// MatMul and Gemm: one native launch  Y = A . B (+ C)
//===----------------------------------------------------------------------===//

namespace {

// Static [.., rows, cols] shapes of A and (B or B^T) of a product.
struct ProductShape {
  int64_t m = 0, k = 0, kb = 0, n = 0;
  bool f16 = false;
};

std::optional<ProductShape> productShape(Operation *op, bool transB) {
  ProductShape s;
  int64_t rb, cb;
  if (failed(matrixShape(op->getOperand(0), s.m, s.k)) ||
      failed(matrixShape(op->getOperand(2), rb, cb)))
    return std::nullopt;
  s.kb = transB ? cb : rb;
  s.n = transB ? rb : cb;
  auto elem = [](Value v) {
    return cast<ShapedType>(v.getType()).getElementType();
  };
  s.f16 = elem(op->getOperand(0)).isF16() && elem(op->getOperand(2)).isF16();
  return s;
}

// A product aisle-tile would split (or has split): f16, [12, 16K] x [16K, 16N].
bool isRedMulEProduct(Operation *op, bool transB) {
  std::optional<ProductShape> s = productShape(op, transB);
  return s && s->f16 && s->m == kM && s->kb == s->k && s->k % kN == 0 &&
         s->n % kK == 0 && s->k > 0 && s->n > 0;
}

// X[12x16] . W[16x16] (+ C): exactly one launch.  A and C may be SPM results
// or (windows of) data memory; B comes from data memory (a constant B is
// resident).  With C this is the firmware's launch_bias, and with C the
// result of the previous K-tile (dead otherwise) it is launch_accumulate.
LogicalResult lowerNativeProduct(Operation *op, Value aValue, Value bValue,
    Value cValue, Value cOrig, bool transB, Type resultType,
    ConversionPatternRewriter &rewriter) {
  std::optional<ProductShape> s = productShape(op, transB);
  if (!s || s->m != kM || s->k != kN || s->kb != kN || s->n != kK)
    return op->emitError()
           << "not a native RedMulE launch (needs [12, 16] x [16, 16], got ["
           << (s ? s->m : -1) << ", " << (s ? s->k : -1) << "] x ["
           << (s ? s->kb : -1) << ", " << (s ? s->n : -1)
           << "]); run aisle-tile first";
  FailureOr<Operand> a = classify(op, aValue, "A", kM, kN);
  FailureOr<Operand> b = classify(op, bValue, "B", kN, kK);
  if (failed(a) || failed(b))
    return failure();
  if (!b->host)
    return op->emitError("RedMulE MatMul/Gemm: B must come from data memory");
  std::optional<Operand> c;
  if (cValue) {
    FailureOr<Operand> co = classify(op, cValue, "C", kM, kK);
    if (failed(co))
      return failure();
    c = *co;
  }

  Emitter emit(rewriter, op->getLoc(), blockName(op, "matmul"));
  Value w = transB ? emit.weightWindow(*b->host, 0, 0, kK, "W",
                         /*transpose=*/true)
                   : emit.weightWindow(*b->host, 0, 0, kN, "W");
  SPMValue x = emit.activation(*a, "X");
  // Y = C (uploaded, or an SPM value in place / copied) or 0.
  const bool cDead = mayOverwrite(cOrig, op);
  SPMValue y = emit.accumulator(c, cDead, ValueRange{});
  SmallVector<Value> deps{y.token};
  if (a->spm)
    deps.push_back(x.token);
  Value t = emit.gemm(x.address, w, y.address, deps);
  rewriter.replaceOp(op,
      makeSPMTensor(rewriter, op->getLoc(), resultType, {y.address, t, y.tile}));
  return success();
}

} // namespace

struct AISLEMatMulOpLowering : public ConversionPattern {
  AISLEMatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(AISLEMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto oldOp = cast<AISLEMatMulOp>(op);
    AISLEMatMulOpAdaptor adaptor(operands);
    return lowerNativeProduct(op, adaptor.getA(), adaptor.getB(), Value(),
        Value(), /*transB=*/false, oldOp.getY().getType(), rewriter);
  }
};

// aisle.GEMM  Y = A . B + C  (alpha = beta = 1, transA = 0) for f16 RedMulE
// shapes.  Benefit 2: tried before the legacy AISLEGEMM -> AISMEMGEMM
// pattern, which keeps handling everything else (e.g. f32).
struct AISLEGEMMOpRedMulELowering : public ConversionPattern {
  AISLEGEMMOpRedMulELowering(MLIRContext *ctx)
      : ConversionPattern(AISLEGEMMOp::getOperationName(), 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto oldOp = cast<AISLEGEMMOp>(op);
    const bool transB = oldOp.getTransB() != 0;
    if (oldOp.getTransA() != 0 || !isRedMulEProduct(op, transB))
      return rewriter.notifyMatchFailure(op, "not a RedMulE-sized f16 Gemm");
    AISLEGEMMOpAdaptor adaptor(operands);
    return lowerNativeProduct(op, adaptor.getA(), adaptor.getB(),
        adaptor.getC(), oldOp.getC(), transB, oldOp.getY().getType(),
        rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Window and Concat (aisle-tile)
//===----------------------------------------------------------------------===//

// aisle.Window of data memory -> memref.subview, which the consumer's upload
// folds into its offsets (no copy).  Of a tiled SPM result -> that tile.
struct AISLEWindowOpLowering : public ConversionPattern {
  AISLEWindowOpLowering(MLIRContext *ctx)
      : ConversionPattern(AISLEWindowOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto window = cast<AISLEWindowOp>(op);
    Value input = operands[0];
    auto outType = cast<RankedTensorType>(window.getOutput().getType());
    ArrayRef<int64_t> offsets = window.getOffsets();
    const int64_t rank = outType.getRank();

    if (std::optional<SmallVector<SPMValue>> tiles = getSPMTiles(input)) {
      // Tiles are [.., 12, 16] side by side along the last axis.
      bool whole = outType.getDimSize(rank - 1) == kK &&
                   offsets[rank - 1] % kK == 0 &&
                   outType.getDimSize(rank - 2) == kM;
      for (int64_t d = 0; whole && d + 1 < rank; ++d)
        whole = offsets[d] == 0;
      const int64_t j = offsets[rank - 1] / kK;
      if (!whole || j >= static_cast<int64_t>(tiles->size()))
        return op->emitError("a Window of an SPM result must select one of "
                             "its 12x16 tiles");
      rewriter.replaceOp(
          op, makeSPMTensor(rewriter, op->getLoc(), outType, (*tiles)[j]));
      return success();
    }

    Value memref = input;
    if (auto cast = input.getDefiningOp<UnrealizedConversionCastOp>())
      if (cast.getInputs().size() == 1)
        memref = cast.getInputs().front();
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || !memrefType.hasStaticShape())
      return op->emitError("a Window needs a static memref or an SPM result");
    SmallVector<int64_t> strides(rank, 1);
    auto view = rewriter.create<memref::SubViewOp>(op->getLoc(), memref,
        offsets, outType.getShape(), strides);
    rewriter.replaceOp(op, rewriter
                               .create<UnrealizedConversionCastOp>(
                                   op->getLoc(), TypeRange{outType},
                                   ValueRange{view.getResult()})
                               .getResult(0));
    return success();
  }
};

// aisle.Concat of SPM tiles along the last axis: the tiles stay where they
// are; one tiled SPM value.
struct AISLEConcatOpLowering : public ConversionPattern {
  AISLEConcatOpLowering(MLIRContext *ctx)
      : ConversionPattern(AISLEConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto concat = cast<AISLEConcatOp>(op);
    auto outType = cast<RankedTensorType>(concat.getOutput().getType());
    if (concat.getAxisAttr().getInt() != outType.getRank() - 1)
      return op->emitError("only a Concat of RedMulE tiles along the last "
                           "axis is supported");
    SmallVector<SPMValue> tiles;
    for (Value v : operands) {
      std::optional<SmallVector<SPMValue>> t = getSPMTiles(v);
      if (!t)
        return op->emitError("a Concat input is not a RedMulE (SPM) result");
      tiles.append(*t);
    }
    rewriter.replaceOp(
        op, makeSPMTiles(rewriter, op->getLoc(), outType, tiles));
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
        return mayOverwrite(orig, op) && origA != origB ? 0 : 3;
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
    if (yOp.spm && yOp.spm->tile == emit.tile() && mayOverwrite(yOrig, op) &&
        origA != origB) {
      y = *yOp.spm;
    } else if (yOp.host || yOp.spm->tile != emit.tile()) {
      y = emit.activation(yOp, "Y"); // uploaded or moved: a private copy
    } else {
      Value copyTo = emit.alloc(kM, "Y");
      y = {copyTo, emit.copy(yOp.spm->address, copyTo, kM, yOp.spm->token),
          emit.tile()};
    }
    deps.push_back(y.token);

    Value t = emit.gemm(identity, w, y.address, deps);
    rewriter.replaceOp(op, makeSPMTensor(rewriter, op->getLoc(),
                               oldOp.getC().getType(), {y.address, t, y.tile}));
    return success();
  }
};

void populateLoweringAISLEMatMulAddOpPatterns(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<AISLEMatMulOpLowering, AISLEGEMMOpRedMulELowering,
      AISLEAddOpLowering, AISLEWindowOpLowering, AISLEConcatOpLowering>(ctx);
}

} // namespace spade
