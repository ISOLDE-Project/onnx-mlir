/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ Transformer.cpp - Lowering ISOLDE transformer blocks ------===//
//
// Lower aisle.MultiHeadAttention and aisle.PositionwiseFeedForward to an
// explicit RedMulE/SPM schedule in AISMEM.  Every matrix product is one native
// RedMulE launch
//
//     Y[12x16] (+)= X[12x16] . W[16x16]        (K=16, M=12, N=16)
//
// and every host-side data transform (K^T, zero padding, column/row tiles,
// ReLU) happens on the copy into SPM (aismem.RedMulEUploadTile), on raw FP16
// bits.  The schedules are the ones of ibex/isolde/sw/radar_attention
// (tformer.py forward_fp16 / tformer_runtime.c):
//
//   MultiHeadAttention (self-attention, one head, L = 12, d = 16)
//     phase 1  tile t in {0,1,2}: X <- Xq|Xkv, W <- Wq|Wk|Wv, Y <- 0, GEMM
//              wait {0,1,2}; download Q, K, V
//     phase 2  tile 0: X <- Q, W <- pad16(K^T), Y <- 0, GEMM; wait; -> S
//     phase 3  tile 0: X <- ReLU(S), W <- pad16(V), Y <- 0, GEMM; wait; -> O
//     phase 4  tile 0: X <- O, W <- Wo, Y <- C | 0, GEMM; wait; -> result
//
//   PositionwiseFeedForward (d = 16, d_ff = 16 * T)
//     up       chunk j on tile j % 3: X <- X, W <- W1[:, 16j:16j+16], Y <- 0,
//              GEMM; wait per wave of <= 3 tiles; download U_j
//     down     tile 0: Y <- C | 0; for j: X <- ReLU(U_j),
//              W <- W2[16j:16j+16, :], GEMM (Y accumulates); wait
//              download -> result
//
// Scale factors are expected to be folded into the weights already
// (ONNXToAISLE does it for constant weights); anything else is diagnosed.
//
//===----------------------------------------------------------------------===//

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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "AISLEToAISMEM_Transformer"

using namespace mlir;

namespace spade {

namespace {

// One native RedMulE launch: Y[M x K] (+)= X[M x N] . W[N x K].
constexpr int32_t kM = 12; // rows of X and Y (ARRAY_HEIGHT * PIPE_REGS)
constexpr int32_t kN = 16; // reduction length, rows of W
constexpr int32_t kK = 16; // columns of W and Y (one SPM row)
constexpr int32_t kTileCount = 3; // RedMulE instances (platform demo_3)
constexpr int32_t kSPMBank = 0;

int32_t tileBit(int32_t tile) { return 1 << tile; }

// Peel the tensor<->memref bridge cast introduced by the Krnl lowering.
Value unwrapCast(Value value) {
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>())
    if (cast.getInputs().size() == 1)
      return cast.getInputs().front();
  return value;
}

// A statically shaped f16 memref viewed as a row-major [rows x cols] matrix.
struct Matrix {
  Value memref;
  int64_t rows = 0;
  int64_t cols = 0;
};

FailureOr<Matrix> asMatrix(Operation *op, Value value, StringRef name) {
  Value memref = unwrapCast(value);
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1)
    return op->emitError() << name << " must be a statically shaped memref";
  if (!type.getElementType().isF16())
    return op->emitError() << name
                           << ": the RedMulE lowering requires f16 tensors";
  Matrix m;
  m.memref = memref;
  m.cols = type.getShape().back();
  m.rows = type.getNumElements() / m.cols;
  return m;
}

LogicalResult expectShape(Operation *op, const Matrix &m, StringRef name,
    int64_t rows, int64_t cols) {
  if (m.rows == rows && m.cols == cols)
    return success();
  return op->emitError() << name << " is " << m.rows << "x" << m.cols
                         << ", the RedMulE lowering needs " << rows << "x"
                         << cols;
}

// Private SPM addresses of one tile: X, then W, then Y (spm_write layout).
struct TileRegion {
  int32_t tile = 0;
  Value x, w, y;
};

// Thin emitter for the scheduled AISMEM RedMulE operations.
class RedMulEEmitter {
public:
  RedMulEEmitter(ConversionPatternRewriter &rewriter, Location loc)
      : rewriter(rewriter), loc(loc) {}

  Value addrStart(int32_t tile) {
    auto op = rewriter.create<AISMEMRedMulEAddrStartOp>(loc,
        TypeRange{rewriter.getI32Type()}, ValueRange{},
        ArrayRef<NamedAttribute>{attr("tile", tile), attr("bank", kSPMBank)});
    return op.getAddress();
  }

  struct Upload {
    Value next;
    Value token;
  };

  // Upload source[rowOff:rowOff+rows, colOff:colOff+cols] (optionally
  // transposed / ReLU'd) into a zero-padded dstRows x 16 SPM matrix.
  Upload uploadTile(int32_t tile, Value source, Value address,
      ValueRange deps, int64_t rowOff, int64_t colOff, int64_t rows,
      int64_t cols, int64_t dstRows, bool transpose = false,
      bool relu = false) {
    SmallVector<Value> operands{source, address};
    operands.append(deps.begin(), deps.end());
    SmallVector<NamedAttribute> attrs{attr("tile", tile),
        attr("row_offset", rowOff), attr("col_offset", colOff),
        attr("rows", rows), attr("cols", cols), attr("dst_rows", dstRows),
        attr("dst_cols", kK),
        rewriter.getNamedAttr("transpose", rewriter.getBoolAttr(transpose)),
        rewriter.getNamedAttr("relu", rewriter.getBoolAttr(relu)),
        rewriter.getNamedAttr("negate", rewriter.getBoolAttr(false))};
    auto op = rewriter.create<AISMEMRedMulEUploadTileOp>(loc,
        TypeRange{rewriter.getI32Type(), rewriter.getNoneType()}, operands,
        attrs);
    return {op.getNextAddress(), op.getNoneVal()};
  }

  // Whole-matrix upload of an m.rows x 16 matrix.
  Upload uploadWhole(int32_t tile, const Matrix &m, Value address,
      ValueRange deps, bool relu = false) {
    return uploadTile(tile, m.memref, address, deps, 0, 0, m.rows, m.cols,
        m.rows, /*transpose=*/false, relu);
  }

  Value zero(int32_t tile, Value address, ValueRange deps, int32_t elements) {
    SmallVector<Value> operands{address};
    operands.append(deps.begin(), deps.end());
    auto op = rewriter.create<AISMEMRedMulEZeroOp>(loc,
        TypeRange{rewriter.getNoneType()}, operands,
        ArrayRef<NamedAttribute>{
            attr("tile", tile), attr("elements", elements)});
    return op.getNoneVal();
  }

  Value gemm(const TileRegion &r, ValueRange deps) {
    SmallVector<Value> operands{r.x, r.w, r.y};
    operands.append(deps.begin(), deps.end());
    auto op = rewriter.create<AISMEMRedMulEGEMMOp>(loc,
        TypeRange{rewriter.getNoneType()}, operands,
        ArrayRef<NamedAttribute>{attr("tile", r.tile), attr("k", kK),
            attr("m", kM), attr("n", kN)});
    return op.getNoneVal();
  }

  Value wait(ValueRange deps, int32_t mask) {
    auto op = rewriter.create<AISMEMRedMulEWaitOp>(loc,
        TypeRange{rewriter.getNoneType()}, deps,
        ArrayRef<NamedAttribute>{attr("mask", mask)});
    return op.getNoneVal();
  }

  Value download(const TileRegion &r, Value destination, Value dep) {
    auto op = rewriter.create<AISMEMRedMulEDownloadOp>(loc,
        TypeRange{rewriter.getNoneType()},
        ValueRange{r.y, destination, dep},
        ArrayRef<NamedAttribute>{
            attr("tile", r.tile), attr("elements", kM * kK)});
    return op.getNoneVal();
  }

  Value alloc(MemRefType type) {
    return aisle_to_aismem::insertAlloc(rewriter, loc, memref::DimOp(), type)
        .getResult();
  }

  Value allocTile() {
    return alloc(MemRefType::get({kM, kK}, rewriter.getF16Type()));
  }

private:
  NamedAttribute attr(StringRef name, int64_t value) {
    return rewriter.getNamedAttr(
        name, rewriter.getI32IntegerAttr(static_cast<int32_t>(value)));
  }

  ConversionPatternRewriter &rewriter;
  Location loc;
};

// Establish X/W/Y addresses of `tile` by performing its first X and W
// uploads.  Returns the region plus the token of the W upload.
std::pair<TileRegion, Value> firstUploads(RedMulEEmitter &emit, int32_t tile,
    Value xSource, int64_t xRows, Value wSource, int64_t wRowOff,
    int64_t wColOff, ValueRange deps) {
  TileRegion r;
  r.tile = tile;
  r.x = emit.addrStart(tile);
  auto x = emit.uploadTile(tile, xSource, r.x, deps, 0, 0, xRows, kN, kM);
  auto w = emit.uploadTile(tile, wSource, x.next, ValueRange{x.token},
      wRowOff, wColOff, kN, kK, kN);
  r.w = x.next;
  r.y = w.next;
  return {r, w.token};
}

// Y <- C, or Y <- 0 without accumulator.
Value initAccumulator(RedMulEEmitter &emit, const TileRegion &r,
    std::optional<Matrix> c, ValueRange deps) {
  if (c)
    return emit.uploadWhole(r.tile, *c, r.y, deps).token;
  return emit.zero(r.tile, r.y, deps, kM * kK);
}

Value bridgeToTensor(ConversionPatternRewriter &rewriter, Location loc,
    Type tensorType, Value memref) {
  return rewriter
      .create<UnrealizedConversionCastOp>(loc, tensorType, ValueRange{memref})
      .getResult(0);
}

} // namespace

//===----------------------------------------------------------------------===//
// MultiHeadAttention
//===----------------------------------------------------------------------===//

struct AISLEMultiHeadAttentionOpLowering : public ConversionPattern {
  using theOperation = spade::AISLEMultiHeadAttentionOp;
  using theAdaptor = spade::AISLEMultiHeadAttentionOpAdaptor;

  AISLEMultiHeadAttentionOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto oldOp = llvm::dyn_cast<theOperation>(op);
    if (!oldOp)
      return failure();
    // Operands only: attributes are read from oldOp.
    theAdaptor adaptor(operands);

    // ---- what this schedule covers --------------------------------------
    if (oldOp.getNumHeads() != 1)
      return op->emitError("RedMulE MultiHeadAttention: num_heads must be 1");
    if (oldOp.getNormalization() != "relu")
      return op->emitError(
          "RedMulE MultiHeadAttention supports normalization=\"relu\" only; "
          "decompose softmax attention at import time "
          "(--functions-to-decompose=MultiHeadAttention)");
    if (oldOp.getScale().convertToFloat() != 1.0f ||
        oldOp.getPostScale().convertToFloat() != 1.0f)
      return op->emitError(
          "RedMulE MultiHeadAttention: scale and post_scale must be folded "
          "into Wq/Wv (needs constant weights)");

    FailureOr<Matrix> xq = asMatrix(op, adaptor.getXq(), "Xq");
    FailureOr<Matrix> xkv = asMatrix(op, adaptor.getXkv(), "Xkv");
    FailureOr<Matrix> wq = asMatrix(op, adaptor.getWq(), "Wq");
    FailureOr<Matrix> wk = asMatrix(op, adaptor.getWk(), "Wk");
    FailureOr<Matrix> wv = asMatrix(op, adaptor.getWv(), "Wv");
    FailureOr<Matrix> wo = asMatrix(op, adaptor.getWo(), "Wo");
    if (failed(xq) || failed(xkv) || failed(wq) || failed(wk) || failed(wv) ||
        failed(wo))
      return failure();
    if (failed(expectShape(op, *xq, "Xq", kM, kN)) ||
        failed(expectShape(op, *xkv, "Xkv", kM, kN)) ||
        failed(expectShape(op, *wq, "Wq", kN, kK)) ||
        failed(expectShape(op, *wk, "Wk", kN, kK)) ||
        failed(expectShape(op, *wv, "Wv", kN, kK)) ||
        failed(expectShape(op, *wo, "Wo", kN, kK)))
      return failure();
    std::optional<Matrix> c;
    if (Value cValue = adaptor.getC()) {
      FailureOr<Matrix> cm = asMatrix(op, cValue, "C");
      if (failed(cm) || failed(expectShape(op, *cm, "C", kM, kK)))
        return failure();
      c = *cm;
    }

    MemRefType resultType =
        aisle_to_aismem::convertTensorToMemRef(oldOp.getY().getType());
    RedMulEEmitter emit(rewriter, loc);
    Value result = emit.alloc(resultType);
    Value q = emit.allocTile(), k = emit.allocTile(), v = emit.allocTile();
    Value s = emit.allocTile(), o = emit.allocTile();

    // ---- phase 1: Q, K, V, one projection per RedMulE -------------------
    const Matrix *xs[3] = {&*xq, &*xkv, &*xkv};
    const Matrix *ws[3] = {&*wq, &*wk, &*wv};
    TileRegion regions[3];
    SmallVector<Value> launches;
    for (int32_t t = 0; t < 3; ++t) {
      auto [region, wToken] = firstUploads(emit, t, xs[t]->memref, kM,
          ws[t]->memref, 0, 0, ValueRange{});
      regions[t] = region;
      Value z = emit.zero(t, region.y, ValueRange{wToken}, kM * kK);
      launches.push_back(emit.gemm(region, ValueRange{z}));
    }
    Value qkvDone = emit.wait(launches, tileBit(0) | tileBit(1) | tileBit(2));
    Value qReady = emit.download(regions[0], q, qkvDone);
    Value kReady = emit.download(regions[1], k, qkvDone);
    Value vReady = emit.download(regions[2], v, qkvDone);

    const TileRegion &r0 = regions[0];
    // ---- phase 2: S = Q . pad16(K^T) ------------------------------------
    auto sx = emit.uploadTile(
        0, q, r0.x, ValueRange{qReady, kReady}, 0, 0, kM, kN, kM);
    auto sw = emit.uploadTile(0, k, r0.w, ValueRange{sx.token}, 0, 0, kM, kN,
        kN, /*transpose=*/true);
    Value sz = emit.zero(0, r0.y, ValueRange{sw.token}, kM * kK);
    Value sDone = emit.wait(emit.gemm(r0, ValueRange{sz}), tileBit(0));
    Value sReady = emit.download(r0, s, sDone);

    // ---- phase 3: O = ReLU(S) . pad16(V) --------------------------------
    // Columns L..15 of S are exactly zero (K^T was zero padded), so the
    // padded rows of V never contribute.
    auto ox = emit.uploadTile(0, s, r0.x, ValueRange{sReady}, 0, 0, kM, kK, kM,
        /*transpose=*/false, /*relu=*/true);
    auto ow = emit.uploadTile(
        0, v, r0.w, ValueRange{ox.token, vReady}, 0, 0, kM, kK, kN);
    Value oz = emit.zero(0, r0.y, ValueRange{ow.token}, kM * kK);
    Value oDone = emit.wait(emit.gemm(r0, ValueRange{oz}), tileBit(0));
    Value oReady = emit.download(r0, o, oDone);

    // ---- phase 4: Y = O . Wo + C (residual preloaded into Y) ------------
    auto yx = emit.uploadTile(0, o, r0.x, ValueRange{oReady}, 0, 0, kM, kN, kM);
    auto yw = emit.uploadWhole(0, *wo, r0.w, ValueRange{yx.token});
    Value yInit = initAccumulator(emit, r0, c, ValueRange{yw.token});
    Value yDone = emit.wait(emit.gemm(r0, ValueRange{yInit}), tileBit(0));
    (void)emit.download(r0, result, yDone);

    rewriter.replaceOp(
        op, bridgeToTensor(rewriter, loc, oldOp.getY().getType(), result));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PositionwiseFeedForward
//===----------------------------------------------------------------------===//

struct AISLEPositionwiseFeedForwardOpLowering : public ConversionPattern {
  using theOperation = spade::AISLEPositionwiseFeedForwardOp;
  using theAdaptor = spade::AISLEPositionwiseFeedForwardOpAdaptor;

  AISLEPositionwiseFeedForwardOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto oldOp = llvm::dyn_cast<theOperation>(op);
    if (!oldOp)
      return failure();
    // Operands only: attributes are read from oldOp.
    theAdaptor adaptor(operands);

    if (oldOp.getActivation() != "relu")
      return op->emitError(
          "RedMulE PositionwiseFeedForward supports activation=\"relu\" only");

    FailureOr<Matrix> x = asMatrix(op, adaptor.getX(), "X");
    FailureOr<Matrix> w1 = asMatrix(op, adaptor.getW1(), "W1");
    FailureOr<Matrix> w2 = asMatrix(op, adaptor.getW2(), "W2");
    if (failed(x) || failed(w1) || failed(w2))
      return failure();
    const int64_t dff = w1->cols;
    if (dff % kK != 0 || dff == 0)
      return op->emitError() << "RedMulE PositionwiseFeedForward needs d_ff "
                                "to be a multiple of 16, got "
                             << dff;
    if (failed(expectShape(op, *x, "X", kM, kN)) ||
        failed(expectShape(op, *w1, "W1", kN, dff)) ||
        failed(expectShape(op, *w2, "W2", dff, kK)))
      return failure();
    std::optional<Matrix> c;
    if (Value cValue = adaptor.getC()) {
      FailureOr<Matrix> cm = asMatrix(op, cValue, "C");
      if (failed(cm) || failed(expectShape(op, *cm, "C", kM, kK)))
        return failure();
      c = *cm;
    }

    MemRefType resultType =
        aisle_to_aismem::convertTensorToMemRef(oldOp.getY().getType());
    RedMulEEmitter emit(rewriter, loc);
    Value result = emit.alloc(resultType);
    const int64_t chunks = dff / kK;
    SmallVector<Value> u;
    for (int64_t j = 0; j < chunks; ++j)
      u.push_back(emit.allocTile());

    // ---- up: U_j = X . W1[:, 16j:16j+16], waves of <= kTileCount ---------
    SmallVector<TileRegion> regions;
    SmallVector<Value> tileFree(kTileCount, Value()); // last token per tile
    SmallVector<Value> downloaded;
    for (int64_t wave = 0; wave < chunks; wave += kTileCount) {
      SmallVector<Value> launches;
      int32_t mask = 0;
      const int64_t waveEnd = std::min<int64_t>(chunks, wave + kTileCount);
      for (int64_t j = wave; j < waveEnd; ++j) {
        const int32_t t = static_cast<int32_t>(j - wave);
        SmallVector<Value> deps;
        if (tileFree[t])
          deps.push_back(tileFree[t]);
        Value wToken;
        if (static_cast<int64_t>(regions.size()) <= t) {
          auto [region, token] = firstUploads(emit, t, x->memref, kM,
              w1->memref, 0, j * kK, deps);
          regions.push_back(region);
          wToken = token;
        } else {
          const TileRegion &r = regions[t];
          auto up = emit.uploadWhole(t, *x, r.x, deps);
          wToken = emit.uploadTile(t, w1->memref, r.w, ValueRange{up.token},
                           0, j * kK, kN, kK, kN)
                       .token;
        }
        Value z = emit.zero(t, regions[t].y, ValueRange{wToken}, kM * kK);
        launches.push_back(emit.gemm(regions[t], ValueRange{z}));
        mask |= tileBit(t);
      }
      Value done = emit.wait(launches, mask);
      for (int64_t j = wave; j < waveEnd; ++j) {
        const int32_t t = static_cast<int32_t>(j - wave);
        tileFree[t] = emit.download(regions[t], u[j], done);
        downloaded.push_back(tileFree[t]);
      }
    }

    // ---- down: Y = C + sum_j ReLU(U_j) . W2[16j:16j+16, :] on tile 0 -----
    // A K-reduction into one accumulator: the launches serialise on one tile
    // with Y preserved between them.
    const TileRegion &r0 = regions[0];
    Value token = initAccumulator(emit, r0, c, downloaded);
    for (int64_t j = 0; j < chunks; ++j) {
      auto ux = emit.uploadTile(0, u[j], r0.x, ValueRange{token}, 0, 0, kM, kN,
          kM, /*transpose=*/false, /*relu=*/true);
      auto uw = emit.uploadTile(0, w2->memref, r0.w, ValueRange{ux.token},
          j * kN, 0, kN, kK, kN);
      token = emit.wait(emit.gemm(r0, ValueRange{uw.token}), tileBit(0));
    }
    (void)emit.download(r0, result, token);

    rewriter.replaceOp(
        op, bridgeToTensor(rewriter, loc, oldOp.getY().getType(), result));
    return success();
  }
};

void populateLoweringAISLETransformerOpPatterns(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  (void)typeConverter;
  patterns.insert<AISLEMultiHeadAttentionOpLowering,
      AISLEPositionwiseFeedForwardOpLowering>(ctx);
}

} // namespace spade
