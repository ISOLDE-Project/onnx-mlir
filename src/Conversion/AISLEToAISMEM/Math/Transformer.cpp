/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ Transformer.cpp - Lowering ISOLDE transformer blocks ------===//
//
// Lower aisle.MultiHeadAttention and aisle.PositionwiseFeedForward to an
// explicit RedMulE schedule whose intermediates stay in the tile-private SPM.
//
// Every matrix product is one native RedMulE launch
//
//     Y[12x16] (+)= X[12x16] . W[16x16]        (K=16, M=12, N=16)
//
// and all buffers are aismem.SPMAlloc row ranges on one tile (single-tile
// chain: no cross-tile traffic).  A GEMM's Y buffer is used directly as the X
// or W operand of the next GEMM; nothing goes back to data memory between
// blocks.  The `aismem-spm-allocate` pass later assigns the rows from buffer
// lifetimes and hoists the weight uploads into `<entry>_preload`.
//
//   MultiHeadAttention (self-attention, one head, L = 12, d = 16)
//     Q = h Wq                        12 rows
//     K = h Wk                        12 rows
//     V = h Wv    into 16 rows whose last 4 are zero  -> already pad16(V)
//     Kt = pad16(K^T)                 SPMTranspose (core)
//     S = Q Kt ; S = ReLU(S)          SPMRelu, in place (core)
//     O = S V
//     Y = O Wo + h                    residual: Y *is* h's buffer when h has
//                                     no other user (in place, no copy)
//
//   PositionwiseFeedForward (d = 16, d_ff = 16 T)
//     U_j = X W1[:, 16j:16j+16] ; U_j = ReLU(U_j)        j < T
//     Y = X + sum_j U_j W2[16j:16j+16, :]                 Y accumulates
//
// Block inputs that arrive from data memory are uploaded once; a block result
// is handed to the next RedMulE block as an SPM address (a 2-operand
// unrealized cast (address, token) -> tensor).  ConvertAISLEToAISMEM
// materializes a download only where such a result reaches anything else,
// e.g. the function's return.
//
// Scale factors must already be folded into the weights (ONNXToAISLE does it
// for constant weights); other configurations are diagnosed, not
// miscompiled.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/AISLEToAISMEM/Math/SPMEmitter.hpp"
#include "src/Conversion/AISLEToAISMEM/helper.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "AISLEToAISMEM_Transformer"

using namespace mlir;

namespace spade {

using namespace spade::spm;


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

    FailureOr<Operand> xq = classify(op, adaptor.getXq(), "Xq", kM, kN);
    FailureOr<Operand> xkv = classify(op, adaptor.getXkv(), "Xkv", kM, kN);
    FailureOr<Operand> wq = classify(op, adaptor.getWq(), "Wq", kN, kK);
    FailureOr<Operand> wk = classify(op, adaptor.getWk(), "Wk", kN, kK);
    FailureOr<Operand> wv = classify(op, adaptor.getWv(), "Wv", kN, kK);
    FailureOr<Operand> wo = classify(op, adaptor.getWo(), "Wo", kN, kK);
    if (failed(xq) || failed(xkv) || failed(wq) || failed(wk) || failed(wv) ||
        failed(wo))
      return failure();
    if (!wq->host || !wk->host || !wv->host || !wo->host)
      return op->emitError(
          "RedMulE MultiHeadAttention: weights must come from data memory");
    std::optional<Operand> c;
    if (Value cValue = adaptor.getC()) {
      FailureOr<Operand> co = classify(op, cValue, "C", kM, kK);
      if (failed(co))
        return failure();
      c = *co;
    }

    Emitter emit(rewriter, loc, blockName(op, "mha"));
    Value Wq = emit.weight(*wq->host, 0, 0, "Wq");
    Value Wk = emit.weight(*wk->host, 0, 0, "Wk");
    Value Wv = emit.weight(*wv->host, 0, 0, "Wv");
    Value Wo = emit.weight(*wo->host, 0, 0, "Wo");

    SPMValue h = emit.activation(*xq, "Xq");
    SPMValue hkv = oldOp.getXkv() == oldOp.getXq()
                       ? h
                       : emit.activation(*xkv, "Xkv");

    // Q, K and V; V lands in a 16-row buffer whose rows 12..15 stay zero, so
    // it is already the padded W operand of O = S V.
    Value q = emit.alloc(kM, "Q");
    Value t = emit.gemm(h.address, Wq, q,
        ValueRange{emit.zero(q, kM, ValueRange{h.token})});
    Value k = emit.alloc(kM, "K");
    t = emit.gemm(hkv.address, Wk, k,
        ValueRange{emit.zero(k, kM, ValueRange{t, hkv.token})});
    Value v = emit.alloc(kN, "V|0");
    t = emit.gemm(hkv.address, Wv, v, ValueRange{emit.zero(v, kN, t)});

    // S = ReLU(Q . pad16(K^T)).  Columns L..15 of S are zero because K^T is
    // zero padded, so the padded rows of V never contribute.
    Value kt = emit.alloc(kN, "K^T|0");
    t = emit.transpose(k, kt, kM, kN, t);
    Value s = emit.alloc(kM, "S");
    t = emit.gemm(q, kt, s, ValueRange{emit.zero(s, kM, t)});
    t = emit.relu(s, kM, t);

    // O = S . V
    Value o = emit.alloc(kM, "O");
    t = emit.gemm(s, v, o, ValueRange{emit.zero(o, kM, t)});

    // Y = O . Wo + C: the residual is the accumulator, in place when C has
    // no other user.
    const bool cDead = oldOp.getC() && onlyUsedBy(oldOp.getC(), op);
    SPMValue y = (c && c->host && cDead && oldOp.getC() == oldOp.getXq())
                     ? h // C is the block input we already uploaded
                     : emit.accumulator(c, cDead, t);
    t = emit.gemm(o, Wo, y.address, ValueRange{t, y.token});

    rewriter.replaceOp(op, makeSPMTensor(rewriter, loc,
                               oldOp.getY().getType(), {y.address, t, kTile}));
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

    auto w1Type = dyn_cast<ShapedType>(oldOp.getW1().getType());
    const int64_t dff =
        w1Type && w1Type.hasStaticShape() ? w1Type.getShape().back() : 0;
    if (dff % kK != 0 || dff == 0)
      return op->emitError() << "RedMulE PositionwiseFeedForward needs d_ff "
                                "to be a multiple of 16, got "
                             << dff;
    FailureOr<Operand> x = classify(op, adaptor.getX(), "X", kM, kN);
    FailureOr<Operand> w1 = classify(op, adaptor.getW1(), "W1", kN, dff);
    FailureOr<Operand> w2 = classify(op, adaptor.getW2(), "W2", dff, kK);
    if (failed(x) || failed(w1) || failed(w2))
      return failure();
    if (!w1->host || !w2->host)
      return op->emitError(
          "RedMulE PositionwiseFeedForward: weights must come from data "
          "memory");
    std::optional<Operand> c;
    if (Value cValue = adaptor.getC()) {
      FailureOr<Operand> co = classify(op, cValue, "C", kM, kK);
      if (failed(co))
        return failure();
      c = *co;
    }

    Emitter emit(rewriter, loc, blockName(op, "ffn"));
    const int64_t chunks = dff / kK;
    SmallVector<Value> w1Tiles, w2Tiles;
    for (int64_t j = 0; j < chunks; ++j) {
      w1Tiles.push_back(
          emit.weight(*w1->host, 0, j * kK, "W1[:," + Twine(j * kK) + "]"));
      w2Tiles.push_back(
          emit.weight(*w2->host, j * kN, 0, "W2[" + Twine(j * kN) + ",:]"));
    }

    SPMValue in = emit.activation(*x, "X");

    // up: U_j = ReLU(X . W1_j)
    SmallVector<Value> u;
    Value t = in.token;
    for (int64_t j = 0; j < chunks; ++j) {
      Value uj = emit.alloc(kM, "U" + Twine(j));
      t = emit.gemm(in.address, w1Tiles[j], uj, ValueRange{emit.zero(uj, kM, t)});
      t = emit.relu(uj, kM, t);
      u.push_back(uj);
    }

    // down: Y = C + sum_j U_j . W2_j, a K-reduction into one accumulator.
    const bool cDead = oldOp.getC() && onlyUsedBy(oldOp.getC(), op);
    SPMValue y = (c && c->host && cDead && oldOp.getC() == oldOp.getX())
                     ? in // C is the block input we already uploaded
                     : emit.accumulator(c, cDead, t);
    for (int64_t j = 0; j < chunks; ++j)
      t = j == 0 ? emit.gemm(u[j], w2Tiles[j], y.address, ValueRange{t, y.token})
                 : emit.gemm(u[j], w2Tiles[j], y.address, ValueRange{t});

    rewriter.replaceOp(op, makeSPMTensor(rewriter, loc,
                               oldOp.getY().getType(), {y.address, t, kTile}));
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
