/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- GEMM.cpp - Lowering GEMM Op -------------------===//
//
// Lowers onnx.Gemm to a tiled chain of AISLE GEMM ops sized for RedMulE.
//
// A single onnx.Gemm
//
//     Y[M,N] = A[M,K] . B[K,N] + C[M,N]        (alpha = beta = 1)
//
// is decomposed along BOTH free dimensions, because the accelerator only
// executes one fixed tile shape (M x Kt x Nt, Kt = Nt = kTileWidth):
//
//   * N-split  (independent output columns) -> the tiles are independent,
//     each produces its own slice of Y and the slices are concatenated:
//         Y = [ Y_0 | Y_1 | ... ]              -> AISLEhstack
//     This is the MLP-up / QKV pattern: one tile per RedMulE, run in
//     parallel, results collected separately.
//
//   * K-split  (contraction reduction) -> the tiles are NOT independent,
//     they accumulate into the same Y:
//         Y = A_0.B_0 + A_1.B_1 + ... + C
//     realised by threading the accumulator through the C operand:
//         t0 = AISLEGEMM(A_0, B_0, C )      "launch_bias"       (C preload)
//         t1 = AISLEGEMM(A_1, B_1, t0)      "launch_accumulate" (Y untouched)
//
//===----------------------------------------------------------------------===//

#include "../helper.hpp"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/SpadeSupport.hpp"
#include <algorithm>
#include <bits/stdint-intn.h>
#include <cstddef>
#include <utility>

#define DEBUG_TYPE "ONNXToAISLE_GEMM"

using namespace mlir;

namespace spade {

namespace {

/// Width of one RedMulE operand tile, in elements.  Both the contraction (K)
/// and the output columns (N) are cut at this granularity.
/// TODO: promote to a pass option so the value is not duplicated between the
/// compiler and OMRM_FP16_PER_ROW / TF_DMODEL in the runtime.
static constexpr int64_t kTileWidth = 16;

/// Shape container expected by onnx_to_aisle::create()'s shape overload,
/// which takes `const SmallVector<int64_t> &`.  A SmallVector<int64_t, 2>
/// is a DIFFERENT type and does not convert, so the inline size must be the
/// default one.
using ShapeVector = llvm::SmallVector<int64_t>;

/// Half-open [begin, end) ranges covering `extent` in `tile`-sized steps.
/// The last range is short when `extent` is not a multiple of `tile`.
static llvm::SmallVector<std::pair<int64_t, int64_t>, 4> tileRanges(
    int64_t extent, int64_t tile) {
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> ranges;
  for (int64_t start = 0; start < extent; start += tile)
    ranges.push_back({start, std::min(start + tile, extent)});
  return ranges;
}

/// Rank-2 constant sub-matrix [r0,r1) x [c0,c1), preserving the element
/// type.  Returns a null Value if `input` is not a float constant.
///
/// This is the F16-safe replacement for
///     TensorRawData::splitMatrix() + createONNXConstantOp().
static Value splitConstant(ConversionPatternRewriter &rewriter, Location loc,
    Value input, int64_t r0, int64_t r1, int64_t c0, int64_t c1,
    ShapeVector &shapeOut) {
  auto constOp = input.getDefiningOp<mlir::ONNXConstantOp>();
  if (!constOp)
    return nullptr;
  auto dense = constOp.getValueAttr().dyn_cast_or_null<DenseElementsAttr>();
  if (!dense)
    return nullptr;
  auto srcType = dense.getType().dyn_cast<RankedTensorType>();
  if (!srcType || srcType.getRank() != 2)
    return nullptr;
  Type elementType = srcType.getElementType();
  if (!elementType.isa<FloatType>())
    return nullptr;

  const int64_t srcCols = srcType.getShape()[1];
  const int64_t rows = r1 - r0;
  const int64_t cols = c1 - c0;
  assert(rows > 0 && cols > 0 && "empty sub-matrix");

  // Materialise the source once; values are kept as APFloat so F16 stays
  // bit-exact (no float round trip).
  llvm::SmallVector<APFloat, 16> all(
      dense.getValues<APFloat>().begin(), dense.getValues<APFloat>().end());

  llvm::SmallVector<APFloat, 16> part;
  part.reserve(rows * cols);
  for (int64_t i = r0; i < r1; ++i)
    for (int64_t j = c0; j < c1; ++j)
      part.push_back(all[i * srcCols + j]); // row-major, as in helper.hpp

  shapeOut.assign({rows, cols});
  auto dstType = RankedTensorType::get(shapeOut, elementType);
  auto dstAttr = DenseElementsAttr::get(dstType, llvm::ArrayRef<APFloat>(part));

  return rewriter.create<mlir::ONNXConstantOp>(
      loc, /*sparse_value=*/Attribute(), /*value=*/dstAttr);
}

/// Runtime slice of a value along `axis`, [lo, hi).  A is the activation and
/// is generally NOT a constant (the encoder's window is a function
/// argument), so it cannot be split at compile time; it needs a real op.
///
/// NOTE: this emits an onnx.Slice.  It is a pure view -- at the AISMEM level
/// it must fold into a base-pointer offset, which is what the runtime does
/// (`window + X_ELEMENTS`), not a copy.  If onnx.Slice is marked illegal by
/// the conversion target, replace this with the equivalent AISLE slice op.
static Value sliceAxis(ConversionPatternRewriter &rewriter, Location loc,
    Value input, int64_t axis, int64_t lo, int64_t hi) {
  auto inType = input.getType().cast<TensorType>();
  auto inShape = inType.getShape();

  // Identity slice: do not clutter the IR.
  if (lo == 0 && hi == inShape[axis])
    return input;

  auto i64Type = rewriter.getIntegerType(64);
  auto attrType = RankedTensorType::get({1}, i64Type);
  auto constant = [&](int64_t v) -> Value {
    return rewriter.create<mlir::ONNXConstantOp>(loc, Attribute(),
        DenseElementsAttr::get(attrType, llvm::ArrayRef<int64_t>{v}));
  };

  llvm::SmallVector<int64_t, 4> outShape(inShape.begin(), inShape.end());
  outShape[axis] = hi - lo;
  auto outType = RankedTensorType::get(outShape, inType.getElementType());

  return rewriter.create<mlir::ONNXSliceOp>(loc, outType, input,
      /*starts=*/constant(lo), /*ends=*/constant(hi),
      /*axes=*/constant(axis), /*steps=*/constant(1));
}

/// Zero tensor of `shape` and `elementType`, used as the Y preload when the
/// Gemm has no C operand (the runtime's launch_zero).
static Value zeroConstant(ConversionPatternRewriter &rewriter, Location loc,
    llvm::ArrayRef<int64_t> shape, Type elementType) {
  auto type = RankedTensorType::get(shape, elementType);
  auto attr = DenseElementsAttr::get(type, rewriter.getZeroAttr(elementType));
  return rewriter.create<mlir::ONNXConstantOp>(
      loc, /*sparse_value=*/Attribute(), /*value=*/attr);
}

} // namespace

struct ONNXGEMMOpLowering : public ConversionPattern {

  using theOperation = mlir::ONNXGemmOp;
  using theAdaptor = mlir::ONNXGemmOpAdaptor;
  using theNewOp = spade::AISLEGEMMOp;

  ONNXGEMMOpLowering(MLIRContext *ctx)
      : ConversionPattern(theOperation::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    theOperation oldOp = llvm::dyn_cast<theOperation>(op);
    theAdaptor operandAdaptor(operands);
    theAdaptor attrAdaptor(oldOp);

    // ---------------------------------------------------------------
    // Attributes.  Only the plain accumulate form is representable on
    // RedMulE: Y = A.B + C.
    // ---------------------------------------------------------------
    double alpha = attrAdaptor.getAlpha().convertToDouble();
    double beta = attrAdaptor.getBeta().convertToDouble();
    if (alpha != 1.0 || beta != 1.0)
      return rewriter.notifyMatchFailure(
          op, "scalar multipliers not supported, alpha and beta must be 1.0");

    int64_t transA = attrAdaptor.getTransA();
    int64_t transB = attrAdaptor.getTransB();
    if (transA != 0)
      return rewriter.notifyMatchFailure(op, "transA != 0 is not implemented");

    // ---------------------------------------------------------------
    // Shapes.  A is [M,K].  B is [K,N] when transB = 0 (what the ONNX
    // frontend emits for a MatMul+Add fusion) and [N,K] when transB = 1.
    // Both are accepted; only the split axis of B changes.
    // ---------------------------------------------------------------
    Value A = operandAdaptor.getA();
    Value B = operandAdaptor.getB();
    Value C = operandAdaptor.getC();

    auto typeA = A.getType().dyn_cast<TensorType>();
    auto typeB = B.getType().dyn_cast<TensorType>();
    if (!typeA || !typeB || !typeA.hasStaticShape() || !typeB.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "static shapes required");
    if (typeA.getRank() != 2 || typeB.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "rank-2 operands required");

    auto shapeA = typeA.getShape();
    auto shapeB = typeB.getShape();
    const int64_t M = shapeA[0];
    const int64_t K = shapeA[1];
    const int64_t N = (transB == 1) ? shapeB[0] : shapeB[1];
    const int64_t Kb = (transB == 1) ? shapeB[1] : shapeB[0];
    if (K != Kb)
      return rewriter.notifyMatchFailure(op, "A and B disagree on K");

    // Element type comes from the operands: the encoder is F16 end to end,
    // so hardcoding F32 here would silently widen every tile.
    Type elementType = typeA.getElementType();

    const bool hasC = C && !C.getType().isa<NoneType>();

    // ---------------------------------------------------------------
    // Tiling plan.
    // ---------------------------------------------------------------
    auto kTiles = tileRanges(K, kTileWidth);
    auto nTiles = tileRanges(N, kTileWidth);

    LLVM_DEBUG(llvm::dbgs() << "[ONNXToAISLE_GEMM] M=" << M << " K=" << K
                            << " N=" << N << " transB=" << transB << " -> "
                            << kTiles.size() << " K-tile(s) x "
                            << nTiles.size() << " N-tile(s)\n");

    llvm::SmallVector<Value, 4> nResults;

    for (auto &n : nTiles) {
      const int64_t n0 = n.first, n1 = n.second;
      const int64_t nWidth = n1 - n0;

      // C slice for this N tile: [M, n0:n1].  Seeds the reduction chain
      // ("launch_bias": the accumulator is preloaded with the real bias).
      Value acc;
      if (hasC) {
        ShapeVector cShape;
        acc = splitConstant(rewriter, loc, C, 0, M, n0, n1, cShape);
        if (!acc)
          return rewriter.notifyMatchFailure(
              op, "C must be a rank-2 float constant to be split");
      }

      for (size_t ki = 0; ki < kTiles.size(); ++ki) {
        const int64_t k0 = kTiles[ki].first, k1 = kTiles[ki].second;

        // A tile: columns [k0,k1) of the activation.
        Value aTile = sliceAxis(rewriter, loc, A, /*axis=*/1, k0, k1);
        ShapeVector aShapeVec{M, k1 - k0};
        Value aShape = onnx_to_aisle::create(
            rewriter, aTile, "A_tile_shape", aShapeVec);

        // B tile: logically [k0:k1, n0:n1]; the physical sub-matrix
        // depends on the storage order selected by transB.
        ShapeVector bShapeVec;
        Value bTile = (transB == 1)
                          ? splitConstant(rewriter, loc, B, n0, n1, k0, k1,
                                bShapeVec)
                          : splitConstant(rewriter, loc, B, k0, k1, n0, n1,
                                bShapeVec);
        if (!bTile)
          return rewriter.notifyMatchFailure(
              op, "B must be a rank-2 float constant to be split");
        Value bShape = onnx_to_aisle::create(
            rewriter, bTile, "B_tile_shape", bShapeVec);

        // Accumulator: the C slice on the first K tile ("launch_bias"),
        // the previous tile's result afterwards ("launch_accumulate").
        // With no C at all the chain starts from zero ("launch_zero").
        Value cIn = acc;
        if (!cIn)
          cIn = zeroConstant(rewriter, loc, {M, nWidth}, elementType);

        llvm::SmallVector<Value, 5> tileOperands{
            aTile, aShape, bTile, bShape, cIn};
        llvm::SmallVector<NamedAttribute, 4> tileAttrs{
            NamedAttribute(
                oldOp.getTransAAttrName(), attrAdaptor.getTransAAttr()),
            NamedAttribute(
                oldOp.getTransBAttrName(), attrAdaptor.getTransBAttr())};

        auto tileType = RankedTensorType::get({M, nWidth}, elementType);
        llvm::SmallVector<Type, 1> tileTypes{tileType};

        acc = rewriter.create<theNewOp>(
            loc, tileTypes, tileOperands, tileAttrs);
      }

      nResults.push_back(acc);
    }

    // ---------------------------------------------------------------
    // Reassemble.  One N tile means the reduction chain already produced
    // the whole result and no concatenation is needed.
    // ---------------------------------------------------------------
    Value result;
    if (nResults.size() == 1) {
      result = nResults.front();
    } else {
      llvm::SmallVector<Type, 1> stackTypes;
      for (auto v : oldOp.getODSResults(0))
        stackTypes.push_back(v.getType());
      llvm::SmallVector<NamedAttribute, 4> stackAttrs;
      result = rewriter.create<spade::AISLEhstack>(
          loc, stackTypes, nResults, stackAttrs);
    }

    rewriter.replaceOp(op, result);

    LLVM_DEBUG({ spade::dumpUsers(op); });

    return ::mlir::success();
  }
};

void populateLoweringONNXToAISLEGEMMOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXGEMMOpLowering>(ctx);
}

} // namespace spade
