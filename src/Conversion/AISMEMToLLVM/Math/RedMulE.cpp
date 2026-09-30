/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- RedMulE.cpp - Lower scheduled RedMulE operations ----------===//
//
// Lowers the explicit AISMEM RedMulE/SPM schedule to calls into the small
// bare-metal ABI implemented by isolde/system/bsp/onnx_redmule_runtime.c.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"

#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

using namespace mlir;

namespace {

constexpr StringLiteral addrStartFunc = "omrm_addr_start";
constexpr StringLiteral uploadFunc = "omrm_upload_f16";
constexpr StringLiteral uploadTileFunc = "omrm_upload_tile_f16";
constexpr StringLiteral spmReluFunc = "omrm_spm_relu_f16";
constexpr StringLiteral spmTransposeFunc = "omrm_spm_transpose_f16";
constexpr StringLiteral spmCopyFunc = "omrm_spm_copy_f16";
constexpr StringLiteral spmMoveFunc = "omrm_spm_move_f16";
constexpr int64_t spmRowBytes = 64; // get_addr_start(row) = row << 6
constexpr StringLiteral zeroFunc = "omrm_zero_f16";
constexpr StringLiteral gemm16x12x16Func = "omrm_gemm_f16_16_12_16";
constexpr StringLiteral waitFunc = "omrm_wait";
constexpr StringLiteral downloadFunc = "omrm_download_f16";

static LLVM::LLVMFuncOp getOrInsertFunction(Operation *anchor,
    ConversionPatternRewriter &rewriter, StringRef name, Type resultType,
    ArrayRef<Type> argumentTypes) {
  ModuleOp module = anchor->getParentOfType<ModuleOp>();
  if (auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return function;

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto functionType =
      LLVM::LLVMFunctionType::get(resultType, argumentTypes, false);
  return rewriter.create<LLVM::LLVMFuncOp>(
      anchor->getLoc(), name, functionType);
}

static Value i32Constant(
    ConversionPatternRewriter &rewriter, Location loc, int64_t value) {
  return rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(), static_cast<int32_t>(value));
}

static Value completedToken(
    ConversionPatternRewriter &rewriter, Location loc) {
  return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI1Type(), 1);
}

static FailureOr<int64_t> getIntegerAttribute(Operation *op, StringRef name) {
  auto attribute = op->getAttrOfType<IntegerAttr>(name);
  if (!attribute)
    return failure();
  return attribute.getInt();
}

static FailureOr<int64_t> getStaticF16ElementCount(Value source) {
  auto memrefType = dyn_cast<MemRefType>(source.getType());
  if (!memrefType || !memrefType.hasStaticShape() ||
      !memrefType.getElementType().isF16())
    return failure();
  return memrefType.getNumElements();
}

class RedMulEAddrStartLowering final : public ConvertToLLVMPattern {
public:
  RedMulEAddrStartLowering(
      LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(
            spade::AISMEMRedMulEAddrStartOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    (void)operands;
    FailureOr<int64_t> tile = getIntegerAttribute(op, "tile");
    FailureOr<int64_t> bank = getIntegerAttribute(op, "bank");
    if (failed(tile) || failed(bank))
      return rewriter.notifyMatchFailure(op, "expected tile and bank attrs");

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    auto function = getOrInsertFunction(
        op, rewriter, addrStartFunc, i32, {i32, i32});
    auto call = rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, *tile),
            i32Constant(rewriter, loc, *bank)});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

class RedMulEUploadLowering final : public ConvertToLLVMPattern {
public:
  RedMulEUploadLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(
            spade::AISMEMRedMulEUploadOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (operands.size() < 3)
      return rewriter.notifyMatchFailure(op, "expected source, shape and addr");

    FailureOr<int64_t> elements = getStaticF16ElementCount(op->getOperand(0));
    FailureOr<int64_t> tile = getIntegerAttribute(op, "tile");
    auto negateAttr = op->getAttrOfType<BoolAttr>("negate");
    if (failed(elements) || failed(tile) || !negateAttr)
      return rewriter.notifyMatchFailure(
          op, "requires a static f16 source and tile/negate attrs");
    if ((*elements % 16) != 0)
      return op->emitError("RedMulE upload requires complete 16-f16 rows");

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto function = getOrInsertFunction(
        op, rewriter, uploadFunc, i32, {i32, i32, ptr, i32, i32});
    auto call = rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, *tile), operands[2], operands[0],
            i32Constant(rewriter, loc, *elements),
            i32Constant(rewriter, loc, negateAttr.getValue() ? 1 : 0)});
    rewriter.replaceOp(op, ValueRange{call.getResult(),
                               completedToken(rewriter, loc)});
    return success();
  }
};

// omrm_upload_tile_f16 flags; keep in sync with onnx_redmule_runtime.h.
constexpr int64_t tileFlagTranspose = 1;
constexpr int64_t tileFlagRelu = 2;
constexpr int64_t tileFlagNegate = 4;

class RedMulEUploadTileLowering final : public ConvertToLLVMPattern {
public:
  RedMulEUploadTileLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(
            spade::AISMEMRedMulEUploadTileOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto tileOp = cast<spade::AISMEMRedMulEUploadTileOp>(op);
    if (operands.size() < 2)
      return rewriter.notifyMatchFailure(op, "expected source and address");
    auto sourceType = dyn_cast<MemRefType>(tileOp.getSource().getType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        !sourceType.getElementType().isF16())
      return rewriter.notifyMatchFailure(op, "requires a static f16 source");

    int64_t flags = 0;
    if (tileOp.getTranspose())
      flags |= tileFlagTranspose;
    if (tileOp.getRelu())
      flags |= tileFlagRelu;
    if (tileOp.getNegate())
      flags |= tileFlagNegate;

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto function = getOrInsertFunction(op, rewriter, uploadTileFunc, i32,
        {i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32});
    auto call = rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, tileOp.getTile()), operands[1],
            operands[0],
            i32Constant(rewriter, loc, sourceType.getShape().back()),
            i32Constant(rewriter, loc, tileOp.getRowOffset()),
            i32Constant(rewriter, loc, tileOp.getColOffset()),
            i32Constant(rewriter, loc, tileOp.getRows()),
            i32Constant(rewriter, loc, tileOp.getCols()),
            i32Constant(rewriter, loc, tileOp.getDstRows()),
            i32Constant(rewriter, loc, tileOp.getDstCols()),
            i32Constant(rewriter, loc, flags)});
    rewriter.replaceOp(
        op, ValueRange{call.getResult(), completedToken(rewriter, loc)});
    return success();
  }
};

// A placed SPM buffer is just its first row's address.
class SPMAllocLowering final : public ConvertToLLVMPattern {
public:
  SPMAllocLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(spade::AISMEMSPMAllocOp::getOperationName(),
            context, converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    (void)operands;
    auto alloc = cast<spade::AISMEMSPMAllocOp>(op);
    std::optional<uint32_t> row = alloc.getRow();
    if (!row)
      return op->emitError("SPM buffer has no row assigned; run the "
                           "aismem-spm-allocate pass before LLVM lowering");
    rewriter.replaceOp(
        op, i32Constant(rewriter, op->getLoc(), *row * spmRowBytes));
    return success();
  }
};

// omrm_spm_relu_f16(tile, addr, rows)
class SPMReluLowering final : public ConvertToLLVMPattern {
public:
  SPMReluLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(spade::AISMEMSPMReluOp::getOperationName(),
            context, converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto relu = cast<spade::AISMEMSPMReluOp>(op);
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function = getOrInsertFunction(
        op, rewriter, spmReluFunc, voidType, {i32, i32, i32});
    rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, relu.getTile()), operands[0],
            i32Constant(rewriter, loc, relu.getRows())});
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

// omrm_spm_transpose_f16(tile, src, dst, rows, dst_rows)
class SPMTransposeLowering final : public ConvertToLLVMPattern {
public:
  SPMTransposeLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(spade::AISMEMSPMTransposeOp::getOperationName(),
            context, converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto tr = cast<spade::AISMEMSPMTransposeOp>(op);
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function = getOrInsertFunction(
        op, rewriter, spmTransposeFunc, voidType, {i32, i32, i32, i32, i32});
    rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, tr.getTile()), operands[0],
            operands[1], i32Constant(rewriter, loc, tr.getRows()),
            i32Constant(rewriter, loc, tr.getDstRows())});
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

// omrm_spm_copy_f16(tile, src, dst, rows)
class SPMCopyLowering final : public ConvertToLLVMPattern {
public:
  SPMCopyLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(spade::AISMEMSPMCopyOp::getOperationName(),
            context, converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto copy = cast<spade::AISMEMSPMCopyOp>(op);
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function = getOrInsertFunction(
        op, rewriter, spmCopyFunc, voidType, {i32, i32, i32, i32});
    rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, copy.getTile()), operands[0],
            operands[1], i32Constant(rewriter, loc, copy.getRows())});
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

// omrm_spm_move_f16(src_tile, src_addr, dst_tile, dst_addr, rows,
//                   dst_rows, flags)   flags: OMRM_TILE_TRANSPOSE
class SPMMoveTileLowering final : public ConvertToLLVMPattern {
public:
  SPMMoveTileLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(spade::AISMEMSPMMoveTileOp::getOperationName(),
            context, converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto move = cast<spade::AISMEMSPMMoveTileOp>(op);
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function = getOrInsertFunction(op, rewriter, spmMoveFunc, voidType,
        {i32, i32, i32, i32, i32, i32, i32});
    rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, move.getSrcTile()), operands[0],
            i32Constant(rewriter, loc, move.getDstTile()), operands[1],
            i32Constant(rewriter, loc, move.getRows()),
            i32Constant(rewriter, loc, move.getDstRows()),
            i32Constant(rewriter, loc,
                move.getTranspose() ? tileFlagTranspose : 0)});
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

class RedMulEZeroLowering final : public ConvertToLLVMPattern {
public:
  RedMulEZeroLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(
            spade::AISMEMRedMulEZeroOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (operands.empty())
      return rewriter.notifyMatchFailure(op, "expected an SPM address");
    FailureOr<int64_t> tile = getIntegerAttribute(op, "tile");
    FailureOr<int64_t> elements = getIntegerAttribute(op, "elements");
    if (failed(tile) || failed(elements))
      return rewriter.notifyMatchFailure(op, "expected tile/elements attrs");
    if ((*elements % 16) != 0)
      return op->emitError("RedMulE zero requires complete 16-f16 rows");

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function = getOrInsertFunction(
        op, rewriter, zeroFunc, voidType, {i32, i32, i32});
    rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, *tile), operands[0],
            i32Constant(rewriter, loc, *elements)});
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

class RedMulEGEMMLowering final : public ConvertToLLVMPattern {
public:
  RedMulEGEMMLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(
            spade::AISMEMRedMulEGEMMOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (operands.size() < 3)
      return rewriter.notifyMatchFailure(op, "expected X/W/Y SPM addresses");
    FailureOr<int64_t> tile = getIntegerAttribute(op, "tile");
    FailureOr<int64_t> k = getIntegerAttribute(op, "k");
    FailureOr<int64_t> m = getIntegerAttribute(op, "m");
    FailureOr<int64_t> n = getIntegerAttribute(op, "n");
    if (failed(tile) || failed(k) || failed(m) || failed(n))
      return rewriter.notifyMatchFailure(op, "expected tile/K/M/N attrs");

    // redmule.gemm encodes K/M/N as instruction immediates.  This milestone
    // deliberately supports the attached model's single hardware tile.
    if (*k != 16 || *m != 12 || *n != 16)
      return op->emitError(
          "MVP RedMulE LLVM lowering supports only K=16, M=12, N=16");

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function = getOrInsertFunction(
        op, rewriter, gemm16x12x16Func, voidType, {i32, i32, i32, i32});
    rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, *tile), operands[0], operands[1],
            operands[2]});
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

class RedMulEWaitLowering final : public ConvertToLLVMPattern {
public:
  RedMulEWaitLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(
            spade::AISMEMRedMulEWaitOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    (void)operands;
    FailureOr<int64_t> mask = getIntegerAttribute(op, "mask");
    if (failed(mask))
      return rewriter.notifyMatchFailure(op, "expected mask attr");

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function =
        getOrInsertFunction(op, rewriter, waitFunc, voidType, {i32});
    rewriter.create<LLVM::CallOp>(
        loc, function, ValueRange{i32Constant(rewriter, loc, *mask)});
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

class RedMulEDownloadLowering final : public ConvertToLLVMPattern {
public:
  RedMulEDownloadLowering(LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(
            spade::AISMEMRedMulEDownloadOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (operands.size() < 2)
      return rewriter.notifyMatchFailure(op, "expected SPM addr/destination");
    FailureOr<int64_t> tile = getIntegerAttribute(op, "tile");
    FailureOr<int64_t> elements = getIntegerAttribute(op, "elements");
    if (failed(tile) || failed(elements))
      return rewriter.notifyMatchFailure(op, "expected tile/elements attrs");
    if ((*elements % 16) != 0)
      return op->emitError("RedMulE download requires complete 16-f16 rows");

    auto downloadOp = cast<spade::AISMEMRedMulEDownloadOp>(op);
    const int64_t dstOffset = downloadOp.getDstOffset();
    const int64_t dstLd = downloadOp.getDstLd();
    const int64_t rows = *elements / 16;
    auto destinationType = dyn_cast<MemRefType>(op->getOperand(1).getType());
    const bool contiguous = dstOffset == 0 && dstLd == 16;
    if (!destinationType || !destinationType.getElementType().isF16() ||
        !destinationType.hasStaticShape() ||
        (contiguous && destinationType.getNumElements() != *elements) ||
        (!contiguous && (dstLd < 16 || dstOffset < 0 ||
                            dstOffset + (rows - 1) * dstLd + 16 >
                                destinationType.getNumElements())))
      return op->emitError(
          "RedMulE download requires a matching static f16 destination");

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function = getOrInsertFunction(
        op, rewriter, downloadFunc, voidType, {i32, i32, ptr, i32});
    if (contiguous) {
      rewriter.create<LLVM::CallOp>(loc, function,
          ValueRange{i32Constant(rewriter, loc, *tile), operands[0],
              operands[1], i32Constant(rewriter, loc, *elements)});
    } else {
      // A tile of a wider result: one SPM row (16 fp16, 64 bytes of SPM)
      // per destination row.
      constexpr int64_t spmRowBytes = 64;
      for (int64_t r = 0; r < rows; ++r) {
        Value spm = rewriter.create<LLVM::AddOp>(loc, operands[0],
            i32Constant(rewriter, loc, r * spmRowBytes));
        Value dst = rewriter.create<LLVM::GEPOp>(loc, ptr,
            rewriter.getF16Type(), operands[1],
            ArrayRef<LLVM::GEPArg>{
                static_cast<int32_t>(dstOffset + r * dstLd)});
        rewriter.create<LLVM::CallOp>(loc, function,
            ValueRange{i32Constant(rewriter, loc, *tile), spm, dst,
                i32Constant(rewriter, loc, 16)});
      }
    }
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

} // namespace

namespace spade {

void populateLoweringAISMEMRedMulEOpPatterns(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<RedMulEAddrStartLowering, RedMulEUploadLowering,
      RedMulEUploadTileLowering, SPMAllocLowering, SPMReluLowering,
      SPMTransposeLowering, SPMCopyLowering, SPMMoveTileLowering,
      RedMulEZeroLowering, RedMulEGEMMLowering, RedMulEWaitLowering,
      RedMulEDownloadLowering>(typeConverter, ctx);
}

} // namespace spade
