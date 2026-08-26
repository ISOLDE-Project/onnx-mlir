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

    auto destinationType = dyn_cast<MemRefType>(op->getOperand(1).getType());
    if (!destinationType || !destinationType.getElementType().isF16() ||
        !destinationType.hasStaticShape() ||
        destinationType.getNumElements() != *elements)
      return op->emitError(
          "RedMulE download requires a matching static f16 destination");

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Type ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto function = getOrInsertFunction(
        op, rewriter, downloadFunc, voidType, {i32, i32, ptr, i32});
    rewriter.create<LLVM::CallOp>(loc, function,
        ValueRange{i32Constant(rewriter, loc, *tile), operands[0], operands[1],
            i32Constant(rewriter, loc, *elements)});
    rewriter.replaceOp(op, completedToken(rewriter, loc));
    return success();
  }
};

} // namespace

namespace spade {

void populateLoweringAISMEMRedMulEOpPatterns(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<RedMulEAddrStartLowering, RedMulEUploadLowering,
      RedMulEZeroLowering, RedMulEGEMMLowering, RedMulEWaitLowering,
      RedMulEDownloadLowering>(typeConverter, ctx);
}

} // namespace spade
