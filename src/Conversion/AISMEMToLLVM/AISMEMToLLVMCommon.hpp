/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Types.h"
#include <optional>

using namespace mlir;
using std::nullopt;
using std::optional;

namespace spade {
struct AISMEMTypeConverter : public ::mlir::LLVMTypeConverter {
  explicit AISMEMTypeConverter(
      mlir::MLIRContext *ctx, const mlir::LowerToLLVMOptions &options)
      : ::mlir::LLVMTypeConverter(ctx, options) {

    addConversion([ctx](MemRefType type) -> std::optional<Type> {
      mlir::Type elemType = type.getElementType();
      if (!elemType)
         return std::nullopt;
      return LLVM::LLVMPointerType::get(ctx);
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> optional<Value> {
      if (inputs.size() != 1)
        return nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> optional<Value> {
      if (inputs.size() != 1)
        return nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
  bool isSignatureLegal(mlir::FunctionType funcType) {
    return llvm::all_of(llvm::concat<const mlir::Type>(
                            funcType.getInputs(), funcType.getResults()),
        [this](mlir::Type type) { return isLegal(type); });
  }
  bool isSignatureLegal(mlir::func::CallOp call) {
    auto f = [this](mlir::Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }
};

struct LLVMConversionTarget : public ConversionTarget {
  explicit LLVMConversionTarget(
      MLIRContext &ctx, AISMEMTypeConverter &typeConverter)
      : ConversionTarget(ctx), typeConverter(typeConverter) {

    addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      // FuncOp is legal only if types have been converted to Std types.
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      // CallOp is legal only if types have been converted to Std types.
      return typeConverter.isLegal(op);
    });

    // Operations that are legal only if types are not MemRef.
    addDynamicallyLegalOp<mlir::func::ReturnOp>([&](Operation *op) {
      return llvm::none_of(op->getOperandTypes(),
          [](Type type) { return type.isa<TensorType>(); });
    });
  }
  AISMEMTypeConverter &typeConverter;
};

} // namespace spade