/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- SPMValue.hpp - SPM-resident tensors in AISLEToAISMEM --------===//
//
// A block lowered to RedMulE leaves its result in the tile-private SPM.  The
// result is handed to the next block as a tensor produced by a two-operand
//
//   builtin.unrealized_conversion_cast %address, %token : i32, none to tensor
//
// tagged with `aismem.spm_tile`.  A consumer that understands SPM (another
// RedMulE block) takes the address and the token; for any other consumer
// materializeSPMResults() inserts a download into data memory.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>

namespace spade {

constexpr llvm::StringLiteral kSPMTileAttr = "aismem.spm_tile";

struct SPMValue {
  mlir::Value address; // i32, first row of the buffer
  mlir::Value token;   // none, completion of the last write
  int32_t tile = 0;
};

inline mlir::Value makeSPMTensor(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Type tensorType, const SPMValue &value) {
  auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc,
      mlir::TypeRange{tensorType}, mlir::ValueRange{value.address, value.token});
  cast->setAttr(kSPMTileAttr, builder.getI32IntegerAttr(value.tile));
  return cast.getResult(0);
}

inline std::optional<SPMValue> getSPMValue(mlir::Value value) {
  auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 2)
    return std::nullopt;
  auto tile = cast->getAttrOfType<mlir::IntegerAttr>(kSPMTileAttr);
  if (!tile)
    return std::nullopt;
  return SPMValue{cast.getInputs()[0], cast.getInputs()[1],
      static_cast<int32_t>(tile.getInt())};
}

// After the conversion: every SPM-resident result that still has users (they
// are not RedMulE blocks, e.g. func.return or a Krnl-lowered op) is
// downloaded into a fresh data-memory buffer.
void materializeSPMResults(mlir::ModuleOp module);

} // namespace spade
