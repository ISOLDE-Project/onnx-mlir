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
// A tiled result (aisle.Concat of N-tiles, [.., 12, 16N]) is one cast with
// 2N operands (address_0, token_0, address_1, token_1, ...) tagged with
// `aismem.spm_tiles` (the RedMulE instance of every tile): tile j holds
// columns [16j, 16j + 16).
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
constexpr llvm::StringLiteral kSPMTilesAttr = "aismem.spm_tiles";

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

// Several 12x16 SPM tiles side by side along the last axis.
inline mlir::Value makeSPMTiles(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Type tensorType, llvm::ArrayRef<SPMValue> tiles) {
  if (tiles.size() == 1)
    return makeSPMTensor(builder, loc, tensorType, tiles.front());
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> ids;
  for (const SPMValue &t : tiles) {
    operands.push_back(t.address);
    operands.push_back(t.token);
    ids.push_back(t.tile);
  }
  auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, mlir::TypeRange{tensorType}, operands);
  cast->setAttr(kSPMTilesAttr, builder.getDenseI32ArrayAttr(ids));
  return cast.getResult(0);
}

// The SPM tiles of a value: one for a plain SPM result, N for a tiled one.
inline std::optional<llvm::SmallVector<SPMValue>> getSPMTiles(
    mlir::Value value) {
  if (std::optional<SPMValue> one = getSPMValue(value))
    return llvm::SmallVector<SPMValue>{*one};
  auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast)
    return std::nullopt;
  auto ids = cast->getAttrOfType<mlir::DenseI32ArrayAttr>(kSPMTilesAttr);
  if (!ids || cast.getInputs().size() != 2 * ids.size())
    return std::nullopt;
  llvm::SmallVector<SPMValue> tiles;
  for (size_t j = 0; j < ids.size(); ++j)
    tiles.push_back({cast.getInputs()[2 * j], cast.getInputs()[2 * j + 1],
        ids[j]});
  return tiles;
}

// After the conversion: every SPM-resident result that still has users (they
// are not RedMulE blocks, e.g. func.return or a Krnl-lowered op) is
// downloaded into a fresh data-memory buffer.
void materializeSPMResults(mlir::ModuleOp module);

} // namespace spade
