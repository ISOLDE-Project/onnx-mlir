/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SPMAllocation.cpp - SPM row management for RedMulE -------===//
//
// aismem-spm-allocate
//
// Assigns SPM rows to every aismem.SPMAlloc of a function, per tile:
//
//   * resident buffers (weights filled once from a krnl.global) get rows at
//     the bottom of the SPM for the whole program; their uploads are moved
//     into a generated `<function>_preload` function, to be called once
//     before the first inference;
//   * all other buffers get rows from a linear scan over their lifetimes in
//     program order (first fit, rows are reused as soon as a buffer is dead).
//     A buffer read by an asynchronous RedMulE launch lives until the wait
//     that retires the launch.
//
// A tile has `rows-per-tile` rows (default 512: the 32 KiB narrow SPM window
// of platform demo_3 at 64 bytes per row).  Running out of rows is an error;
// with --print-map the pass prints the SPM map of every function.
//
// Also: materializeSPMResults(), used by convert-aisle-to-aismem to download
// SPM-resident block results that reach non-RedMulE users.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "src/Conversion/AISLEToAISMEM/Math/SPMValue.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>

#define DEBUG_TYPE "aismem-spm-allocate"

using namespace mlir;

namespace spade {

//===----------------------------------------------------------------------===//
// Download of SPM-resident results with non-RedMulE users
//===----------------------------------------------------------------------===//

void materializeSPMResults(ModuleOp module) {
  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk([&](UnrealizedConversionCastOp cast) {
    if (getSPMValue(cast.getResult(0)))
      casts.push_back(cast);
  });
  for (UnrealizedConversionCastOp cast : casts) {
    if (cast.use_empty()) {
      cast.erase();
      continue;
    }
    SPMValue value = *getSPMValue(cast.getResult(0));
    auto tensorType = cast.getResult(0).getType().cast<RankedTensorType>();
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    OpBuilder builder(cast);
    builder.setInsertionPointAfter(cast);
    Location loc = cast.getLoc();
    auto alloc = builder.create<memref::AllocOp>(loc, memrefType);
    alloc.setAlignmentAttr(builder.getI64IntegerAttr(16));
    builder.create<AISMEMRedMulEDownloadOp>(loc,
        TypeRange{builder.getNoneType()},
        ValueRange{value.address, alloc.getResult(), value.token},
        ArrayRef<NamedAttribute>{
            builder.getNamedAttr("tile", builder.getI32IntegerAttr(value.tile)),
            builder.getNamedAttr("elements",
                builder.getI32IntegerAttr(
                    static_cast<int32_t>(memrefType.getNumElements())))});
    Value tensor =
        builder
            .create<UnrealizedConversionCastOp>(
                loc, TypeRange{tensorType}, ValueRange{alloc.getResult()})
            .getResult(0);
    cast.getResult(0).replaceAllUsesWith(tensor);
    cast.erase();
  }
}

//===----------------------------------------------------------------------===//
// Row allocation
//===----------------------------------------------------------------------===//

namespace {

struct Buffer {
  AISMEMSPMAllocOp op;
  int64_t tile = 0;
  int64_t rows = 0;
  int64_t def = 0;  // program position of the allocation
  int64_t last = 0; // last position at which the rows are needed
  bool resident = false;
  AISMEMRedMulEUploadTileOp fill; // resident: its only writer
  int64_t row = -1;
};

std::string bufferName(Buffer b) {
  if (std::optional<StringRef> name = b.op.getName())
    return name->str();
  return "<unnamed>";
}

// The single upload that fills a resident buffer from a krnl.global, if the
// buffer is otherwise only read (as a GEMM X or W operand).
AISMEMRedMulEUploadTileOp residentFill(AISMEMSPMAllocOp alloc) {
  AISMEMRedMulEUploadTileOp fill;
  for (OpOperand &use : alloc.getAddress().getUses()) {
    Operation *user = use.getOwner();
    if (auto upload = dyn_cast<AISMEMRedMulEUploadTileOp>(user)) {
      if (fill || use.getOperandNumber() != 1)
        return {};
      Operation *src = upload.getSource().getDefiningOp();
      if (!src || src->getName().getStringRef() != "krnl.global" ||
          !upload.getNoneVal().use_empty())
        return {};
      fill = upload;
      continue;
    }
    auto gemm = dyn_cast<AISMEMRedMulEGEMMOp>(user);
    if (!gemm || use.getOperandNumber() == 2) // Y is written
      return {};
  }
  return fill;
}

struct SPMAllocationPass
    : public PassWrapper<SPMAllocationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SPMAllocationPass)

  StringRef getArgument() const override { return "aismem-spm-allocate"; }
  StringRef getDescription() const override {
    return "Assign RedMulE SPM rows to aismem.SPMAlloc buffers and hoist "
           "resident weight uploads into <function>_preload.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spade::AISMEMDialect, func::FuncDialect>();
  }

  SPMAllocationPass() = default;
  SPMAllocationPass(const SPMAllocationPass &pass)
      : PassWrapper<SPMAllocationPass, OperationPass<ModuleOp>>() {}

  Option<unsigned> rowsPerTile{*this, "rows-per-tile",
      llvm::cl::desc("SPM rows per RedMulE tile (64 bytes each)"),
      llvm::cl::init(512)};
  Option<bool> residentWeights{*this, "resident-weights",
      llvm::cl::desc("Keep constant weights in SPM, uploaded once by "
                     "<function>_preload"),
      llvm::cl::init(true)};
  Option<bool> printMap{*this, "print-map",
      llvm::cl::desc("Print the SPM map of every function"),
      llvm::cl::init(false)};

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());
    for (func::FuncOp f : funcs)
      if (failed(allocate(module, f)))
        return signalPassFailure();
  }

  LogicalResult allocate(ModuleOp module, func::FuncOp f) {
    if (f.isExternal())
      return success();
    SmallVector<AISMEMSPMAllocOp> allocs;
    f.walk([&](AISMEMSPMAllocOp op) { allocs.push_back(op); });
    if (allocs.empty())
      return success();
    Block &body = f.front();
    for (AISMEMSPMAllocOp op : allocs)
      if (op->getBlock() != &body)
        return op.emitError("SPM buffers must be allocated at function scope");

    DenseMap<Operation *, int64_t> position;
    int64_t n = 0;
    for (Operation &op : body)
      position[&op] = n++;
    auto pos = [&](Operation *op) {
      while (op->getBlock() != &body)
        op = op->getParentOp();
      return position[op];
    };

    // ---- lifetimes ------------------------------------------------------
    SmallVector<Buffer> buffers;
    for (AISMEMSPMAllocOp op : allocs) {
      Buffer b;
      b.op = op;
      b.tile = op.getTile();
      b.rows = op.getRows();
      b.def = b.last = pos(op);
      SmallVector<Value> work{op.getAddress()};
      while (!work.empty()) {
        Value v = work.pop_back_val();
        for (Operation *user : v.getUsers()) {
          b.last = std::max(b.last, pos(user));
          if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user))
            for (Value r : cast.getResults())
              work.push_back(r);
          // An asynchronous launch reads its operands until it is retired.
          if (auto gemm = dyn_cast<AISMEMRedMulEGEMMOp>(user))
            for (Operation *waiter : gemm.getNoneVal().getUsers())
              if (isa<AISMEMRedMulEWaitOp>(waiter))
                b.last = std::max(b.last, pos(waiter));
        }
      }
      if (residentWeights && op.getResident())
        if ((b.fill = residentFill(op)))
          b.resident = true;
      buffers.push_back(b);
    }

    // ---- placement, per tile ------------------------------------------------
    const int64_t capacity = rowsPerTile;
    std::map<int64_t, SmallVector<Buffer *>> byTile;
    for (Buffer &b : buffers)
      byTile[b.tile].push_back(&b);
    std::map<int64_t, int64_t> peakRows;
    for (auto &[tile, list] : byTile) {
      int64_t base = 0; // resident region [0, base)
      for (Buffer *b : list)
        if (b->resident) {
          b->row = base;
          base += b->rows;
        }
      SmallVector<Buffer *> transient;
      for (Buffer *b : list)
        if (!b->resident)
          transient.push_back(b);
      llvm::stable_sort(transient,
          [](Buffer *a, Buffer *b) { return a->def < b->def; });
      SmallVector<Buffer *> active;
      int64_t peak = base;
      for (Buffer *b : transient) {
        llvm::erase_if(active, [&](Buffer *a) { return a->last < b->def; });
        llvm::sort(active, [](Buffer *x, Buffer *y) { return x->row < y->row; });
        int64_t row = base;
        for (Buffer *a : active) {
          if (a->row - row >= b->rows)
            break;
          row = std::max(row, a->row + a->rows);
        }
        b->row = row;
        active.push_back(b);
        peak = std::max(peak, row + b->rows);
      }
      peakRows[tile] = peak;
      if (peak > capacity) {
        InFlightDiagnostic diag = f.emitError()
                                  << "SPM of tile " << tile << " overflows: "
                                  << peak << " rows needed, " << capacity
                                  << " available (" << base
                                  << " resident)";
        for (Buffer *b : list)
          diag.attachNote(b->op.getLoc())
              << bufferName(*b) << ": rows [" << b->row << ", "
              << b->row + b->rows << ")" << (b->resident ? " resident" : "");
        return failure();
      }
    }

    OpBuilder builder(f.getContext());
    for (Buffer &b : buffers)
      b.op.setRowAttr(builder.getI32IntegerAttr(static_cast<int32_t>(b.row)));

    // ---- preload function ---------------------------------------------------
    SmallVector<Buffer *> hoisted;
    for (Buffer &b : buffers)
      if (b.resident)
        hoisted.push_back(&b);
    if (!hoisted.empty()) {
      builder.setInsertionPointAfter(f);
      std::string name = (f.getName() + "_preload").str();
      auto preload = builder.create<func::FuncOp>(
          f.getLoc(), name, builder.getFunctionType({}, {}));
      preload->setAttr("aismem.preload", builder.getUnitAttr());
      if (f->hasAttr("llvm.emit_c_interface"))
        preload->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
      Block *entry = preload.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      for (Buffer *b : hoisted) {
        AISMEMRedMulEUploadTileOp fill = b->fill;
        Operation *global = fill.getSource().getDefiningOp();
        Value source;
        if (global->hasOneUse()) {
          global->moveBefore(entry, entry->end());
          source = global->getResult(0);
        } else {
          Operation *copy = builder.clone(*global);
          copy->setAttr("name",
              builder.getStringAttr(
                  global->getAttrOfType<StringAttr>("name").str() +
                  "_preload"));
          source = copy->getResult(0);
        }
        builder.setInsertionPointToEnd(entry);
        Value address = builder.clone(*b->op)->getResult(0);
        IRMapping map;
        map.map(fill.getSource(), source);
        map.map(fill.getSpmAddress(), address);
        builder.clone(*fill, map);
        fill.erase();
      }
      builder.create<func::ReturnOp>(f.getLoc());
    }

    // ---- report -------------------------------------------------------------
    SmallVector<NamedAttribute> used;
    for (auto &[tile, peak] : peakRows)
      used.push_back(builder.getNamedAttr(
          ("tile" + llvm::Twine(tile)).str(), builder.getI64IntegerAttr(peak)));
    f->setAttr("aismem.spm_rows_used", builder.getDictionaryAttr(used));

    if (printMap) {
      llvm::raw_ostream &os = llvm::outs();
      os << "SPM map of @" << f.getName() << " (" << capacity
         << " rows per tile, 64 B per row)\n";
      for (auto &[tile, list] : byTile) {
        os << "  tile " << tile << ": " << peakRows[tile] << " rows used\n";
        SmallVector<Buffer *> sorted(list);
        llvm::stable_sort(sorted, [](Buffer *a, Buffer *b) {
          return a->row != b->row ? a->row < b->row : a->def < b->def;
        });
        for (Buffer *b : sorted) {
          os << "    rows [" << b->row << ", " << b->row + b->rows << ")  "
             << bufferName(*b);
          if (b->resident)
            os << "  resident";
          else
            os << "  live " << b->def << ".." << b->last;
          os << "\n";
        }
      }
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createSPMAllocationPass() {
  return std::make_unique<SPMAllocationPass>();
}

} // namespace spade
