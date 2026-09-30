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
// A tile has `rows-per-tile` rows (onnx-mlir --redmule-spm-rows; default
// 256).  The narrow window of platform demo_3 is 32 KiB (512 rows of 64
// bytes), but the tile's bank memories are addressed with TCDM_AW = 10 bits
// (isolde_tcdm_pkg): row r and row r + 256 are the same memory, so a
// schedule for 512 rows silently corrupts itself on that RTL.
// If the rows do not suffice, resident weights are demoted to per-call
// uploads (with a warning) until they do; running out of rows even then is
// an error.  With print-map the pass prints the SPM map of every function.
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
#include "src/Pass/Passes.hpp"
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
    if (cast->getNumResults() == 1 && getSPMTiles(cast.getResult(0)))
      casts.push_back(cast);
  });
  for (UnrealizedConversionCastOp cast : casts) {
    if (cast.use_empty()) {
      cast.erase();
      continue;
    }
    SmallVector<SPMValue> tiles = *getSPMTiles(cast.getResult(0));
    auto tensorType = cast.getResult(0).getType().cast<RankedTensorType>();
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    OpBuilder builder(cast);
    builder.setInsertionPointAfter(cast);
    Location loc = cast.getLoc();
    auto alloc = builder.create<memref::AllocOp>(loc, memrefType);
    alloc.setAlignmentAttr(builder.getI64IntegerAttr(16));
    // One tile: a contiguous copy of its rows (just its first cols if the
    // result is narrower than 16).  N tiles side by side: tile j fills
    // columns [16j, 16j + 16) of every row.
    const int64_t rowLength = tensorType.getShape().back();
    const int64_t rows = memrefType.getNumElements() / rowLength;
    const int64_t tileCols = rowLength / static_cast<int64_t>(tiles.size());
    for (auto [j, tile] : llvm::enumerate(tiles)) {
      auto i32 = [&](int64_t v) {
        return builder.getI32IntegerAttr(static_cast<int32_t>(v));
      };
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("tile", i32(tile.tile)),
          builder.getNamedAttr("elements", i32(rows * tileCols))};
      if (tiles.size() > 1 || tileCols != 16) {
        attrs.push_back(builder.getNamedAttr(
            "dst_offset", i32(static_cast<int64_t>(j) * tileCols)));
        attrs.push_back(builder.getNamedAttr("dst_ld", i32(rowLength)));
        attrs.push_back(builder.getNamedAttr("cols", i32(tileCols)));
      }
      builder.create<AISMEMRedMulEDownloadOp>(loc,
          TypeRange{builder.getNoneType()},
          ValueRange{tile.address, alloc.getResult(), tile.token}, attrs);
    }
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

// A read-only upload: fills a fresh buffer from an immutable source (a
// function argument or a krnl.global) and the buffer is only read as a GEMM
// X or W operand afterwards, with no one waiting on the upload's token.
bool isReadOnlyUpload(AISMEMRedMulEUploadTileOp upload) {
  auto alloc = upload.getSpmAddress().getDefiningOp<AISMEMSPMAllocOp>();
  if (!alloc || !upload.getNoneVal().use_empty() ||
      !upload.getDependencies().empty())
    return false;
  Value src = upload.getSource();
  Operation *def = src.getDefiningOp();
  if (!isa<BlockArgument>(src) &&
      !(def && def->getName().getStringRef() == "krnl.global"))
    return false;
  for (OpOperand &use : alloc.getAddress().getUses()) {
    Operation *user = use.getOwner();
    if (user == upload.getOperation())
      continue;
    if (!isa<AISMEMRedMulEGEMMOp>(user) || use.getOperandNumber() > 1)
      return false;
  }
  return true;
}

// Tiling reads the same window more than once (the X tile of every N-tile):
// keep the first read-only upload of each window and let later readers use
// its buffer (the allocator extends its lifetime accordingly).
void reuseUploads(func::FuncOp f) {
  SmallVector<AISMEMRedMulEUploadTileOp> kept;
  SmallVector<AISMEMRedMulEUploadTileOp> uploads;
  f.walk([&](AISMEMRedMulEUploadTileOp u) { uploads.push_back(u); });
  for (AISMEMRedMulEUploadTileOp u : uploads) {
    if (!isReadOnlyUpload(u))
      continue;
    auto alloc = u.getSpmAddress().getDefiningOp<AISMEMSPMAllocOp>();
    auto same = llvm::find_if(kept, [&](AISMEMRedMulEUploadTileOp k) {
      auto kAlloc = k.getSpmAddress().getDefiningOp<AISMEMSPMAllocOp>();
      return k.getSource() == u.getSource() &&
             k->getAttrDictionary() == u->getAttrDictionary() &&
             kAlloc.getRows() == alloc.getRows() &&
             kAlloc.getTile() == alloc.getTile() &&
             kAlloc.getResident() == alloc.getResident() &&
             k->getBlock() == u->getBlock();
    });
    if (same == kept.end()) {
      kept.push_back(u);
      continue;
    }
    alloc.getAddress().replaceAllUsesExcept(
        same->getSpmAddress(), u.getOperation());
    u.erase();
    alloc.erase();
  }
}

struct SPMAllocationPass
    : public PassWrapper<SPMAllocationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SPMAllocationPass)

  StringRef getArgument() const override { return "aismem-spm-allocate"; }
  StringRef getDescription() const override {
    return "Assign RedMulE SPM rows to aismem.SPMAlloc buffers (after "
           "merging repeated read-only uploads of the same window) and hoist "
           "resident weight uploads into <function>_preload.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spade::AISMEMDialect, func::FuncDialect>();
  }

  SPMAllocationPass() = default;
  SPMAllocationPass(const SPMAllocationPass &pass)
      : PassWrapper<SPMAllocationPass, OperationPass<ModuleOp>>() {}
  SPMAllocationPass(unsigned rows, bool resident, bool map) {
    rowsPerTile = rows;
    residentWeights = resident;
    printMap = map;
  }

  Option<unsigned> rowsPerTile{*this, "rows-per-tile",
      llvm::cl::desc("SPM rows per RedMulE tile (64 bytes each)"),
      llvm::cl::init(kDefaultSPMRowsPerTile)};
  Option<bool> residentWeights{*this, "resident-weights",
      llvm::cl::desc("Keep constant weights in SPM, uploaded once by "
                     "<function>_preload"),
      llvm::cl::init(true)};
  Option<bool> printMap{*this, "print-map",
      llvm::cl::desc("Print the SPM map of every function"),
      llvm::cl::init(false)};

  // Rows for `list` (one tile): resident buffers at the bottom, [0, base),
  // the others first fit over their lifetimes.  Returns the peak row count.
  static int64_t place(ArrayRef<Buffer *> list, int64_t &base) {
    base = 0;
    for (Buffer *b : list)
      if (b->resident) {
        b->row = base;
        base += b->rows;
      }
    SmallVector<Buffer *> transient;
    for (Buffer *b : list)
      if (!b->resident)
        transient.push_back(b);
    llvm::stable_sort(
        transient, [](Buffer *a, Buffer *b) { return a->def < b->def; });
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
    return peak;
  }

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
    reuseUploads(f);
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
    // Resident buffers first, at the bottom; the others first fit over their
    // lifetimes.  If a tile does not fit, resident weights are demoted to
    // ordinary buffers, uploaded on every call where they are used (as the
    // hand-written firmware does), one at a time -- each time the one whose
    // demotion gives the smallest peak -- until the tile fits.
    const int64_t capacity = rowsPerTile;
    std::map<int64_t, SmallVector<Buffer *>> byTile;
    for (Buffer &b : buffers)
      byTile[b.tile].push_back(&b);
    std::map<int64_t, int64_t> peakRows;
    std::map<int64_t, int64_t> residentRows;
    std::map<int64_t, SmallVector<Buffer *>> demoted;
    for (auto &[tile, list] : byTile) {
      int64_t base = 0;
      int64_t peak = place(list, base);
      while (peak > capacity) {
        Buffer *best = nullptr;
        int64_t bestPeak = 0;
        for (Buffer *b : list) {
          if (!b->resident)
            continue;
          b->resident = false;
          int64_t unusedBase;
          const int64_t p = place(list, unusedBase);
          b->resident = true;
          if (!best || p < bestPeak ||
              (p == bestPeak && b->def > best->def)) {
            best = b;
            bestPeak = p;
          }
        }
        if (!best)
          break;
        best->resident = false;
        demoted[tile].push_back(best);
        peak = place(list, base);
      }
      peakRows[tile] = peak;
      residentRows[tile] = base;
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
    for (auto &[tile, list] : demoted) {
      int64_t rows = 0;
      for (Buffer *b : list)
        rows += b->rows;
      // Not an MLIR diagnostic: the onnx-mlir driver does not show warnings.
      llvm::errs() << "warning: @" << f.getName() << ": SPM of tile " << tile
                   << " (" << capacity << " rows): " << list.size()
                   << " constant buffer(s), " << rows
                   << " rows, are uploaded on every call instead of once by "
                      "the preload function\n";
    }

    OpBuilder builder(f.getContext());
    for (Buffer &b : buffers) {
      b.op.setRowAttr(builder.getI32IntegerAttr(static_cast<int32_t>(b.row)));
      if (!b.resident && b.fill)
        b.op.setResidentAttr(builder.getBoolAttr(false)); // demoted
    }

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
      // One copy of every weight global in the preload function, however
      // many resident windows read it.
      DenseMap<Operation *, Value> preloaded;
      for (Buffer *b : hoisted) {
        AISMEMRedMulEUploadTileOp fill = b->fill;
        Operation *global = fill.getSource().getDefiningOp();
        Value &source = preloaded[global];
        if (!source) {
          bool onlyResidentFills = llvm::all_of(global->getUsers(),
              [&](Operation *user) {
                return llvm::any_of(hoisted,
                    [&](Buffer *h) { return h->fill == user; });
              });
          if (onlyResidentFills) {
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
            os << "  live " << b->def << ".." << b->last
               << (b->fill ? "  (constant, uploaded per call)" : "");
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

std::unique_ptr<Pass> createSPMAllocationPass(
    unsigned rowsPerTile, bool residentWeights, bool printMap) {
  return std::make_unique<SPMAllocationPass>(
      rowsPerTile, residentWeights, printMap);
}

} // namespace spade
