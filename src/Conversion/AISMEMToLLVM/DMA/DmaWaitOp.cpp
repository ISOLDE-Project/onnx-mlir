#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Support/SpadeSupport.hpp"

#define DEBUG_TYPE "AISMEMToLLVM_DmaWaitOp"

using namespace mlir;


namespace {
struct DmaWaitOpLowering : public ConvertToLLVMPattern {
  using theOperation = memref::DmaWaitOp;
  using theAdaptor = memref::DmaWaitOpAdaptor;

  explicit DmaWaitOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            theOperation::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    theOperation oldOp = llvm::dyn_cast<theOperation>(op);
    theAdaptor operandAdaptor(operands);
    theAdaptor attrAdaptor(oldOp);
    const LLVMTypeConverter *typeConverter =
        static_cast<const LLVMTypeConverter *>(getTypeConverter());
    // auto llvmPointerType = LLVM::LLVMPointerType::get(rewriter.getContext());
    // auto srcPtr = rewriter.create<LLVM::BitcastOp>(
    //     op->getLoc(), llvmPointerType, srcMemRef);
    // auto dstPtr = rewriter.create<LLVM::BitcastOp>(
    //     op->getLoc(), llvmPointerType, dstMemRef);
    // Call an external DMA function (assume it is defined elsewhere)

    // Define the LLVM function type for the DMA transfer
    // void dma_transfer(void *dst, void *src, size_t num_elements);
    auto resultType = LLVM::LLVMVoidType::get(rewriter.getContext());
    ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
    for (auto type : oldOp->getOperandTypes())
      tblgen_types.push_back(typeConverter->convertType(type));

    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(resultType, tblgen_types, false);

    // Get or insert the external DMA function
    auto module = op->getParentOfType<ModuleOp>();
    auto llvmFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!llvmFunc) {
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      //
      llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
    }
    if (!llvmFunc)
      return failure();
    LLVM::CallOp newCallOp;
    // Create the call to the external DMA function
    llvm::SmallVector<::mlir::Value> tblgen_operands(
        operandAdaptor.getOperands().begin(),
        operandAdaptor.getOperands().end());
    newCallOp = rewriter.create<LLVM::CallOp>(loc, llvmFunc, tblgen_operands);

    LLVM_DEBUG({
      llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
      spade::dumpBlock(op);
      llvm::errs() << "---\n";
    });

    SmallVector<Value, 4> newValues(
        newCallOp->getResults().begin(), newCallOp->getResults().end());
    // Remove the original DMA start operation
    rewriter.replaceOp(oldOp, newValues);
    bool deadOp = mlir::isOpTriviallyDead(oldOp);
    bool useEmpty = oldOp.use_empty();
    if (deadOp && useEmpty) {
      oldOp->dropAllUses();
      rewriter.eraseOp(oldOp);
    }
    LLVM_DEBUG({
      llvm::errs() << " ** " << __FILE__ << "(" << __LINE__ << ")\n";
      spade::dumpBlock(op);
      llvm::errs() << "---\n";
    });
    return success();
  }

private:
  inline static std::string funcName = "_dma_wait";
};
} // namespace
namespace spade {
void populateMemrefDmaWaitOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<DmaWaitOpLowering>(typeConverter, ctx);
}
} // namespace spade