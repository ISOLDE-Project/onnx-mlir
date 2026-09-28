
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "patterns.h"
#include "src/Conversion/AISLEToAISMEM/Math/SPMValue.hpp"
#include <optional>

#include "mlir/Pass/Pass.h"
#include "llvm/Pass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "src/Dialect/AISLE/AISLEDialect.hpp"
#include "src/Dialect/AISLE/AISLEOps.hpp"
#include "src/Dialect/AISMEM/AISMEMDialect.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;
using std::nullopt;
using std::optional;

namespace spade {

void populateAISLEToAISMEMConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {

  populateLoweringAISLEQConstantOpPattern(patterns, typeConverter, ctx);

  populateLoweringAISLEComplexGEMMOpPattern(patterns, typeConverter, ctx);

  populateLoweringAISLETransformerOpPatterns(patterns, typeConverter, ctx);

  populateLoweringAISLEMatMulAddOpPatterns(patterns, typeConverter, ctx);

  populateLoweringAISLEGEMMOpPattern(patterns, typeConverter, ctx);

  populateLoweringAISLEhstackOpPattern(patterns, typeConverter, ctx);
}

//===----------------------------------------------------------------------===//
// AISLE to AISMEM Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
struct AISLEToAISMEMLoweringPass
    : public PassWrapper<AISLEToAISMEMLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AISLEToAISMEMLoweringPass)

  StringRef getArgument() const override { return "convert-aisle-to-aismem"; }

  StringRef getDescription() const override {
    return "Lower (some)AISLE ops to AISMEM dialect.";
  }

  // Needed when the input IR does not already mention AISMEM (onnx-mlir-opt).
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spade::AISMEMDialect, memref::MemRefDialect>();
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  AISLEToAISMEMLoweringPass() = default;

  void runOnOperation();

public:
};

void AISLEToAISMEMLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<KrnlDialect, affine::AffineDialect,
      arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
      math::MathDialect, memref::MemRefDialect, shape::ShapeDialect,
      scf::SCFDialect, spade::AISMEMDialect>();

  // ComplexGEMM expands to an explicit scheduled RedMulE sequence in AISMEM,
  // while its newly allocated result memrefs are bridged back to the tensor
  // result types expected at the AISLE boundary. These temporary bridge casts
  // are reconciled later in the pipeline.
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Require complex GEMM to cross the AISLE/AISMEM boundary and materialize
  // the explicit four-launch schedule. applyPartialConversion is otherwise
  // allowed to leave unclassified AISLE operations in the IR.
  target.addIllegalOp<spade::AISLEComplexGEMMOp>();

  // Same for the transformer blocks: an unsupported configuration must be
  // reported here rather than silently reaching the LLVM lowering.
  target.addIllegalOp<spade::AISLEMultiHeadAttentionOp,
      spade::AISLEPositionwiseFeedForwardOp, spade::AISLEMatMulOp,
      spade::AISLEAddOp>();

  RewritePatternSet patterns(&getContext());

  // Create a TypeConverter
  TypeConverter spadeTypeConverter;

  // Add source materialization
  spadeTypeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> optional<Value> {
        if (inputs.size() != 1)
          return nullopt;

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });

  // Add target materialization
  spadeTypeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> optional<Value> {
        if (inputs.size() != 1)
          return nullopt;

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });

  // Define patterns.
  populateAISLEToAISMEMConversionPattern(
      patterns, spadeTypeConverter, &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // RedMulE blocks hand their results to each other in SPM; download only
  // those that reach anything else (e.g. the function's return).
  materializeSPMResults(module);
}

std::unique_ptr<Pass> createLowerToAISMEMPass() {
  return std::make_unique<AISLEToAISMEMLoweringPass>();
}

} // namespace spade
