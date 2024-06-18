
#include "patterns.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
using std::optional;
using std::nullopt;

namespace spade {

void populateAISLEToAISMEMConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {

  populateLoweringAISLEQConstantOpPattern(patterns, typeConverter, ctx); 
  
  populateLoweringAISLEGEMMOpPattern(patterns, typeConverter, ctx, true);
}

//===----------------------------------------------------------------------===//
// AISLE to AISMEM Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
struct AISLEToAISMEMLoweringPass
    : public PassWrapper<AISLEToAISMEMLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AISLEToAISMEMLoweringPass)

  StringRef getArgument() const { return "convert-contirv-to-contimem"; }

  StringRef getDescription() const {
    return "Lower (some)AISLE ops to AISMEM dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  AISLEToAISMEMLoweringPass() = default;

  void runOnOperation();

public:
  // Some ops (RNN ops for example) are lowered to other ONNX ops such as
  // ONNXMatMulOp, ONNXSplitOp, ONNXTransposeOp, etc. These ONNX ops are then
  // lowered into krnl ops in this pass.
  //
  // To write LIT tests for operations that are lowered to other ONNX
  // operations, we do not need to check the final generated krnl code (which is
  // lengthy). It is more convenient to check the intermediate generated code
  // including ONNX ops. We trust the lowering of the other ONNX ops.
  //
  // This flag is used in LIT tests to stop the lowering of the other ONNX ops.
  // Usage: onnx-mlir-opt --convert-onnx-to-krnl='emit-intermediate-ir'
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
  // Needed to support unsigned int computations. To be removed if we use a
  // scheme that does not rely on the UnrealizedConversionCastOp.
  // target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
  // Make ONNXNoneOp legal so that other ONNX ops can use it during the
  // lowering. ONNXNoneOp will be dangling and removed by calling
  // canonicalization after the lowering.
  // target.addLegalOp<::mlir::ONNXNoneOp>();
//  target.addLegalOp<::spade::AISLEQConstantOp>();
  // target.addIllegalOp<::conti::AISLEConvOp>();
  // target.addIllegalOp<::conti::AISLEReluOp>();
  // target.addIllegalOp<::conti::AISLEMaxPoolOp>();

  /*
    if (emitIntermediateIR) {
      // Only used for writing LIT tests for ONNX operations that are lowered to
      // other ONNX operations. The following operations are prevented from
    being
      // lowered further. See the comment in the declaration of
      // 'emitIntermediateIR' for more details.
      target.addLegalOp<ONNXMatMulOp>();
      target.addLegalOp<ONNXReshapeOp>();
      target.addLegalOp<ONNXSplitV11Op>();
      target.addLegalOp<ONNXSqueezeV11Op>();
      target.addLegalOp<ONNXTransposeOp>();
    }
  */
  // Conversion target for accelerators.
  // for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
  //  accel->conversionTargetONNXToKrnl(target);

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

 // Create a TypeConverter
  TypeConverter spadeTypeConverter;

  // Add source materialization
  spadeTypeConverter.addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                                  ValueRange inputs, Location loc) -> optional<Value> {
    if (inputs.size() != 1)
      return nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
  });

  // Add target materialization
  spadeTypeConverter.addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                                  ValueRange inputs, Location loc) -> optional<Value> {
    if (inputs.size() != 1)
      return nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
  });

  // Define patterns.
  populateAISLEToAISMEMConversionPattern(
      patterns, spadeTypeConverter, &getContext());

  // Rewrite patterns for accelerators.
  // for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
  //  accel->rewritePatternONNXToKrnl(patterns, krnlTypeConverter,
  //  &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToAISMEMPass() {
  return std::make_unique<AISLEToAISMEMLoweringPass>();
}

} // namespace spade