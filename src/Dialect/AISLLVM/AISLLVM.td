// SPDX-License-Identifier: Apache-2.0

//===--- AISLLVM.td -- AISLLVM Dialect Operation Definitions ----*- tablegen -===//
//
// Copyleft
//
// =============================================================================
//
// Defines AISLLVM Dialect Definitions, Types, and Operations.
//
//
// After changes that impact the documentation of the ops, run
// "make onnx-mlir-docs".
//
//===----------------------------------------------------------------------===//

#ifndef AISLLVM_OPS
#define AISLLVM_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/Interface/ResultTypeInferenceOpInterface.td"
include "src/Interface/HasOnnxSubgraphOpInterface.td"
include "src/IR/AttrBase.td"

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
//include "mlir/Dialect/LLVMIR/LLVMIntrinsicOps.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
//===----------------------------------------------------------------------===//
// Definition of the AISLLVM dialect.
//===----------------------------------------------------------------------===//

def AISLLVM_Dialect : Dialect {
  let name = "aisllvm";
  let summary = "A intermediate-level dialect for the AISLLVM specification";
  let cppNamespace = "::spade";
  //let useDefaultTypePrinterParser = 1;
  //let useDefaultAttributePrinterParser = 1;
    let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
  let extraClassDeclaration = [{
      private:
    // Register the builtin Attributes.
    void registerAttributes();
    // Register the builtin Types.
    void registerTypes();
  public:
  }];
}



//===----------------------------------------------------------------------===//
//  Operations
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// AISLLVM  op definitions
//===----------------------------------------------------------------------===//

class AISLLVM_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<AISLLVM_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// AISLLVM intrinsic operations
//===----------------------------------------------------------------------===//


class AISLLVM_IntrOp<string mnem, list<Trait> traits,
                  int numResults>
  : LLVM_IntrOpBase<AISLLVM_Dialect, mnem, "spade_" # !subst(".", "_", mnem),
                    /*list<int> overloadedResults=*/[],
                    /*list<int> overloadedOperands=*/[],
                    traits, numResults>;


// 26.06.2024 droped in favor of mlir::VectorType
//def QConstant: LLVMType<"LLVMFixedVector", "vec">{}
def QConstant: VectorOfRankAndType<[1], [I32]>;

 //GEMM
// 26.06.2024 it should be an intrinsic, however, the llvm is not updated yet 
//below a workaround to test the lowering 
//def AISLLVMGEMMOp: AISLLVM_IntrOp< "spade.gemmf32",[],1>{ 
def AISLLVMGEMMOp: AISLLVM_Op< "spade.gemm">{ 
  let arguments = (ins
    LLVMPointerType: $Y,
    LLVMPointerType: $A,
    QConstant: $A_shape,
    LLVMPointerType: $B,
    QConstant: $B_shape,
    LLVMPointerType: $C,
    I32: $transA,
    I32: $transB
  );
  let results = (outs  QConstant: $res);    
} 



#endif // AISLLVM_OPS