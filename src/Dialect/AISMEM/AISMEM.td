// SPDX-License-Identifier: Apache-2.0

//===--- AISMEM.td -- AISMEM Dialect Operation Definitions ----*- tablegen -===//
//
// Copyleft
//
// =============================================================================
//
// Defines AISMEM Dialect Definitions, Types, and Operations.
//
//
// After changes that impact AISMEM, run "make OMAISMEMOpsIncTranslation".
// After changes that impact the documentation of the ops, run
// "make onnx-mlir-docs".
//
//===----------------------------------------------------------------------===//

#ifndef AISMEMREF_OPS
#define AISMEMREF_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/Interface/ResultTypeInferenceOpInterface.td"
include "src/Interface/HasOnnxSubgraphOpInterface.td"
include "src/IR/AttrBase.td"


//===----------------------------------------------------------------------===//
// Definition of the AISMEM dialect.
//===----------------------------------------------------------------------===//

def AISMEM_Dialect : Dialect {
  let name = "aismem";
  let summary = "A intermediate-level dialect for the AISMEM specification";
  let cppNamespace = "::spade";
  //let useDefaultTypePrinterParser = 1;
  //let useDefaultAttributePrinterParser = 1;
    let dependentDialects = [
    "memref::MemRefDialect"
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
// Base class for ONNX dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class AISMEM_Op<string mnemonic, list<Trait> traits = []> :
  Op<AISMEM_Dialect, mnemonic, traits> ;

def AISMEMConvOp:AISMEM_Op<"Conv",
  [Pure]> {
  let summary = "AISMEM Conv operation";
  let description = [{
  The convolution operator consumes an input tensor and a filter, and
  computes the output.
  Y= conv(X,W)
  }];
  let arguments = (ins 
    Arg<AnyMemRef, "input", [MemRead]>:$X,
    Arg<AnyMemRef, "[N,IH,IW,IC]", [MemRead]>:$X_shape,
    Arg<AnyMemRef, "weight", [MemRead]>:$W,
    Arg<AnyMemRef, "[OC,KH,KW,IC]", [MemRead]>:$W_shape,
    Arg<AnyMemRef, "[pad_top, pad_bottom, pad_left, pad_right]", [MemRead]>:$padding,
    Arg<AnyMemRef, "[stride_y, stride_x, dilation_y, dilation_x]", [MemRead]>:$stride_dilation,
    Arg<AnyMemRef, "output", [MemWrite]>:$Y
    );
  let results = (outs NoneType:$none_val);
  //let hasVerifier = 1;
}

def AISMEMConvExOp:AISMEM_Op<"ConvEx",
  [Pure]> {
  let summary = "AISMEM ConvEx operation";
  let description = [{
  The convolution operator consumes an input tensor and a filter, and
  computes the output.
  Y= conv(X,W)
  }];
  let arguments = (ins 
    Arg<AnyMemRef, "input", [MemRead]>:$X,
    Arg<AnyMemRef, "[N,IH,IW,IC]", [MemRead]>:$X_shape,
    Arg<AnyMemRef, "weight", [MemRead]>:$W,
    Arg<AnyMemRef, "[OC,KH,KW,IC]", [MemRead]>:$W_shape,
    Arg<AnyMemRef, "[pad_top, pad_bottom, pad_left, pad_right]", [MemRead]>:$padding,
    Arg<AnyMemRef, "[stride_y, stride_x, dilation_y, dilation_x]", [MemRead]>:$stride_dilation,
    Arg<AnyMemRef, "bias", [MemRead]>:$B,
    Arg<AnyMemRef, "output", [MemWrite]>:$Y
    );
  let results = (outs NoneType:$none_val);
  //let hasVerifier = 1;
}
//Relu
def AISMEMReluOp:AISMEM_Op<"Relu",
  [Pure]> {
  let summary = "AISMEM Relu operation";
  let description = [{
    rectified linear function
   Relu(input,io_shape,output)
  }];
  let arguments = (ins 
    Arg<AnyMemRef, "input", [MemRead]>:$X,
    Arg<AnyMemRef, "io_shape", [MemRead]>:$X_shape,
    Arg<AnyMemRef, "output", [MemWrite]>:$Y
    );
  let results = (outs NoneType:$none_val);
  //let hasVerifier = 1;
}


def AISMEMMaxPoolOp:AISMEM_Op<"MaxPool",
  [Pure]> {
  let summary = "AISMEM MaxPool operation";
  let description = [{
   Max pooling consisting of computing the max on all values of a subset of the input tensor according
   to the kernel size and downsampling the data into the output tensor Y for further processing.
  }];
  let arguments = (ins 
    Arg<AnyMemRef, "input", [MemRead]>:$X,
    Arg<AnyMemRef, "[N,IH,IW,IC]", [MemRead]>:$X_shape,
    Arg<AnyMemRef, "[K1,K2,S1,S2]", [MemRead]>:$KernelStrides,
    Arg<AnyMemRef, "padding", [MemRead]>:$Padding,
    Arg<AnyMemRef, "output", [MemWrite]>:$Y
    );
  let results = (outs NoneType:$none_val);
  //let hasVerifier = 1;
}

//Reshape
def AISMEMReshapeOp: AISMEM_Op<"Reshape", [Pure]> {
  let summary = "AISMEM Reshape Operator";
  let description = [{
      Reshape the input tensor similar to numpy.reshape. 
      First input is the data tensor, second input is a shape tensor which specifies the output shape. 
      It outputs the reshaped tensor.
  }];
  let arguments = (ins
    Arg<AnyMemRef, "X", [MemRead]>:$X,
    Arg<AnyMemRef, "X_Shape", [MemRead]>:$X_shape,
    Arg<AnyMemRef, "Y_Shape", [MemRead]>:$Y_shape,
    Arg<AnyMemRef, "Y", [MemWrite]>:$Y
  );
  let results = (outs NoneType:$none_val);
}

//Transpose
def AISMEMTransposeOp: AISMEM_Op<"Transpose", [Pure]> {
  let summary = "AISMEM Transpose Operator";
  let description = [{The Transpose operator stores data from input in output, but with the new shape, respectively output_shape}];
  let arguments = (ins
    Arg<AnyMemRef, "X", [MemRead]>:$X,
    Arg<AnyMemRef, "X_Shape", [MemRead]>:$X_shape,
    Arg<AnyMemRef, "Perm", [MemRead]>:$Perm,
    Arg<AnyMemRef, "Y", [MemWrite]>:$Y
  );
  let results = (outs NoneType:$none_val);
}

//MatMul
def AISMEMMatMulOp: AISMEM_Op<"MatMul", [Pure]> {
  let summary = "AISMEM MatMul Operator";
  let description = [{
    Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
    }];
  let arguments = (ins
    Arg<AnyMemRef, "A", [MemRead]>:$A,
    Arg<AnyMemRef, "A_Shape", [MemRead]>:$A_shape,
    Arg<AnyMemRef, "B", [MemRead]>:$B,
    Arg<AnyMemRef, "B_Shape", [MemRead]>:$B_shape,
    Arg<AnyMemRef, "C", [MemWrite]>:$C
  );
  let results = (outs NoneType:$none_val);
}

//GEMM
def AISMEMGEMMOp: AISMEM_Op<"GEMM", [Pure]> {
  let summary = "AISMEM GEMM Operator";
  let description = [{
    General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
    with the following hardcoded attributes
    alpha = 1.0, betha = 1.0
    transA = transB = 0 , i.e. no transpose
    }];
  let arguments = (ins
    Arg<AnyMemRef, "A", [MemRead]>:$A,
    Arg<AnyMemRef, "A_Shape", [MemRead]>:$A_shape,
    Arg<AnyMemRef, "B", [MemRead]>:$B,
    Arg<AnyMemRef, "B_Shape", [MemRead]>:$B_shape,
    Arg<AnyMemRef, "B", [MemRead]>:$C,
    Arg<AnyMemRef, "C", [MemWrite]>:$Y,
    DefaultValuedAttr<SI64Attr, "0">:$transA,
    DefaultValuedAttr<SI64Attr, "0">:$transB
  );
  let results = (outs NoneType:$none_val);
}

//Softmax
def AISMEMSoftmaxOp: AISMEM_Op<"Softmax", [Pure]> {
  let summary = "AISMEM Softmax Operator";
  let description = [{
    The operator computes the normalized exponential values for the given input:
    }];
  let arguments = (ins
    Arg<AnyMemRef, "X", [MemRead]>:$X,
    Arg<AnyMemRef, "X_Shape", [MemRead]>:$X_shape,
    Arg<AnyMemRef, "Y", [MemWrite]>:$Y
  );
  let results = (outs NoneType:$none_val);
}

//Add
def AISMEMAddOp: AISMEM_Op<"Add", [Pure]> {
  let summary = "AISMEM Add Operator";
  let description = [{
    Performs element-wise binary addition (no support for Numpy-style broadcasting support).
    }];
  let arguments = (ins
    Arg<AnyMemRef, "A", [MemRead]>:$A,
    Arg<AnyMemRef, "A_Shape", [MemRead]>:$A_shape,
    Arg<AnyMemRef, "B", [MemRead]>:$B,
    Arg<AnyMemRef, "B_Shape", [MemRead]>:$B_shape,
    Arg<AnyMemRef, "C", [MemWrite]>:$C
  );
  let results = (outs NoneType:$none_val);
}

//ReduceMean
def AISMEMReduceMeanOp: AISMEM_Op<"ReduceMean", [Pure]> {
  let summary = "AISMEM ReduceMean Operator";
  let description = [{
  Computes the mean of the input tensor's element along the provided axes.
    }];
  let arguments = (ins
    Arg<AnyMemRef, "X", [MemRead]>:$X,
    Arg<AnyMemRef, "X_Shape", [MemRead]>:$X_shape,
    Arg<AnyMemRef, "Axes", [MemRead]>:$Axes,
    Arg<AnyMemRef, "Y", [MemWrite]>:$Y
  );
  let results = (outs NoneType:$none_val);
}

def AISMEMQConstantOp : AISMEM_Op< "qconstant", [Pure, MemRefsNormalizable, OpInterface<"mlir::ResultTypeInferenceOpInterface">]> {
  let summary = "AISMEM quad-word constant operation";
  let description = [{
    Operation for holding constant data values. A quadword constant can have a
    meaningful name recorded as its `name` attribute. Its content is stored
    in the `value` dense/opaque element attribute.
  }];

  let arguments = (ins AnyAttr:$shape, StrAttr:$name, AnyAttr:$value);
  let results = (outs   AnyTypeOf<[AnyMemRef]>:$output);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 0;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {-1};
    }

    std::vector<mlir::Type> resultTypeInference() {
      std::vector<mlir::Type> resultTypes;
      if (auto attr = getValueAttr()) {
        resultTypes.push_back(attr.cast<mlir::TypedAttr>().getType());
      } 
      return resultTypes;
    }

    static ::mlir::Attribute  buildShapeAttr(mlir::OpBuilder& b, llvm::ArrayRef<int64_t>&shape){
       return b.getI64ArrayAttr(shape);
    }

    static mlir::Type convertTensorToMemRef(::mlir::Value value) {
      mlir::TensorType type = value.getType().cast<::mlir::TensorType>();
      assert(type.hasRank() && "expected only ranked shapes");
      return mlir::MemRefType::get(type.getShape(), type.getElementType());
    }
    
      }];
   let builders = [
  OpBuilder<(ins  "::mlir::StringAttr&":$name, "::mlir::DenseElementsAttr&":$value), [{
   
    ::llvm::SmallVector<std::int64_t> shape(2);
    shape[0]=1;
    shape[1]=value.size();;
    auto output =   ::mlir::MemRefType::get(llvm::ArrayRef(shape),value.getElementType());
    auto shapeAttr = $_builder.getI64ArrayAttr(::llvm::ArrayRef(shape));
    build($_builder, $_state, output, shapeAttr,name,value);
   
  }]>,
  OpBuilder<(ins   "spade::AISLEQConstantOp&":$op), [{
    ::mlir::DenseElementsAttr value = op.getValueAttr().cast<::mlir::DenseElementsAttr>();
    ::llvm::SmallVector<std::int64_t> shape(2);
    shape[0]=1;
    shape[1]=value.size();;
    auto output =   ::mlir::MemRefType::get(llvm::ArrayRef(shape),value.getElementType());
    build($_builder, $_state, output, op.getShapeAttr(),op.getNameAttr(),op.getValueAttr());
  }]>
  ];
}

#endif // AISMEMREF_OPS