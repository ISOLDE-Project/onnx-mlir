// SPDX-License-Identifier: Apache-2.0

//===--- AISLE.td -- AISLE Dialect Operation Definitions ----*- tablegen -===//
//
// Copyleft
//
// =============================================================================
//
// Defines AISLE Dialect Definitions, Types, and Operations.
//
//
// After changes that impact the documentation of the ops, run
// "make onnx-mlir-docs".
//
//===----------------------------------------------------------------------===//

#ifndef AISLE_OPS
#define AISLE_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/Interface/ResultTypeInferenceOpInterface.td"
include "src/Interface/HasOnnxSubgraphOpInterface.td"
include "src/IR/AttrBase.td"

//===----------------------------------------------------------------------===//
// Definition of the AISLE dialect.
//===----------------------------------------------------------------------===//

def AISLE_Dialect : Dialect {
  let name = "aisle";
  let summary = "A high-level dialect for the AISLE specification";
  let cppNamespace = "spade";
  //let useDefaultTypePrinterParser = 1;
  //let useDefaultAttributePrinterParser = 1;
  let dependentDialects = ["func::FuncDialect"];
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

// Base class for AISLE dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class AISLE_Op<string mnemonic, list<Trait> traits = []> :
  Op<AISLE_Dialect, mnemonic, traits> ;

include "mlir/Interfaces/SideEffectInterfaces.td"

def AISLEConvOp:AISLE_Op<"Conv",
  [Pure]> {
  let summary = "AISLE Conv operation";
  let description = [{
  The convolution operator consumes an input tensor and a filter, and
  computes the output.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I32]>:$X_shape,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
    TensorOf<[I32]>:$W_shape,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
    TensorOf<[I32]>:$padding,
    TensorOf<[I32]>:$stride_dilation
    );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 7;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
  //let hasVerifier = 1;
}

//Reshape
def AISLEReshapeOp: AISLE_Op<"Reshape", [Pure]> {
  let summary = "AISLE Reshape Operator";
  let description = [{
      Reshape the input tensor similar to numpy.reshape. 
      First input is the data tensor, second input is a shape tensor which specifies the output shape. 
      It outputs the reshaped tensor.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I32]>:$X_shape,
    TensorOf<[I32]>:$Y_shape
  );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 3;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

//Relu
def AISLEReluOp:AISLE_Op<"Relu",
  [Pure]> {
  let summary = "AISLE Relu operation";
  let description = [{
      Relu takes one input data (Tensor) and produces one output data (Tensor) where 
      the rectified linear function, y = max(0, x), is applied to the tensor elementwise.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I32]>:$X_shape
    );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
  //let hasVerifier = 1;
}

//MaxPool
def AISLEMaxPoolOp:AISLE_Op<"MaxPool",
  [Pure]> {
  let summary = "AISLE MaxPool operation";
  let description = [{
  Max pooling consisting of computing the max on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I32]>:$X_shape,
    TensorOf<[I32]>:$KernelStrides,
    TensorOf<[I32]>:$Padding
    );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 4;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
  //let hasVerifier = 1;
}


//Transpose
def AISLETransposeOp: AISLE_Op<"Transpose", [Pure]> {
  let summary = "AISLE Transpose Operator";
  let description = [{The Transpose operator stores data from input in output, but with the new shape, respectively output_shape}];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I32]>:$X_shape,
    TensorOf<[I32]>:$Perm
  );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 3;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

//MatMul
def AISLEMatMulOp: AISLE_Op<"MatMul", [Pure]> {
  let summary = "AISLE MatMul Operator";
  let description = [{
    Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
    }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
    TensorOf<[I32]>:$A_shape,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
    TensorOf<[I32]>:$B_shape
  );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 4;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

//GEMM
def AISLEGEMMOp: AISLE_Op<"GEMM", [Pure]> {
  let summary = "AISLE GEMM Operator";
  let description = [{
    General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
    with the following hardcoded attributes
    alpha = 1.0, betha = 1.0
    transA = transB = 0 , i.e. no transpose
    }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
    TensorOf<[I32]>:$A_shape,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
    TensorOf<[I32]>:$B_shape,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C,
    DefaultValuedAttr<SI64Attr, "0">:$transA,
    DefaultValuedAttr<SI64Attr, "0">:$transB
  );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 5;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}
//Softmax
def AISLESoftmaxOp: AISLE_Op<"Softmax", [Pure]> {
  let summary = "AISLE Softmax Operator";
  let description = [{
    The operator computes the normalized exponential values for the given input:
    }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I32]>:$X_shape
  );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

//Add
def AISLEAddOp:AISLE_Op<"Add",
  [Pure]> {
  let summary = "AISLE Add operation";
  let description = [{
  Performs element-wise binary addition (no support for Numpy-style broadcasting support).
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
    TensorOf<[I32]>:$A_shape,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
    TensorOf<[I32]>:$B_shape
    );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  
    let extraClassDeclaration = [{
      static int getNumberOfOperands() {
        return 2;
      }
      static int getNumberOfResults() {
        return 1;
      }
      static std::vector<int> getTypeMap() {
        return {20};
      }
    }];

}

//ReduceMean
def AISLEReduceMeanOp:AISLE_Op<"ReduceMean",
  [Pure]> {
  let summary = "AISLE ReduceMean operation";
  let description = [{
  Computes the mean of the input tensor's element along the provided axes.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I32]>:$X_shape,
    TensorOf<[I32]>:$Axes
    );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  
    let extraClassDeclaration = [{
      static int getNumberOfOperands() {
        return 1;
      }
      static int getNumberOfResults() {
        return 1;
      }
      static std::vector<int> getTypeMap() {
        return {20};
      }
    }];
}

//hstack
def AISLEhstack:AISLE_Op<"hstack",
  [Pure]> {
  let summary = "AISLE hstack operation";
  let description = [{
  Stack arrays in sequence horizontally (column wise). Functional equivalent to numpy.hstack, see https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
        AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B
    );
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  
    let extraClassDeclaration = [{
      static int getNumberOfOperands() {
        return 2;
      }
      static int getNumberOfResults() {
        return 1;
      }
      static std::vector<int> getTypeMap() {
        return {20};
      }
    }];
}


//const
def AISLEQConstantOp : AISLE_Op< "qconstant",  [Pure, OpInterface<"mlir::ResultTypeInferenceOpInterface">] > {
  let summary = "AISLE constant quadword operation";
  let description = [{
    Operation for holding constant data values. A quadword constant can have a
    meaningful name recorded as its `name` attribute. Its content is stored
    in the `value` dense/opaque element attribute.
  }];

  let arguments = (ins AnyAttr:$shape, StrAttr:$name, AnyAttr:$value);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]> ]>:$output);
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
    static mlir::Type buildOutputType( mlir::Type& elemType,llvm::ArrayRef<int64_t>&shape){
      return mlir::RankedTensorType::get(shape,elemType);
    }
      }];
   let builders = [
  OpBuilder<(ins  "const char*":$name, "::mlir::DenseElementsAttr":$value), [{
   
    ::llvm::SmallVector<std::int64_t> shape(2);
    shape[0]=1;
    shape[1]=value.size();;
    auto output= ::mlir::RankedTensorType::get(llvm::ArrayRef(shape),value.getElementType());
    auto shapeAttr = $_builder.getI64ArrayAttr(::llvm::ArrayRef(shape));
    auto nameAtr =$_builder.getStringAttr(name);
    build($_builder, $_state, output, shapeAttr,nameAtr,value);
   
  }]>
  ];
}

#endif // AISLE_OPS