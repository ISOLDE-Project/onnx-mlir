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

//ComplexGEMM
// Split-complex matrix multiplication with explicit output buffers.
// Kept as a semantic AISMEM operation for compatibility.  The current
// AISLE->AISMEM lowering expands aisle.ComplexGEMM directly into the scheduled
// RedMulE operations below, so new complex GEMMs do not need to materialize
// this operation.
def AISMEMComplexGEMMOp: AISMEM_Op<"ComplexGEMM", [Pure]> {
  let summary = "AISMEM split-complex GEMM operator";
  let description = [{
    Computes a split-complex matrix multiplication:

      Cr = Ar * Br - Ai * Bi
      Ci = Ar * Bi + Ai * Br

    Ar/Ai and Br/Bi are split real/imaginary input buffers. Cr and Ci are
    explicit output buffers. This operation preserves complex-GEMM semantics
    at the AISMEM level; RedMulE tile assignment, SPM residency, phase
    scheduling, and sign-bit negation are lower-level decisions.
  }];
  let arguments = (ins
    Arg<AnyMemRef, "Ar", [MemRead]>:$Ar,
    Arg<AnyMemRef, "Ar_Shape", [MemRead]>:$Ar_shape,
    Arg<AnyMemRef, "Ai", [MemRead]>:$Ai,
    Arg<AnyMemRef, "Ai_Shape", [MemRead]>:$Ai_shape,
    Arg<AnyMemRef, "Br", [MemRead]>:$Br,
    Arg<AnyMemRef, "Br_Shape", [MemRead]>:$Br_shape,
    Arg<AnyMemRef, "Bi", [MemRead]>:$Bi,
    Arg<AnyMemRef, "Bi_Shape", [MemRead]>:$Bi_shape,
    Arg<AnyMemRef, "Cr", [MemWrite]>:$Cr,
    Arg<AnyMemRef, "Ci", [MemWrite]>:$Ci
  );
  let results = (outs NoneType:$none_val);
}

//===----------------------------------------------------------------------===//
// RedMulE scheduled operations
//===----------------------------------------------------------------------===//

// Return get_addr_start(bank) for one RedMulE tile.  The tile attribute keeps
// TILESEL explicit so AISMEM -> LLVM can emit isolde_set_tile(tile) before
// querying the private SPM address window.
def AISMEMRedMulEAddrStartOp: AISMEM_Op<"RedMulEAddrStart"> {
  let summary = "Get the first private SPM address for a RedMulE tile";
  let arguments = (ins
    I32Attr:$tile,
    DefaultValuedAttr<I32Attr, "0">:$bank
  );
  let results = (outs I32:$address);
}

// Host -> private SPM transfer.  spm_address is the destination address in the
// selected tile's private SPM.  next_address models the address returned by
// spm_write and is used to lay out X, then W, then Y exactly as in the C
// programming model.
//
// negate=true performs raw FP16 sign-bit negation while copying.  This belongs
// on the upload operation (not GEMM), because the reference implementation
// forms -Bi while writing Bi into SPM.
def AISMEMRedMulEUploadOp: AISMEM_Op<"RedMulEUpload", [MemoryEffects<[MemWrite]>]> {
  let summary = "Upload a host matrix into private RedMulE SPM";
  let arguments = (ins
    Arg<AnyMemRef, "source", [MemRead]>:$source,
    Arg<AnyMemRef, "source shape", [MemRead]>:$source_shape,
    I32:$spm_address,
    Variadic<NoneType>:$dependencies,
    I32Attr:$tile,
    DefaultValuedAttr<BoolAttr, "false">:$negate
  );
  let results = (outs
    I32:$next_address,
    NoneType:$none_val
  );
}

// Host -> private SPM transfer of a 2-D window, with the FP16 bit-level
// transforms the transformer schedules need applied on the way in:
//
//   S       = source viewed row-major as [numElements / ld, ld],
//             ld = the innermost static dimension of `source`
//   Win     = S[row_offset : row_offset + rows, col_offset : col_offset + cols]
//   Win'    = transpose ? Win^T : Win
//   dst[i][j] = f(Win'[i][j])  if (i, j) is inside Win', else +0.0
//             for 0 <= i < dst_rows, 0 <= j < dst_cols
//   f(x)    = relu ? (sign(x) ? +0.0 : x) : x, then negate ? -x : x
//
// All transforms act on raw binary16 bits (sign mask / sign flip / moves), so
// no scalar floating point is needed on the core.  dst_cols is one RedMulE row
// (16).  next_address follows the same spm_write convention as RedMulEUpload.
// Typical uses: zero-padded K^T and V for attention, the column tiles of W1
// and row tiles of W2 for the feed-forward block, and the ReLU between two
// GEMMs.
// The SPM is not modelled as a memref, so the write into it is declared on
// the default resource: an upload whose results are unused (a resident
// weight) must not be erased as dead.
def AISMEMRedMulEUploadTileOp: AISMEM_Op<"RedMulEUploadTile",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Upload a transformed 2-D window of a host matrix into private RedMulE SPM";
  let arguments = (ins
    Arg<AnyMemRef, "source", [MemRead]>:$source,
    I32:$spm_address,
    Variadic<NoneType>:$dependencies,
    I32Attr:$tile,
    I32Attr:$row_offset,
    I32Attr:$col_offset,
    I32Attr:$rows,
    I32Attr:$cols,
    I32Attr:$dst_rows,
    DefaultValuedAttr<I32Attr, "16">:$dst_cols,
    DefaultValuedAttr<BoolAttr, "false">:$transpose,
    DefaultValuedAttr<BoolAttr, "false">:$relu,
    DefaultValuedAttr<BoolAttr, "false">:$negate
  );
  let results = (outs
    I32:$next_address,
    NoneType:$none_val
  );
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SPM-resident buffers
//
// Tensors that are produced and consumed by RedMulE live in the tile-private
// scratchpad instead of travelling SPM -> DMEM -> SPM.  A buffer is a range of
// SPM rows (64-byte address stride, 16 binary16 payload values per row).
//
//   aismem.SPMAlloc      symbolic buffer; the `aismem-spm-allocate` pass
//                        assigns `row` from buffer lifetimes
//   aismem.SPMRelu       in-place ReLU (sign-bit mask)
//   aismem.SPMTranspose  zero-padded transpose, SPM -> SPM, same tile
//   aismem.SPMCopy       row copy, SPM -> SPM, same tile
//
// The three transforms run on the core through the narrow SPM window today
// (omrm_spm_*_f16); they are separate ops so a loader-based implementation can
// replace them without touching the schedules.
//===----------------------------------------------------------------------===//

def AISMEMSPMAllocOp: AISMEM_Op<"SPMAlloc"> {
  let summary = "A buffer of `rows` rows in the private SPM of `tile`";
  let description = [{
    Returns the SPM address of the buffer's first row.  Until the
    `aismem-spm-allocate` pass has run, `row` is absent and the buffer is only
    a name; the pass assigns rows from the buffer lifetimes and capacity of
    the tile.  `resident` buffers are filled once from constants by the
    generated `<entry>_preload` function and keep their rows for the whole
    program (weights).  `name` is informational (SPM map, debugging).

    Deliberately not Pure: two identical allocations are two buffers.
  }];
  let arguments = (ins
    I32Attr:$tile,
    I32Attr:$rows,
    OptionalAttr<I32Attr>:$row,
    DefaultValuedAttr<BoolAttr, "false">:$resident,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs I32:$address);
  let hasVerifier = 1;
}

def AISMEMSPMReluOp: AISMEM_Op<"SPMRelu"> {
  let summary = "In-place ReLU on `rows` SPM rows (raw binary16 sign test)";
  let arguments = (ins
    I32:$spm_address,
    Variadic<NoneType>:$dependencies,
    I32Attr:$tile,
    I32Attr:$rows
  );
  let results = (outs NoneType:$none_val);
}

def AISMEMSPMTransposeOp: AISMEM_Op<"SPMTranspose"> {
  let summary = "dst[i][j] = j < rows ? src[j][i] : +0.0, i < dst_rows";
  let description = [{
    Transposes a `rows` x 16 SPM matrix into a zero-padded `dst_rows` x 16 one
    on the same tile (e.g. K[12x16] -> pad16(K^T)[16x16] for Q K^T).
  }];
  let arguments = (ins
    I32:$source,
    I32:$destination,
    Variadic<NoneType>:$dependencies,
    I32Attr:$tile,
    I32Attr:$rows,
    DefaultValuedAttr<I32Attr, "16">:$dst_rows
  );
  let results = (outs NoneType:$none_val);
}

def AISMEMSPMCopyOp: AISMEM_Op<"SPMCopy"> {
  let summary = "Copy `rows` SPM rows on one tile";
  let arguments = (ins
    I32:$source,
    I32:$destination,
    Variadic<NoneType>:$dependencies,
    I32Attr:$tile,
    I32Attr:$rows
  );
  let results = (outs NoneType:$none_val);
}

// Cross-tile transfer: a RedMulE instance only sees its own SPM, so a value
// produced on one tile goes through data memory to reach another.
def AISMEMSPMMoveTileOp: AISMEM_Op<"SPMMoveTile", [MemoryEffects<[MemWrite]>]> {
  let summary = "Move `rows` SPM rows from one tile to another";
  let description = [{
    Reads rows x 16 fp16 at `source` in the SPM of `src_tile` and writes them
    to `destination` in the SPM of `dst_tile` as dst_rows x 16, zero padded;
    with `transpose` the 16 x rows transpose is written instead.  Lowered to
    omrm_spm_move_f16 (SPM -> data-memory staging -> SPM).  The source tile
    must be idle: its dependencies include the wait of the launch that
    produced the source.
  }];
  let arguments = (ins
    I32:$source,
    I32:$destination,
    Variadic<NoneType>:$dependencies,
    I32Attr:$src_tile,
    I32Attr:$dst_tile,
    I32Attr:$rows,
    I32Attr:$dst_rows,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );
  let results = (outs NoneType:$none_val);
}

// Initialize the persistent Y region in private SPM to zero.  This operation
// is emitted only before phase 1.  Phase 2 intentionally omits it so that the
// second GEMM accumulates into the phase-1 Y value.
def AISMEMRedMulEZeroOp: AISMEM_Op<"RedMulEZero"> {
  let summary = "Zero a RedMulE private SPM region";
  let arguments = (ins
    I32:$spm_address,
    Variadic<NoneType>:$dependencies,
    I32Attr:$tile,
    I32Attr:$elements
  );
  let results = (outs NoneType:$none_val);
}

// Launch one asynchronous redmule.gemm using addresses that already refer to
// private SPM.  K/M/N match the redmule_gemm_async programming-model argument
// order and are explicit i32 attributes because they eventually become
// instruction immediates.
def AISMEMRedMulEGEMMOp: AISMEM_Op<"RedMulEGEMM"> {
  let summary = "Launch GEMM from already-populated RedMulE private SPM";
  let arguments = (ins
    I32:$x_spm_address,
    I32:$w_spm_address,
    I32:$y_spm_address,
    Variadic<NoneType>:$dependencies,
    I32Attr:$tile,
    I32Attr:$k,
    I32Attr:$m,
    I32Attr:$n
  );
  let results = (outs NoneType:$none_val);
}

// Barrier for one or more asynchronous RedMulE launches.  mask uses the same
// bit convention as REDMULE_BIT(tile) in the C programming model.
def AISMEMRedMulEWaitOp: AISMEM_Op<"RedMulEWait"> {
  let summary = "Wait for a mask of RedMulE tiles";
  let arguments = (ins
    Variadic<NoneType>:$dependencies,
    I32Attr:$mask
  );
  let results = (outs NoneType:$none_val);
}

// Private SPM -> host transfer.  spm_address is normally the persistent Y
// address established during phase 1.  The dependency is the final wait token,
// preventing the readback from moving before accelerator completion.
def AISMEMRedMulEDownloadOp: AISMEM_Op<"RedMulEDownload"> {
  let summary = "Download a RedMulE private SPM region to host memory";
  let description = [{
    Copies elements / cols SPM rows starting at `spm_address`: the first
    `cols` fp16 of SPM row r go to destination elements
    [dst_offset + r * dst_ld, dst_offset + r * dst_ld + cols).  With the
    defaults (cols 16, dst_offset 0, dst_ld 16) that is a plain contiguous
    copy of whole rows; a tile of a wider result uses dst_offset = its first
    column and dst_ld = the row length; a result narrower than 16 (or with
    fewer than 12 rows) copies just its cols (rows).
  }];
  let arguments = (ins
    I32:$spm_address,
    Arg<AnyMemRef, "destination", [MemWrite]>:$destination,
    NoneType:$dependency,
    I32Attr:$tile,
    I32Attr:$elements,
    DefaultValuedAttr<I32Attr, "0">:$dst_offset,
    DefaultValuedAttr<I32Attr, "16">:$dst_ld,
    DefaultValuedAttr<I32Attr, "16">:$cols
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