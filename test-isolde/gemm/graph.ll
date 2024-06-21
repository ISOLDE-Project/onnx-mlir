; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"

@_entry_point_1_graph = constant [21 x i8] c"run_main_graph_graph\00"
@_entry_point_1_in_sig_graph = constant [61 x i8] c"[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8] , \22name\22 : \22A\22 }\0A\0A]\00"
@_entry_point_1_out_sig_graph = constant [61 x i8] c"[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22Y\22 }\0A\0A]\00"
@_entry_point_0_graph = constant [15 x i8] c"run_main_graph\00"
@_entry_point_0_in_sig_graph = constant [61 x i8] c"[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8] , \22name\22 : \22A\22 }\0A\0A]\00"
@_entry_point_0_out_sig_graph = constant [61 x i8] c"[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22Y\22 }\0A\0A]\00"
@constant_0_graph = internal constant [1 x [10 x float]] [[10 x float] [float 0x3FDD937340000000, float 0x3FDDD84A20000000, float 0x3FD771F220000000, float 0x3FB7C9BF60000000, float 0x3FDDDBF7A0000000, float 0x3FE233EE00000000, float 0x3FE56AD480000000, float 0x3FEC150FE0000000, float 0x3FEA42C4C0000000, float 0x3FD28538E0000000]], align 16
@_entry_point_arrays_graph = internal constant [3 x ptr] [ptr @_entry_point_0_graph, ptr @_entry_point_1_graph, ptr null]

declare i32 @strncmp(ptr, ptr, i64)

declare void @omGetExternalConstantAddr(ptr, ptr, i64)

declare void @omMMapBinaryFile(ptr, ptr, i64, i64)

declare i64 @omTensorListGetSize(ptr)

declare void @omTensorPrint(ptr, ptr)

declare ptr @omTensorListGetOmtArray(ptr)

declare void @omTensorSetDataType(ptr, i64)

declare i64 @omTensorGetDataType(ptr)

declare ptr @omTensorGetStrides(ptr)

declare ptr @omTensorGetShape(ptr)

declare i64 @omTensorGetRank(ptr)

declare void @omTensorSetDataPtr(ptr, i64, ptr, ptr)

declare ptr @omTensorGetDataPtr(ptr)

declare void @omTensorDestroy(ptr)

declare ptr @omTensorCreateUntyped(i64)

declare ptr @omTensorListCreate(ptr, i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main_graph_graph(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } { ptr @constant_0_graph, ptr @constant_0_graph, i64 0, [2 x i64] [i64 1, i64 10], [2 x i64] [i64 10, i64 1] }
}

define void @_mlir_ciface_main_graph_graph(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 1
  %11 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @main_graph_graph(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, ptr %0, align 8
  ret void
}

define ptr @run_main_graph_graph(ptr %0) {
  %2 = call ptr @omTensorListGetOmtArray(ptr %0)
  %3 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  %4 = load ptr, ptr %2, align 4
  %5 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  %6 = call ptr @omTensorGetDataPtr(ptr %4)
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %6, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, i64 0, 2
  %10 = call ptr @omTensorGetShape(ptr %4)
  %11 = call ptr @omTensorGetStrides(ptr %4)
  %12 = load i64, ptr %10, align 8
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 %12, 3, 0
  %14 = load i64, ptr %11, align 8
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %14, 4, 0
  %16 = getelementptr i64, ptr %10, i32 1
  %17 = load i64, ptr %16, align 8
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 %17, 3, 1
  %19 = getelementptr i64, ptr %11, i32 1
  %20 = load i64, ptr %19, align 8
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %20, 4, 1
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, ptr %5, align 8
  call void @_mlir_ciface_main_graph_graph(ptr %3, ptr %5)
  %22 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %3, align 8
  %23 = alloca ptr, i64 1, align 4
  %24 = call ptr @omTensorCreateUntyped(i64 2)
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 0
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  call void @omTensorSetDataPtr(ptr %24, i64 0, ptr %25, ptr %26)
  call void @omTensorSetDataType(ptr %24, i64 1)
  %27 = call ptr @omTensorGetShape(ptr %24)
  %28 = call ptr @omTensorGetStrides(ptr %24)
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 3, 0
  store i64 %29, ptr %27, align 8
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 4, 0
  store i64 %30, ptr %28, align 8
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 3, 1
  %32 = getelementptr i64, ptr %27, i32 1
  store i64 %31, ptr %32, align 8
  %33 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 4, 1
  %34 = getelementptr i64, ptr %28, i32 1
  store i64 %33, ptr %34, align 8
  store ptr %24, ptr %23, align 4
  %35 = call ptr @omTensorListCreate(ptr %23, i64 1)
  ret ptr %35
}

define ptr @run_main_graph(ptr %0) {
  %2 = call ptr @run_main_graph_graph(ptr %0)
  ret ptr %2
}

define ptr @omQueryEntryPoints_graph(ptr %0) {
  %2 = icmp ne ptr %0, null
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  store i64 2, ptr %0, align 8
  br label %4

4:                                                ; preds = %3, %1
  ret ptr @_entry_point_arrays_graph
}

define ptr @omQueryEntryPoints(ptr %0) {
  %2 = call ptr @omQueryEntryPoints_graph(ptr %0)
  ret ptr %2
}

define ptr @omInputSignature_graph(ptr %0) {
  %2 = call i32 @strncmp(ptr %0, ptr @_entry_point_0_graph, i64 15)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  ret ptr @_entry_point_0_in_sig_graph

5:                                                ; preds = %1
  %6 = call i32 @strncmp(ptr %0, ptr @_entry_point_1_graph, i64 21)
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %5
  ret ptr @_entry_point_1_in_sig_graph

9:                                                ; preds = %5
  ret ptr null
}

define ptr @omInputSignature(ptr %0) {
  %2 = call ptr @omInputSignature_graph(ptr %0)
  ret ptr %2
}

define ptr @omOutputSignature_graph(ptr %0) {
  %2 = call i32 @strncmp(ptr %0, ptr @_entry_point_0_graph, i64 15)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  ret ptr @_entry_point_0_out_sig_graph

5:                                                ; preds = %1
  %6 = call i32 @strncmp(ptr %0, ptr @_entry_point_1_graph, i64 21)
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %5
  ret ptr @_entry_point_1_out_sig_graph

9:                                                ; preds = %5
  ret ptr null
}

define ptr @omOutputSignature(ptr %0) {
  %2 = call ptr @omOutputSignature_graph(ptr %0)
  ret ptr %2
}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"Product Major Version", i32 0}
!2 = !{i32 2, !"Product Minor Version", i32 0}
!3 = !{i32 2, !"Product Patchlevel", i32 0}
!4 = !{i32 2, !"Product Id", !"NOT_SPECIFIED"}
!5 = !{!"onnx-mlir version 0.4.2 (ec3f410a)"}
