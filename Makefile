ROOT_DIR := $(shell git rev-parse --show-toplevel)/../..

num_cores         := $(shell nproc)
num_cores_half    := $(shell echo "$$(($(num_cores) / 2))")
num_cores_quarter := $(shell echo "$$(($(num_cores) / 4))")

INSTALL_PREFIX          ?= install
INSTALL_DIR             ?= ${ROOT_DIR}/${INSTALL_PREFIX}
LLVM_INSTALL_DIR        ?= ${INSTALL_DIR}/riscv-llvm
ONNX_INSTALL_DIR        ?= ${INSTALL_DIR}/onnx-mlir
PROTOC_INSTALL_DIR      ?= ${INSTALL_DIR}/protoc
CMAKE_INSTALL_DIR       ?= ${INSTALL_DIR}/cmake
MLIR_DIR                ?= ${ROOT_DIR}/toolchain/riscv-llvm/build/lib/cmake/mlir
PROTOC_DIR              ?= ${PROTOC_INSTALL_DIR}/bin

CC  := clang-10
CXX := clang++-10

CMAKE ?=  $(CMAKE_INSTALL_DIR)/bin/cmake




#all: toolchain-llvm-main

toolchain-onnx-mlir: Makefile
	export PATH=$(PROTOC_DIR):$(PATH) && \
	cd $(ROOT_DIR)/toolchain/onnx-mlir && rm -rf build && mkdir -p build && cd build && \
	$(CMAKE)   \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
	-DONNX_MLIR_BUILD_TESTS=OFF \
	-DONNX_MLIR_ACCELERATORS=OFF \
	-DONNX_MLIR_ENABLE_STABLEHLO=OFF \
	-DCMAKE_C_COMPILER=$(CC) \
	-DCMAKE_CXX_COMPILER=$(CXX) \
	-DCMAKE_INSTALL_PREFIX=$(ONNX_INSTALL_DIR) \
	-DMLIR_DIR=${MLIR_DIR} \
	-DCMAKE_BUILD_TYPE="Debug" \
	..
	cd $(ROOT_DIR)/toolchain/onnx-mlir && \
	$(CMAKE) --build build --target onnx-mlir -j$(num_cores_half)

onnx-mlir:
	$(CMAKE) --build build --target onnx-mlir -j$(num_cores_half)

.PHONY: test test-clean
test:
	make ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/apps/ResNet-18/aida_48_48 graph.test
test-clean:
	make ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/apps/ResNet-18/aida_48_48 clean
	