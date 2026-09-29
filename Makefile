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
MLIR_DIR                ?= ${LLVM_INSTALL_DIR}/lib/cmake/mlir
PROTOC_DIR              ?= ${PROTOC_INSTALL_DIR}/bin
export PATH             := $(PROTOC_DIR):$(PATH) 
CC  := clang
CXX := clang++

CMAKE ?=  cmake

ONNX_MLIR_BUILD_TYPE    ?= "Debug"
ONNX_MLIR_CMAKE_TARGET  ?= onnx-mlir
# Executables `make install` copies into $(ONNX_INSTALL_DIR)/bin.  Only these
# are installed: `cmake --install` would also want every library and the
# Python modules, which the ISOLDE flow never builds.
ONNX_MLIR_INSTALL_BINS  ?= onnx-mlir onnx-mlir-opt
# onnx-mlir puts its executables in build/<build type>/bin.
ONNX_MLIR_BIN_DIR       ?= build/$(strip $(subst ",,$(ONNX_MLIR_BUILD_TYPE)))/bin

.PHONY: compiler
compiler:
	$(CMAKE) --build build --target $(ONNX_MLIR_CMAKE_TARGET) -j$(num_cores_half)


config: 
	rm -rf build && mkdir -p build && cd build && \
	$(CMAKE)   \
	-DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
	-DONNX_MLIR_BUILD_TESTS=OFF \
	-DONNX_MLIR_ACCELERATORS=OFF \
	-DONNX_MLIR_ENABLE_STABLEHLO=OFF \
	-DCMAKE_C_COMPILER=$(CC) \
	-DCMAKE_CXX_COMPILER=$(CXX) \
	-DCMAKE_CXX_FLAGS="-include cstdint" \
	-DCMAKE_INSTALL_PREFIX=$(ONNX_INSTALL_DIR) \
	-DMLIR_DIR=${MLIR_DIR} \
	-DCMAKE_BUILD_TYPE=$(ONNX_MLIR_BUILD_TYPE) \
	..


toolchain-onnx-mlir: config
	cd $(ROOT_DIR)/toolchain/onnx-mlir && \
	$(CMAKE) --build build --target $(ONNX_MLIR_CMAKE_TARGET) -j$(num_cores_half)

## (re)build the installed executables incrementally and copy them into
## $(ONNX_INSTALL_DIR)/bin (install/onnx-mlir of the ibex repo links there);
## run from the onnx-mlir checkout, after `make config` once
.PHONY: install
install:
	@test -d build || (echo "No build/ here: run make config first"; exit 1)
	$(CMAKE) --build build --target $(ONNX_MLIR_INSTALL_BINS) -j$(num_cores_half)
	mkdir -p $(ONNX_INSTALL_DIR)/bin
	for bin in $(ONNX_MLIR_INSTALL_BINS); do \
	  install -m 755 $(ONNX_MLIR_BIN_DIR)/$$bin $(ONNX_INSTALL_DIR)/bin/$$bin || exit 1; \
	done
	@echo "installed from $$(git rev-parse --short HEAD) ($$(git rev-parse --abbrev-ref HEAD)) into $(ONNX_INSTALL_DIR)/bin:"
	@ls -l $(addprefix $(ONNX_INSTALL_DIR)/bin/,$(ONNX_MLIR_INSTALL_BINS))
	@if strings $(ONNX_INSTALL_DIR)/bin/onnx-mlir | grep -q aisle-tile; then \
	  echo "onnx-mlir has the RedMulE tiling (aisle-tile)"; \
	else \
	  echo "WARNING: onnx-mlir has no aisle-tile pass (branch without the ISOLDE tiling patches?)"; \
	fi

.PHONY: test test-clean
test:
#	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.onnx 
#	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.aisle
#	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.aismem
	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.aisllvmir
#	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.aisllvm

test-all:
	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.onnx 
	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.aisle
	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.aismem
	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.aisllvmir
#	make  ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm graph.test.aisllvm

test-clean:
	make ROOT_DIR=$(ROOT_DIR) -C $(ROOT_DIR)/toolchain/onnx-mlir/test-isolde/gemm clean
	