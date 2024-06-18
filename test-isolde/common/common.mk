


ifndef ROOT_DIR
$(error ROOT_DIR is not defined. Please execute'source eth.sh' from top folder.)
endif
INSTALL_PREFIX          ?= install
LLVM_INSTALL_DIR ?=$(ROOT_DIR)/$(INSTALL_PREFIX)/riscv-llvm
LLVM_INCLUDE_DIR ?=$(ROOT_DIR)/$(INSTALL_PREFIX)/riscv-llvm






ECHO      = /usr/bin/echo

RISCV_XLEN    ?= 32
RISCV_ARCH    ?= rv$(RISCV_XLEN)gcv
RISCV_ABI     ?= ilp32
RISCV_TARGET  ?= riscv$(RISCV_XLEN)-unknown-elf

# Use LLVM
RISCV_PREFIX  ?= $(LLVM_INSTALL_DIR)/bin/
RISCV_CC      ?= $(RISCV_PREFIX)clang
RISCV_CXX     ?= $(RISCV_PREFIX)clang++
RISCV_LLC     ?= $(RISCV_PREFIX)llc
RISCV_OBJDUMP ?= $(RISCV_PREFIX)llvm-objdump
RISCV_OBJCOPY ?= $(RISCV_PREFIX)llvm-objcopy
RISCV_AS      ?= $(RISCV_PREFIX)llvm-as
RISCV_AR      ?= $(RISCV_PREFIX)llvm-ar
RISCV_LD      ?= $(RISCV_PREFIX)ld.lld
RISCV_STRIP   ?= $(RISCV_PREFIX)llvm-strip
#
ifdef RISCV_GCC

	CC      := $(PULP_RISCV_GCC_TOOLCHAIN)/bin/riscv32-unknown-elf-gcc
	CXX     := $(PULP_RISCV_GCC_TOOLCHAIN)/bin/riscv32-unknown-elf-g++
	LD      := $(PULP_RISCV_GCC_TOOLCHAIN)/bin/riscv32-unknown-elf-ld
	AR      := $(PULP_RISCV_GCC_TOOLCHAIN)/bin/riscv32-unknown-elf-ar
	OBJDUMP := $(PULP_RISCV_GCC_TOOLCHAIN)/bin/riscv32-unknown-elf-objdump
else
	CC      := $(RISCV_CC)
	CXX     := $(RISCV_CXX)
	LD      := $(RISCV_LD)
	AR      := $(RISCV_AR)
	OBJDUMP := $(RISCV_OBJDUMP)
endif
# Common flags
RISCV_WARNINGS += -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument # -Werror
 
# LLVM Flags
LLVM_INCLUDES  ?= $(LLVM_INCLUDE_DIR)/riscv$(RISCV_XLEN)-unknown-elf/include
LLVM_LIBS      ?= $(LLVM_INSTALL_DIR)/riscv$(RISCV_XLEN)-unknown-elf/lib
LLVM_RT_LIBS   ?= $(LLVM_INSTALL_DIR)/lib/linux
LLVM_FLAGS     ?= --target=riscv32 -march=rv32gv  -menable-experimental-extensions -mabi=$(RISCV_ABI) -mno-relax -fuse-ld=lld
GCC_FLAGS      ?=  -march=rv32gv   -mabi=$(RISCV_ABI) -mno-relax -fuse-ld=lld
#LLVM_V_FLAGS   ?= -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -Xclang -target-feature -Xclang +no-optimized-zero-stride-load
RISCV_FLAGS    ?= $(GCC_FLAGS) $(LLVM_V_FLAGS) -mcmodel=medany  -O0 -ffast-math  -g  $(DEFINES) $(RISCV_WARNINGS)
RISCV_CCFLAGS  ?= $(RISCV_FLAGS) -std=gnu99  -ffunction-sections -fdata-sections
RISCV_CCFLAGS_SPIKE  ?= $(RISCV_FLAGS) $(SPIKE_CCFLAGS) -ffunction-sections -fdata-sections
RISCV_CXXFLAGS ?= $(GCC_FLAGS) -ffunction-sections -fdata-sections
#RISCV_LDFLAGS  ?= -static -L$(LLVM_LIBS) -L$(LLVM_RT_LIBS) -lc -lgloss -lclang_rt.builtins-riscv32
RISCV_LDFLAGS  ?= -static -L$(LLVM_LIBS) -L$(LLVM_RT_LIBS)   
LD_SCRIPT      ?= -T link.ld
LLC_FLAGS      ?=  -mtriple=riscv32 -mattr=+v -target-abi=ilp32 

RISCV_OBJDUMP_FLAGS ?= --mattr=v
OBJDUMP_FLAGS ?= 
RUNTIME_LLVM  ?= crt0-llvm.S.o 




#ONNX_INSTALL_DIR        ?= ${ROOT_DIR}/${INSTALL_PREFIX}/onnx-mlir
ONNX_INSTALL_DIR        ?= ${ROOT_DIR}/toolchain/onnx-mlir/build/Debug
TOOLS_INSTALL_DIR       ?= ${ROOT_DIR}/install/onnx-mlir/py-codegen
EXPORT_ELF              ?= ${ROOT_DIR}/HLS/aida/build/bin/export_elf



%.cpp.o : ../models/%.cpp
	$(CXX) -c    -march=rv32gv -Iinclude -I. -I$(LLVM_INCLUDES) $(RISCV_CXXFLAGS) -o  $(^F).o   $<


%.cpp.o : ../src/%.cpp
	$(CXX) -c    -march=rv32gv -Iinclude -I. -I$(LLVM_INCLUDES) $(RISCV_CXXFLAGS) -o  $(^F).o   $<
#	$(CXX) -cc1  -target-feature +v  -S -O0  -emit-llvm  -I. -I$(LLVM_INCLUDES)  -o  $(^F).ll   $<
#	$(CXX) -cc1  -target-feature +v -S -O0  -Iinclude -I. -I$(LLVM_INCLUDES)     -o  $(^F).S    $<

%.cpp.o : %.cpp
	$(RISCV_CXX) -c    -march=rv32gv -Iinclude -I. -I$(LLVM_INCLUDES) $(RISCV_CXXFLAGS) -o  $(^F).o   $<
#	$(RISCV_CXX) -cc1  -target-feature +v  -S -O0  -emit-llvm  -I. -I$(LLVM_INCLUDES)  -o  $(^F).ll   $<
#	$(RISCV_CXX) -cc1  -target-feature +v -S -O0  -Iinclude -I. -I$(LLVM_INCLUDES)     -o  $(^F).S    $<

%.c.o : ../src/%.c
	$(RISCV_CC) -c    -Iinclude -I. -I$(LLVM_INCLUDES) $(RISCV_CCFLAGS) -o  $(^F).o   $<

%.c.o : ../../common/%.c
	$(CC) -c    -Iinclude -I. -I$(LLVM_INCLUDES) $(RISCV_CCFLAGS) -o  $(^F).o   $<

%.cpp.o : ../../common/%.cpp
	$(RISCV_CXX) -c    -march=rv32gv -Iinclude -I. -I$(LLVM_INCLUDES) $(RISCV_CXXFLAGS) -o  $(^F).o   $<
	



libsim.a : startup.c.o 
	$(AR) rcs $@ $^



graph :  $(ONNX_MODEL) 
	@echo +++ $(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitObj -o $@  $<
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitObj -o $@  $<
	@echo +++ $(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitLLVMIR -o $@  $<
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitLLVMIR -o $@  $<
	@echo +++ $(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitONNXIR -o $@  $<
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitONNXIR -o $@  $<
graph.test :  $(ONNX_MODEL) 	
	@echo +++ $(ONNX_INSTALL_DIR)/bin/onnx-mlir 
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitONNXIR -o graph  $<
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitMLIR   -o graph  $<
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitLLVM   -o graph  $<

graph.test.aisle :  $(ONNX_MODEL) 	
	@echo +++ $(ONNX_INSTALL_DIR)/bin/onnx-mlir 
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitSPADEIR -o graph  $<

graph.test.aismem :  $(ONNX_MODEL) 	
	@echo +++ $(ONNX_INSTALL_DIR)/bin/onnx-mlir 
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitSPADEMLIR -o graph  $<


graph.test.onnx :  $(ONNX_MODEL) 	
	@echo +++ $(ONNX_INSTALL_DIR)/bin/onnx-mlir 
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir --mtriple=riscv32-unknown-elf --EmitONNXIR -o graph  $<
	

.PHONY: clean
clean:
	rm -f graph.* libsim.a *.o *.riscv32* *.ll *.log *.py *.S *.csv *.tar.gz MemManager.cpp *.yaml *.mlir *.inc *.tmp *.npy

.PHONY: rm_onnx
rm_onnx:
	rm -f *.py *.riscv32* graph.* *.cpp *.yaml *.mlir *.inc *.tmp MemManager.cpp.o pretty_print.cpp.o exec.log

exec.log: $(APP).riscv$(XLEN)
	. ./runTest.sh


export_elf: exec.log
	${EXPORT_ELF} -f $(APP).riscv$(XLEN)
	python3 $(TOOLS_INSTALL_DIR)/convert.py
	mv code_image.npy $(APP).npy
	mv exec.log $(APP)_exec.log
	tar -czvf $(APP).tar.gz  $(APP)_exec.log $(APP).npy  $(APP).riscv32 $(APP).riscv32.dump $(APP).riscv32.headers $(APP).riscv32.map	

.PHONY: print_shared_library_deps print_config
print_shared_library_deps:
	@echo ldd - print shared library dependencies
	ldd $(ONNX_INSTALL_DIR)/bin/onnx-mlir 


print_config:
	@echo ROOT_DIR=$(ROOT_DIR)
	@echo CC=$(CC)
	@echo CXX=$(CXX)
	@echo OBJDUMP=$(OBJDUMP)