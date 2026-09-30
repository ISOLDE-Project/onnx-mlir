


ifndef ROOT_DIR
$(error ROOT_DIR is not defined. Please execute'source eth.sh' from top folder.)
endif
TEST_CASE           ?=  undefined
TEST_CASE_DIR       :=  $(ROOT_DIR)/test-isolde/$(TEST_CASE)
BANNER              :=  "💡 🏲  $(TEST_CASE) 🏲"

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
 

DEBUG_DIALECT_CONVERSION ?= no

ifeq ($(DEBUG_DIALECT_CONVERSION),yes)
DIALECT_DEBUG := dialect-conversion,pattern-application
else
DIALECT_DEBUG :=
endif


# Optional onnx-mlir diagnostics (all disabled by default).
# ONNX_DEBUG_ONLY selects the exact, case-sensitive C++ DEBUG_TYPE strings.
# A comma-separated or quoted space-separated list is accepted. If nonempty,
# it enables selective logging by itself and takes precedence over ONNX_DEBUG.
# The existing DEBUG_DIALECT_CONVERSION=yes switch adds dialect-conversion
# to that list. DIALECT_DEBUG above remains available to other Makefiles.
# LLVM_DEBUG code must be compiled in: use an assertions-enabled compiler and
# LLVM build. Defining DEBUG_TYPE alone does not print anything; LLVM_DEBUG
# statements in that category must execute. RISC-V -g flags are unrelated.
ONNX_DEBUG              ?= no
ONNX_DEBUG_ONLY         ?=
ONNX_IR_DUMP            ?= none
ONNX_IR_MODULE_SCOPE    ?= no
# Empty: stderr stays on the terminal. Otherwise each invocation overwrites
# <directory>/<target>.log; stdout stays on the terminal and failures propagate.
ONNX_DEBUG_LOG_DIR      ?=

# IR modes: none, before, after, all (before + after), failure.
# failure is deliberately separate from the other after-printing options.
ifeq ($(filter $(ONNX_IR_DUMP),none before after all failure),)
$(error ONNX_IR_DUMP must be one of: none before after all failure)
endif

onnx_debug_empty :=
onnx_debug_space := $(onnx_debug_empty) $(onnx_debug_empty)
onnx_debug_comma := ,
onnx_debug_types = $(sort $(subst $(onnx_debug_comma), ,$(ONNX_DEBUG_ONLY)) $(if $(filter yes,$(DEBUG_DIALECT_CONVERSION)),$(DIALECT_DEBUG)))
onnx_debug_trace_flags = $(if $(onnx_debug_types),--debug-only=$(subst $(onnx_debug_space),$(onnx_debug_comma),$(onnx_debug_types)),$(if $(filter yes,$(ONNX_DEBUG)),--debug))
onnx_debug_ir_none :=
onnx_debug_ir_before := --mlir-print-ir-before-all
onnx_debug_ir_after := --mlir-print-ir-after-all
onnx_debug_ir_all := --mlir-print-ir-before-all --mlir-print-ir-after-all
onnx_debug_ir_failure := --mlir-print-ir-after-failure
onnx_debug_ir_flags = $(onnx_debug_ir_$(ONNX_IR_DUMP)) $(if $(filter yes,$(ONNX_IR_MODULE_SCOPE)),--mlir-print-ir-module-scope)

# Disable compiler multithreading while tracing, both for readable output and
# for module-scope IR printing. Keep user-supplied ONNX_MLIR_FLAGS untouched,
# including when supplied on make's command line.
ONNX_DEBUG_FLAGS = $(strip $(onnx_debug_trace_flags) $(onnx_debug_ir_flags) \
    $(if $(strip $(onnx_debug_trace_flags) $(onnx_debug_ir_flags)),--mlir-disable-threading))


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
#ONNX_INSTALL_DIR        ?= ${ROOT_DIR}/toolchain/onnx-mlir/build/Debug
ONNX_INSTALL_DIR        ?= ${ROOT_DIR}/build/Debug
ONNX_MLIR_FLAGS			?=	
TOOLS_INSTALL_DIR       ?= ${ROOT_DIR}/install/onnx-mlir/py-codegen
EXPORT_ELF              ?= ${ROOT_DIR}/HLS/aida/build/bin/export_elf


# Shared by all graph stages so diagnostics are applied consistently.
define onnx_mlir_run
$(if $(strip $(ONNX_DEBUG_LOG_DIR)),@mkdir -p -- "$(ONNX_DEBUG_LOG_DIR)")
$(if $(strip $(ONNX_DEBUG_LOG_DIR)),@echo "Compiler stderr: $(ONNX_DEBUG_LOG_DIR)/$@.log")
$(ONNX_INSTALL_DIR)/bin/onnx-mlir $(ONNX_MLIR_FLAGS) $(ONNX_DEBUG_FLAGS) --mtriple=riscv32-unknown-elf $(1) -o graph $< $(if $(strip $(ONNX_DEBUG_LOG_DIR)),2>"$(ONNX_DEBUG_LOG_DIR)/$@.log")
endef


check-conda-%:
	@if [ "$$CONDA_DEFAULT_ENV" != "$*" ]; then \
		echo "Error: Conda environment '$*' must be active."; \
		echo "Run: source ./eth.sh"; \
		exit 1; \
	fi

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




.PHONY: graph
## Test complete lowering	         Layer  0 -> Layer -3 ->llvm
graph: print_config graph.test.onnx graph.test.aisle graph.test.aismem graph.test.aisllvmir graph.test.aisllvm 


## Emit ONNX IR                    Layer  0: ONNX Dialect
graph.test.onnx:  $(ONNX_MODEL) 	
	$(call onnx_mlir_run,--EmitONNXIR)
	@echo "🔔 $(TEST_CASE_DIR)/graph.onnx.onnxir"

## Emit AISLE / SPADE IR - AISLE - Layer -1: AutomotIve demonStrator mLir dialEct 
graph.test.aisle:  $(ONNX_MODEL) 	
	$(call onnx_mlir_run,--EmitSPADEIR)
	@echo "🔔 $(TEST_CASE_DIR)/graph.spade.aisle"

## Emit AISMEM / SPADE MLIR        Layer -2: AISMEM AutomotIve DemonStrator MEMref dialect
graph.test.aismem:  $(ONNX_MODEL) 	
	$(call onnx_mlir_run,--EmitSPADEMLIR)
	@echo "🔔 $(TEST_CASE_DIR)/graph.spade.mlir"

## Emit AISLLVM IR                 Layer -3: AISLLVM AutomotIve DemonStrator LLVM dialect
graph.test.aisllvmir:  $(ONNX_MODEL) 	
	$(call onnx_mlir_run,--EmitSPADELLVMIR)
	@echo "🔔 $(TEST_CASE_DIR)/graph.spade.llvm"

## Emit llvm and obj
graph.test.aisllvm:  $(ONNX_MODEL) 	
	$(call onnx_mlir_run,--EmitSPADELLVM)
	@echo "🔔 $(TEST_CASE_DIR)/graph.ll"


.PHONY: clean
clean:
	rm -f graph.* libsim.a *.o *.riscv32* *.ll *.log *.py *.S *.csv *.tar.gz MemManager.cpp *.yaml  *.inc *.tmp *.npy

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

## print configuration
print_config:
	@echo ROOT_DIR=$(ROOT_DIR)
	@echo $(BANNER)
	@echo onnx-mlir=$(ONNX_INSTALL_DIR)/bin/onnx-mlir
	@echo "ONNX_MLIR_FLAGS=$(ONNX_MLIR_FLAGS)"
	@echo "ONNX_DEBUG_FLAGS=$(ONNX_DEBUG_FLAGS)"
	@echo "ONNX_DEBUG_LOG_DIR=$(ONNX_DEBUG_LOG_DIR)"
	@echo TEST_CASE_DIR=$(TEST_CASE_DIR)
	@echo 💡 ONNX_MODEL=$(ONNX_MODEL)
# 	@echo $(BANNER)
	@echo CC=$(CC)
	@echo CXX=$(CXX)
# 	@echo OBJDUMP=$(OBJDUMP)

.PHONY: help-debug
## Show optional onnx-mlir debug settings and examples
help-debug:
	@printf '%s\n' \
	  'Optional settings (debugging is off by default):' \
	  '  ONNX_DEBUG_ONLY=<types>       Exact DEBUG_TYPE names, comma-separated' \
	  '  ONNX_DEBUG=yes               All LLVM debug categories if no types selected' \
	  '  DEBUG_DIALECT_CONVERSION=yes Add dialect-conversion,pattern-application to the selected types' \
	  '  ONNX_IR_DUMP=<mode>          none | before | after | all | failure' \
	  '  ONNX_IR_MODULE_SCOPE=yes     Print the whole module for requested IR dumps' \
	  '  ONNX_DEBUG_LOG_DIR=<dir>     Write stderr to <dir>/<target>.log (overwrite)' \
	  '' \
	  'Examples (add ONNX_MODEL=projection.onnx if needed):' \
	  '  make graph.test.aismem ONNX_DEBUG_ONLY=AISLEToAISMEM_MatMulAdd' \
	  '  make graph.test.aismem ONNX_DEBUG_ONLY=AISLEToAISMEM_MatMulAdd DEBUG_DIALECT_CONVERSION=yes' \
	  '  make graph.test.aisllvm ONNX_IR_DUMP=after ONNX_IR_MODULE_SCOPE=yes ONNX_DEBUG_LOG_DIR=debug' \
	  '  make graph.test.aisllvm ONNX_IR_DUMP=failure ONNX_DEBUG_LOG_DIR=debug' \
	  '' \
	  'Tracing automatically adds --mlir-disable-threading.' \
	  'LLVM_DEBUG logging requires an assertions-enabled compiler/LLVM build.' \
	  'The selected pass and its LLVM_DEBUG statements must actually execute.' \
	  'ONNX_MLIR_FLAGS may still supply other compiler options.' \
	  'Avoid conflicting debug/IR flags there when using these settings.' \
	  'Use make -n <target> ... to inspect the generated command.' \
	  'Use make -j1 graph when emitting all stages: they share the graph output prefix.'

help: Makefile
	@printf "Available targets:\n------------------\n"
	@for mkfile in $(MAKEFILE_LIST); do \
		awk '\
		/^[a-zA-Z0-9_.-]+[[:space:]]*:/ { \
			if (match(lastLine, /^##[[:space:]]+(.*)/)) { \
				target = $$0; \
				sub(/[[:space:]]*:.*$$/, "", target); \
				helpMessage = substr(lastLine, RSTART + 3, RLENGTH - 3); \
				printf "%-24s %s\n", target, helpMessage; \
			} \
		} \
		{ lastLine = $$0 }' $$mkfile; \
	done

.PHONY: help