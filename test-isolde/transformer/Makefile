TOP_LEVEL_DIR       :=  $(shell git rev-parse --show-toplevel)
TEST_CASE           :=  transformer
mkfile_path         :=  $(TOP_LEVEL_DIR)/test-isolde/$(TEST_CASE)

XLEN                ?=  32
# layer | mha | ffn | proj | proj_layer
BLOCK               ?=  layer
D_FF                ?=  48
DST_DIR             :=  $(mkfile_path)/models
ONNX_MODEL          ?=  $(DST_DIR)/transformer_$(BLOCK).onnx

include ${TOP_LEVEL_DIR}/test-isolde/common/common.mk

all: graph.test.aismem

## (re)generate the ONNX model (com.isolde functions) and its .npz references
golden: $(DST_DIR)
	python3 $(mkfile_path)/models/generate_transformer.py \
	  --block $(BLOCK) --d-ff $(D_FF) --out $(ONNX_MODEL)

$(ONNX_MODEL): | $(DST_DIR)
	python3 $(mkfile_path)/models/generate_transformer.py \
	  --block $(BLOCK) --d-ff $(D_FF) --out $@

## run the emitted AISMEM RedMulE schedule on a host model of the SPM and
## compare it bit-exactly with the FP16 reference from `golden`
check: $(ONNX_MODEL)
	$(ONNX_INSTALL_DIR)/bin/onnx-mlir $(ONNX_MLIR_FLAGS) \
	  --mtriple=riscv32-unknown-elf --EmitSPADEMLIR -o graph $<
	python3 $(mkfile_path)/check_aismem.py graph.spade.mlir \
	  $(basename $(ONNX_MODEL)).npz

# Ensure the destination directory exists
$(DST_DIR):
	@mkdir -p $@

.PHONY: clean-model golden check
clean-model:
	rm -f $(DST_DIR)/*.onnx $(DST_DIR)/*.npz
