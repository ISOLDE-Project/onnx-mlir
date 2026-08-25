TOP_LEVEL_DIR       :=  $(shell git rev-parse --show-toplevel)
mkfile_path         :=  $(TOP_LEVEL_DIR)/test-isolde/complex_gemm
XLEN                ?=  32
ONNX_MODEL          ?=  $(mkfile_path)/models/redmule_complex_gemm_12x16x16.onnx
# APP=resnet18
# 
# Directories
DST_DIR             := $(mkfile_path)/models

include ${TOP_LEVEL_DIR}/test-isolde/common/common.mk

all: graph.test.aismem

## generate ONNX model
golden:  check-conda-ibex  $(DST_DIR)
	python3 $(mkfile_path)/models/generate_redmule_complex_gemm.py \
    --m 12 \
    --n 16 \
    --k 16 \
    --dtype f16 \
    --out $(DST_DIR)/redmule_complex_gemm_12x16x16.onnx






# Ensure the destination directory exists
$(DST_DIR):
	@mkdir -p $@


.PHONY:  clean-model	
clean-model:
	rm -rf $(DST_DIR) 


