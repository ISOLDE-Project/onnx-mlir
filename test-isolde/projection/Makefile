TOP_LEVEL_DIR       :=  $(shell git rev-parse --show-toplevel)
TEST_CASE           :=  projection
mkfile_path         :=  $(TOP_LEVEL_DIR)/test-isolde/$(TEST_CASE)

XLEN                ?=  32
# Directories
DST_DIR             :=  $(mkfile_path)/models

# Trained, folded weights of the radar_attention encoder.  The 2-layer export
# is written by `make model` in ibex/isolde/sw/radar_attention (inc/ is not in
# git there); a copy lives in models/inc/.  LAYERS is checked against the
# header's TF_LAYERS, so a 1-layer file (inc_l1/) fails instead of silently
# producing a 1-layer graph.
WEIGHTS             ?=  $(DST_DIR)/inc/tformer_weights.h
LAYERS              ?=  2
ONNX_MODEL          ?=  $(DST_DIR)/projection.onnx

include ${TOP_LEVEL_DIR}/test-isolde/common/common.mk

all: graph.test.aisle

## generate the ONNX model (com.isolde MultiHeadAttention / PositionwiseFeedForward)
golden: check-conda-torch_env $(DST_DIR)
	python3 $(DST_DIR)/proj_onnx.py \
	  --seed 4785 \
	  --out $(ONNX_MODEL) 

# Ensure the destination directory exists
$(DST_DIR):
	@mkdir -p $@

.PHONY: golden clean-model
clean-model:
	rm -f $(DST_DIR)/*.onnx