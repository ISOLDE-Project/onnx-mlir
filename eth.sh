#!/usr/bin/env bash
# Copyleft

# Define environment variables
MINICONDA=$HOME/hdd1/miniconda3/etc/profile.d/conda.sh
MINICONDA_ENV=ibex
# MINICONDA_ENV=onnx-mlir

# To activate this environment, use
#
#     $ conda activate snitch
#
# To deactivate an active environment, use
#
#     $ conda deactivate

# Get the root directory of the Git repository
# export ROOT_DIR=$(git rev-parse --show-toplevel)
source $MINICONDA
conda activate $MINICONDA_ENV

