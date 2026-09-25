PYTHON ?= python3.9
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
REQS ?= requirements.txt
PIP_EXTRA_INDEX_URL ?= https://download.pytorch.org/whl/cu124
INSTALL_STAMP := $(VENV)/.requirements-installed
BUILD_STAMP := $(VENV)/.cython-built

DATA_DIR ?= images2
OUTPUT_DIR ?= submission
WEIGHTS ?= weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last_1
GPUS ?= 1
MASTER_PORT ?= 29500
RGB_SUFFIX ?= png
BATCH_SIZE ?= 1
DOWNSAMPLE ?= 1
UNIT ?= cm
TILE_SIZE ?= 2048
COMPRESS ?= true
CLUSTER_PARTITION ?= hiperf
CLUSTER_NODE ?= tesla-a100
CLUSTER_NODES ?= 1
CLUSTER_GPUS ?= 8
CLUSTER_CPUS ?= 10
CLUSTER_MEMORY ?= 40000
CLUSTER_TIME ?= 20:00:00
CLUSTER_JOB_NAME ?= geopose-predict
CLUSTER_LOG_DIR ?= logs
CLUSTER_GRES ?= gpu:$(CLUSTER_GPUS)
CLUSTER_CUDA_DEVICES ?=
SBATCH_BIN ?= sbatch

.PHONY: help all venv install setup build predict predict-cluster clean clean-venv
.DEFAULT_GOAL := help

help:
	@printf '%s\n' \
		'Targets:' \
		'  make venv       Create the virtual environment' \
		'  make install    Install dependencies from requirements.txt' \
		'  make setup      Create the environment and install dependencies' \
		'  make build      Build the Cython extension' \
		'  make predict    Run large-image prediction' \
		'  make predict-cluster Submit prediction to a Slurm cluster' \
		'  make all        Set up dependencies and build the extension' \
		'  make clean      Remove generated build artifacts' \
		'  make clean-venv Remove the virtual environment'

all: setup build

$(VENV_PYTHON):
	@"$(PYTHON)" -m venv "$(VENV)"

venv: $(VENV_PYTHON)

$(INSTALL_STAMP): $(REQS) $(VENV_PYTHON)
	@test -f "$(REQS)" || { printf 'requirements file not found: %s\n' "$(REQS)"; exit 1; }
	@"$(VENV_PYTHON)" -m pip install --upgrade pip
	@"$(VENV_PYTHON)" -m pip install --extra-index-url "$(PIP_EXTRA_INDEX_URL)" -r "$(REQS)"
	@touch "$@"

install: $(INSTALL_STAMP)

setup: install

$(BUILD_STAMP): $(INSTALL_STAMP) utilities/cythonize_invert_flow.py utilities/invert_flow.pyx
	@PYTHONPATH=. "$(VENV_PYTHON)" utilities/cythonize_invert_flow.py build_ext --inplace
	@touch "$@"

build: $(BUILD_STAMP)

predict: build
	@test -n "$(strip $(WEIGHTS))" || { printf 'WEIGHTS must contain at least one checkpoint\n' >&2; exit 1; }
	@GPUS="$(GPUS)" MASTER_PORT="$(MASTER_PORT)" WEIGHTS="$(WEIGHTS)" PYTHON="$(VENV_PYTHON)" SKIP_BUILD=1 ./predict.sh \
		--dataset-dir "$(DATA_DIR)" \
		--predictions-dir "$(OUTPUT_DIR)" \
		--rgb-suffix "$(RGB_SUFFIX)" \
		--batch-size "$(BATCH_SIZE)" \
		--downsample "$(DOWNSAMPLE)" \
		--unit "$(UNIT)" \
		--tile-size "$(TILE_SIZE)" \
		--convert-predictions-to-cm-and-compress "$(COMPRESS)"

predict-cluster: build
	@test -n "$(strip $(WEIGHTS))" || { printf 'WEIGHTS must contain at least one checkpoint\n' >&2; exit 1; }
	@mkdir -p "$(CLUSTER_LOG_DIR)"
	@DATA_DIR="$(DATA_DIR)" \
		OUTPUT_DIR="$(OUTPUT_DIR)" \
		WEIGHTS="$(WEIGHTS)" \
		RGB_SUFFIX="$(RGB_SUFFIX)" \
		BATCH_SIZE="$(BATCH_SIZE)" \
		DOWNSAMPLE="$(DOWNSAMPLE)" \
		UNIT="$(UNIT)" \
		TILE_SIZE="$(TILE_SIZE)" \
		COMPRESS="$(COMPRESS)" \
		MASTER_PORT="$(MASTER_PORT)" \
		CLUSTER_PARTITION="$(CLUSTER_PARTITION)" \
		CLUSTER_NODE="$(CLUSTER_NODE)" \
		CLUSTER_NODES="$(CLUSTER_NODES)" \
		CLUSTER_GPUS="$(CLUSTER_GPUS)" \
		CLUSTER_CPUS="$(CLUSTER_CPUS)" \
		CLUSTER_MEMORY="$(CLUSTER_MEMORY)" \
		CLUSTER_TIME="$(CLUSTER_TIME)" \
		CLUSTER_JOB_NAME="$(CLUSTER_JOB_NAME)" \
		CLUSTER_LOG_DIR="$(CLUSTER_LOG_DIR)" \
		CLUSTER_GRES="$(CLUSTER_GRES)" \
		CLUSTER_CUDA_DEVICES="$(CLUSTER_CUDA_DEVICES)" \
		SBATCH_BIN="$(SBATCH_BIN)" \
		PYTHON="$(VENV_PYTHON)" \
		SKIP_BUILD=1 \
		./predict_cluster.sh

clean:
	rm -f utilities/invert_flow*.so
	rm -rf utilities/build
	rm -f "$(BUILD_STAMP)"

clean-venv:
	rm -rf "$(VENV)"
