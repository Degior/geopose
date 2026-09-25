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

.PHONY: help all venv install setup build predict clean clean-venv
.DEFAULT_GOAL := help

help:
	@printf '%s\n' \
		'Targets:' \
		'  make venv       Create the virtual environment' \
		'  make install    Install dependencies from requirements.txt' \
		'  make setup      Create the environment and install dependencies' \
		'  make build      Build the Cython extension' \
		'  make predict    Run large-image prediction' \
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

clean:
	rm -f utilities/invert_flow*.so
	rm -rf utilities/build
	rm -f "$(BUILD_STAMP)"

clean-venv:
	rm -rf "$(VENV)"
