#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
DATA_DIR="${1:?Usage: train.sh DATA_DIR WDATA_DIR GPUS}"
WDATA_DIR="${2:?WDATA_DIR is required}"
GPUS="${3:-1}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
        PYTHON="$ROOT_DIR/.venv/bin/python"
    else
        PYTHON=python
    fi
fi
export PYTHON
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
[[ -d "$DATA_DIR" ]] || { printf 'input directory not found: %s\n' "$DATA_DIR" >&2; exit 1; }

echo 'Check training dependencies'
if ! "$PYTHON" -c 'from training.train_segmentor import main'; then
    printf 'Training dependencies could not be imported. Install requirements.txt with this Python: %s\n' "$PYTHON" >&2
    exit 1
fi

mkdir -p logs weights
if [[ "${SKIP_PREPARE:-0}" == 1 ]]; then
    [[ -d "$WDATA_DIR" ]] || { printf 'prepared directory not found: %s\n' "$WDATA_DIR" >&2; exit 1; }
    printf 'Use prepared images: %s\n' "$WDATA_DIR"
else
    mkdir -p "$WDATA_DIR"
    echo 'Convert images'
    "$PYTHON" utilities/downsample_images.py --indir "$DATA_DIR" --outdir "$WDATA_DIR" \
        --rgb-suffix "${TRAIN_RGB_SUFFIX:-j2k}"
fi

echo 'Start training'
for fold in 5 6; do
    if [[ -d preds ]]; then
        rm -f -- preds/*_AGL.tif preds/*_VFLOW.json
    fi
    bash ./dist_train.sh "$DATA_DIR" "$WDATA_DIR" "$fold" "$GPUS" > "logs/l$fold" 2>&1
    bash ./dist_train_tune.sh "$DATA_DIR" "$WDATA_DIR" "$fold" "$GPUS" \
        "weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_${fold}_r2" >> "logs/l$fold" 2>&1
done
echo 'Done training'
