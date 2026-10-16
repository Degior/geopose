#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
DATA="${1:?Usage: dist_train.sh DATA_DIR WDATA_DIR FOLD GPUS}"
WDATA="${2:?WDATA_DIR is required}"
FOLD="${3:?FOLD is required}"
GPUS="${4:-1}"
PYTHON_BIN="${PYTHON:-python}"
if [[ -z "${PYTHON:-}" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -u -m torch.distributed.run \
 --nproc_per_node=$GPUS \
 --master_port "${MASTER_PORT:-9901}" \
  training/train_segmentor.py \
 --world-size $GPUS  \
 --distributed \
 --config configs/v2l.json \
 --workers 8 \
 --multiplier 4 \
 --data-dir "$WDATA" \
 --train-dir "$DATA" \
 --test_every 2 \
 --crop 896 \
 --freeze-epochs 0 \
 --fold $FOLD \
 --prefix folds_
