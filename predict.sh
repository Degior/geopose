#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

fail() {
    printf 'predict.sh: %s\n' "$1" >&2
    exit 2
}

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || fail "Python executable not found: $PYTHON_BIN"
else
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"
fi

GPUS_VALUE="${GPUS:-1}"
MASTER_PORT_VALUE="${MASTER_PORT:-29500}"
DATA_DIR_VALUE="${DATA_DIR:-images2}"
OUTPUT_DIR_VALUE="${OUTPUT_DIR:-submission}"
RGB_SUFFIX_VALUE="${RGB_SUFFIX:-png}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-1}"
DOWNSAMPLE_VALUE="${DOWNSAMPLE:-1}"
UNIT_VALUE="${UNIT:-cm}"
TILE_SIZE_VALUE="${TILE_SIZE:-2048}"
COMPRESS_VALUE="${COMPRESS:-true}"

if [[ -n "${WEIGHTS:-}" ]]; then
    read -r -a DEFAULT_WEIGHTS <<< "$WEIGHTS"
else
    DEFAULT_WEIGHTS=(
        weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2
        weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2
    )
fi

FORWARDED_ARGS=()
while (($# > 0)); do
    case "$1" in
        --gpus)
            (($# >= 2)) || fail "--gpus requires a value"
            GPUS_VALUE="$2"
            FORWARDED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --gpus=*)
            GPUS_VALUE="${1#*=}"
            FORWARDED_ARGS+=("$1")
            shift
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ "$GPUS_VALUE" =~ ^[1-9][0-9]*$ ]] || fail "GPUS must be a positive integer"
[[ "$MASTER_PORT_VALUE" =~ ^[1-9][0-9]*$ ]] || fail "MASTER_PORT must be a positive integer"
((MASTER_PORT_VALUE <= 65535)) || fail "MASTER_PORT must be at most 65535"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" utilities/cythonize_invert_flow.py build_ext --inplace
fi

DEFAULT_ARGS=(
    --predict
    --world_size "$GPUS_VALUE"
    --gpus "$GPUS_VALUE"
    --model-path "${DEFAULT_WEIGHTS[@]}"
    --dataset-dir "$DATA_DIR_VALUE"
    --predictions-dir "$OUTPUT_DIR_VALUE"
    --rgb-suffix "$RGB_SUFFIX_VALUE"
    --batch-size "$BATCH_SIZE_VALUE"
    --downsample "$DOWNSAMPLE_VALUE"
    --unit "$UNIT_VALUE"
    --tile-size "$TILE_SIZE_VALUE"
    --convert-predictions-to-cm-and-compress "$COMPRESS_VALUE"
)

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -u -m torch.distributed.run \
    --nproc_per_node="$GPUS_VALUE" \
    --master_port "$MASTER_PORT_VALUE" \
    "$ROOT_DIR/main.py" \
    "${DEFAULT_ARGS[@]}" \
    "${FORWARDED_ARGS[@]}"
