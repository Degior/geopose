#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CLUSTER_PARTITION="${CLUSTER_PARTITION:-hiperf}"
CLUSTER_NODE="${CLUSTER_NODE:-}"
CLUSTER_NODES="${CLUSTER_NODES:-1}"
CLUSTER_GPUS="${CLUSTER_GPUS:-8}"
CLUSTER_CPUS="${CLUSTER_CPUS:-8}"
CLUSTER_MEMORY="${CLUSTER_MEMORY:-40000}"
CLUSTER_TIME="${CLUSTER_TIME:-20:00:00}"
CLUSTER_JOB_NAME="${CLUSTER_JOB_NAME:-geopose-predict}"
CLUSTER_LOG_DIR="${CLUSTER_LOG_DIR:-logs}"
CLUSTER_GRES="${CLUSTER_GRES:-gpu:${CLUSTER_GPUS}}"
CLUSTER_CUDA_DEVICES="${CLUSTER_CUDA_DEVICES:-}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

DATA_DIR="${DATA_DIR:-images2}"
OUTPUT_DIR="${OUTPUT_DIR:-submission}"
WEIGHTS="${WEIGHTS:-weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last_1}"
RGB_SUFFIX="${RGB_SUFFIX:-png}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DOWNSAMPLE="${DOWNSAMPLE:-1}"
UNIT="${UNIT:-cm}"
TILE_SIZE="${TILE_SIZE:-2048}"
COMPRESS="${COMPRESS:-true}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SKIP_BUILD="${SKIP_BUILD:-1}"

fail() {
    printf 'predict_cluster.sh: %s\n' "$1" >&2
    exit 2
}

[[ -n "$WEIGHTS" ]] || fail "WEIGHTS must contain at least one checkpoint"
read -r -a checkpoint_paths <<< "$WEIGHTS"
for checkpoint_path in "${checkpoint_paths[@]}"; do
    [[ -f "$checkpoint_path" ]] || fail "checkpoint not found: $checkpoint_path"
done
[[ -d "$DATA_DIR" ]] || fail "input directory not found: $DATA_DIR"
[[ "$CLUSTER_NODES" =~ ^[1-9][0-9]*$ ]] || fail "CLUSTER_NODES must be a positive integer"
[[ "$CLUSTER_GPUS" =~ ^[1-9][0-9]*$ ]] || fail "CLUSTER_GPUS must be a positive integer"
[[ "$CLUSTER_CPUS" =~ ^[1-9][0-9]*$ ]] || fail "CLUSTER_CPUS must be a positive integer"
[[ "$CLUSTER_MEMORY" =~ ^[1-9][0-9]*$ ]] || fail "CLUSTER_MEMORY must be a positive integer"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || fail "MASTER_PORT must be a positive integer"
((MASTER_PORT <= 65535)) || fail "MASTER_PORT must be at most 65535"
command -v "$SBATCH_BIN" >/dev/null 2>&1 || fail "Slurm submit command not found: $SBATCH_BIN"

if [[ "$PYTHON" == */* && "$PYTHON" != /* ]]; then
    PYTHON="$ROOT_DIR/$PYTHON"
fi
if [[ "$CLUSTER_LOG_DIR" != /* ]]; then
    CLUSTER_LOG_DIR="$ROOT_DIR/$CLUSTER_LOG_DIR"
fi
mkdir -p -- "$CLUSTER_LOG_DIR"

if [[ -z "$CLUSTER_CUDA_DEVICES" ]]; then
    devices=()
    for ((index = 0; index < CLUSTER_GPUS; index++)); do
        devices+=("$index")
    done
    CLUSTER_CUDA_DEVICES="$(IFS=,; printf '%s' "${devices[*]}")"
fi

GPUS="$CLUSTER_GPUS"
CUDA_VISIBLE_DEVICES="$CLUSTER_CUDA_DEVICES"
export DATA_DIR OUTPUT_DIR WEIGHTS RGB_SUFFIX BATCH_SIZE DOWNSAMPLE UNIT TILE_SIZE COMPRESS MASTER_PORT PYTHON SKIP_BUILD GPUS CUDA_VISIBLE_DEVICES

sbatch_args=(
    --nodes="$CLUSTER_NODES"
    --partition="$CLUSTER_PARTITION"
    --gres="$CLUSTER_GRES"
    --cpus-per-task="$CLUSTER_CPUS"
    --mem="$CLUSTER_MEMORY"
    --time="$CLUSTER_TIME"
    --job-name="$CLUSTER_JOB_NAME"
    --output="$CLUSTER_LOG_DIR/${CLUSTER_JOB_NAME}_%j.out"
    --error="$CLUSTER_LOG_DIR/${CLUSTER_JOB_NAME}_%j.err"
)
if [[ -n "$CLUSTER_NODE" ]]; then
    sbatch_args+=(--nodelist="$CLUSTER_NODE")
fi

job_command="$(printf '%q ' "$ROOT_DIR/predict.sh")"
sbatch_args+=(--chdir="$ROOT_DIR" --export=ALL --wrap="$job_command")
"$SBATCH_BIN" "${sbatch_args[@]}"
