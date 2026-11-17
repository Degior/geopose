#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
DATA_DIR="${1:-${DATA_DIR:-}}"
WDATA_DIR="${2:-${WDATA_DIR:-}}"
GPUS="${3:-${CLUSTER_GPUS:-1}}"
[[ -n "$DATA_DIR" && -n "$WDATA_DIR" ]] || {
    printf 'Usage: run_run.sh DATA_DIR WDATA_DIR [GPUS]\n' >&2
    exit 2
}
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { printf 'GPUS must be a positive integer\n' >&2; exit 2; }
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
fi
export PYTHON
mkdir -p logs

sbatch_args=(
    --nodes=1
    --ntasks=1
    --partition="${CLUSTER_PARTITION:-hiperf}"
    --gres="${CLUSTER_GRES:-gpu:$GPUS}"
    --cpus-per-task="${CLUSTER_CPUS:-8}"
    --mem="${CLUSTER_MEMORY:-40000}"
    --time="${CLUSTER_TIME:-20:00:00}"
    --job-name=geopose-train
    --output="$ROOT_DIR/logs/train_%j.log"
    --chdir="$ROOT_DIR"
    --export=ALL
)
if [[ -n "${CLUSTER_NODE-tesla-a100}" ]]; then
    sbatch_args+=(--nodelist="${CLUSTER_NODE-tesla-a100}")
fi
job_command="$(printf '%q ' bash "$ROOT_DIR/run.sh" "$DATA_DIR" "$WDATA_DIR" "$GPUS")"
sbatch_args+=(--wrap="$job_command")
"${SBATCH_BIN:-sbatch}" "${sbatch_args[@]}"
