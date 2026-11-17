#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${DATA_DIR:-}}"
WDATA_DIR="${2:-${WDATA_DIR:-}}"
GPUS="${3:-${GPUS:-1}}"
[[ -n "$DATA_DIR" && -n "$WDATA_DIR" ]] || {
    printf 'Usage: run.sh DATA_DIR WDATA_DIR [GPUS]\n' >&2
    exit 2
}
exec bash "$ROOT_DIR/train.sh" "$DATA_DIR" "$WDATA_DIR" "$GPUS"
