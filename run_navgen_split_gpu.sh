#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/gengshuang/miniconda3/envs/lhvln-cu128/bin/python}"
SIM_GPU_DEVICE="${NAVGEN_SIM_GPU_DEVICE:-0}"
RAM_DEVICE="${NAVGEN_RAM_DEVICE:-cuda:${SIM_GPU_DEVICE}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found at $PYTHON_BIN" >&2
  exit 1
fi

# main.py derives its default task/log paths from the current working directory.
cd "$PROJECT_ROOT/nav_gen"

exec env \
  MAGNUM_DEVICE=HeadlessEGL \
  NAVGEN_SIM_GPU_DEVICE="$SIM_GPU_DEVICE" \
  NAVGEN_RAM_DEVICE="$RAM_DEVICE" \
  "$PYTHON_BIN" run_split_steps.py "$@"
