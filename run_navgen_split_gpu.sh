#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/gs/anaconda3/bin/conda}"
SIM_GPU_DEVICE="${NAVGEN_SIM_GPU_DEVICE:-0}"
RAM_DEVICE="${NAVGEN_RAM_DEVICE:-cuda:${SIM_GPU_DEVICE}}"

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "conda executable not found at $CONDA_EXE" >&2
  exit 1
fi

# main.py derives its default task/log paths from the current working directory.
cd "$PROJECT_ROOT/nav_gen"

exec "$CONDA_EXE" run -n lhvln-cu128 env \
  NAVGEN_SIM_GPU_DEVICE="$SIM_GPU_DEVICE" \
  NAVGEN_RAM_DEVICE="$RAM_DEVICE" \
  python run_split_steps.py "$@"
