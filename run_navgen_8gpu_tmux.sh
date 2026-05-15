#!/usr/bin/env bash
# ============================================================================
# NavGen 8-GPU Parallel Data Generation — tmux persistent runner
# ============================================================================
# Runs the 8-GPU generation inside a tmux session so that it survives
# SSH disconnects, terminal closures, or network instability.
#
# Usage:
#   ./run_navgen_8gpu_tmux.sh [same options as run_navgen_8gpu.sh]
#
# Common example:
#   ./run_navgen_8gpu_tmux.sh --total-target 4000 --max-step 500 --training-data-mode --no-remote
#
# After launching:
#   tmux attach -t navgen_8gpu    # watch live output
#   Ctrl+b, d                     # detach (process keeps running)
#
# To check progress without attaching:
#   tail -f /mnt/data/gengshuang/vln_cot/nav_gen/remote_batches/navgen_8gpu_*/logs/gpu0.log
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="navgen_8gpu"

# --- Check tmux is available ---
if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed. Install it first: apt install tmux" >&2
  exit 1
fi

# --- Check if session already exists ---
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[WARN] tmux session '$SESSION_NAME' already exists!"
  echo "  To attach:  tmux attach -t $SESSION_NAME"
  echo "  To kill:    tmux kill-session -t $SESSION_NAME"
  echo "  To check:   tmux list-sessions"
  exit 1
fi

# --- Create tmux session and run the 8-GPU script ---
echo "============================================"
echo " Launching 8-GPU NavGen in tmux session"
echo "============================================"
echo " Session name : $SESSION_NAME"
echo " Command      : run_navgen_8gpu.sh $*"
echo ""
echo " To attach (watch live):"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo " To detach (keep running):"
echo "   Ctrl+b, then d"
echo ""
echo " To kill the session:"
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo " To list sessions:"
echo "   tmux list-sessions"
echo "============================================"

tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50 \
  "bash ${SCRIPT_DIR}/run_navgen_8gpu.sh $*; echo ''; echo '=== PROCESS FINISHED ==='; echo 'Press Enter to close this session...'; read"

echo ""
echo "Session '$SESSION_NAME' launched successfully."
echo "The 8-GPU generation is now running inside tmux and will survive disconnects."