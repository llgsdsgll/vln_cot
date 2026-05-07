#!/usr/bin/env bash
set -euo pipefail

START_TIME=$(date +%s)

usage() {
  cat <<'EOF'
Usage:
  ./run_navgen_remote_batch.sh [wrapper-options] [navgen-main.py args...]

Wrapper options:
  --run-name NAME              Override the batch run name.
  --local-root DIR             Local parent directory for temporary outputs.
  --remote-user USER           Remote SSH user. Default: root
  --remote-host HOST           Remote SSH host. Default: 139.196.171.150
  --remote-port PORT           Remote SSH port. Default: 6222
  --remote-base-dir DIR        Remote parent directory.
                               Default: /mnt/data-cpfs/gengshuang/lhvln_dataset
  --conda-env NAME             Conda environment used to run NavGen.
                               Default: lhvln-cu128
  --conda-exe PATH             Conda executable path.
                               Default: /home/gs/anaconda3/bin/conda
  --sim-gpu-device ID          Passed as NAVGEN_SIM_GPU_DEVICE. Default: 0
  --ram-device DEVICE          Passed as NAVGEN_RAM_DEVICE.
                               Default: cuda:<sim-gpu-device>
  --export-videos              After each successful NavGen item, export
                               trajectory RGB videos from every
                               */success/trial_1/task.json found in it.
  --video-subdir DIRNAME       Per-item local video output subdir.
                               Default: videos_action1fps_boxes
  --video-fps N                Exported video fps. Default: 1
  --video-width N              Exported video width. Default: 1280
  --video-height N             Exported video height. Default: 744
  --render-sensor-height M     Shared camera / semantic sensor height in meters
                               for both NavGen and exported videos. Default: 1.0
  --video-sensor-height M      Alias of --render-sensor-height.
  --video-hfov DEG             Horizontal FOV in degrees. Default: 86.0
  --video-frame-mode MODE      state|action. Default: action
  --video-codec CODEC          h264|mp4v. Default: h264
  --video-annotate             Overlay task/action text on exported videos.
                               Default: on
  --no-video-annotate          Disable text overlay on exported videos.
  --video-target-boxes         Draw 2D boxes for target objects. Default: on
  --no-video-target-boxes      Disable target-object 2D boxes.
  --video-target-box-scope S   instance|class. Default: instance
  --max-attempts COUNT         Maximum total attempts used to reach the target
                               number of successful tasks. Default: 10 * --loop
                               Use 0 for no limit.
  --keep-local-item            Keep each per-task local directory after it has
                               been synced. Default: off
  --cleanup-local              Remove the local batch directory after a
                               successful sync.
  --help                       Show this help message.

All unrecognized arguments are forwarded to:
  python main.py

Wrapper behavior:
  The forwarded --loop value is treated as the target number of successful
  NavGen tasks. This wrapper runs `python main.py --loop 1` repeatedly until
  that many successful tasks are produced, or until --max-attempts is reached.

Example:
  ./run_navgen_remote_batch.sh \
    --run-name batch_20260427 \
    --export-videos \
    --loop 100 \
    --max-attempts 1000 \
    --max_step 500
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/gs/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-lhvln-cu128}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-139.196.171.150}"
REMOTE_PORT="${REMOTE_PORT:-6222}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/mnt/data-cpfs/gengshuang/lhvln_dataset}"
SIM_GPU_DEVICE="${NAVGEN_SIM_GPU_DEVICE:-0}"
RAM_DEVICE="${NAVGEN_RAM_DEVICE:-cuda:${SIM_GPU_DEVICE}}"
LOCAL_ROOT_DEFAULT="$PROJECT_ROOT/nav_gen/remote_batches"
LOCAL_ROOT="${LOCAL_ROOT:-$LOCAL_ROOT_DEFAULT}"
RUN_NAME_DEFAULT="navgen_batch_$(date +%Y%m%d_%H%M%S)"
RUN_NAME=""
KEEP_LOCAL_ITEM=0
CLEANUP_LOCAL=0
VIDEO_EXPORT=1
VIDEO_SUBDIR="videos_action1fps_boxes"
VIDEO_FPS=1
VIDEO_WIDTH=1280
VIDEO_HEIGHT=744
RENDER_SENSOR_HEIGHT=1.0
VIDEO_HFOV=86.0
VIDEO_FRAME_MODE="action"
VIDEO_CODEC="h264"
VIDEO_ANNOTATE=1
VIDEO_TARGET_BOXES=1
VIDEO_TARGET_BOX_SCOPE="instance"
VIZ_EXPORT=1
VIZ_SUBDIR="viz_topdown"
NAVGEN_ARGS=()
FORWARDED_ARGS=()
LOOP_COUNT=""
MAX_ATTEMPTS=""
HAS_LOCAL_RSYNC=0
SYNC_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="${2:?missing value for --run-name}"
      shift 2
      ;;
    --local-root)
      LOCAL_ROOT="${2:?missing value for --local-root}"
      shift 2
      ;;
    --remote-user)
      REMOTE_USER="${2:?missing value for --remote-user}"
      shift 2
      ;;
    --remote-host)
      REMOTE_HOST="${2:?missing value for --remote-host}"
      shift 2
      ;;
    --remote-port)
      REMOTE_PORT="${2:?missing value for --remote-port}"
      shift 2
      ;;
    --remote-base-dir)
      REMOTE_BASE_DIR="${2:?missing value for --remote-base-dir}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="${2:?missing value for --conda-env}"
      shift 2
      ;;
    --conda-exe)
      CONDA_EXE="${2:?missing value for --conda-exe}"
      shift 2
      ;;
    --sim-gpu-device)
      SIM_GPU_DEVICE="${2:?missing value for --sim-gpu-device}"
      shift 2
      ;;
    --ram-device)
      RAM_DEVICE="${2:?missing value for --ram-device}"
      shift 2
      ;;
    --export-videos)
      VIDEO_EXPORT=1
      shift
      ;;
    --video-subdir)
      VIDEO_SUBDIR="${2:?missing value for --video-subdir}"
      shift 2
      ;;
    --video-fps)
      VIDEO_FPS="${2:?missing value for --video-fps}"
      shift 2
      ;;
    --video-width)
      VIDEO_WIDTH="${2:?missing value for --video-width}"
      shift 2
      ;;
    --video-height)
      VIDEO_HEIGHT="${2:?missing value for --video-height}"
      shift 2
      ;;
    --render-sensor-height|--video-sensor-height)
      RENDER_SENSOR_HEIGHT="${2:?missing value for $1}"
      shift 2
      ;;
    --video-hfov)
      VIDEO_HFOV="${2:?missing value for --video-hfov}"
      shift 2
      ;;
    --video-frame-mode)
      VIDEO_FRAME_MODE="${2:?missing value for --video-frame-mode}"
      shift 2
      ;;
    --video-codec)
      VIDEO_CODEC="${2:?missing value for --video-codec}"
      shift 2
      ;;
    --video-annotate)
      VIDEO_ANNOTATE=1
      shift
      ;;
    --no-video-annotate)
      VIDEO_ANNOTATE=0
      shift
      ;;
    --video-target-boxes)
      VIDEO_TARGET_BOXES=1
      shift
      ;;
    --no-video-target-boxes)
      VIDEO_TARGET_BOXES=0
      shift
      ;;
    --video-target-box-scope)
      VIDEO_TARGET_BOX_SCOPE="${2:?missing value for --video-target-box-scope}"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="${2:?missing value for --max-attempts}"
      shift 2
      ;;
    --keep-local-item)
      KEEP_LOCAL_ITEM=1
      shift
      ;;
    --cleanup-local)
      CLEANUP_LOCAL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      [[ -n "$1" ]] && NAVGEN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$RUN_NAME_DEFAULT"
fi

while [[ ${#NAVGEN_ARGS[@]} -gt 0 ]]; do
  arg="${NAVGEN_ARGS[0]}"
  NAVGEN_ARGS=("${NAVGEN_ARGS[@]:1}")

  case "$arg" in
    --loop)
      if [[ ${#NAVGEN_ARGS[@]} -eq 0 ]]; then
        echo "missing value for --loop" >&2
        exit 1
      fi
      LOOP_COUNT="${NAVGEN_ARGS[0]}"
      NAVGEN_ARGS=("${NAVGEN_ARGS[@]:1}")
      ;;
    --loop=*)
      LOOP_COUNT="${arg#--loop=}"
      ;;
    --task_path|--step_task_path|--ram_logs|--split_save_path)
      if [[ ${#NAVGEN_ARGS[@]} -eq 0 ]]; then
        echo "missing value for $arg" >&2
        exit 1
      fi
      echo "[WARN] Ignoring forwarded $arg because this wrapper manages per-item paths." >&2
      NAVGEN_ARGS=("${NAVGEN_ARGS[@]:1}")
      ;;
    --render_sensor_height)
      if [[ ${#NAVGEN_ARGS[@]} -eq 0 ]]; then
        echo "missing value for --render_sensor_height" >&2
        exit 1
      fi
      RENDER_SENSOR_HEIGHT="${NAVGEN_ARGS[0]}"
      NAVGEN_ARGS=("${NAVGEN_ARGS[@]:1}")
      ;;
    --render_sensor_height=*)
      RENDER_SENSOR_HEIGHT="${arg#--render_sensor_height=}"
      ;;
    --task_path=*|--step_task_path=*|--ram_logs=*|--split_save_path=*)
      echo "[WARN] Ignoring forwarded ${arg%%=*} because this wrapper manages per-item paths." >&2
      ;;
    *)
      FORWARDED_ARGS+=("$arg")
      ;;
  esac
done

if [[ -z "$LOOP_COUNT" ]]; then
  LOOP_COUNT=1
fi

if ! [[ "$LOOP_COUNT" =~ ^[0-9]+$ ]] || [[ "$LOOP_COUNT" -lt 1 ]]; then
  echo "--loop must be a positive integer, got: $LOOP_COUNT" >&2
  exit 1
fi

if [[ -z "$MAX_ATTEMPTS" ]]; then
  MAX_ATTEMPTS=$((LOOP_COUNT * 10))
fi

if ! [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]]; then
  echo "--max-attempts must be a non-negative integer, got: $MAX_ATTEMPTS" >&2
  exit 1
fi

if [[ "$MAX_ATTEMPTS" -gt 0 ]] && [[ "$MAX_ATTEMPTS" -lt "$LOOP_COUNT" ]]; then
  echo "[WARN] --max-attempts (${MAX_ATTEMPTS}) is smaller than target successes (${LOOP_COUNT}); the run may stop early." >&2
fi

if ! [[ "$VIDEO_FPS" =~ ^[0-9]+$ ]] || [[ "$VIDEO_FPS" -lt 1 ]]; then
  echo "--video-fps must be a positive integer, got: $VIDEO_FPS" >&2
  exit 1
fi

if ! [[ "$VIDEO_WIDTH" =~ ^[0-9]+$ ]] || [[ "$VIDEO_WIDTH" -lt 1 ]]; then
  echo "--video-width must be a positive integer, got: $VIDEO_WIDTH" >&2
  exit 1
fi

if ! [[ "$VIDEO_HEIGHT" =~ ^[0-9]+$ ]] || [[ "$VIDEO_HEIGHT" -lt 1 ]]; then
  echo "--video-height must be a positive integer, got: $VIDEO_HEIGHT" >&2
  exit 1
fi

case "$VIDEO_FRAME_MODE" in
  state|action)
    ;;
  *)
    echo "--video-frame-mode must be one of: state, action" >&2
    exit 1
    ;;
esac

case "$VIDEO_CODEC" in
  h264|mp4v)
    ;;
  *)
    echo "--video-codec must be one of: h264, mp4v" >&2
    exit 1
    ;;
esac

case "$VIDEO_TARGET_BOX_SCOPE" in
  instance|class)
    ;;
  *)
    echo "--video-target-box-scope must be one of: instance, class" >&2
    exit 1
    ;;
esac

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "conda executable not found at $CONDA_EXE" >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required but not found in PATH" >&2
  exit 1
fi

if command -v rsync >/dev/null 2>&1; then
  HAS_LOCAL_RSYNC=1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required but not found in PATH" >&2
  exit 1
fi

if ! command -v scp >/dev/null 2>&1; then
  echo "scp is required but not found in PATH" >&2
  exit 1
fi

mkdir -p "$LOCAL_ROOT"
LOCAL_RUN_DIR="$LOCAL_ROOT/$RUN_NAME"
LOCAL_LOG_DIR="$LOCAL_RUN_DIR/logs"
SUMMARY_FILE="$LOCAL_RUN_DIR/batch_summary.tsv"

REMOTE_SPEC="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_RUN_DIR="${REMOTE_BASE_DIR%/}/${RUN_NAME}"

mkdir -p "$LOCAL_RUN_DIR" "$LOCAL_LOG_DIR"
printf "item_id\tstatus\tsuccess_progress\tstep_task_json_count\tvideo_count\tvideo_status\tlocal_item_dir\tremote_run_dir\n" > "$SUMMARY_FILE"

ensure_remote_layout() {
  ssh -p "$REMOTE_PORT" "$REMOTE_SPEC" \
    "mkdir -p '$REMOTE_RUN_DIR/task' '$REMOTE_RUN_DIR/step_task' '$REMOTE_RUN_DIR/logs' '$REMOTE_RUN_DIR/videos'"
}

detect_sync_mode() {
  if [[ -n "$SYNC_MODE" ]]; then
    return
  fi

  if [[ "$HAS_LOCAL_RSYNC" -eq 1 ]] && ssh -p "$REMOTE_PORT" "$REMOTE_SPEC" "command -v rsync >/dev/null 2>&1"; then
    SYNC_MODE="rsync"
    return
  fi

  if ssh -p "$REMOTE_PORT" "$REMOTE_SPEC" "command -v tar >/dev/null 2>&1"; then
    SYNC_MODE="tar"
    return
  fi

  echo "[ERROR] Remote host is missing both rsync and tar, so directory sync cannot proceed." >&2
  exit 1
}

sync_file() {
  local src_file="$1"
  local remote_file="$2"

  case "$SYNC_MODE" in
    rsync)
      rsync -az \
        -e "ssh -p ${REMOTE_PORT}" \
        "$src_file" \
        "${REMOTE_SPEC}:${remote_file}"
      ;;
    tar)
      ssh -p "$REMOTE_PORT" "$REMOTE_SPEC" "cat > '$remote_file'" < "$src_file"
      ;;
    *)
      echo "[ERROR] Unsupported sync mode: $SYNC_MODE" >&2
      exit 1
      ;;
  esac
}

sync_dir_contents() {
  local src_dir="$1"
  local remote_dir="$2"

  case "$SYNC_MODE" in
    rsync)
      rsync -az --partial --info=progress2 \
        -e "ssh -p ${REMOTE_PORT}" \
        "${src_dir}/" \
        "${REMOTE_SPEC}:${remote_dir}/"
      ;;
    tar)
      tar -C "$src_dir" -cf - . | ssh -p "$REMOTE_PORT" "$REMOTE_SPEC" "tar -xf - -C '$remote_dir'"
      ;;
    *)
      echo "[ERROR] Unsupported sync mode: $SYNC_MODE" >&2
      exit 1
      ;;
  esac
}

sync_summary() {
  sync_file "$SUMMARY_FILE" "${REMOTE_RUN_DIR}/batch_summary.tsv"
}

append_summary() {
  local item_id="$1"
  local status="$2"
  local success_progress="$3"
  local step_task_json_count="$4"
  local video_count="$5"
  local video_status="$6"
  local item_dir="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$item_id" \
    "$status" \
    "$success_progress" \
    "$step_task_json_count" \
    "$video_count" \
    "$video_status" \
    "$item_dir" \
    "${REMOTE_SPEC}:${REMOTE_RUN_DIR}" >> "$SUMMARY_FILE"
}

sync_item() {
  local item_id="$1"
  local item_dir="$2"
  local item_task_dir="$item_dir/task"
  local item_step_task_dir="$item_dir/step_task"
  local item_log_dir="$item_dir/logs"
  local item_video_dir="$item_dir/$VIDEO_SUBDIR"

  echo "[INFO] Syncing ${item_id} to ${REMOTE_SPEC}:${REMOTE_RUN_DIR}/"

  ensure_remote_layout

  if [[ -d "$item_task_dir" ]] && [[ -n "$(find "$item_task_dir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    sync_dir_contents "$item_task_dir" "${REMOTE_RUN_DIR}/task"
  fi

  if [[ -d "$item_step_task_dir" ]] && [[ -n "$(find "$item_step_task_dir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    sync_dir_contents "$item_step_task_dir" "${REMOTE_RUN_DIR}/step_task"
  fi

  if [[ -d "$item_log_dir" ]] && [[ -n "$(find "$item_log_dir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    ssh -p "$REMOTE_PORT" "$REMOTE_SPEC" "mkdir -p '$REMOTE_RUN_DIR/logs/$item_id'"
    sync_dir_contents "$item_log_dir" "${REMOTE_RUN_DIR}/logs/${item_id}"
  fi

  if [[ -d "$item_video_dir" ]] && [[ -n "$(find "$item_video_dir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    ssh -p "$REMOTE_PORT" "$REMOTE_SPEC" "mkdir -p '$REMOTE_RUN_DIR/videos/$item_id/$VIDEO_SUBDIR'"
    sync_dir_contents "$item_video_dir" "${REMOTE_RUN_DIR}/videos/${item_id}/${VIDEO_SUBDIR}"
  fi

  local item_viz_dir="$item_dir/$VIZ_SUBDIR"
  if [[ -d "$item_viz_dir" ]] && [[ -n "$(find "$item_viz_dir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    ssh -p "$REMOTE_PORT" "$REMOTE_SPEC" "mkdir -p '$REMOTE_RUN_DIR/viz/$item_id/$VIZ_SUBDIR'"
    sync_dir_contents "$item_viz_dir" "${REMOTE_RUN_DIR}/viz/${item_id}/${VIZ_SUBDIR}"
  fi

  sync_summary
}

cleanup_local_item() {
  local item_dir="$1"
  if [[ "$KEEP_LOCAL_ITEM" -eq 0 ]]; then
    rm -rf "$item_dir"
  fi
}

count_step_task_jsons() {
  local item_step_task_dir="$1"
  if [[ ! -d "$item_step_task_dir" ]]; then
    echo 0
    return
  fi

  find "$item_step_task_dir" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l | tr -d '[:space:]'
}

item_has_success_output() {
  local item_dir="$1"
  local item_task_dir="$item_dir/task"
  local item_step_task_dir="$item_dir/step_task"

  if [[ "$(count_step_task_jsons "$item_step_task_dir")" -gt 0 ]]; then
    return 0
  fi

  if [[ -d "$item_task_dir" ]] && [[ -n "$(find "$item_task_dir" -path '*/success/trial_1/task.json' -print -quit 2>/dev/null)" ]]; then
    return 0
  fi

  return 1
}

LAST_VIDEO_EXPORT_COUNT=0
LAST_VIDEO_EXPORT_STATUS="disabled"

export_item_videos() {
  local item_id="$1"
  local item_dir="$2"
  local item_task_dir="$item_dir/task"
  local item_video_dir="$item_dir/$VIDEO_SUBDIR"
  local item_video_log="$item_dir/logs/video_export.log"
  local found_task_jsons=0
  local exported_videos=0

  LAST_VIDEO_EXPORT_COUNT=0
  LAST_VIDEO_EXPORT_STATUS="disabled"

  if [[ "$VIDEO_EXPORT" -eq 0 ]]; then
    return 0
  fi

  mkdir -p "$item_video_dir"
  : > "$item_video_log"

  while IFS= read -r -d '' task_json; do
    found_task_jsons=$((found_task_jsons + 1))
    local rel_task_json="${task_json#$item_task_dir/}"
    local rel_task_dir
    rel_task_dir="$(dirname "$rel_task_json")"
    local output_video="$item_video_dir/$rel_task_dir/trajectory_rgb.mp4"

    echo "[INFO] Exporting video for ${item_id}: ${rel_task_json}" | tee -a "$item_video_log"

    local export_cmd=(
      "$CONDA_EXE" run -n "$CONDA_ENV" env
      NAVGEN_SIM_GPU_DEVICE="$SIM_GPU_DEVICE"
      python export_trajectory_rgb_video.py
      --task_json "$task_json"
      --output_video "$output_video"
      --fps "$VIDEO_FPS"
      --width "$VIDEO_WIDTH"
      --height "$VIDEO_HEIGHT"
      --sensor_height "$RENDER_SENSOR_HEIGHT"
      --hfov "$VIDEO_HFOV"
      --sim_gpu_device "$SIM_GPU_DEVICE"
      --frame_mode "$VIDEO_FRAME_MODE"
      --output_codec "$VIDEO_CODEC"
    )

    if [[ "$VIDEO_ANNOTATE" -eq 1 ]]; then
      export_cmd+=(--annotate)
    fi

    if [[ "$VIDEO_TARGET_BOXES" -eq 1 ]]; then
      export_cmd+=(--target_boxes --target_box_scope "$VIDEO_TARGET_BOX_SCOPE")
    fi

    set +e
    "${export_cmd[@]}" 2>&1 | tee -a "$item_video_log"
    local export_status=${PIPESTATUS[0]}
    set -e

    if [[ "$export_status" -ne 0 ]]; then
      LAST_VIDEO_EXPORT_COUNT="$exported_videos"
      LAST_VIDEO_EXPORT_STATUS="export_failed:${export_status}"
      echo "[ERROR] Video export failed for ${rel_task_json}; exit=${export_status}" | tee -a "$item_video_log" >&2
      return 1
    fi

    exported_videos=$((exported_videos + 1))
  done < <(find "$item_task_dir" -path '*/success/trial_1/task.json' -print0 2>/dev/null)

  LAST_VIDEO_EXPORT_COUNT="$exported_videos"

  if [[ "$found_task_jsons" -eq 0 ]]; then
    LAST_VIDEO_EXPORT_STATUS="no_success_trial_task_json"
    echo "[WARN] No success/trial_1/task.json found for ${item_id}; skipping video export." | tee -a "$item_video_log" >&2
    return 1
  fi

  LAST_VIDEO_EXPORT_STATUS="ok:${exported_videos}"
  return 0
}

export_item_viz() {
  local item_id="$1"
  local item_dir="$2"
  local item_task_dir="$item_dir/task"
  local item_viz_dir="$item_dir/$VIZ_SUBDIR"
  local item_viz_log="$item_dir/logs/viz_export.log"

  if [[ "$VIZ_EXPORT" -eq 0 ]]; then
    return 0
  fi

  mkdir -p "$item_viz_dir"
  : > "$item_viz_log"

  while IFS= read -r -d '' task_json; do
    local rel_task_json="${task_json#$item_task_dir/}"
    local rel_task_dir
    rel_task_dir="$(dirname "$rel_task_json")"
    local out_png="$item_viz_dir/$rel_task_dir/topdown.png"
    mkdir -p "$(dirname "$out_png")"

    echo "[INFO] Exporting viz for ${item_id}: ${rel_task_json}" | tee -a "$item_viz_log"

    local viz_cmd=(
      "$CONDA_EXE" run -n "$CONDA_ENV" env
      NAVGEN_SIM_GPU_DEVICE="$SIM_GPU_DEVICE"
      python visualize_goal_viewpoints_topdown.py
      --task-json "$task_json"
      --output "$out_png"
      --render-sensor-height "$RENDER_SENSOR_HEIGHT"
      --allow-occluded-goal-fallback
      --sim-gpu-device "$SIM_GPU_DEVICE"
    )

    local traj_meta
    traj_meta="$(dirname "$task_json")/trajectory_rgb.json"
    if [[ -f "$traj_meta" ]]; then
      viz_cmd+=(--trajectory-meta "$traj_meta")
    fi

    set +e
    "${viz_cmd[@]}" >> "$item_viz_log" 2>&1
    local viz_status=$?
    set -e

    if [[ "$viz_status" -ne 0 ]]; then
      echo "[WARN] Viz export failed for ${rel_task_json}; exit=${viz_status}" | tee -a "$item_viz_log" >&2
    fi
  done < <(find "$item_task_dir" -path '*/success/trial_1/task.json' -print0 2>/dev/null)

  while IFS= read -r -d '' config_json; do
    local config_dir; config_dir="$(dirname "$config_json")"
    [[ -d "$config_dir/success" ]] && continue
    local rel_config="${config_json#$item_task_dir/}"
    local rel_dir; rel_dir="$(dirname "$rel_config")"
    local out_png="$item_viz_dir/$rel_dir/topdown.png"
    mkdir -p "$(dirname "$out_png")"
    echo "[INFO] Exporting viz (fail) for ${item_id}: ${rel_config}" | tee -a "$item_viz_log"
    local fail_viz_cmd=(
      "$CONDA_EXE" run -n "$CONDA_ENV" env
      NAVGEN_SIM_GPU_DEVICE="$SIM_GPU_DEVICE"
      python visualize_goal_viewpoints_topdown.py
      --task-json "$config_json"
      --output "$out_png"
      --render-sensor-height "$RENDER_SENSOR_HEIGHT"
      --allow-occluded-goal-fallback
      --sim-gpu-device "$SIM_GPU_DEVICE"
    )
    set +e
    "${fail_viz_cmd[@]}" >> "$item_viz_log" 2>&1
    local fail_viz_status=$?
    set -e
    if [[ "$fail_viz_status" -ne 0 ]]; then
      echo "[WARN] Viz (fail) export failed for ${rel_config}; exit=${fail_viz_status}" | tee -a "$item_viz_log" >&2
    fi
  done < <(find "$item_task_dir" -name 'config.json' -print0 2>/dev/null)
}

echo "[INFO] Project root      : $PROJECT_ROOT"
echo "[INFO] Conda env         : $CONDA_ENV"
echo "[INFO] Local batch dir   : $LOCAL_RUN_DIR"
echo "[INFO] Remote batch dir  : ${REMOTE_SPEC}:${REMOTE_RUN_DIR}"
echo "[INFO] SIM GPU device    : $SIM_GPU_DEVICE"
echo "[INFO] RAM device        : $RAM_DEVICE"
echo "[INFO] Export videos     : $VIDEO_EXPORT"
if [[ "$VIDEO_EXPORT" -eq 1 ]]; then
  echo "[INFO] Video subdir      : $VIDEO_SUBDIR"
  echo "[INFO] Video params      : fps=${VIDEO_FPS}, size=${VIDEO_WIDTH}x${VIDEO_HEIGHT}, sensor_height=${RENDER_SENSOR_HEIGHT}, hfov=${VIDEO_HFOV}, frame_mode=${VIDEO_FRAME_MODE}, codec=${VIDEO_CODEC}, annotate=${VIDEO_ANNOTATE}, target_boxes=${VIDEO_TARGET_BOXES}, scope=${VIDEO_TARGET_BOX_SCOPE}"
fi
echo "[INFO] Sensor height    : $RENDER_SENSOR_HEIGHT"
echo "[INFO] Target successes  : $LOOP_COUNT"
if [[ "$MAX_ATTEMPTS" -eq 0 ]]; then
  echo "[INFO] Max attempts      : unlimited"
else
  echo "[INFO] Max attempts      : $MAX_ATTEMPTS"
fi
echo "[INFO] Keep local item   : $KEEP_LOCAL_ITEM"
echo "[INFO] Forwarded args    : ${FORWARDED_ARGS[*]:-(none)}"

detect_sync_mode
echo "[INFO] Remote sync mode  : $SYNC_MODE"

ensure_remote_layout
sync_summary

cd "$PROJECT_ROOT/nav_gen"

attempt_count=0
success_count=0

while [[ "$success_count" -lt "$LOOP_COUNT" ]]; do
  attempt_count=$((attempt_count + 1))

  if [[ "$MAX_ATTEMPTS" -gt 0 ]] && [[ "$attempt_count" -gt "$MAX_ATTEMPTS" ]]; then
    break
  fi

  item_id="$(printf 'item_%04d' "$attempt_count")"
  item_dir="$LOCAL_RUN_DIR/$item_id"
  item_task_dir="$item_dir/task"
  item_step_task_dir="$item_dir/step_task"
  item_log_dir="$item_dir/logs"
  item_trail_list="$item_task_dir/trail_list.txt"
  item_ram_log="$item_log_dir/step_task_logs.txt"
  item_stdout_log="$item_log_dir/navgen_main.log"
  item_video_dir="$item_dir/$VIDEO_SUBDIR"

  mkdir -p "$item_task_dir" "$item_step_task_dir" "$item_log_dir" "$item_video_dir"

  echo "[INFO] Running ${item_id} (attempt ${attempt_count}, success ${success_count}/${LOOP_COUNT})"

  set +e
  "$CONDA_EXE" run -n "$CONDA_ENV" env \
    NAVGEN_SIM_GPU_DEVICE="$SIM_GPU_DEVICE" \
    NAVGEN_RAM_DEVICE="$RAM_DEVICE" \
    python main.py \
      "${FORWARDED_ARGS[@]}" \
      --loop 1 \
      --render_sensor_height "$RENDER_SENSOR_HEIGHT" \
      --task_path "${item_task_dir}/" \
      --step_task_path "${item_step_task_dir}/" \
      --ram_logs "$item_ram_log" \
      --split_save_path "$item_trail_list" 2>&1 | tee "$item_stdout_log"
  item_status=${PIPESTATUS[0]}
  set -e

  step_task_json_count="$(count_step_task_jsons "$item_step_task_dir")"
  item_success=0
  item_summary_status=""
  video_count=0
  video_status="disabled"

  if [[ "$item_status" -eq 0 ]] && item_has_success_output "$item_dir"; then
    if [[ "$VIDEO_EXPORT" -eq 1 ]]; then
      if export_item_videos "$item_id" "$item_dir"; then
        video_count="$LAST_VIDEO_EXPORT_COUNT"
        video_status="$LAST_VIDEO_EXPORT_STATUS"
        item_success=1
        success_count=$((success_count + 1))
        item_summary_status="ok"
      else
        video_count="$LAST_VIDEO_EXPORT_COUNT"
        video_status="$LAST_VIDEO_EXPORT_STATUS"
        item_summary_status="video_failed:${video_status}"
      fi
    else
      item_success=1
      success_count=$((success_count + 1))
      item_summary_status="ok"
    fi
  elif [[ "$item_status" -eq 0 ]]; then
    item_summary_status="no_success_task"
    if [[ "$VIDEO_EXPORT" -eq 1 ]]; then
      video_status="skipped"
    fi
  else
    item_summary_status="navgen_failed:${item_status}"
    if [[ "$VIDEO_EXPORT" -eq 1 ]]; then
      video_status="skipped"
    fi
  fi

  export_item_viz "$item_id" "$item_dir"

  if [[ "$item_success" -eq 1 ]]; then
    echo "[INFO] ${item_id} produced a successful task (${success_count}/${LOOP_COUNT}); step-task jsons: ${step_task_json_count}; videos: ${video_count}"
  else
    echo "[WARN] ${item_id} did not produce a successful task; status=${item_summary_status}, step-task jsons=${step_task_json_count}, video_status=${video_status}" >&2
  fi

  append_summary "$item_id" "$item_summary_status" "${success_count}/${LOOP_COUNT}" "$step_task_json_count" "$video_count" "$video_status" "$item_dir"
  sync_item "$item_id" "$item_dir"

  cleanup_local_item "$item_dir"
done

if [[ "$success_count" -lt "$LOOP_COUNT" ]]; then
  echo "[ERROR] Only ${success_count}/${LOOP_COUNT} successful tasks were produced." >&2
  if [[ "$MAX_ATTEMPTS" -gt 0 ]]; then
    echo "[ERROR] Reached the maximum attempt limit: ${MAX_ATTEMPTS}" >&2
  fi
  exit 1
fi

if [[ "$CLEANUP_LOCAL" -eq 1 ]]; then
  find "$LOCAL_RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'item_*' -exec rm -rf {} +
  rmdir "$LOCAL_LOG_DIR" 2>/dev/null || true
  if [[ "$KEEP_LOCAL_ITEM" -eq 0 ]]; then
    rm -f "$SUMMARY_FILE"
    rmdir "$LOCAL_RUN_DIR" 2>/dev/null || true
  fi
fi

echo "[INFO] NavGen batch completed successfully."
echo "[INFO] Success count : ${success_count}/${LOOP_COUNT}"
echo "[INFO] Attempt count : ${attempt_count}"
echo "[INFO] Elapsed time  : $(( $(date +%s) - START_TIME ))s"
echo "[INFO] Local output : $LOCAL_RUN_DIR"
echo "[INFO] Remote output: ${REMOTE_SPEC}:${REMOTE_RUN_DIR}"
