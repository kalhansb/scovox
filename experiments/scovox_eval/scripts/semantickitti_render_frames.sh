#!/usr/bin/env bash
# Render SCovox SemanticKITTI sequence frames to PNG (and optional PLY).
#
# This script launches scovox_mapping (LiDAR mode), starts the frame renderer,
# and replays SemanticKITTI seq 08 to produce frame_XXXXXX.{png,ply}.
#
# Defaults can be overridden via environment variables:
#   WS, KITTI_ROOT, SEQ, ROBOT, RESOLUTION, REPLAY_HZ, N_SCANS, SEMANTIC_MODE,
#   LABELS_SUBDIR, OUT_DIR, MAX_POINTS, VOXEL_SIZE, MAX_FRAMES,
#   EVERY_N, POINT_STRIDE, WIDTH, HEIGHT, POINT_SIZE, DYNAMIC_VIEW, BG,
#   WRITE_PLY, PLY_ONLY, SETTLE_SECS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WS:-$(realpath "$SCRIPT_DIR/../../../../..")}"

SEQ="${SEQ:-08}"
ROBOT="${ROBOT:-atlas}"
RESOLUTION="${RESOLUTION:-0.10}"
REPLAY_HZ="${REPLAY_HZ:-0.5}"
N_SCANS="${N_SCANS:--1}"
SEMANTIC_MODE="${SEMANTIC_MODE:-dirichlet}"
LABELS_SUBDIR="${LABELS_SUBDIR:-labels}"
KITTI_ROOT="${KITTI_ROOT:-$WS/data/semantickitti/dataset}"
OUT_DIR="${OUT_DIR:-$WS/results/semantickitti_seq${SEQ}_frames}"

MAX_POINTS="${MAX_POINTS:-500000}"
VOXEL_SIZE="${VOXEL_SIZE:-0.0}"
EVERY_N="${EVERY_N:-1}"
POINT_STRIDE="${POINT_STRIDE:-1}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
POINT_SIZE="${POINT_SIZE:-2.0}"
DYNAMIC_VIEW="${DYNAMIC_VIEW:-0}"
BG="${BG:-0 0 0}"
WRITE_PLY="${WRITE_PLY:-0}"
PLY_ONLY="${PLY_ONLY:-0}"
SETTLE_SECS="${SETTLE_SECS:-5}"

if [[ -z "${MAX_FRAMES:-}" ]]; then
  if [[ "$N_SCANS" -gt 0 ]]; then
    MAX_FRAMES="$N_SCANS"
  else
    MAX_FRAMES="-1"
  fi
fi

MAP_LOG="$OUT_DIR/scovox_semantickitti_map.log"
RENDER_LOG="$OUT_DIR/scovox_semantickitti_render.log"

mkdir -p "$OUT_DIR"

cleanup() {
  if [[ -n "${RENDER_PID:-}" ]] && kill -0 "$RENDER_PID" 2>/dev/null; then
    kill "$RENDER_PID" 2>/dev/null || true
  fi
  if [[ -n "${MAP_PID:-}" ]] && kill -0 "$MAP_PID" 2>/dev/null; then
    kill "$MAP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Strip conda from PATH so system Python (rclpy) is used.
export PATH
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')"

# ROS + workspace overlays
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export PYTHONPATH="$WS/src/robot_sw/distributed_mapping/scovox_eval:$PYTHONPATH"

# --- Launch SCovox mapping node (LiDAR mode) ---
ros2 launch scovox_mapping semantickitti_eval.launch.py \
  robot_name:="$ROBOT" \
  resolution:="$RESOLUTION" \
  semantic_mode:="$SEMANTIC_MODE" \
  occupancy_vis_threshold:=0.5 \
  carve_band:=0.1 \
  > "$MAP_LOG" 2>&1 &
MAP_PID=$!

# --- Launch renderer ---
RENDER_ARGS=(
  --topic "/$ROBOT/scovox_node/pointcloud"
  --sync-topic "/$ROBOT/velodyne_points"
  --out-dir "$OUT_DIR"
  --every-n "$EVERY_N"
  --max-frames "$MAX_FRAMES"
  --max-points "$MAX_POINTS"
  --point-stride "$POINT_STRIDE"
  --voxel-size "$VOXEL_SIZE"
  --width "$WIDTH"
  --height "$HEIGHT"
  --point-size "$POINT_SIZE"
  --bg $BG
)
if [[ "$DYNAMIC_VIEW" == "1" ]]; then
  RENDER_ARGS+=(--dynamic-view)
fi
if [[ "$WRITE_PLY" == "1" ]]; then
  RENDER_ARGS+=(--write-ply)
fi
if [[ "$PLY_ONLY" == "1" ]]; then
  RENDER_ARGS+=(--ply-only)
fi

python3 -m scovox_eval.render_pointcloud_frames "${RENDER_ARGS[@]}" \
  > "$RENDER_LOG" 2>&1 &
RENDER_PID=$!

# --- Replay SemanticKITTI ---
python3 -m scovox_eval.semantickitti_replay_node --ros-args \
  -p dataset_path:="$KITTI_ROOT" \
  -p sequence:="$SEQ" \
  -p rate_hz:="$REPLAY_HZ" \
  -p robot_name:="$ROBOT" \
  -p labels_subdir:="$LABELS_SUBDIR" \
  -p n_scans:="$N_SCANS"

# Allow final publish to flush
sleep "$SETTLE_SECS"

# Cleanup handled by trap

echo "Done. Frames in: $OUT_DIR"
