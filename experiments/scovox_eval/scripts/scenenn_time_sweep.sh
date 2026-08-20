#!/bin/bash
# Decompose SCovox frame time on SceneNN by varying one knob at a time.
#
# Each config replays the SAME first N frames of scene 016, so map size (and
# therefore per-frame cost) is comparable across runs. Replay is pushed faster
# than integration on purpose -- the dataset-mode queue is KeepLast(1000), so
# with N < 1000 nothing is dropped and wall-clock is pure integration time.
#
# Usage: scenenn_time_sweep.sh [n_frames] [scene_id]
set -o pipefail
N="${1:-400}"
SCENE_ID="${2:-016}"

WS=/home/jetsondevkit/jetbot-slam/scovox
EVAL_PKG="${WS}/experiments/scovox_eval"
DATA=/home/jetsondevkit/datasets/scenenn
SCENE="${DATA}/${SCENE_ID}"
OUTDIR="${WS}/experiments/results/scenenn/${SCENE_ID}/sweep"
ROBOT=scenenn
RATE=25.0   # outrun integration; queue absorbs it since N < 1000

[ "${N}" -lt 1000 ] || { echo "ABORT: N must stay under the 1000-message queue"; exit 1; }
mkdir -p "${OUTDIR}"

source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

# name : extra launch args. Baseline is TSDF-off stride-1; every other row
# changes exactly one thing so the delta is attributable.
run_one() {
  local name="$1"; shift
  local log="${OUTDIR}/${name}.log"
  local npz="${OUTDIR}/${name}.npz"
  echo "=== ${name}: $* ==="

  ros2 launch scovox_mapping scenenn_eval.launch.py \
      robot_name:=${ROBOT} semantic_mode:=dirichlet \
      nyu_colormap:="${DATA}/nyu_color.xml" "$@" > "${log}" 2>&1 &
  local lpid=$!
  sleep 6

  python3 -m scovox_eval.scenenn_replay_node --ros-args \
      -p scene_dir:="${SCENE}" -p intrinsics:="${DATA}/asus.ini" \
      -p nyu_colormap:="${DATA}/nyu_color.xml" -p labels_dir:="${SCENE}/labels" \
      -p rate_hz:=${RATE} -p robot_name:=${ROBOT} -p n_scans:=${N} >> "${log}" 2>&1

  # Wait for the backlog to drain rather than guessing.
  local prev=-1 stable=0 waited=0 last
  while [ ${waited} -lt 600 ]; do
    last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1)
    if [ "${last:-0}" -eq "${prev}" ]; then stable=$((stable+1)); else stable=0; fi
    [ ${stable} -ge 3 ] && break
    prev=${last:-0}; sleep 2; waited=$((waited+2))
  done

  timeout 90 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/${ROBOT}/scovox_node/pointcloud -p output:="${npz}" >> "${log}" 2>&1 \
      || echo "  WARN: capture failed"

  kill ${lpid} 2>/dev/null || true; wait ${lpid} 2>/dev/null || true
  pkill -9 -f 'scovox_mapping_node|scenenn_replay_node|pointcloud_to_npz' 2>/dev/null || true
  sleep 2
  echo "  recv=${prev}  $(grep -oP 'frame_ms=\K[\d.]+' "${log}" | tail -1) ms last frame"
}

# --- baseline: TSDF off, stride 1, full-ray carve --------------------------
run_one base        stride:=1 enable_tsdf:=false
# --- what TSDF costs inside this same window -------------------------------
run_one tsdf_on     stride:=1 enable_tsdf:=true
# --- per-ray cost: ray count scales 1/stride^2 -----------------------------
run_one stride2     stride:=2 enable_tsdf:=false
run_one stride4     stride:=4 enable_tsdf:=false
# --- walk length: voxels per ray scales 1/resolution -----------------------
run_one res10       stride:=1 enable_tsdf:=false resolution:=0.10
# --- walk length: shorter rays, same voxel size ----------------------------
run_one maxrange2   stride:=1 enable_tsdf:=false max_range:=2.0
# --- THE hypothesis: full-ray free carve vs the Replica/KITTI 0.1 m band ---
run_one carve01     stride:=1 enable_tsdf:=false carve_band:=0.1
# --- attribute TSDF vs semantic separately (two-DDA path) ------------------
run_one nofuse      stride:=1 enable_tsdf:=true fused_walker:=false

echo
echo "logs + npz in ${OUTDIR}"
