#!/bin/bash
# A/B one build of scovox_core against another on identical SceneNN replays.
#
# Run it once per build with a different LABEL, then diff the two result dirs.
# Every config replays the SAME first N frames, so map size -- and therefore
# per-frame cost -- is comparable across builds. Replay outruns integration on
# purpose: the dataset-mode queue is KeepLast(1000), so with N < 1000 nothing is
# dropped and wall-clock is pure integration time.
#
# Usage: scenenn_ab.sh <label> [n_frames] [scene_id]
set -o pipefail
LABEL="${1:?usage: scenenn_ab.sh <label> [n_frames] [scene_id]}"
N="${2:-400}"
SCENE_ID="${3:-016}"

WS=/home/jetsondevkit/jetbot-slam/scovox
EVAL_PKG="${WS}/experiments/scovox_eval"
DATA=/home/jetsondevkit/datasets/scenenn
SCENE="${DATA}/${SCENE_ID}"
OUTDIR="${WS}/experiments/results/scenenn/${SCENE_ID}/ab/${LABEL}"
ROBOT=scenenn
RATE=25.0

[ "${N}" -lt 1000 ] || { echo "ABORT: N must stay under the 1000-message queue"; exit 1; }
mkdir -p "${OUTDIR}"

source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

run_one() {
  local name="$1"; shift
  local log="${OUTDIR}/${name}.log"
  local npz="${OUTDIR}/${name}.npz"
  echo "=== ${LABEL}/${name}: $* ==="

  ros2 launch scovox_mapping scenenn_eval.launch.py \
      robot_name:=${ROBOT} semantic_mode:=dirichlet \
      nyu_colormap:="${DATA}/nyu_color.xml" "$@" > "${log}" 2>&1 &
  local lpid=$!
  sleep 6

  python3 -m scovox_eval.scenenn_replay_node --ros-args \
      -p scene_dir:="${SCENE}" -p intrinsics:="${DATA}/asus.ini" \
      -p nyu_colormap:="${DATA}/nyu_color.xml" -p labels_dir:="${SCENE}/labels" \
      -p rate_hz:=${RATE} -p robot_name:=${ROBOT} -p n_scans:=${N} >> "${log}" 2>&1

  local prev=-1 stable=0 waited=0 last
  while [ ${waited} -lt 900 ]; do
    last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1)
    if [ "${last:-0}" -eq "${prev}" ]; then stable=$((stable+1)); else stable=0; fi
    [ ${stable} -ge 3 ] && break
    prev=${last:-0}; sleep 2; waited=$((waited+2))
  done

  timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/${ROBOT}/scovox_node/pointcloud -p output:="${npz}" >> "${log}" 2>&1 \
      || echo "  WARN: capture failed"

  kill ${lpid} 2>/dev/null || true; wait ${lpid} 2>/dev/null || true
  pkill -9 -f 'scovox_mapping_node|scenenn_replay_node|pointcloud_to_npz' 2>/dev/null || true
  sleep 2

  # Mean over the last 200 integrated frames -- skips the growing-map transient
  # so the number reflects steady state, not warm-up.
  python3 - "${log}" "${name}" "${prev}" <<'PY'
import re, sys
log, name, recv = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(log, errors="ignore").read()
fr  = [float(x) for x in re.findall(r'frame_ms=([\d.]+)', txt)]
rss = [float(x) for x in re.findall(r'rss_mb=([\d.]+)', txt)]
tail = fr[-200:] if len(fr) > 200 else fr
mean = sum(tail)/len(tail) if tail else float('nan')
print(f"  recv={recv} frames={len(fr)} mean_ms={mean:.1f} "
      f"hz={1000.0/mean if mean else 0:.2f} peak_rss={max(rss) if rss else 0:.1f}")
PY
}

# baseline config for this project: TSDF off (nothing reads the grid), stride 1
run_one base      stride:=1 enable_tsdf:=false
# TSDF on, to confirm the delta holds on the other side of that knob
run_one tsdf_on   stride:=1 enable_tsdf:=true
# short carve band -- staging is ~30x smaller here, so the win should shrink
run_one carve01   stride:=1 enable_tsdf:=false carve_band:=0.1
# publish EVERY allocated voxel: the only honest cross-build memory metric
run_one thresh0   stride:=1 enable_tsdf:=false occupancy_vis_threshold:=0.0

echo
echo "logs + npz in ${OUTDIR}"
