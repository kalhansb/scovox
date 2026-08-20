#!/bin/bash
# E7 -- embedded feasibility (C5) on Jetson AGX Orin, SceneNN 016 (RES env).
#
# Operating point: the fastest configuration that still keeps a usable TSDF
# ("trunc1" in scenenn_fps_sweep.sh) -- 5.40 Hz at 5 cm. It is NOT
# the shipped default: it narrows the TSDF band to +/-1 voxel and turns the
# batched free-space carve off. Both trades are stated in the results doc; the
# shipped default is measured here too, as the honest floor.
#
# Three things are captured per run:
#   FPS  -- frame_ms percentiles from the node's own per-frame log
#   CPU  -- process CPU% and per-core busy% from jetson_telemetry.py
#   MEM  -- process RSS (telemetry + per-frame log) and, in the mem run only,
#           the [memSplit]/[memBlocks] grid-byte breakdown
#
# Memory instrumentation is deliberately NOT on during the timing runs:
# scheduleMemUsage() spawns a whole-grid walk every 10 frames under a shared
# lock, which perturbs frame_ms. It runs once as a separate 4th run, and the
# script reports the perturbation rather than assuming it away.
#
# n=3 timing runs per the plan's principal protocol (mean +/- SD across runs).
#
# Usage: scenenn_e7_embedded.sh [n_frames] [scene_id]
#
# RES env selects the voxel size (default 0.05). The results dir defaults to
# e7 at 5 cm and e7_r<res> otherwise, so resolutions never overwrite each other.
set -o pipefail
N="${1:-400}"; SCENE_ID="${2:-016}"
RES="${RES:-0.05}"
WS=/home/jetsondevkit/jetbot-slam/scovox
EVAL_PKG="${WS}/experiments/scovox_eval"; DATA=/home/jetsondevkit/datasets/scenenn
SCENE="${DATA}/${SCENE_ID}"
DEF_TAG="e7"; [ "${RES}" = "0.05" ] || DEF_TAG="e7_r${RES#0.}"
OUTDIR="${WS}/experiments/results/scenenn/${SCENE_ID}/${E7_TAG:-${DEF_TAG}}"
TELEM="${EVAL_PKG}/jetbot/scripts/jetson_telemetry.py"
ROBOT=scenenn; RATE=25.0          # SceneNN sensor rate -> RTF period 40 ms
[ "${N}" -lt 1000 ] || { echo "ABORT: N must stay under the 1000-message queue"; exit 1; }
mkdir -p "${OUTDIR}"
source /opt/ros/humble/setup.bash; source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

# --- platform provenance: an embedded claim is meaningless without it -------
{
  echo "model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
  echo "nproc=$(nproc)"
  echo "nvpmodel=$(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)"
  echo "mem_total_mb=$(awk '/MemTotal/{printf "%.0f",$2/1024}' /proc/meminfo)"
  echo "kernel=$(uname -r)"
  echo "scene=${SCENE_ID} n_frames=${N} rate_hz=${RATE} resolution=${RES} stride=1"
} > "${OUTDIR}/platform.txt"

run_one() {
  local name="$1"; shift
  local log="${OUTDIR}/${name}.log" csv="${OUTDIR}/${name}.telem.csv"
  [ -s "${log}" ] && { echo "=== ${name}: cached, skipping"; return 0; }
  echo "=== ${name}: $*"
  pkill -9 -f 'scovox_mapping[_]node' 2>/dev/null; sleep 1

  ros2 launch scovox_mapping scenenn_eval.launch.py robot_name:=${ROBOT} \
      semantic_mode:=dirichlet resolution:=${RES} stride:=1 \
      nyu_colormap:="${DATA}/nyu_color.xml" "$@" > "${log}" 2>&1 &
  local lpid=$!; sleep 6

  # Telemetry starts only after the node is up so the CSV covers the run, not
  # the launch. It samples /proc at 1 Hz -- negligible against a ~185 ms frame.
  python3 "${TELEM}" --out "${csv}" --node-pattern 'scovox_mapping_node' \
      --interval 1.0 > /dev/null 2>&1 &
  local tpid=$!

  python3 -m scovox_eval.scenenn_replay_node --ros-args -p scene_dir:="${SCENE}" \
      -p intrinsics:="${DATA}/asus.ini" -p nyu_colormap:="${DATA}/nyu_color.xml" \
      -p labels_dir:="${SCENE}/labels" -p rate_hz:=${RATE} -p robot_name:=${ROBOT} \
      -p n_scans:=${N} > /dev/null 2>&1

  local prev=-1 stable=0 waited=0 last
  while [ ${waited} -lt 900 ]; do
    last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1)
    if [ "${last:-0}" -eq "${prev}" ]; then stable=$((stable+1)); else stable=0; fi
    [ ${stable} -ge 4 ] && break
    prev=${last:-0}; sleep 2; waited=$((waited+2))
  done

  kill ${tpid} 2>/dev/null; wait ${tpid} 2>/dev/null
  kill ${lpid} 2>/dev/null; wait ${lpid} 2>/dev/null
  pkill -9 -f 'scovox_mapping[_]node' 2>/dev/null
  pkill -9 -f 'scenenn_replay[_]node' 2>/dev/null
  sleep 2
}

# --- the 5.40 Hz operating point, n=3 --------------------------------------
CFG_FAST=(enable_tsdf:=true min_range:=0.5 max_range:=3.5
          batch_free_carve:=false sdf_trunc_voxels:=1)
for i in 1 2 3; do run_one "fast_r${i}" "${CFG_FAST[@]}"; done

# --- same point, grid-byte accounting on (perturbs timing; reported) -------
run_one fast_mem "${CFG_FAST[@]}" log_mem_usage:=true

# --- shipped default at the same range: the floor, for context -------------
run_one shipped_mem enable_tsdf:=true min_range:=0.5 max_range:=3.5 log_mem_usage:=true

echo; echo "results -> ${OUTDIR}"

# --- 2-way stage attribution at the same point ------------------------------
# Under the fused walker tsdf_ms absorbs the whole walk and sembeta_ms reads
# 0.0, so the shipped log cannot attribute cost. fused_walker:=false splits the
# walk into its two brackets. It is SLOWER (the far-voxel skip lives only on the
# fused path) -- this arm is for attribution, never for the headline rate.
run_one fast_nofuse "${CFG_FAST[@]}" fused_walker:=false
echo; echo "results -> ${OUTDIR}"
