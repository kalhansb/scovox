#!/bin/bash
# SceneNN (HKUST) GT-semantics run on the Jetson.
# Usage: run_scenenn.sh [scene_id] [resolution] [rate_hz] [stride] [enable_tsdf] [carve_band] [tag]
#
# Prereqs, in order (see docs/scenenn_jetson_run.md):
#   1. scene downloaded from https://hkust-vgd.ust.hk/scenenn/ (oni + trajectory.log + *_nyu.ply)
#   2. frames extracted:  playback <scene>.oni <scene>/frames
#   3. mesh registered:   scenenn_register_mesh.py  -> mesh_align.json
#   4. labels rendered:   scenenn_make_labels.py    -> <scene>/labels/
set -o pipefail
SCENE_ID="${1:-016}"
RES="${2:-0.05}"
# Replay outruns integration, and the backlog is bounded by the dataset-mode
# input queue (KeepLast(1000), reliable, both ends). Measured on 016: stride 1
# integrates at ~2.6 Hz, so 6 Hz replay leaves a ~500-frame backlog at the end
# of a 1300-frame scene -- under the 1000 limit, but not by much. Raise RATE
# only together with stride, or frames are lost silently rather than slowly.
RATE="${3:-6.0}"
STRIDE="${4:-2}"
# This config never reads the TsdfMap grid (share_tsdf=0, no tsdf_pointcloud
# publisher, no extract_mesh), so enable_tsdf:=false only drops the per-voxel
# band writes inside the fused walker. Verified on 016: the two npz files are
# bit-identical across all 13 fields, at 15% less frame time and 45% less RSS.
# Leave this true only if you need ~/tsdf_pointcloud or ~/extract_mesh.
ENABLE_TSDF="${5:-true}"
# Optional suffix so A/B runs do not overwrite each other's log/npz/telemetry.
TAG="${6:-}"
# Anything after the tag is forwarded verbatim to `ros2 launch`, so one-off
# ablations need no edit here:
#   run_scenenn.sh 016 0.05 6.0 1 true _trunc030 sdf_trunc_voxels:=6
EXTRA=("${@:7}")

WS=/home/jetsondevkit/jetbot-slam/scovox
EVAL_PKG="${WS}/experiments/scovox_eval"
DATA=/home/jetsondevkit/datasets/scenenn
SCENE="${DATA}/${SCENE_ID}"
RES_ROOT="${WS}/experiments/results/scenenn/${SCENE_ID}"
ROBOT=scenenn

mkdir -p "${RES_ROOT}"
LOG="${RES_ROOT}/scovox${TAG}.log"; OUT="${RES_ROOT}/scovox${TAG}.npz"

for f in "${SCENE}/trajectory.log" "${DATA}/asus.ini" "${DATA}/nyu_color.xml"; do
  [ -f "$f" ] || { echo "[run_scenenn] ABORT: missing $f"; exit 1; }
done
NFRAMES=$(ls "${SCENE}/frames/depth" 2>/dev/null | wc -l)
NLABELS=$(ls "${SCENE}/labels" 2>/dev/null | wc -l)
[ "${NFRAMES}" -gt 0 ] || { echo "[run_scenenn] ABORT: no extracted frames in ${SCENE}/frames/depth"; exit 1; }
echo "[run_scenenn] scene ${SCENE_ID}: ${NFRAMES} frames, ${NLABELS} label PNGs, res=${RES} rate=${RATE}Hz stride=${STRIDE} enable_tsdf=${ENABLE_TSDF} extra=${EXTRA[*]}"
[ "${NLABELS}" -gt 0 ] || echo "[run_scenenn] WARNING: no labels -- map will be occupancy-only"

source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

ros2 launch scovox_mapping scenenn_eval.launch.py \
    robot_name:=${ROBOT} resolution:=${RES} semantic_mode:=dirichlet stride:=${STRIDE} \
    enable_tsdf:=${ENABLE_TSDF} \
    nyu_colormap:="${DATA}/nyu_color.xml" "${EXTRA[@]}" > "${LOG}" 2>&1 &
LPID=$!
sleep 6

# System-level cost: the node logs frame_ms/rss_mb itself but never CPU.
# Sample the process and the 12 Orin cores at 1 Hz alongside it.
TELEM="${RES_ROOT}/telemetry${TAG}.csv"
python3 "${EVAL_PKG}/jetbot/scripts/jetson_telemetry.py" \
    --out "${TELEM}" --node-pattern scovox_mapping_node >> "${LOG}" 2>&1 &
TPID=$!

python3 -m scovox_eval.scenenn_replay_node --ros-args \
    -p scene_dir:="${SCENE}" -p intrinsics:="${DATA}/asus.ini" \
    -p nyu_colormap:="${DATA}/nyu_color.xml" -p labels_dir:="${SCENE}/labels" \
    -p rate_hz:=${RATE} -p robot_name:=${ROBOT} >> "${LOG}" 2>&1

# Let integration drain: the node logs recv=<n> per frame, so wait until it
# stops advancing rather than guessing a fixed sleep.
PREV=-1; STABLE=0; WAITED=0
while [ ${WAITED} -lt 300 ]; do
  LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
  if [ "${LAST:-0}" -eq "${PREV}" ]; then STABLE=$((STABLE+1)); else STABLE=0; fi
  [ ${STABLE} -ge 3 ] && break
  PREV=${LAST:-0}; sleep 3; WAITED=$((WAITED+3))
done
echo "[run_scenenn] integration settled at recv=${PREV} after ${WAITED}s"

timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/${ROBOT}/scovox_node/pointcloud -p output:="${OUT}" \
    || echo "[run_scenenn] WARN capture timeout"

# SIGTERM so the telemetry writer flushes its CSV before the node dies.
kill ${TPID} 2>/dev/null || true; wait ${TPID} 2>/dev/null || true
kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node|scenenn_replay_node|pointcloud_to_npz' 2>/dev/null || true

echo "[run_scenenn] map -> ${OUT}"
[ -f "${TELEM}" ] && echo "[run_scenenn] telemetry -> ${TELEM} ($(wc -l < "${TELEM}") rows)"
ls -la "${OUT}" 2>/dev/null || echo "[run_scenenn] no npz produced -- check ${LOG}"
