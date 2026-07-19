#!/bin/bash
# Run scovox on one SemanticKITTI sequence and capture the map as npz.
# Generic over label source (GT `labels/` vs PolarSeg `predictions/`).
#
# Usage: run_kitti.sh <seq2digit> <labels_subdir> <robot_name> <out_npz> <log>
#   e.g. run_kitti.sh 06 labels      atlas_gt  results/kitti_gt/06/scovox.npz  ...
#        run_kitti.sh 06 predictions atlas_ps  results/kitti_polarseg/06/scovox.npz ...
# NOTE: deliberately no `set -e` — SIGKILL of the launch child and the trailing
# pkill/wait return non-zero after a fully successful capture; set -e would mark
# the whole (successful) run as failed.
set -o pipefail

SEQ="$1"; LABELS_SUBDIR="$2"; ROBOT="$3"; OUT_NPZ="$4"; LOG="$5"
RESOLUTION="${6:-0.10}"
SEM_MODE="${7:-dirichlet}"
N_SCANS=100
REPLAY_HZ=0.5   # paper rate; gives the mapping node headroom so it doesn't drop clouds

WS=/home/kalhan/Projects/scovox_ws
KITTI_ROOT="${WS}/data/semantickitti/dataset"
EVAL_PKG="${WS}/experiments/scovox_eval"

# ROS env; strip conda so rclpy uses system python3.10
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/scovox/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUT_NPZ}")" "$(dirname "${LOG}")"

echo "[$(date +%T)] seq=${SEQ} labels=${LABELS_SUBDIR} robot=${ROBOT} res=${RESOLUTION}"
ros2 launch scovox_mapping semantickitti_eval.launch.py \
    robot_name:="${ROBOT}" \
    resolution:="${RESOLUTION}" \
    semantic_mode:="${SEM_MODE}" \
    > "${LOG}" 2>&1 &
LPID=$!
sleep 9   # let the mapping node fully subscribe before replay starts (else early frames are missed)

SEQ_INT=$((10#${SEQ}))   # strip leading zero (ROS int param)
python3 -m scovox_eval.semantickitti_replay_node --ros-args \
    -p dataset_path:="${KITTI_ROOT}" \
    -p sequence:=${SEQ_INT} \
    -p rate_hz:=${REPLAY_HZ} \
    -p robot_name:="${ROBOT}" \
    -p max_range:=30.0 \
    -p min_range:=1.0 \
    -p labels_subdir:="${LABELS_SUBDIR}" \
    -p n_scans:=${N_SCANS}

# Replay ran in the foreground, so all frames are already published. Now wait for
# the mapping node to drain its integration backlog. A few early frames are always
# missed during node startup, so recv plateaus below N_SCANS — wait for the recv
# counter to stop climbing (plateau) rather than for a fixed target.
PREV=-1; STABLE=0; WAITED=0
while true; do
    LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    [ -z "${LAST}" ] && LAST=0
    if [ "${LAST}" -eq "${PREV}" ] 2>/dev/null; then
        STABLE=$((STABLE+1))
    else
        STABLE=0
    fi
    if [ "${STABLE}" -ge 3 ] && [ "${LAST}" -gt 0 ] 2>/dev/null; then
        echo "  recv plateaued at ${LAST}/${N_SCANS}, grace 8s"; sleep 8; break
    fi
    PREV=${LAST}
    sleep 3; WAITED=$((WAITED+3))
    if [ ${WAITED} -ge 300 ]; then echo "  WARN timeout recv=${LAST}"; break; fi
done

timeout 90 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:="/${ROBOT}/scovox_node/pointcloud" \
    -p output:="${OUT_NPZ}" || echo "  WARN capture timeout"

kill ${LPID} 2>/dev/null || true
wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
sleep 2

if [[ -f "${OUT_NPZ}" ]]; then
    python3 -c "import numpy as np; d=np.load('${OUT_NPZ}'); print('  captured', len(d['points']), 'voxels ->', '${OUT_NPZ}')"
else
    echo "  ERROR: no npz produced for seq ${SEQ}"; exit 1
fi
