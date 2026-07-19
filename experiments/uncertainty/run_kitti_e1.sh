#!/bin/bash
# E1: deterministic KITTI replay -> capture the split Beta/Dir map snapshot.
# Uses the split substrate (use_split=true) the uncertainty plan concerns, in
# rolling mode (enables ScovoxMapBinary) with a 1 Hz share timer so a post-replay
# capture subscriber triggers a FULL snapshot (a_occ/a_free for occupied AND
# carved-free voxels). During replay there is no bin subscriber, so the timer is
# a cheap no-op and recv stays 100/100.
#
# Usage: run_kitti_e1.sh <seq2digit> [robot] [out_dir] [res] [sem_mode]
set -o pipefail
SEQ="${1:-06}"; ROBOT="${2:-atlas_e1}"
OUT_DIR="${3:-/home/kalhan/Projects/scovox_ws/experiments/results/kitti_e1/${SEQ}}"
RES="${4:-0.10}"; SEM_MODE="${5:-dirichlet}"
N_SCANS=100; REPLAY_HZ=0.5

WS=/home/kalhan/Projects/scovox_ws
KITTI_ROOT="${WS}/data/semantickitti/dataset"
EVAL_PKG="${WS}/experiments/scovox_eval"
LOG="${OUT_DIR}/run.log"
BIN_NPZ="${OUT_DIR}/map_bin.npz"
mkdir -p "${OUT_DIR}"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/scovox/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

pgrep -af 'scovox_mapping_node|semantickitti_replay_node|scovox_bin_capture' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2

echo "[$(date +%T)] E1 seq=${SEQ} robot=${ROBOT} res=${RES} split+rolling"
ros2 launch scovox_mapping semantickitti_eval.launch.py \
    robot_name:="${ROBOT}" resolution:="${RES}" semantic_mode:="${SEM_MODE}" \
    use_split:=true map_mode:=rolling share_rate_hz:=1.0 num_classes:=20 \
    > "${LOG}" 2>&1 &
LPID=$!
sleep 9

SEQ_INT=$((10#${SEQ}))
python3 -m scovox_eval.semantickitti_replay_node --ros-args \
    -p dataset_path:="${KITTI_ROOT}" -p sequence:=${SEQ_INT} \
    -p rate_hz:=${REPLAY_HZ} -p robot_name:="${ROBOT}" \
    -p max_range:=30.0 -p min_range:=1.0 \
    -p labels_subdir:=labels -p n_scans:=${N_SCANS} \
    > "${OUT_DIR}/replay.log" 2>&1

# Wait for the mapping node to drain its integration backlog (recv plateau).
PREV=-1; STABLE=0; WAITED=0
while true; do
    LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    [ -z "${LAST}" ] && LAST=0
    if [ "${LAST}" -eq "${PREV}" ] 2>/dev/null; then STABLE=$((STABLE+1)); else STABLE=0; fi
    if [ "${STABLE}" -ge 3 ] && [ "${LAST}" -gt 0 ] 2>/dev/null; then
        echo "  recv plateaued at ${LAST}/${N_SCANS}, grace 6s"; sleep 6; break; fi
    PREV=${LAST}; sleep 3; WAITED=$((WAITED+3))
    if [ ${WAITED} -ge 300 ]; then echo "  WARN timeout recv=${LAST}"; break; fi
done
echo "RECV=${LAST}/${N_SCANS}" | tee "${OUT_DIR}/recv.txt"

# Connect the capture subscriber -> triggers a full snapshot on the next 1 Hz tick.
echo "  capturing binary snapshot ..."
timeout 60 python3 -m scovox_eval.scovox_bin_capture --ros-args \
    -p topic:="/${ROBOT}/scovox_node/scovox_bin" -p output:="${BIN_NPZ}" \
    2>&1 | tail -3 || echo "  WARN bin capture timeout"

kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
pgrep -af 'scovox_mapping_node|semantickitti_replay_node|scovox_bin_capture' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2

if [[ -f "${BIN_NPZ}" ]]; then
    python3 - "${BIN_NPZ}" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1])
ao, af = d["a_occ"], d["a_free"]; s = ao+af; p = ao/np.where(s>0,s,1)
print(f"  captured beta={len(ao)} dir={len(d['dir_xyz'])} res={float(d['resolution'])}")
print(f"  p_occ: free(<.5)={np.mean(p<0.5):.1%} occ(>.5)={np.mean(p>0.5):.1%} "
      f"med_s={np.median(s):.1f}")
PY
else
    echo "  ERROR: no binary npz for seq ${SEQ}"; exit 1
fi
echo "DONE E1 seq ${SEQ}"
