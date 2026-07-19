#!/bin/bash
# E1 (SceneNet): deterministic RGB-D replay -> capture the split Beta/Dir map
# snapshot. Uses the split substrate (use_split=true) in rolling mode (enables
# ScovoxMapBinary) with a 1 Hz share timer so a post-replay capture subscriber
# triggers a FULL snapshot (a_occ/a_free for occupied AND carved-free voxels).
# During replay there is no bin subscriber, so the timer is a cheap no-op and
# recv stays 300/300. Depth/seg replay uses reliable QoS (deep queue) so frames
# buffer if integration lags -> deterministic delivery.
#
# Usage: run_scenenet_e1.sh <seq> [robot] [out_dir] [res]
set -o pipefail
SEQ="${1:-0_223}"; ROBOT="${2:-atlas_e1}"
OUT_DIR="${3:-/home/kalhan/Projects/scovox_ws/experiments/results/scenenet_e1/${SEQ}}"
RES="${4:-0.05}"
N_FRAMES=300; REPLAY_HZ=4.0

WS=/home/kalhan/Projects/scovox_ws
SN_ROOT="${WS}/data/scenenet_val_layout"
EVAL_PKG="${WS}/experiments/scovox_eval"
LOG="${OUT_DIR}/run.log"
BIN_NPZ="${OUT_DIR}/map_bin.npz"
mkdir -p "${OUT_DIR}"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/scovox/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

pgrep -af 'scovox_mapping_node|scenenet_replay_node|scovox_bin_capture' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2

echo "[$(date +%T)] E1 seq=${SEQ} robot=${ROBOT} res=${RES} split+rolling (scenenet)"
ros2 launch scovox_mapping scenenet_eval.launch.py \
    robot_name:="${ROBOT}" resolution:="${RES}" semantic_mode:=dirichlet \
    use_split:=true map_mode:=rolling share_rate_hz:=1.0 scovox_publish_rate:=0.2 \
    > "${LOG}" 2>&1 &
LPID=$!
sleep 9

python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${SN_ROOT}" -p sequence:="${SEQ}" \
    -p rate_hz:=${REPLAY_HZ} -p robot_name:="${ROBOT}" \
    -p use_gt_labels:=true -p n_scans:=${N_FRAMES} \
    > "${OUT_DIR}/replay.log" 2>&1

# Wait for the mapping node to drain its integration backlog (recv plateau).
PREV=-1; STABLE=0; WAITED=0
while true; do
    LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    [ -z "${LAST}" ] && LAST=0
    if [ "${LAST}" -eq "${PREV}" ] 2>/dev/null; then STABLE=$((STABLE+1)); else STABLE=0; fi
    if [ "${STABLE}" -ge 3 ] && [ "${LAST}" -gt 0 ] 2>/dev/null; then
        echo "  recv plateaued at ${LAST}/${N_FRAMES}, grace 6s"; sleep 6; break; fi
    PREV=${LAST}; sleep 3; WAITED=$((WAITED+3))
    if [ ${WAITED} -ge 360 ]; then echo "  WARN timeout recv=${LAST}"; break; fi
done
echo "RECV=${LAST}/${N_FRAMES}" | tee "${OUT_DIR}/recv.txt"

echo "  capturing binary snapshot ..."
timeout 60 python3 -m scovox_eval.scovox_bin_capture --ros-args \
    -p topic:="/${ROBOT}/scovox_node/scovox_bin" -p output:="${BIN_NPZ}" \
    2>&1 | tail -3 || echo "  WARN bin capture timeout"

kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
pgrep -af 'scovox_mapping_node|scenenet_replay_node|scovox_bin_capture' \
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
