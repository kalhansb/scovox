#!/bin/bash
# KITTI verification: confirm new defaults (sat=1000, dir_min=0.5) don't
# regress the LiDAR anchor. Expected: mIoU ~ 0.41-0.42 (within noise).
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
KITTI_ROOT="${WS}/data/semantickitti/dataset"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RESULTS_ROOT="${EVAL_PKG}/results/ablations_kitti_seq08_polarseg"
SEQ_INT=8
RES=0.10
N_SCANS=100
REPLAY_HZ=0.5

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

CELL_NAME="kitti_new_defaults"
CELL_DIR="${RESULTS_ROOT}/${CELL_NAME}"
OUT="${CELL_DIR}/scovox.npz"
LOG="${CELL_DIR}/scovox_run.log"
mkdir -p "${CELL_DIR}"
[[ -f "${OUT}" ]] && { echo "[skip] ${CELL_NAME}"; exit 0; }

echo "━━ ${CELL_NAME}  (KITTI w/ new defaults) ━━"
ros2 launch scovox_mapping semantickitti_eval.launch.py \
    robot_name:=ablation \
    resolution:=${RES} \
    semantic_mode:=dirichlet \
    semantic_occ_gate:=0.5 \
    range_decay_length:=50.0 \
    kappa0:=2.0 \
    w_occ:=6.0 \
    w_free:=1.0 \
    > "${LOG}" 2>&1 &
LPID=$!
sleep 4

python3 -m scovox_eval.semantickitti_replay_node --ros-args \
    -p dataset_path:="${KITTI_ROOT}" \
    -p sequence:=${SEQ_INT} \
    -p rate_hz:=${REPLAY_HZ} \
    -p robot_name:=ablation \
    -p max_range:=30.0 \
    -p min_range:=1.0 \
    -p labels_subdir:=predictions \
    -p n_scans:=${N_SCANS}

MIN_RECV=$((N_SCANS * 99 / 100)); WAITED=0
while true; do
    LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 10; break; fi
    sleep 3; WAITED=$((WAITED+3))
    [ ${WAITED} -ge 600 ] && { sleep 10; break; }
done

timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/ablation/scovox_node/pointcloud \
    -p output:="${OUT}" || true

kill ${LPID} 2>/dev/null || true
wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true

echo "=== ${CELL_NAME} done ==="
