#!/bin/bash
# Step-Bayesian: hard early-return at p_occ < 0.5 + Bayesian-soft
# (weight=p_occ) for accepted voxels + evidence_saturation=1000.
# No smooth-gate sigmoid (gate_k=0).
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
REPLICA_ROOT="${WS}/data/replica_niceslam"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RESULTS_ROOT="${EVAL_PKG}/results/ablations_replica_room0_m2f"
SCENE=room0
RES=0.05
LABEL_DIR=semantic_m2f_ade

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

CELL_NAME="res_step_bayesian"
CELL_DIR="${RESULTS_ROOT}/${CELL_NAME}"
OUT="${CELL_DIR}/scovox.npz"
LOG="${CELL_DIR}/scovox_run.log"
mkdir -p "${CELL_DIR}"
[[ -f "${OUT}" ]] && { echo "[skip] ${CELL_NAME}"; exit 0; }

echo "━━ ${CELL_NAME}  (dir_min=0.5, sat=1000, gate_k=0 = Bayesian-soft, no smooth gate) ━━"
ros2 launch scovox_mapping replica_eval.launch.py \
    robot_name:=ablation \
    resolution:=${RES} \
    semantic_occ_gate:=0.5 \
    range_decay_length:=-1.0 \
    kappa0:=2.0 \
    w_occ:=2.0 \
    w_free:=1.0 \
    carve_skip_occ_threshold:=0.7 \
    dirichlet_min_p_occ:=0.5 \
    evidence_saturation:=1000 \
    gate_k:=0.0 \
    > "${LOG}" 2>&1 &
LPID=$!
sleep 4

python3 -m scovox_eval.replica_replay_node --ros-args \
    -p dataset_path:="${REPLICA_ROOT}/${SCENE}" \
    -p rate_hz:=2.0 \
    -p robot_name:=ablation \
    -p camera_poses:=true \
    -p semantic_subdir:=${LABEL_DIR} \
    -p n_scans:=2000

MIN_RECV=1980; WAITED=0
while true; do
    LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 30; break; fi
    sleep 5; WAITED=$((WAITED+5))
    [ ${WAITED} -ge 1800 ] && { sleep 30; break; }
done

timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/ablation/scovox_node/pointcloud \
    -p output:="${OUT}" || true

kill ${LPID} 2>/dev/null || true
wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
pkill -9 -f 'replica_replay_node' 2>/dev/null || true
pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true

echo "=== ${CELL_NAME} done ==="
