#!/bin/bash
# Soft-probability ablation on Replica room0 with M2F+ADE150 collapsed to
# the 18-class indoor SCovox space. Single cell — same parameters as the
# post-NG baseline used in scovox_ablation_replica_room0_m2f.sh.
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
REPLICA_ROOT="${WS}/data/replica_niceslam"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
SCENE="${SCENE_OVERRIDE:-room0}"
RESULTS_ROOT="${RESULTS_ROOT_OVERRIDE:-${EVAL_PKG}/results/softprob_replica_${SCENE}_m2f}"
TOPK_DIR="${REPLICA_ROOT}/${SCENE}/semantic_m2f_topk"
RES=0.05
LABEL_DIR=semantic_m2f_ade
N_SCANS="${N_SCANS_OVERRIDE:-2000}"
# Step-9 / D7 — split-grid v2 mode toggle. Default false matches the
# legacy soft-prob baseline. smoke_split_refactor.sh and the FPS probe
# pass true via this env var when running the split path.
USE_SPLIT="${USE_SPLIT_OVERRIDE:-false}"
SHARE_TSDF="${SHARE_TSDF_OVERRIDE:-false}"
# Step 12.10 — fused single-DDA ray walker (default true per scovox_node).
# Set FUSED_WALKER_OVERRIDE=false for the legacy two-DDA split path A/B.
FUSED_WALKER="${FUSED_WALKER_OVERRIDE:-true}"

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RESULTS_ROOT}"

CELL_NAME="baseline_softprob"
CELL_DIR="${RESULTS_ROOT}/${CELL_NAME}"
mkdir -p "${CELL_DIR}"
OUT="${CELL_DIR}/scovox.npz"
LOG="${CELL_DIR}/scovox_run.log"

echo "━━ ${CELL_NAME} ━━"
ros2 launch scovox_mapping replica_eval.launch.py \
    robot_name:=ablation \
    resolution:=${RES} \
    semantic_occ_gate:=0.0 \
    range_decay_length:=-1.0 \
    w_occ:=2.0 \
    w_free:=1.0 \
    carve_skip_occ_threshold:=0.7 \
    kappa0:=2.0 \
    evidence_saturation:=1000.0 \
    semantic_min_confidence:=0.1 \
    topk_probs_dir:="${TOPK_DIR}" \
    use_split:=${USE_SPLIT} \
    share_tsdf:=${SHARE_TSDF} \
    fused_walker:=${FUSED_WALKER} \
    > "${LOG}" 2>&1 &
LPID=$!
sleep 4

python3 -m scovox_eval.replica_replay_node --ros-args \
    -p dataset_path:="${REPLICA_ROOT}/${SCENE}" \
    -p rate_hz:=2.0 \
    -p robot_name:=ablation \
    -p camera_poses:=true \
    -p semantic_subdir:=${LABEL_DIR} \
    -p n_scans:=${N_SCANS}

MIN_RECV=$((N_SCANS * 99 / 100)); WAITED=0
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

echo "=== Replica room0 soft-prob done. NPZ at ${OUT} ==="
