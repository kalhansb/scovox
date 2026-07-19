#!/bin/bash
# Soft-probability ablation on SemanticKITTI seq 08 with PolarSeg
# top-K (K=5) per-point distributions. Single cell — same parameters as
# the post-NG baseline used in scovox_ablation_kitti_seq08_polarseg.sh.
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
KITTI_ROOT="${WS}/data/semantickitti/dataset"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
SEQ="${SEQ_OVERRIDE:-08}"
RESULTS_ROOT="${RESULTS_ROOT_OVERRIDE:-${EVAL_PKG}/results/softprob_kitti_seq${SEQ}_polarseg}"
TOPK_DIR="${KITTI_ROOT}/sequences/${SEQ}/predictions_topk"
SEQ_INT=$((10#${SEQ}))
RES=0.10
N_SCANS="${N_SCANS_OVERRIDE:-100}"
REPLAY_HZ=0.5
# Step-9 / D7 — split-grid v2 mode toggle. Default false matches the
# legacy soft-prob baseline. smoke_split_refactor.sh passes true via
# this env var when running the --gate KITTI check on the split path.
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
ros2 launch scovox_mapping semantickitti_eval.launch.py \
    robot_name:=ablation \
    resolution:=${RES} \
    semantic_mode:=dirichlet \
    semantic_occ_gate:=0.0 \
    range_decay_length:=50.0 \
    w_occ:=6.0 \
    w_free:=1.0 \
    carve_skip_occ_threshold:=0.4 \
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

python3 -m scovox_eval.semantickitti_replay_node --ros-args \
    -p dataset_path:="${KITTI_ROOT}" \
    -p sequence:=${SEQ_INT} \
    -p rate_hz:=${REPLAY_HZ} \
    -p robot_name:=ablation \
    -p max_range:=30.0 \
    -p min_range:=1.0 \
    -p labels_subdir:=predictions \
    -p n_scans:=${N_SCANS} \
    -p soft_prob_passthrough:=true

MIN_RECV=$((N_SCANS * 99 / 100)); WAITED=0
while true; do
    LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 10; break; fi
    sleep 3; WAITED=$((WAITED+3))
    [ ${WAITED} -ge 600 ] && { echo "  WARN: timeout @ recv=${LAST}"; break; }
done

timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/ablation/scovox_node/pointcloud \
    -p output:="${OUT}" || echo "  WARN: capture timeout"

kill ${LPID} 2>/dev/null || true
wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true

echo "=== KITTI seq 08 soft-prob done. NPZ at ${OUT} ==="
