#!/bin/bash
# TSDF integration smoke test: run the baseline cell on
#   - SemanticKITTI seq 08 (polarseg predictions, 100 frames, 10 cm)
#   - Replica room0 (m2f ADE labels, 2000 frames, 5 cm)
# with TSDF integration enabled (sdf_trunc_voxels=3, default), and write to
# `tsdf_baseline/` subdirectories so the pre-TSDF `baseline/` logs are
# preserved for direct comparison via extract_run_stats.py.
#
# Mirrors scovox_ablation_kitti_seq08_polarseg.sh and
# scovox_ablation_replica_room0_m2f.sh exactly except for the cell name and
# the TSDF params. Re-run-safe via `[[ -f $OUT ]] && skip` guards.
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
REPLICA_ROOT="${WS}/data/replica_niceslam"
KITTI_RES="${EVAL_PKG}/results/ablations_kitti_seq08_polarseg/tsdf_baseline"
REPL_RES="${EVAL_PKG}/results/ablations_replica_room0_m2f/tsdf_baseline"

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${KITTI_RES}" "${REPL_RES}"

run_kitti() {
    local OUT="${KITTI_RES}/scovox.npz"
    local LOG="${KITTI_RES}/scovox_run.log"
    if [[ -f "${OUT}" ]]; then echo "[kitti tsdf_baseline] skip"; return; fi
    echo "━━ kitti seq08 tsdf_baseline ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=ablation \
        resolution:=0.10 \
        semantic_mode:=dirichlet \
        semantic_occ_gate:=0.0 \
        range_decay_length:=50.0 \
        w_occ:=6.0 \
        w_free:=1.0 \
        carve_skip_occ_threshold:=0.4 \
        kappa0:=2.0 \
        evidence_saturation:=1000.0 \
        semantic_min_confidence:=0.1 \
        sdf_trunc_voxels:=3 \
        tsdf_space_carving:=false \
        > "${LOG}" 2>&1 &
    local LPID=$!
    sleep 4
    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="${KITTI_ROOT}" \
        -p sequence:=8 \
        -p rate_hz:=0.5 \
        -p robot_name:=ablation \
        -p max_range:=30.0 \
        -p min_range:=1.0 \
        -p labels_subdir:=predictions \
        -p n_scans:=100
    local MIN_RECV=99 WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then echo "  recv=${LAST}/100, grace 10s"; sleep 10; break; fi
        sleep 3; WAITED=$((WAITED+3))
        [ ${WAITED} -ge 600 ] && { echo "  WARN: timed out at recv=${LAST}"; sleep 10; break; }
    done
    timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${OUT}" || echo "  WARN: capture timeout"
    kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' 2>/dev/null || true
    sleep 3
}

run_replica() {
    local OUT="${REPL_RES}/scovox.npz"
    local LOG="${REPL_RES}/scovox_run.log"
    if [[ -f "${OUT}" ]]; then echo "[replica tsdf_baseline] skip"; return; fi
    echo "━━ replica room0 m2f tsdf_baseline ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        robot_name:=ablation \
        resolution:=0.05 \
        semantic_occ_gate:=0.0 \
        range_decay_length:=-1.0 \
        w_occ:=2.0 \
        w_free:=1.0 \
        carve_skip_occ_threshold:=0.7 \
        kappa0:=2.0 \
        evidence_saturation:=1000.0 \
        semantic_min_confidence:=0.1 \
        sdf_trunc_voxels:=3 \
        tsdf_space_carving:=false \
        > "${LOG}" 2>&1 &
    local LPID=$!
    sleep 4
    python3 -m scovox_eval.replica_replay_node --ros-args \
        -p dataset_path:="${REPLICA_ROOT}/room0" \
        -p rate_hz:=2.0 \
        -p robot_name:=ablation \
        -p camera_poses:=true \
        -p semantic_subdir:=semantic_m2f_ade \
        -p n_scans:=2000
    local MIN_RECV=1980 WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then echo "  recv=${LAST}/2000, grace 30s"; sleep 30; break; fi
        sleep 5; WAITED=$((WAITED+5))
        [ ${WAITED} -ge 1800 ] && { echo "  WARN: timed out at recv=${LAST}"; sleep 30; break; }
    done
    timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${OUT}" || echo "  WARN: capture timeout"
    kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|replica_replay_node|pointcloud_to_npz' 2>/dev/null || true
    ros2 daemon stop 2>/dev/null || true; sleep 4; ros2 daemon start 2>/dev/null || true
    sleep 2
}

case "${1:-both}" in
    kitti)   run_kitti ;;
    replica) run_replica ;;
    both|"") run_kitti; run_replica ;;
    *) echo "usage: $0 [kitti|replica|both]"; exit 1 ;;
esac
echo "done. logs/npz at: ${KITTI_RES} and ${REPL_RES}"
