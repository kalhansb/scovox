#!/bin/bash
# B7 ablation: sensor-physics-derived (w_occ, w_free) on SemanticKITTI seq 08 PolarSeg.
#
# Key purpose: confirm whether the post-cleanup mIoU regression seen on
# Replica room0 also appears on LiDAR. Pre-cleanup baseline lives in
# results/ablations_kitti_seq08_polarseg/baseline/ (2026-04-26).
#
# Three cells:
#   b7_kitti_old_default     (6.00, 1.00)  — pre-cleanup KITTI default; recheck
#   b7_kitti_lidar_physics   (8.00, 4.67)  — sensor-physics LiDAR derivation
#   b7_kitti_moderate        (2.00, 1.00)  — RGB-D modality mismatch (negative ctrl)
#
# 100 scans @ 0.5 Hz ≈ 4-5 min/cell.
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

mkdir -p "${RESULTS_ROOT}"

# Post-cleanup-only param set (no sat/smc/csk).
BASE_ARGS=(
  robot_name:=ablation
  resolution:=${RES}
  semantic_mode:=dirichlet
  semantic_occ_gate:=0.5
  range_decay_length:=50.0
  kappa0:=2.0
)

# (cell_name, w_occ, w_free)
CELLS=(
  "b7_kitti_old_default|6.00|1.00"
  "b7_kitti_lidar_physics|8.00|4.67"
  "b7_kitti_moderate|2.00|1.00"
)

run_cell() {
    local cell_name="$1"; local w_occ="$2"; local w_free="$3"
    local CELL_DIR="${RESULTS_ROOT}/${cell_name}"
    local OUT="${CELL_DIR}/scovox.npz"
    local LOG="${CELL_DIR}/scovox_run.log"
    mkdir -p "${CELL_DIR}"
    if [[ -f "${OUT}" ]]; then echo "[${cell_name}] skip"; return; fi
    echo "━━ ${cell_name}  (w_occ=${w_occ}, w_free=${w_free}) ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        "${BASE_ARGS[@]}" w_occ:=${w_occ} w_free:=${w_free} \
        > "${LOG}" 2>&1 &
    local LPID=$!
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
    local MIN_RECV=$((N_SCANS * 99 / 100)); local WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
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
    sleep 2
}

for cell in "${CELLS[@]}"; do
    IFS='|' read -r name w_occ w_free <<< "$cell"
    run_cell "$name" "$w_occ" "$w_free"
done

echo "=== KITTI B7 done. NPZs in ${RESULTS_ROOT} ==="
