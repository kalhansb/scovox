#!/bin/bash
# B7 ablation: sensor-physics-derived (w_occ, w_free) on Replica room0 m2f.
#
# Runs three cells:
#   b7_moderate_recheck    (2.00, 1.00)  — current default; sanity check vs baseline_newgate_05
#   b7_conservative        (1.33, 0.50)  — OctoMap defaults (prob_hit=0.7, prob_miss=0.4)
#   b7_lidar               (8.00, 4.67)  — LiDAR-tuned (off-modality, sanity for KITTI param)
#
# Run from workspace root. Output: results/ablations_replica_room0_m2f/<cell>/scovox.npz
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

mkdir -p "${RESULTS_ROOT}"

# Post-cleanup baseline. Only params that still exist after the
# bayesian-mapping-cleanup branch (tau/k_gate/s_min/sat/smc/csk removed).
BASE_ARGS=(
  robot_name:=ablation
  resolution:=${RES}
  semantic_occ_gate:=0.5
  range_decay_length:=-1.0
  kappa0:=2.0
)

# (cell_name, w_occ, w_free)
CELLS=(
  "b7_moderate_recheck|2.00|1.00"
  "b7_conservative|1.33|0.50"
  "b7_lidar|8.00|4.67"
)

run_cell() {
    local cell_name="$1"; local w_occ="$2"; local w_free="$3"
    local CELL_DIR="${RESULTS_ROOT}/${cell_name}"
    local OUT="${CELL_DIR}/scovox.npz"
    local LOG="${CELL_DIR}/scovox_run.log"
    mkdir -p "${CELL_DIR}"
    if [[ -f "${OUT}" ]]; then echo "[${cell_name}] skip (npz exists)"; return; fi
    echo "━━ ${cell_name}  (w_occ=${w_occ}, w_free=${w_free}) ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        "${BASE_ARGS[@]}" w_occ:=${w_occ} w_free:=${w_free} \
        > "${LOG}" 2>&1 &
    local LPID=$!
    sleep 4
    python3 -m scovox_eval.replica_replay_node --ros-args \
        -p dataset_path:="${REPLICA_ROOT}/${SCENE}" \
        -p rate_hz:=2.0 \
        -p robot_name:=ablation \
        -p camera_poses:=true \
        -p semantic_subdir:=${LABEL_DIR} \
        -p n_scans:=2000
    local MIN_RECV=1980; local WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
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
    sleep 3
}

for cell in "${CELLS[@]}"; do
    IFS='|' read -r name w_occ w_free <<< "$cell"
    run_cell "$name" "$w_occ" "$w_free"
done

echo "=== B7 done. NPZs in ${RESULTS_ROOT} ==="
