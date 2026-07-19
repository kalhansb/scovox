#!/bin/bash
# Residual attribution for the post-revert mIoU gap.
# Compares against `post_revert_default` (mIoU 0.3394). Each cell restores
# one pre-cleanup feature so we can isolate which contributes to the
# remaining −0.0252 gap vs the original baseline (0.3646).
#
#   res_smc          + semantic_min_confidence = 0.1   (rest off)
#   res_sat          + evidence_saturation = 1000      (rest off)
#   res_smooth_gate  + gate_k = 12 (smooth sigmoid)    (rest off)
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

# Common args (post_revert defaults).
BASE_ARGS=(
  robot_name:=ablation
  resolution:=${RES}
  semantic_occ_gate:=0.5
  range_decay_length:=-1.0
  kappa0:=2.0
  w_occ:=2.0
  w_free:=1.0
  carve_skip_occ_threshold:=0.7
)

# (cell_name, extra_overrides)
CELLS=(
  "res_smc|semantic_min_confidence:=0.1"
  "res_sat|evidence_saturation:=1000"
  "res_smooth_gate|gate_k:=12.0 gate_tau:=0.5"
)

run_cell() {
    local cell_name="$1"; shift
    local overrides=("$@")
    local CELL_DIR="${RESULTS_ROOT}/${cell_name}"
    local OUT="${CELL_DIR}/scovox.npz"
    local LOG="${CELL_DIR}/scovox_run.log"
    mkdir -p "${CELL_DIR}"
    if [[ -f "${OUT}" ]]; then echo "[${cell_name}] skip"; return; fi
    echo "━━ ${cell_name}  (extras: ${overrides[*]}) ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        "${BASE_ARGS[@]}" "${overrides[@]}" \
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
    name="${cell%%|*}"
    rest="${cell#*|}"
    if [[ -n "$rest" ]]; then
        # shellcheck disable=SC2206
        overrides=( $rest )
    else
        overrides=()
    fi
    run_cell "$name" "${overrides[@]}"
done

echo "=== residual attribution done. NPZs in ${RESULTS_ROOT} ==="
