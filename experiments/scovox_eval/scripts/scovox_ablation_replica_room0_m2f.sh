#!/bin/bash
# Single-knob ablation sweep on Replica room0 with m2f predictions.
# See docs/issues/ablations_punch_list.md for the candidate list.
#
# Each cell: name=<knob>=<val>, with all other knobs at the post-NG baseline:
#   range_decay=-1, w_occ=2, w_free=1, kappa0=2, evidence_saturation=1000,
#   semantic_min_confidence=0.1, carve_skip=0.7, semantic_occ_gate=0 (NG).
# (Note: replica_eval default w_occ is 6 in the launch file but the paper
#  baseline for indoor RGB-D is 2.0 — see default_params.yaml. We pass 2.0
#  explicitly to align with the documented baseline.)
#
# Outputs: results/ablations_replica_room0_m2f/<cell>/scovox.npz + .log
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
REPLICA_ROOT="${WS}/data/replica_niceslam"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RESULTS_ROOT="${RESULTS_ROOT_OVERRIDE:-${EVAL_PKG}/results/ablations_replica_room0_m2f}"
SCENE=room0
RES=0.05
LABEL_DIR=semantic_m2f_ade

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RESULTS_ROOT}"

# Baseline (everything default for the NG paper config)
BASE_ARGS=(
  robot_name:=ablation
  resolution:=${RES}
  semantic_occ_gate:=0.0
  range_decay_length:=-1.0
  w_occ:=2.0
  w_free:=1.0
  carve_skip_occ_threshold:=0.7
  kappa0:=2.0
  evidence_saturation:=1000.0
  semantic_min_confidence:=0.1
)

# Cells: cell_name|override1=val [override2=val ...]
# Override syntax matches launch arg=value pairs and is appended verbatim.
CELLS=(
  "baseline|"
  # A1 range_decay
  "rdec_5|range_decay_length:=5.0"
  "rdec_10|range_decay_length:=10.0"
  "rdec_50|range_decay_length:=50.0"
  # A2 w_occ/w_free
  "wsym_1_1|w_occ:=1.0 w_free:=1.0"
  "wfree_2|w_occ:=1.0 w_free:=2.0"
  "wocc_4|w_occ:=4.0 w_free:=1.0"
  # A3 kappa0
  "kappa_05|kappa0:=0.5"
  "kappa_1|kappa0:=1.0"
  "kappa_4|kappa0:=4.0"
  # A4 evidence_saturation
  "sat_100|evidence_saturation:=100.0"
  "sat_500|evidence_saturation:=500.0"
  "sat_5000|evidence_saturation:=5000.0"
  # A5 semantic_min_confidence
  "smc_0|semantic_min_confidence:=0.0"
  "smc_25|semantic_min_confidence:=0.25"
  "smc_50|semantic_min_confidence:=0.5"
  # A6 carve_skip_occ_threshold
  "csk_05|carve_skip_occ_threshold:=0.5"
  "csk_09|carve_skip_occ_threshold:=0.9"
)

run_cell() {
    local cell_name="$1"; shift
    local overrides=("$@")
    local CELL_DIR="${RESULTS_ROOT}/${cell_name}"
    local OUT="${CELL_DIR}/scovox.npz"
    local LOG="${CELL_DIR}/scovox_run.log"
    mkdir -p "${CELL_DIR}"
    if [[ -f "${OUT}" ]]; then echo "[${cell_name}] skip"; return; fi
    echo "━━ ${cell_name} ━━"
    # Apply overrides AFTER base so they win.
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

echo "=== Replica room0 m2f ablation done. NPZs in ${RESULTS_ROOT} ==="
