#!/bin/bash
# Single-knob ablation sweep on SemanticKITTI seq 08 with PolarSeg
# predictions, 100 frames, 10 cm voxel.
# See docs/issues/ablations_punch_list.md for the candidate list.
#
# Baseline (post-NG): range_decay=50, w_occ=6, w_free=1, kappa0=2,
#   evidence_saturation=1000, semantic_min_confidence=0.1, carve_skip=0.4,
#   semantic_occ_gate=0 (NG).
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
KITTI_ROOT="${WS}/data/semantickitti/dataset"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RESULTS_ROOT="${RESULTS_ROOT_OVERRIDE:-${EVAL_PKG}/results/ablations_kitti_seq08_polarseg}"
SEQ=08
SEQ_INT=8
RES=0.10
N_SCANS=100
REPLAY_HZ=0.5

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RESULTS_ROOT}"

BASE_ARGS=(
  robot_name:=ablation
  resolution:=${RES}
  semantic_mode:=dirichlet
  semantic_occ_gate:=0.0
  range_decay_length:=50.0
  w_occ:=6.0
  w_free:=1.0
  carve_skip_occ_threshold:=0.4
  kappa0:=2.0
  evidence_saturation:=1000.0
  semantic_min_confidence:=0.1
)

CELLS=(
  "baseline|"
  # A1 range_decay
  "rdec_off|range_decay_length:=-1.0"
  "rdec_10|range_decay_length:=10.0"
  "rdec_30|range_decay_length:=30.0"
  # A2 w_occ/w_free
  "wsym_1_1|w_occ:=1.0 w_free:=1.0"
  "wfree_2|w_occ:=1.0 w_free:=2.0"
  "wocc_2|w_occ:=2.0 w_free:=1.0"
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
  "csk_07|carve_skip_occ_threshold:=0.7"
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
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        "${BASE_ARGS[@]}" "${overrides[@]}" \
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

echo "=== KITTI seq 08 PolarSeg ablation done. NPZs in ${RESULTS_ROOT} ==="
