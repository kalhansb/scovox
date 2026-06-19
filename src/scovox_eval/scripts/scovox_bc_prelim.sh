#!/bin/bash
# B/C-bucket preliminary sweep on the standard A1–A6 anchors.
#
# Goal: identify which open B/C ablations actually move mIoU on
# Replica room0 m2f and KITTI seq08 PolarSeg. Items not exercised on these
# anchors (B2 needs recompile, B6 needs instrumentation, C1–C4 need
# multi-robot fusion) are deferred and not run here.
#
# Replica cells (added to results/ablations_replica_room0_m2f/):
#   grazon_03    B4 — grazing_angle_threshold=0.3 (default value, on)
#   q_both_03    B5 — range_decay=10 AND grazing=0.3 (full quality scaling)
#
# KITTI cells (added to results/ablations_kitti_seq08_polarseg/):
#   trans_678    B3 — dynamic_classes=[6,7,8] routed to transient layer
#
# Baseline parameters mirror scovox_ablation_replica_room0_m2f.sh /
# scovox_ablation_kitti_seq08_polarseg.sh, post-NG.
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
REPLICA_ROOT="${WS}/data/replica_niceslam"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
REPL_OUT_ROOT="${EVAL_PKG}/results/ablations_replica_room0_m2f"
KITTI_OUT_ROOT="${EVAL_PKG}/results/ablations_kitti_seq08_polarseg"
SCENE=room0
LABEL_DIR=semantic_m2f_ade

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${REPL_OUT_ROOT}" "${KITTI_OUT_ROOT}"

REPL_BASE_ARGS=(
  robot_name:=ablation
  resolution:=0.05
  semantic_occ_gate:=0.0
  range_decay_length:=-1.0
  w_occ:=2.0
  w_free:=1.0
  carve_skip_occ_threshold:=0.7
  kappa0:=2.0
  evidence_saturation:=1000.0
  semantic_min_confidence:=0.1
)

KITTI_BASE_ARGS=(
  robot_name:=ablation
  resolution:=0.10
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

run_replica_cell() {
    local cell_name="$1"; shift
    local overrides=("$@")
    local CELL_DIR="${REPL_OUT_ROOT}/${cell_name}"
    local OUT="${CELL_DIR}/scovox.npz"
    local LOG="${CELL_DIR}/scovox_run.log"
    mkdir -p "${CELL_DIR}"
    if [[ -f "${OUT}" ]]; then echo "[replica ${cell_name}] skip"; return; fi
    echo "━━ replica ${cell_name} ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        "${REPL_BASE_ARGS[@]}" "${overrides[@]}" \
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
    local MIN_RECV=1980 WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 30; break; fi
        sleep 5; WAITED=$((WAITED+5))
        [ ${WAITED} -ge 1800 ] && { sleep 30; break; }
    done
    timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${OUT}" || true
    kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|replica_replay_node|pointcloud_to_npz' 2>/dev/null || true
    ros2 daemon stop 2>/dev/null || true; sleep 4; ros2 daemon start 2>/dev/null || true
}

run_kitti_cell() {
    local cell_name="$1"; shift
    local overrides=("$@")
    local CELL_DIR="${KITTI_OUT_ROOT}/${cell_name}"
    local OUT="${CELL_DIR}/scovox.npz"
    local LOG="${CELL_DIR}/scovox_run.log"
    mkdir -p "${CELL_DIR}"
    if [[ -f "${OUT}" ]]; then echo "[kitti ${cell_name}] skip"; return; fi
    echo "━━ kitti ${cell_name} ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        "${KITTI_BASE_ARGS[@]}" "${overrides[@]}" \
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
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 10; break; fi
        sleep 3; WAITED=$((WAITED+3))
        [ ${WAITED} -ge 600 ] && { sleep 10; break; }
    done
    timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${OUT}" || true
    kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' 2>/dev/null || true
    sleep 3
}

# ---- Cells -----------------------------------------------------------------

# B4: grazing-angle weighting on (default 0.3)
run_replica_cell grazon_03 \
    grazing_angle_threshold:=0.3

# B5: full quality scaling (range_decay + grazing both on). Other corners
# (q=1, range-only, angle-only) are covered by baseline / A1 rdec_10 / B4 above.
run_replica_cell q_both_03 \
    range_decay_length:=10.0 \
    grazing_angle_threshold:=0.3

# B3: route persons/bicyclists/motorcyclists to transient layer on KITTI.
# Learning IDs 6, 7, 8 in semantic_kitti's 20-class taxonomy.
run_kitti_cell trans_678 \
    "dynamic_classes:=[6,7,8]"

echo "=== B/C prelim cells done ==="
