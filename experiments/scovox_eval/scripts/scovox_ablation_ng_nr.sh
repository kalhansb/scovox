#!/bin/bash
# Ablation: SCovox-NG (semantic_occ_gate=0) and SCovox-NR (range_decay_length=-1)
# on a subset of Replica (NG only — Replica baseline already has range decay off)
# and SemanticKITTI seq 08 (NG and NR).
#
# Hypothesis:
#   NG  — without the gate, free-space rays vote on labels → semantic ECE worse.
#   NR  — without range decay, far-range noisy observations carry equal weight
#         → mIoU degrades at map edges.
#
# Usage: bash scovox_ablation_ng_nr.sh
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
REPLICA_ROOT="${WS}/data/replica_niceslam"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT="${WS}/data/semantickitti/dataset"

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

# ── Replica × SCovox-NG (3 scenes) ─────────────────────────────────────
REPLICA_SCENES=(room0 office0 office3)
REPLICA_RES=0.05
REPLICA_OUT_NG="${EVAL_PKG}/results/replica_ng_5cm"
mkdir -p "${REPLICA_OUT_NG}"

for scene in "${REPLICA_SCENES[@]}"; do
    SCENE_RES="${REPLICA_OUT_NG}/${scene}"
    mkdir -p "${SCENE_RES}"
    OUT="${SCENE_RES}/scovox.npz"
    LOG="${SCENE_RES}/scovox_run.log"
    if [[ -f "${OUT}" ]]; then
        echo "[Replica-NG ${scene}] skip"; continue
    fi
    echo "━━ Replica-NG ${scene} ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        robot_name:=ablation resolution:=${REPLICA_RES} \
        semantic_occ_gate:=0.0 range_decay_length:=-1.0 \
        > "${LOG}" 2>&1 &
    LPID=$!
    sleep 4
    python3 -m scovox_eval.replica_replay_node --ros-args \
        -p dataset_path:="${REPLICA_ROOT}/${scene}" \
        -p rate_hz:=2.0 \
        -p robot_name:=ablation \
        -p camera_poses:=true \
        -p semantic_subdir:=semantic_gt_fixed \
        -p n_scans:=2000
    MIN_RECV=1980; WAITED=0
    while true; do
        LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then
            sleep 30; break
        fi
        sleep 5; WAITED=$((WAITED+5))
        [ ${WAITED} -ge 1800 ] && { sleep 30; break; }
    done
    timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${OUT}" || true
    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    # Force-kill any stray scovox/replay nodes so the next run's /ablation
    # topic has no zombie publishers contaminating the capture.
    pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
    pkill -9 -f 'replica_replay_node' 2>/dev/null || true
    pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
    pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
    sleep 3
done

# ── SemanticKITTI seq 08 × SCovox-NG and SCovox-NR ──────────────────────
KITTI_RES=0.10
KITTI_OUT_NG="${EVAL_PKG}/results/semantickitti_ng_10cm"
KITTI_OUT_NR="${EVAL_PKG}/results/semantickitti_nr_10cm"
mkdir -p "${KITTI_OUT_NG}" "${KITTI_OUT_NR}"

run_kitti_ablation() {
    local tag="$1" sem_gate="$2" range_decay="$3" out_dir="$4"
    local OUT="${out_dir}/scovox.npz"
    local LOG="${out_dir}/scovox_run.log"
    if [[ -f "${OUT}" ]]; then echo "[KITTI-${tag}] skip"; return; fi
    echo "━━ KITTI-${tag} seq 08 ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=ablation resolution:=${KITTI_RES} \
        semantic_mode:=dirichlet carve_band:=0 \
        semantic_occ_gate:=${sem_gate} range_decay_length:=${range_decay} \
        > "${LOG}" 2>&1 &
    local LPID=$!
    sleep 4
    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="${KITTI_ROOT}" \
        -p sequence:=8 \
        -p rate_hz:=0.5 \
        -p robot_name:=ablation \
        -p use_predictions:=false
    MIN_RECV=4060; WAITED=0
    while true; do
        LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then
            sleep 30; break
        fi
        sleep 10; WAITED=$((WAITED+10))
        [ ${WAITED} -ge 3600 ] && { sleep 30; break; }
    done
    timeout 180 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${OUT}" || true
    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    # Force-kill any stray scovox/replay nodes so the next run's /ablation
    # topic has no zombie publishers contaminating the capture.
    pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
    pkill -9 -f 'replica_replay_node' 2>/dev/null || true
    pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
    pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
    sleep 3
}

run_kitti_ablation "NG" "0.0" "50.0" "${KITTI_OUT_NG}"
run_kitti_ablation "NR" "0.6" "-1.0" "${KITTI_OUT_NR}"

echo "== Ablation runs done =="
