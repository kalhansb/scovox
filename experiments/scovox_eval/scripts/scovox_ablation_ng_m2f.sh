#!/bin/bash
# SCovox-NG ablation under m2f predictions (Replica, 3 scenes).
#
# Mirrors scovox_ablation_ng_nr.sh's NG block but feeds Mask2Former ADE-150
# predictions (semantic_subdir=semantic_m2f_ade) instead of GT labels. Goal:
# show whether the occupancy gate's protective value materialises under
# noisy segmentation, complementing the GT finding (gate=0 helped + 0.32 mIoU
# on GT — gate hurts under clean labels).
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
REPLICA_ROOT="${WS}/data/replica_niceslam"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

REPLICA_SCENES=(room0 office0 office3)
REPLICA_RES=0.05
REPLICA_OUT_NG="${EVAL_PKG}/results/replica_ng_m2f_5cm"
mkdir -p "${REPLICA_OUT_NG}"

for scene in "${REPLICA_SCENES[@]}"; do
    SCENE_RES="${REPLICA_OUT_NG}/${scene}"
    mkdir -p "${SCENE_RES}"
    OUT="${SCENE_RES}/scovox.npz"
    LOG="${SCENE_RES}/scovox_run.log"
    if [[ -f "${OUT}" ]]; then
        echo "[Replica-NG-m2f ${scene}] skip"; continue
    fi
    echo "━━ Replica-NG-m2f ${scene} ━━"
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
        -p semantic_subdir:=semantic_m2f_ade \
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
    pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
    pkill -9 -f 'replica_replay_node' 2>/dev/null || true
    pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
    sleep 3
done

echo "=== Replica-NG-m2f done. NPZs in ${REPLICA_OUT_NG}/<scene>/scovox.npz ==="
