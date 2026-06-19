#!/bin/bash
# Extend the SCovox-NG ablation to all 8 Replica scenes for both label
# regimes (semantic_gt_fixed and semantic_m2f_ade). Skips scenes that
# already have an NPZ. The 3-scene baseline NG runs (room0, office0,
# office3) live in:
#   replica_ng_5cm/      (GT-oracle labels)
#   replica_ng_m2f_5cm/  (m2f predictions)
# This script adds the remaining 5 scenes (room1, room2, office1, office2,
# office4) to each, so the NG mIoU column is comparable to the 8-scene
# baseline (gate=0.6) numbers in exp3 / paper_experiments tables.
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
REPLICA_ROOT="${WS}/data/replica_niceslam"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
EXTRA_SCENES=(room1 room2 office1 office2 office4)
RES=0.05

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

run_replica_ng() {
    local label_dir="$1"     # semantic_gt_fixed | semantic_m2f_ade
    local out_root="$2"      # results/replica_ng_5cm | replica_ng_m2f_5cm
    mkdir -p "${out_root}"
    for scene in "${EXTRA_SCENES[@]}"; do
        local SCENE_RES="${out_root}/${scene}"
        mkdir -p "${SCENE_RES}"
        local OUT="${SCENE_RES}/scovox.npz"
        local LOG="${SCENE_RES}/scovox_run.log"
        if [[ -f "${OUT}" ]]; then echo "[$(basename ${out_root}) ${scene}] skip"; continue; fi
        echo "━━ $(basename ${out_root}) ${scene} ━━"
        ros2 launch scovox_mapping replica_eval.launch.py \
            robot_name:=ablation resolution:=${RES} \
            semantic_occ_gate:=0.0 range_decay_length:=-1.0 \
            > "${LOG}" 2>&1 &
        local LPID=$!
        sleep 4
        python3 -m scovox_eval.replica_replay_node --ros-args \
            -p dataset_path:="${REPLICA_ROOT}/${scene}" \
            -p rate_hz:=2.0 \
            -p robot_name:=ablation \
            -p camera_poses:=true \
            -p semantic_subdir:=${label_dir} \
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
    done
}

echo "=== Stage A: Replica GT × NG (extend to 8 scenes) ==="
run_replica_ng "semantic_gt_fixed" "${EVAL_PKG}/results/replica_ng_5cm"

echo "=== Stage B: Replica m2f × NG (extend to 8 scenes) ==="
run_replica_ng "semantic_m2f_ade" "${EVAL_PKG}/results/replica_ng_m2f_5cm"

echo "=== Replica NG extension done. ==="
