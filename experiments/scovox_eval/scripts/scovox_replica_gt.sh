#!/bin/bash
# SCovox runs on all 8 Replica scenes using GT-fixed semantic labels
# (semantic_gt_fixed/), matching the SLIM-VDB GT-oracle condition.
#
# Usage: bash scovox_replica_gt.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.05 — matches paper)
set -eo pipefail

RESOLUTION="${1:-0.05}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
REPLICA_ROOT="${WS}/data/replica_niceslam"
RESULTS_ROOT="${WS}/src/robot_sw/distributed_mapping/scovox_eval/results/replica_gt_${RES_CM}cm"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
TOTAL_FRAMES=2000
REPLAY_HZ=2.0

VARIANTS=(
    "scovox:scovox_mapping:replica_eval.launch.py:/scovox_node/pointcloud"
)

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RESULTS_ROOT}"

echo "========================================================"
echo "  SCovox × Replica (GT labels) — ${RES_CM}cm"
echo "  Scenes: ${SCENES[*]}   Frames each: ${TOTAL_FRAMES}"
echo "  Results: ${RESULTS_ROOT}"
echo "========================================================"

for scene in "${SCENES[@]}"; do
    SCENE_RES="${RESULTS_ROOT}/${scene}"
    mkdir -p "${SCENE_RES}"
    for variant_spec in "${VARIANTS[@]}"; do
        IFS=':' read -r name pkg launch_file pc_topic <<< "$variant_spec"
        OUT_NPZ="${SCENE_RES}/${name}.npz"
        MAP_LOG="${SCENE_RES}/${name}_run.log"

        if [[ -f "${OUT_NPZ}" ]]; then
            echo "  [${scene}/${name}] already captured, skipping"
            continue
        fi

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  scene ${scene}   variant ${name}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        ros2 launch ${pkg} ${launch_file} \
            robot_name:=atlas \
            resolution:=${RESOLUTION} \
            > "${MAP_LOG}" 2>&1 &
        LAUNCH_PID=$!
        sleep 4

        python3 -m scovox_eval.replica_replay_node --ros-args \
            -p dataset_path:="${REPLICA_ROOT}/${scene}" \
            -p rate_hz:=${REPLAY_HZ} \
            -p robot_name:=atlas \
            -p camera_poses:=true \
            -p semantic_subdir:=semantic_gt_fixed \
            -p n_scans:=${TOTAL_FRAMES}

        MIN_RECV=$((TOTAL_FRAMES * 99 / 100))
        WAITED=0
        while true; do
            LAST_RECV=$(grep -oP 'recv=\K[0-9]+' "${MAP_LOG}" 2>/dev/null | tail -1 || echo "0")
            if [ "${LAST_RECV}" -ge "${MIN_RECV}" ] 2>/dev/null; then
                echo "  [${scene}/${name}] recv=${LAST_RECV}/${TOTAL_FRAMES}, grace 30s..."
                sleep 30
                break
            fi
            sleep 5
            WAITED=$((WAITED + 5))
            if [ ${WAITED} -ge 1800 ]; then
                echo "  [${scene}/${name}] WARN: timed out at recv=${LAST_RECV}"
                sleep 30
                break
            fi
        done

        timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
            -p topic:=/atlas${pc_topic} \
            -p output:="${OUT_NPZ}" || echo "  WARN: capture timeout"

        if [[ -f "${OUT_NPZ}" ]]; then
            VOX=$(python3 -c "import numpy as np; print(len(np.load('${OUT_NPZ}')['points']))")
            echo "  [${scene}/${name}] captured ${VOX} voxels"
        else
            echo "  [${scene}/${name}] ERROR: no npz"
        fi

        kill ${LAUNCH_PID} 2>/dev/null || true
        wait ${LAUNCH_PID} 2>/dev/null || true
        # Flush any residual ROS state to avoid cross-scene topic contamination
        ros2 daemon stop 2>/dev/null || true
        sleep 4
        ros2 daemon start 2>/dev/null || true
        sleep 2
    done
done

echo ""
echo "=== done. outputs in ${RESULTS_ROOT}/<scene>/<variant>.npz ==="
