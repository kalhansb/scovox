#!/bin/bash
# Run SCovox variants on SemanticKITTI sequence 08.
# Processes each variant sequentially.
#
# Usage: bash semantickitti_create_dataset.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.05)
#   Examples:
#     bash semantickitti_create_dataset.sh          # 5cm (default)
#     bash semantickitti_create_dataset.sh 0.10     # 10cm
#
# Prerequisites:
#   - Workspace built: colcon build && source install/setup.bash
#   - SemanticKITTI seq 08 at $KITTI_ROOT
#   - scovox_eval package installed (pip install -e .)

RESOLUTION="${1:-0.05}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

KITTI_ROOT="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/semantickitti/dataset"
BASE_RESULTS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/src/robot_sw/distributed_mapping/scovox_eval/results"
RESULTS_ROOT="$BASE_RESULTS/semantickitti_${RES_CM}cm"
WS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG="$WS/src/robot_sw/distributed_mapping/scovox_eval"
SEQUENCE="08"
REPLAY_HZ=0.5
TOTAL_FRAMES=4071

# SCovox variants: name:semantic_mode
VARIANTS=(
    "scovox:dirichlet"
    "scovox_mv:majority_vote"
    "scovox_np:naive"
)

# Strip conda from PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export PYTHONPATH="$EVAL_PKG:$PYTHONPATH"

mkdir -p "$RESULTS_ROOT"

echo "============================================"
echo "  SemanticKITTI Evaluation (seq $SEQUENCE)"
echo "  ${#VARIANTS[@]} variants"
echo "  Resolution: ${RESOLUTION}m (${RES_CM}cm)"
echo "  Results: $RESULTS_ROOT"
echo "============================================"
echo ""

for variant_spec in "${VARIANTS[@]}"; do
    IFS=':' read -r name sem_mode <<< "$variant_spec"
    OUT_NPZ="$RESULTS_ROOT/${name}.npz"

    if [ -f "$OUT_NPZ" ]; then
        echo "  [$name] Already captured, skipping"
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  VARIANT: $name (semantic_mode=$sem_mode)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo "  [$name] Launching mapping node..."
    MAP_LOG="${RESULTS_ROOT}/${name}_run.log"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=atlas \
        resolution:=$RESOLUTION \
        semantic_mode:=$sem_mode \
        > "$MAP_LOG" 2>&1 &
    LAUNCH_PID=$!
    sleep 4

    echo "  [$name] Replaying $TOTAL_FRAMES frames at ${REPLAY_HZ}Hz..."
    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="$KITTI_ROOT" \
        -p sequence:=8 \
        -p rate_hz:=$REPLAY_HZ \
        -p robot_name:=atlas \
        -p max_range:=30.0 \
        -p min_range:=1.0
    echo "  [$name] Replay finished, waiting for integration..."

    # Wait until mapping node integrates >=99% of frames
    MIN_RECV=$((TOTAL_FRAMES * 99 / 100))
    WAITED=0
    while true; do
        LAST_RECV=$(grep -oP 'recv=\K[0-9]+' "$MAP_LOG" 2>/dev/null | tail -1 || echo "0")
        if [ "$LAST_RECV" -ge "$MIN_RECV" ] 2>/dev/null; then
            echo "  [$name] recv=${LAST_RECV}/${TOTAL_FRAMES} (>=99%), waiting 30s..."
            sleep 30
            break
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        echo "  [$name] ... recv=${LAST_RECV}/${TOTAL_FRAMES} (waited ${WAITED}s)"
        if [ $WAITED -ge 3600 ]; then
            echo "  [$name] WARNING: timed out at recv=${LAST_RECV}"
            sleep 30
            break
        fi
    done

    echo "  [$name] Capturing pointcloud..."
    timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/atlas/scovox_node/pointcloud \
        -p output:="$OUT_NPZ" || echo "  WARN: capture timed out"

    if [ -f "$OUT_NPZ" ]; then
        VOXELS=$(python3 -c "import numpy as np; print(len(np.load('$OUT_NPZ')['points']))")
        echo "  [$name] Captured $VOXELS voxels"
    else
        echo "  [$name] ERROR: capture failed!"
    fi

    kill $LAUNCH_PID 2>/dev/null
    wait $LAUNCH_PID 2>/dev/null
    sleep 2
done

echo ""
echo "=== DATA COLLECTION DONE ==="
echo "Results in: $RESULTS_ROOT"
