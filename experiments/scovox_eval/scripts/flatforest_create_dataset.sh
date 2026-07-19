#!/bin/bash
# Replay pre-recorded flatforest RGB-D dataset through SCovox + CoVox + LogOdds.
# Captures PointCloud2 output as .npz for calibration evaluation (Exp 2, task 4.5).
#
# Usage: bash flatforest_create_dataset.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.10)
#   Examples:
#     bash flatforest_create_dataset.sh          # 10cm (default)
#     bash flatforest_create_dataset.sh 0.05     # 5cm
#
# Prerequisites:
#   - Workspace built: colcon build && source install/setup.bash
#   - Flatforest rendered data at data/flatforest_rendered/
#     (5611 frames: color/*.npy, depth/*.npy, semantic/*.npy, poses.json, camera.json)

RESOLUTION="${1:-0.10}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

WS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws"
DATASET="$WS/data/flatforest_rendered"
EVAL_PKG="$WS/src/robot_sw/distributed_mapping/scovox_eval"
BASE_RESULTS="$EVAL_PKG/results"
RESULTS_DIR="$BASE_RESULTS/flatforest_${RES_CM}cm"
REPLAY_HZ=2.0
TOTAL_FRAMES=5611

VARIANTS=(
    "scovox:scovox_mapping:flatforest_eval.launch.py:/scovox_node/pointcloud"
    "covox:covox_mapping:flatforest_eval_covox.launch.py:/covox_node/pointcloud"
    "logodds:log_odds_mapping:flatforest_eval_logodds.launch.py:/log_odds_node/pointcloud"
)

# Strip conda from PATH to avoid Python conflicts
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export PYTHONPATH="$EVAL_PKG:$PYTHONPATH"

echo "============================================"
echo "  Flatforest Evaluation (pre-rendered dataset)"
echo "  ${#VARIANTS[@]} variants at ${RESOLUTION}m (${RES_CM}cm)"
echo "  Dataset: $DATASET ($TOTAL_FRAMES frames)"
echo "  Results: $RESULTS_DIR"
echo "============================================"

# Verify dataset exists
if [ ! -f "$DATASET/poses.json" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "  Expected: poses.json, camera.json, depth/*.npy, semantic/*.npy"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

for variant_spec in "${VARIANTS[@]}"; do
    IFS=':' read -r name pkg launch_file pc_topic <<< "$variant_spec"
    OUT_NPZ="$RESULTS_DIR/${name}.npz"

    if [ -f "$OUT_NPZ" ]; then
        VOXELS=$(python3 -c "import numpy as np; print(len(np.load('$OUT_NPZ')['points']))")
        echo "  [$name] Already captured ($VOXELS voxels), skipping"
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  VARIANT: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo "  [$name] Launching mapping node..."
    MAP_LOG="${RESULTS_DIR}/${name}_run.log"
    ros2 launch $pkg $launch_file robot_name:=atlas resolution:=$RESOLUTION > "$MAP_LOG" 2>&1 &
    LAUNCH_PID=$!
    sleep 4

    echo "  [$name] Replaying $TOTAL_FRAMES frames at ${REPLAY_HZ}Hz..."
    python3 -m scovox_eval.gazebo_replay_node --ros-args \
        -p dataset_path:="$DATASET" \
        -p rate_hz:=$REPLAY_HZ \
        -p robot_name:=atlas
    echo "  [$name] Replay finished, waiting for integration..."

    # Wait until mapping node integrates >=99% of frames, then wait 30s
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
        # Safety timeout: 60 min max (5611 frames is ~47 min at 2Hz)
        if [ $WAITED -ge 3600 ]; then
            echo "  [$name] WARNING: timed out at recv=${LAST_RECV}"
            sleep 30
            break
        fi
    done

    echo "  [$name] Capturing pointcloud..."
    timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/atlas${pc_topic} \
        -p output:="$OUT_NPZ" || echo "  WARN: capture timed out"

    # Verify capture
    if [ -f "$OUT_NPZ" ]; then
        VOXELS=$(python3 -c "import numpy as np; print(len(np.load('$OUT_NPZ')['points']))")
        echo "  [$name] Captured $VOXELS voxels -> $OUT_NPZ"
    else
        echo "  [$name] ERROR: capture failed!"
    fi

    kill $LAUNCH_PID 2>/dev/null
    wait $LAUNCH_PID 2>/dev/null
    sleep 2
done

echo ""
echo "=== FLATFOREST DATA COLLECTION DONE ==="
echo "Results in: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Export voxel positions for GT query:"
echo "     python3 scripts/npz_to_query_csv.py $RESULTS_DIR/scovox.npz $RESULTS_DIR/query_points.csv"
echo "  2. Run WorldQuerySystem in Gazebo to get per-voxel GT"
echo "  3. Convert GT results:"
echo "     python3 scripts/gz_gt_to_npz.py <world_query_output.csv> $RESULTS_DIR/gt_voxels.npz"
