#!/bin/bash
# Run SCovox + CoVox + LogOdds on 8 standard Replica scenes (NICE-SLAM trajectories).
# Processes one scene at a time, cleans up rendered data after each.
#
# Usage: bash replica_create_dataset.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.05)
#   Examples:
#     bash replica_create_dataset.sh          # 5cm (default)
#     bash replica_create_dataset.sh 0.10     # 10cm
#     bash replica_create_dataset.sh 0.20     # 20cm
#
# Prerequisites:
#   - Workspace built: colcon build && source install/setup.bash
#   - NICE-SLAM Replica data at ~/Projects/datasets/Replica/
#   - System Python has trimesh, scipy, Pillow

# Don't use set -e — polling loops use grep which returns non-zero on no match

RESOLUTION="${1:-0.05}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

NICESLAM_ROOT="$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/replica_niceslam"
RENDERED_ROOT="$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/replica_niceslam"
BASE_RESULTS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/src/robot_sw/distributed_mapping/scovox_eval/results"
RESULTS_ROOT="$BASE_RESULTS/replica_${RES_CM}cm"
WS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG="$WS/src/robot_sw/distributed_mapping/scovox_eval"
CAM_PARAMS="$NICESLAM_ROOT/cam_params.json"
REPLAY_HZ=2.0
TOTAL_FRAMES=2000

SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)

VARIANTS=(
    "scovox:scovox_mapping:replica_eval.launch.py:/scovox_node/pointcloud"
    "covox:covox_mapping:replica_eval_covox.launch.py:/covox_node/pointcloud"
    "logodds:log_odds_mapping:replica_eval_logodds.launch.py:/log_odds_node/pointcloud"
)

# Strip conda from PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export PYTHONPATH="$EVAL_PKG:$PYTHONPATH"

echo "============================================"
echo "  Replica Evaluation (NICE-SLAM trajectories)"
echo "  ${#SCENES[@]} scenes × ${#VARIANTS[@]} variants"
echo "  Resolution: ${RESOLUTION}m (${RES_CM}cm)"
echo "  Results: $RESULTS_ROOT"
echo "============================================"
echo "  Disk free: $(df -h / | tail -1 | awk '{print $4}')"
echo ""

for scene in "${SCENES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SCENE: $scene"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    RENDERED="$RENDERED_ROOT/$scene"
    RESULT_DIR="$RESULTS_ROOT/$scene"
    GT_FILE="$RESULT_DIR/gt_sampled_ros.npz"
    GT_MESH="$RENDERED/mesh.ply"
    mkdir -p "$RESULT_DIR"

    # ── Check if scene is fully done ─────────────────────────────
    ALL_DONE=true
    for variant_spec in "${VARIANTS[@]}"; do
        IFS=':' read -r name pkg launch_file pc_topic <<< "$variant_spec"
        if [ ! -f "$RESULT_DIR/${name}.npz" ]; then
            ALL_DONE=false
            break
        fi
    done
    if $ALL_DONE && [ -f "$GT_FILE" ]; then
        echo "  All variants already captured, skipping"
        continue
    fi

    # ── Step A: Verify pre-rendered dataset exists ────────────────
    if [ ! -d "$RENDERED" ] || [ ! -f "$RENDERED/poses.txt" ]; then
        echo "  ERROR: Pre-rendered dataset not found at $RENDERED"
        echo "  Run render_replica_dataset.py first."
        continue
    else
        echo "  Dataset: $RENDERED"
    fi

    # ── Step B: Generate GT point cloud (reuse from replica/ if exists) ──
    if [ ! -f "$GT_FILE" ]; then
        EXISTING_GT="$BASE_RESULTS/replica/$scene/gt_sampled_ros.npz"
        if [ -f "$EXISTING_GT" ]; then
            cp "$EXISTING_GT" "$GT_FILE"
            echo "  Reused GT from replica/"
        elif [ ! -f "$GT_MESH" ]; then
            echo "  WARNING: GT mesh not found at $GT_MESH"
        else
            echo "  Sampling GT mesh..."
            python3 -c "
import trimesh, numpy as np
mesh = trimesh.load('$GT_MESH')
pts = mesh.sample(100000)
np.savez_compressed('$GT_FILE', points=pts.astype(np.float32))
print(f'  {len(pts)} GT points saved')
"
        fi
    fi

    # ── Step C: Run each variant ─────────────────────────────────
    for variant_spec in "${VARIANTS[@]}"; do
        IFS=':' read -r name pkg launch_file pc_topic <<< "$variant_spec"
        OUT_NPZ="$RESULT_DIR/${name}.npz"

        if [ -f "$OUT_NPZ" ]; then
            echo "  [$name] Already captured"
            continue
        fi

        echo "  [$name] Launching mapping node..."
        MAP_LOG="${RESULT_DIR}/${name}_run.log"
        ros2 launch $pkg $launch_file robot_name:=atlas resolution:=$RESOLUTION > "$MAP_LOG" 2>&1 &
        LAUNCH_PID=$!
        sleep 4

        echo "  [$name] Replaying $TOTAL_FRAMES frames at ${REPLAY_HZ}Hz..."
        python3 -m scovox_eval.replica_replay_node --ros-args \
            -p dataset_path:="$RENDERED" \
            -p rate_hz:=$REPLAY_HZ \
            -p robot_name:=atlas \
            -p camera_poses:=true
        echo "  [$name] Replay finished, waiting for integration..."

        # Wait until mapping node integrates >=99% of frames, then wait 30s
        MIN_RECV=$((TOTAL_FRAMES * 99 / 100))  # 1980 for 2000 frames
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
            # Safety timeout: 30 min max
            if [ $WAITED -ge 1800 ]; then
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
            echo "  [$name] Captured $VOXELS voxels"
        else
            echo "  [$name] ERROR: capture failed!"
        fi

        kill $LAUNCH_PID 2>/dev/null
        wait $LAUNCH_PID 2>/dev/null
        sleep 2
    done

    # ── Step D: Clean up rendered data to save disk ──────────────
    # Disabled: RENDERED now points at source data under data/replica_niceslam/,
    # so deleting it would destroy the dataset. Re-enable only if RENDERED_ROOT
    # is switched back to a disposable rendered copy.
    # rm -rf "$RENDERED"
    echo "  Disk free: $(df -h / | tail -1 | awk '{print $4}')"
    echo ""
done

echo ""
echo "=== DATA COLLECTION DONE ==="
echo "Run metrics with: bash scripts/compute_replica_metrics.sh"
