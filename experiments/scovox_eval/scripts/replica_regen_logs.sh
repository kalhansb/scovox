#!/bin/bash
# Re-run mapping nodes on existing scenes to regenerate logs with timing breakdown.
# Does NOT re-capture npz files — only produces new *_run.log files.
#
# Usage: bash replica_regen_logs.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.05)
#
# Prerequisites:
#   - Workspace built with timing breakdown: colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
#   - Rendered data must be re-created (script handles this)
#   - Existing npz files in results directory (will not be touched)

RESOLUTION="${1:-0.05}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

NICESLAM_ROOT="/home/kalhan/Projects/datasets/Replica"
RENDERED_ROOT="/home/kalhan/Projects/datasets/replica_rendered"
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
echo "  Replica Log Regeneration (timing breakdown)"
echo "  ${#SCENES[@]} scenes × ${#VARIANTS[@]} variants"
echo "  Resolution: ${RESOLUTION}m (${RES_CM}cm)"
echo "  Results: $RESULTS_ROOT"
echo "============================================"
echo ""

for scene in "${SCENES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SCENE: $scene"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    RENDERED="$RENDERED_ROOT/$scene"
    RESULT_DIR="$RESULTS_ROOT/$scene"
    mkdir -p "$RESULT_DIR"

    # ── Step A: Convert NICE-SLAM → SCovox format (if not cached) ──
    if [ ! -d "$RENDERED" ] || [ ! -f "$RENDERED/poses.json" ]; then
        echo "  Converting NICE-SLAM data..."
        python3 -m scovox_eval.niceslam_to_scovox \
            --input "$NICESLAM_ROOT/$scene" \
            --cam-params "$CAM_PARAMS" \
            --output "$RENDERED"
    else
        echo "  Already converted"
    fi

    # ── Step B: Run each variant (log only, skip npz capture) ──────
    for variant_spec in "${VARIANTS[@]}"; do
        IFS=':' read -r name pkg launch_file pc_topic <<< "$variant_spec"
        MAP_LOG="${RESULT_DIR}/${name}_run.log"

        if [ -f "$MAP_LOG" ]; then
            echo "  [$name] Log already exists, skipping"
            continue
        fi

        echo "  [$name] Launching mapping node..."
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

        # Wait until mapping node integrates >=99% of frames, then wait 10s
        MIN_RECV=$((TOTAL_FRAMES * 99 / 100))
        WAITED=0
        while true; do
            LAST_RECV=$(grep -oP 'recv=\K[0-9]+' "$MAP_LOG" 2>/dev/null | tail -1 || echo "0")
            if [ "$LAST_RECV" -ge "$MIN_RECV" ] 2>/dev/null; then
                echo "  [$name] recv=${LAST_RECV}/${TOTAL_FRAMES} (>=99%), done."
                sleep 10
                break
            fi
            sleep 5
            WAITED=$((WAITED + 5))
            echo "  [$name] ... recv=${LAST_RECV}/${TOTAL_FRAMES} (waited ${WAITED}s)"
            if [ $WAITED -ge 1800 ]; then
                echo "  [$name] WARNING: timed out at recv=${LAST_RECV}"
                sleep 10
                break
            fi
        done

        # No pointcloud capture — just kill the node
        kill $LAUNCH_PID 2>/dev/null
        wait $LAUNCH_PID 2>/dev/null
        sleep 2

        # Verify log has timing breakdown
        if grep -q "integrate_ms=" "$MAP_LOG"; then
            NFRAMES=$(grep -c "frame_ms=" "$MAP_LOG")
            echo "  [$name] Log captured: $NFRAMES frames with timing breakdown"
        else
            echo "  [$name] WARNING: log missing timing breakdown fields"
        fi
    done

    # ── Step C: Clean up rendered data to save disk ──────────────
    echo "  Cleaning up rendered data for $scene..."
    rm -rf "$RENDERED"
    echo ""
done

echo ""
echo "=== LOG REGENERATION DONE ==="
echo "Run metrics with: bash scripts/replica_compute_metrics.sh $RESOLUTION"
