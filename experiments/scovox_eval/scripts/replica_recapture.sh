#!/bin/bash
# Re-capture Replica pointclouds with a_occ/a_free fields.
#
# This script re-runs ONLY the mapping + capture steps (not the data conversion).
# The rendered data must already exist from a previous run of replica_create_dataset.sh.
#
# Usage:
#   bash scripts/replica_recapture.sh [RESOLUTION]
#   Examples:
#     bash scripts/replica_recapture.sh 0.05    # 5cm
#     bash scripts/replica_recapture.sh 0.10    # 10cm
#     bash scripts/replica_recapture.sh 0.20    # 20cm
#     bash scripts/replica_recapture.sh          # all three

RESOLUTION="${1:-all}"

NICESLAM_ROOT="/home/kalhan/Projects/datasets/Replica"
RENDERED_ROOT="/home/kalhan/Projects/datasets/replica_rendered"
BASE_RESULTS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/src/robot_sw/distributed_mapping/scovox_eval/results"
WS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG="$WS/src/robot_sw/distributed_mapping/scovox_eval"
CAM_PARAMS="$NICESLAM_ROOT/cam_params.json"
REPLAY_HZ=2.0
TOTAL_FRAMES=2000

SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)

VARIANTS=(
    "scovox:scovox_mapping:replica_eval.launch.py:/scovox_node/pointcloud"
    "covox:covox_mapping:replica_eval_covox.launch.py:/covox_node/pointcloud"
)

# Strip conda from PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export PYTHONPATH="$EVAL_PKG:$PYTHONPATH"

if [ "$RESOLUTION" = "all" ]; then
    RESOLUTIONS=(0.05 0.10 0.20)
else
    RESOLUTIONS=($RESOLUTION)
fi

for RES in "${RESOLUTIONS[@]}"; do
    RES_CM=$(python3 -c "print(int(float('$RES')*100))")
    RESULTS_ROOT="$BASE_RESULTS/replica_${RES_CM}cm"

    echo "============================================"
    echo "  Replica Re-capture — ${RES_CM}cm"
    echo "  ${#SCENES[@]} scenes × ${#VARIANTS[@]} variants"
    echo "  Results: $RESULTS_ROOT"
    echo "============================================"

    for scene in "${SCENES[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  SCENE: $scene @ ${RES_CM}cm"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        RENDERED="$RENDERED_ROOT/$scene"
        RESULT_DIR="$RESULTS_ROOT/$scene"

        # Check rendered data exists
        if [ ! -d "$RENDERED" ] || [ ! -f "$RENDERED/poses.json" ]; then
            # Try to convert from NICE-SLAM format
            if [ -d "$NICESLAM_ROOT/$scene" ]; then
                echo "  Converting NICE-SLAM data..."
                python3 -m scovox_eval.niceslam_to_scovox \
                    --input "$NICESLAM_ROOT/$scene" \
                    --cam-params "$CAM_PARAMS" \
                    --output "$RENDERED"
            else
                echo "  SKIP: no rendered data at $RENDERED"
                continue
            fi
        fi

        for variant_spec in "${VARIANTS[@]}"; do
            IFS=':' read -r name pkg launch_file pc_topic <<< "$variant_spec"
            OUT_NPZ="$RESULT_DIR/${name}.npz"

            # Back up old NPZ
            if [ -f "$OUT_NPZ" ]; then
                mv "$OUT_NPZ" "${OUT_NPZ}.bak"
                echo "  [$name] Backed up old NPZ"
            fi

            echo "  [$name] Launching mapping node..."
            MAP_LOG="${RESULT_DIR}/${name}_run.log"
            ros2 launch $pkg $launch_file robot_name:=atlas resolution:=$RES > "$MAP_LOG" 2>&1 &
            LAUNCH_PID=$!
            sleep 4

            echo "  [$name] Replaying $TOTAL_FRAMES frames at ${REPLAY_HZ}Hz..."
            python3 -m scovox_eval.replica_replay_node --ros-args \
                -p dataset_path:="$RENDERED" \
                -p rate_hz:=$REPLAY_HZ \
                -p robot_name:=atlas \
                -p camera_poses:=true
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
                if [ $WAITED -ge 1800 ]; then
                    echo "  [$name] WARNING: timed out at recv=${LAST_RECV}"
                    sleep 30
                    break
                fi
            done

            echo "  [$name] Capturing pointcloud (with a_occ/a_free)..."
            timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
                -p topic:=/atlas${pc_topic} \
                -p output:="$OUT_NPZ" || echo "  WARN: capture timed out"

            # Verify new fields
            if [ -f "$OUT_NPZ" ]; then
                python3 -c "
import numpy as np
d = np.load('$OUT_NPZ')
keys = list(d.keys())
n = len(d['points'])
has_beta = 'a_occ' in keys and 'a_free' in keys
print(f'  [{\"$name\"}] {n} voxels, keys={keys}')
if has_beta:
    print(f'  [{\"$name\"}] a_occ: [{d[\"a_occ\"].min():.2f}, {d[\"a_occ\"].max():.2f}], a_free: [{d[\"a_free\"].min():.2f}, {d[\"a_free\"].max():.2f}]')
else:
    print(f'  [{\"$name\"}] WARNING: a_occ/a_free NOT found!')
"
            else
                echo "  [$name] ERROR: capture failed!"
                # Restore backup
                if [ -f "${OUT_NPZ}.bak" ]; then
                    mv "${OUT_NPZ}.bak" "$OUT_NPZ"
                fi
            fi

            kill $LAUNCH_PID 2>/dev/null
            wait $LAUNCH_PID 2>/dev/null
            sleep 2
        done

        # Clean up rendered data to save disk
        echo "  Cleaning rendered data..."
        rm -rf "$RENDERED"
    done
done

echo ""
echo "=== RE-CAPTURE DONE ==="
echo "NPZ files now include a_occ, a_free, posterior_variance, eig fields."
echo "Run calibration metrics with:"
echo "  compute-unknown-ambiguous results/replica_5cm/room0/scovox.npz \\"
echo "    /home/kalhan/Projects/datasets/Replica/room0_mesh.ply --resolution 0.05"
