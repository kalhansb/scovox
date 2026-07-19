#!/usr/bin/env bash
# Run SceneNet RGB-D evaluation for all mapping variants.
#
# Replays scene through each variant, captures the output pointcloud to NPZ,
# then computes mIoU + runtime stats.
#
# Usage:
#   ./scenenet_run_eval.sh [DATA_ROOT] [SEQUENCE] [RESOLUTION]
#
# Defaults:
#   DATA_ROOT = $HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/scenenet
#   SEQUENCE  = 2
#   RESOLUTION = 0.05

set -eo pipefail

DATA_ROOT="${1:-$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/scenenet}"
SEQUENCE="${2:-2}"
RESOLUTION="${3:-0.05}"
RESULTS_DIR="$(dirname "$0")/../results/scenenet_${RESOLUTION//.}cm"
ROBOT="atlas"
RATE_HZ=10.0

# Strip miniconda from PATH so system python3 (with rclpy) is found.
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV PYTHONNOUSERSITE
WS=/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${WS}/src/robot_sw/distributed_mapping/scovox_eval:${PYTHONPATH:-}"

mkdir -p "$RESULTS_DIR"

echo "=== SceneNet Eval: seq=${SEQUENCE}, res=${RESOLUTION}m, data=${DATA_ROOT} ==="
echo "Results dir: $RESULTS_DIR"

# --- Helper: run a single variant ---
run_variant() {
    local VARIANT="$1"      # scovox_dirichlet | scovox_mv | scovox_np | logodds
    local LAUNCH_PKG="$2"
    local LAUNCH_FILE="$3"
    local PC_TOPIC="$4"     # pointcloud topic to capture
    local EXTRA_ARGS="${5:-}"
    local NPZ_OUT="$RESULTS_DIR/${VARIANT}.npz"
    local LOG_OUT="$RESULTS_DIR/${VARIANT}_run.log"

    echo ""
    echo "--- Running variant: $VARIANT ---"

    # 1. Launch mapping node in background
    ros2 launch "$LAUNCH_PKG" "$LAUNCH_FILE" \
        robot_name:="$ROBOT" resolution:="$RESOLUTION" $EXTRA_ARGS \
        > "$LOG_OUT" 2>&1 &
    MAPPING_PID=$!
    sleep 3  # let it initialise

    # 2. Launch replay node (blocks until all frames published)
    python3 -m scovox_eval.scenenet_replay_node --ros-args \
        -p data_root:="$DATA_ROOT" \
        -p sequence:="$SEQUENCE" \
        -p rate_hz:="$RATE_HZ" \
        -p robot_name:="$ROBOT" \
        -p use_gt_labels:=true \
        >> "$LOG_OUT" 2>&1 || true

    echo "  Replay finished. Waiting 5s for integration to drain..."
    sleep 5

    # 3. Capture pointcloud to NPZ
    echo "  Capturing pointcloud..."
    timeout 30 python3 -m scovox_eval.pointcloud_to_npz \
        --ros-args -p topic:="$PC_TOPIC" \
                   -p output:="$NPZ_OUT" \
                   -p wait_secs:=5.0 \
        >> "$LOG_OUT" 2>&1 || echo "  WARN: pointcloud capture timed out"

    # 4. Kill mapping node (both ros2 launch and scovox_mapping_node child)
    kill "$MAPPING_PID" 2>/dev/null || true
    wait "$MAPPING_PID" 2>/dev/null || true
    pkill -f "scovox_mapping_node" 2>/dev/null || true
    pkill -f "log_odds_node" 2>/dev/null || true
    sleep 3
    ros2 daemon stop 2>/dev/null || true
    rm -f /dev/shm/sem.fastrtps_* /dev/shm/fastrtps_* 2>/dev/null
    sleep 1
    ros2 daemon start 2>/dev/null || true
    sleep 2

    echo "  Done. Log: $LOG_OUT"
    echo "  NPZ: $NPZ_OUT"
}

# --- Run all variants ---

# SCovox (Dirichlet)
run_variant "scovox_dirichlet" "scovox_mapping" "scenenet_eval.launch.py" \
    "/$ROBOT/scovox_node/pointcloud" "semantic_mode:=dirichlet"

# SCovox-MV (Majority Vote)
run_variant "scovox_mv" "scovox_mapping" "scenenet_eval.launch.py" \
    "/$ROBOT/scovox_node/pointcloud" "semantic_mode:=majority_vote"

# SCovox-NP (Naive Projection)
run_variant "scovox_np" "scovox_mapping" "scenenet_eval.launch.py" \
    "/$ROBOT/scovox_node/pointcloud" "semantic_mode:=naive"

# LogOdds (occupancy only — no semantic eval, runtime/memory baseline)
run_variant "logodds" "log_odds_mapping" "scenenet_eval_logodds.launch.py" \
    "/$ROBOT/log_odds_node/pointcloud"

echo ""
echo "=== All variants complete. Results in $RESULTS_DIR ==="
echo ""
echo "Next steps:"
echo "  1. Build GT:  python3 scripts/scenenet_build_gt.py $DATA_ROOT $SEQUENCE $RESULTS_DIR/gt.npz"
echo "  2. Compute metrics: python3 scripts/scenenet_compute_metrics.py $RESULTS_DIR"
