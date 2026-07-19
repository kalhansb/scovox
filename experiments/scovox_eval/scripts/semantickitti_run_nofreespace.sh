#!/bin/bash
# SemanticKITTI ablation: freespace carving vs endpoint-only (task 4.9f).
#
# The existing 4.9 runs (in results/semantickitti_<res>cm/) used carve_band=0
# (endpoint-only, NO freespace carving). This script re-runs SCovox (Dirichlet)
# under two explicit freespace configs so we can compare.
#
# Variant:
#   scovox_dirichlet_fullray  — carve_band=-1 (full ray freespace carving)
#
# The nocarve baseline already exists as scovox.npz in results/semantickitti_<res>cm/
# (task 4.9, carve_band=0). No need to re-run it.
#
# Output directory:
#   results/semantickitti_<res>cm_freespace_ablation/
#     scovox_dirichlet_nocarve.npz       — captured voxel map
#     scovox_dirichlet_nocarve_run.log   — mapping node log (timing/memory)
#     scovox_dirichlet_fullray.npz
#     scovox_dirichlet_fullray_run.log
#     gt.npz                             — symlinked from main results dir
#
# This is a separate directory from the main 4.9 results to keep experiments
# cleanly isolated. File names include the semantic mode (dirichlet) and the
# carving config (nocarve/fullray) so they are self-documenting.
#
# Metrics are computed via semantickitti_compute_metrics.sh pointed at
# this directory, with VARIANTS overridden to match the file names above.
#
# Usage: bash semantickitti_run_nofreespace.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.10)

set -eo pipefail

RESOLUTION="${1:-0.10}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

KITTI_ROOT="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/semantickitti/dataset"
BASE_RESULTS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/src/robot_sw/distributed_mapping/scovox_eval/results"
RESULTS_ROOT="$BASE_RESULTS/semantickitti_${RES_CM}cm_freespace_ablation"
MAIN_RESULTS="$BASE_RESULTS/semantickitti_${RES_CM}cm"
WS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG="$WS/src/robot_sw/distributed_mapping/scovox_eval"
SEQUENCE="08"
REPLAY_HZ=0.5
TOTAL_FRAMES=4071

# Ablation variants: name:carve_band
# carve_band=0  → endpoint only (no freespace carving)
# carve_band=-1 → full ray (freespace carving enabled)
VARIANTS=(
    "scovox_dirichlet_fullray:-1.0"
)

# Strip conda from PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export PYTHONPATH="$EVAL_PKG:$PYTHONPATH"

mkdir -p "$RESULTS_ROOT"

# Symlink GT from main results dir so metrics script can find it
if [ -f "$MAIN_RESULTS/gt.npz" ] && [ ! -e "$RESULTS_ROOT/gt.npz" ]; then
    ln -s "$MAIN_RESULTS/gt.npz" "$RESULTS_ROOT/gt.npz"
    echo "Symlinked GT from $MAIN_RESULTS/gt.npz"
fi

echo "============================================"
echo "  SemanticKITTI Freespace Ablation (seq $SEQUENCE)"
echo "  Resolution: ${RESOLUTION}m (${RES_CM}cm)"
echo "  Results: $RESULTS_ROOT"
echo "============================================"
echo ""

for variant_spec in "${VARIANTS[@]}"; do
    IFS=':' read -r name carve_band <<< "$variant_spec"
    OUT_NPZ="$RESULTS_ROOT/${name}.npz"

    if [ -f "$OUT_NPZ" ]; then
        echo "  [$name] Already captured, skipping"
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  VARIANT: $name (carve_band=$carve_band)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo "  [$name] Launching mapping node..."
    MAP_LOG="${RESULTS_ROOT}/${name}_run.log"

    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=atlas \
        resolution:=$RESOLUTION \
        semantic_mode:=dirichlet \
        carve_band:=$carve_band \
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
        if [ $WAITED -ge 7200 ]; then
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
echo "=== FREESPACE ABLATION DATA COLLECTION DONE ==="
echo "Results in: $RESULTS_ROOT"
echo ""
echo "To compute metrics, run semantickitti_compute_metrics.sh with"
echo "VARIANTS overridden to (scovox_dirichlet_nocarve scovox_dirichlet_fullray)."
