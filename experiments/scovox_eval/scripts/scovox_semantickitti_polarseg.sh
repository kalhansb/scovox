#!/bin/bash
# SCovox runs on SemanticKITTI seq 06-10 × first 100 frames, using PolarSeg
# predictions (predictions/*.label) instead of GT labels. Matches the SLIM-VDB
# paper protocol so the SCovox / SLIM-VDB rows of Table II are directly
# comparable (same scenes, same frame counts, same predicted input).
#
# Usage: bash scovox_semantickitti_polarseg.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.10 — matches paper)
set -eo pipefail

RESOLUTION="${1:-0.10}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
KITTI_ROOT="${WS}/data/semantickitti/dataset"
RESULTS_ROOT="${WS}/src/robot_sw/distributed_mapping/scovox_eval/results/semantickitti_polarseg_${RES_CM}cm"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
SEQUENCES=(06 07 08 09 10)
N_SCANS=100
REPLAY_HZ=0.5

VARIANTS=(
    "scovox:dirichlet"
    "scovox_mv:majority_vote"
    "scovox_np:naive"
)

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RESULTS_ROOT}"

echo "========================================================"
echo "  SCovox × SemanticKITTI (PolarSeg labels) — ${RES_CM}cm"
echo "  Sequences: ${SEQUENCES[*]}   Frames each: ${N_SCANS}"
echo "  Results: ${RESULTS_ROOT}"
echo "========================================================"

for seq in "${SEQUENCES[@]}"; do
    SEQ_RES="${RESULTS_ROOT}/${seq}"
    mkdir -p "${SEQ_RES}"
    for variant_spec in "${VARIANTS[@]}"; do
        IFS=':' read -r name sem_mode <<< "$variant_spec"
        OUT_NPZ="${SEQ_RES}/${name}.npz"
        if [[ -f "${OUT_NPZ}" ]]; then
            echo "  [${seq}/${name}] already captured, skipping"
            continue
        fi

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  seq ${seq}   variant ${name}   mode ${sem_mode}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        MAP_LOG="${SEQ_RES}/${name}_run.log"
        ros2 launch scovox_mapping semantickitti_eval.launch.py \
            robot_name:=atlas \
            resolution:=${RESOLUTION} \
            semantic_mode:=${sem_mode} \
            > "${MAP_LOG}" 2>&1 &
        LAUNCH_PID=$!
        sleep 4

        SEQ_INT=$((10#$seq))  # strip leading zero for ROS2 param (avoids float parse)
        python3 -m scovox_eval.semantickitti_replay_node --ros-args \
            -p dataset_path:="${KITTI_ROOT}" \
            -p sequence:=${SEQ_INT} \
            -p rate_hz:=${REPLAY_HZ} \
            -p robot_name:=atlas \
            -p max_range:=30.0 \
            -p min_range:=1.0 \
            -p labels_subdir:=predictions \
            -p n_scans:=${N_SCANS}

        # Wait for mapping node to have integrated ~all frames.
        MIN_RECV=$((N_SCANS * 99 / 100))
        WAITED=0
        while true; do
            LAST_RECV=$(grep -oP 'recv=\K[0-9]+' "${MAP_LOG}" 2>/dev/null | tail -1 || echo "0")
            if [ "${LAST_RECV}" -ge "${MIN_RECV}" ] 2>/dev/null; then
                echo "  [${seq}/${name}] recv=${LAST_RECV}/${N_SCANS}, grace 10s..."
                sleep 10
                break
            fi
            sleep 3
            WAITED=$((WAITED + 3))
            if [ ${WAITED} -ge 600 ]; then
                echo "  [${seq}/${name}] WARN: timed out at recv=${LAST_RECV}"
                break
            fi
        done

        timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
            -p topic:=/atlas/scovox_node/pointcloud \
            -p output:="${OUT_NPZ}" || echo "  WARN: capture timeout"

        if [[ -f "${OUT_NPZ}" ]]; then
            VOX=$(python3 -c "import numpy as np; print(len(np.load('${OUT_NPZ}')['points']))")
            echo "  [${seq}/${name}] captured ${VOX} voxels -> ${OUT_NPZ}"
        else
            echo "  [${seq}/${name}] ERROR: no npz produced"
        fi

        kill ${LAUNCH_PID} 2>/dev/null || true
        wait ${LAUNCH_PID} 2>/dev/null || true
        sleep 2
    done
done

echo ""
echo "=== done. outputs in ${RESULTS_ROOT}/<seq>/<variant>.npz ==="
