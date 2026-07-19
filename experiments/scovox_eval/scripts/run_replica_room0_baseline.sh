#!/bin/bash
# (A) Replica room0 sanity check — verify today's HEAD reproduces mIoU 0.46
# baseline from project_softprob_pipeline_2026_05_04.md.
# Default config: 5 cm voxels, M2F semantics, dirichlet, 2000 frames at 2 Hz.
set -eo pipefail

WS=/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws
RESULTS_ROOT="${WS}/src/robot_sw/distributed_mapping/scovox_eval/results/post_refactor_replica_room0_2026_05_08"
OUT_DIR="${RESULTS_ROOT}"
LOG="${OUT_DIR}/scovox_run.log"
NPZ="${OUT_DIR}/scovox.npz"
mkdir -p "${OUT_DIR}"

cd "${WS}"
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONPATH="${WS}/src/robot_sw/distributed_mapping/scovox_eval:${PYTHONPATH:-}"

pkill -9 -f scovox_mapping_node 2>/dev/null || true
pkill -9 -f replica_replay_node 2>/dev/null || true
sleep 2

# Match scovox_replica_m2f.sh defaults exactly.
ros2 launch scovox_mapping replica_eval.launch.py \
    robot_name:=atlas \
    resolution:=0.05 \
    > "${LOG}" 2>&1 &
LAUNCH_PID=$!
sleep 4

python3 -m scovox_eval.replica_replay_node --ros-args \
    -p dataset_path:="${WS}/data/replica_niceslam/room0" \
    -p rate_hz:=2.0 \
    -p robot_name:=atlas \
    -p camera_poses:=true \
    -p semantic_subdir:=semantic_m2f_ade \
    -p n_scans:=2000 > /dev/null 2>&1

MIN_RECV=1980
WAITED=0
while true; do
    last=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    if [ "${last}" -ge "${MIN_RECV}" ] 2>/dev/null; then
        echo "  recv=${last}/2000, grace 30s..."
        sleep 30
        break
    fi
    sleep 8
    WAITED=$((WAITED+8))
    [ ${WAITED} -ge 1800 ] && { echo "  TIMEOUT at recv=${last}"; sleep 30; break; }
done

timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/atlas/scovox_node/pointcloud \
    -p output:="${NPZ}" 2>&1 | tail -2

kill ${LAUNCH_PID} 2>/dev/null || true
pkill -9 -f scovox_mapping_node 2>/dev/null || true
pkill -9 -f replica_replay_node 2>/dev/null || true
sleep 2

ls -la "${NPZ}" 2>&1
echo "DONE_REPLICA_ROOM0_BASELINE"
