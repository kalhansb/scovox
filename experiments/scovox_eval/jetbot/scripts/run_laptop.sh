#!/bin/bash
# Laptop-side orchestrator for the JetBot SceneNet experiment.
#
# Brings up, in this order:
#   1. static TF  map -> robotB/odom
#   2. scovox_node B (publishes /robotB/scovox_node/scovox_bin over DDS)
#   3. scenenet_replay for robot_a   (frames [0..200))
#   4. scenenet_replay for robot_b   (frames [100..300))
#
# robot_a's static TF and dscovox_node live on the Jetson (run_jetbot.sh).
#
# Usage (laptop, from workspace root, AFTER Jetson is up):
#   src/robot_sw/distributed_mapping/scovox_eval/jetbot/scripts/run_laptop.sh \
#       [sequence] [rate_hz]
#
# Examples:
#   ./run_laptop.sh                # 0_223 @ 10 Hz (throughput run)
#   ./run_laptop.sh 0_223 2.0      # 0_223 @ 2  Hz (quality run, Jetson keeps up)

# Source ROS BEFORE enabling strict mode — AMENT scripts touch unset vars.
ROS_SETUP=${ROS_SETUP:-/opt/ros/humble/setup.bash}
source "${ROS_SETUP}"

set -euo pipefail

SEQUENCE=${1:-0_223}
RATE_HZ=${2:-10.0}
ROBOT_A=robotA
ROBOT_B=robotB
START_A=0
START_B=100
N_SCANS=200
RESOLUTION=0.05

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS=$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)
JETBOT_DIR=${WS}/src/robot_sw/distributed_mapping/scovox_eval/jetbot
EVAL_PKG=${WS}/src/robot_sw/distributed_mapping/scovox_eval
DATA_ROOT=${WS}/data/scenenet_val_layout

# DDS — must match the Jetson.
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${JETBOT_DIR}/cyclonedds.xml"

# Workspace overlay (ROS base already sourced above).
set +u  # AMENT setup scripts touch unset vars
source ${WS}/install/setup.bash
set -u
# Drop conda from PATH per the existing batch convention.
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV PYTHONNOUSERSITE
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "[laptop] WS=${WS}"
echo "[laptop] DATA_ROOT=${DATA_ROOT}"
echo "[laptop] sequence=${SEQUENCE} rate_hz=${RATE_HZ}"
echo "[laptop] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[laptop] CYCLONEDDS_URI=${CYCLONEDDS_URI}"

if [[ ! -d "${DATA_ROOT}/train/${SEQUENCE}" ]]; then
  echo "ERROR: ${DATA_ROOT}/train/${SEQUENCE} does not exist."
  exit 1
fi

PIDS=()
cleanup() {
  echo ""
  echo "[laptop] cleanup..."
  for p in "${PIDS[@]:-}"; do kill -INT "$p" 2>/dev/null || true; done
  sleep 1
  for p in "${PIDS[@]:-}"; do kill -KILL "$p" 2>/dev/null || true; done
  pkill -9 -f "scovox_mapping_node.*${ROBOT_B}" 2>/dev/null || true
  pkill -9 -f "scenenet_replay_node" 2>/dev/null || true
  pkill -9 -f "static_transform_publisher.*${ROBOT_B}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 1. static TF map -> robotB/odom ───────────────────────────────────────
ros2 run tf2_ros static_transform_publisher \
    --frame-id map --child-frame-id ${ROBOT_B}/odom \
    --x 0 --y 0 --z 0 &
PIDS+=($!)
sleep 0.5

# ── 2. scovox_node B (laptop's peer scovox) ──────────────────────────────
# NYU14 colour LUT keys/classes — must match the Jetson scovox_a and the
# scenenet_replay_node colour emissions.
NYU14_KEYS="[0,255,15292720,55808,9765104,14610712,16764622,57573,6981836,7675177,15737835,42908,16354048,14804418]"
NYU14_CLASSES="[0,1,2,3,4,5,6,7,8,9,10,11,12,13]"

ros2 run scovox_mapping scovox_mapping_node --ros-args \
    -r __ns:=/${ROBOT_B} -r __node:=scovox_node \
    -p dataset_mode:=true \
    -p mode:=rolling \
    -p submap_max_distance:=9999.0 \
    -p resolution:=${RESOLUTION} \
    -p w_free:=1.0 -p w_occ:=6.0 \
    -p kappa0:=2.0 \
    -p semantic_top_k:=10 \
    -p semantic_occ_gate:=0.6 \
    -p semantic_mode:=dirichlet \
    -p max_semantic_classes:=14 \
    -p range_decay_length:=-1.0 \
    -p min_range:=0.1 -p max_range:=10.0 \
    -p grazing_angle_threshold:=-1.0 \
    -p transient_decay_rate:=0.0 \
    -p base_frame:=${ROBOT_B}/base_link \
    -p integration_frame:=${ROBOT_B}/odom \
    -p map_frame:=map \
    -p depth_topic:=rgbd_camera_depth_image \
    -p depth_info_topic:=rgbd_camera_info \
    -p seg_topic:=segmentation/colored \
    -p stride:=1 -p min_depth:=0.1 -p max_depth:=10.0 \
    -p trace_no_return_rays:=false \
    -p robot_id:=${ROBOT_B} \
    -p publish_pointcloud:=true \
    -p occupancy_vis_threshold:=0.5 \
    -p scovox_publish_rate:=1.0 \
    -p publish_planning_map:=false \
    -p "semantic_color_map_keys:=${NYU14_KEYS}" \
    -p "semantic_color_map_classes:=${NYU14_CLASSES}" \
    -p use_split:=true \
    -p share_tsdf:=false \
    -p fused_walker:=true \
    -p wire_format:=v3 \
    -p num_classes:=14 \
    -p dirichlet_prior:=0.01 &
PIDS+=($!)
sleep 4

# ── 3. scenenet replay for robot_a ────────────────────────────────────────
python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${DATA_ROOT}" \
    -p sequence:="${SEQUENCE}" \
    -p rate_hz:="${RATE_HZ}" \
    -p robot_name:="${ROBOT_A}" \
    -p use_gt_labels:=true \
    -p start_frame:=${START_A} \
    -p n_scans:=${N_SCANS} &
PIDS+=($!)

# ── 4. scenenet replay for robot_b ────────────────────────────────────────
python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${DATA_ROOT}" \
    -p sequence:="${SEQUENCE}" \
    -p rate_hz:="${RATE_HZ}" \
    -p robot_name:="${ROBOT_B}" \
    -p use_gt_labels:=true \
    -p start_frame:=${START_B} \
    -p n_scans:=${N_SCANS} &
PIDS+=($!)

echo "[laptop] all four processes up; PIDs=${PIDS[*]}"
echo "[laptop] will exit when both replays finish; Ctrl-C to abort."

# Wait on the two replay PIDs (the last two we appended). They exit when
# the 200-frame split finishes.
REPLAY_A_PID=${PIDS[-2]}
REPLAY_B_PID=${PIDS[-1]}
wait ${REPLAY_A_PID} 2>/dev/null || true
wait ${REPLAY_B_PID} 2>/dev/null || true

# Let scovox_b drain its last publish before we kill it.
sleep 4
echo "[laptop] replays complete."
