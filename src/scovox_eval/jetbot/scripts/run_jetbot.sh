#!/bin/bash
# Run the Jetson-side launch (scovox_a + dscovox + static TF for robot_a).
#
# Prereqs (done once):
#   - Source workspace synced to ~/jetbot_ws/src on the Jetson
#   - Humble container running with: -v ~/jetbot_ws:/jetbot_ws --network host
#   - colcon build inside the container completed (see jetbot/README.md)
#
# Usage (inside the running container on the Jetson):
#   /jetbot_ws/src/robot_sw/distributed_mapping/scovox_eval/jetbot/scripts/run_jetbot.sh

set -euo pipefail

# Override with: WS=/some/path /path/to/run_jetbot.sh
WS=${WS:-/workspace}
JETBOT_DIR=${WS}/src/robot_sw/distributed_mapping/scovox_eval/jetbot

# DDS — Cyclone with unicast peer list (Wi-Fi multicast is unreliable).
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${JETBOT_DIR}/cyclonedds.xml"

# Source ROS — dustynv image puts the overlay at /opt/ros/humble/install,
# upstream apt puts it at /opt/ros/humble.
if [[ -f /opt/ros/humble/install/setup.bash ]]; then
  source /opt/ros/humble/install/setup.bash
else
  source /opt/ros/humble/setup.bash
fi
source ${WS}/install/setup.bash

echo "[jetbot] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[jetbot] CYCLONEDDS_URI=${CYCLONEDDS_URI}"
echo "[jetbot] launching scovox_a + dscovox..."

# Optional telemetry capture in the background; writes /tmp/jetson_timing.csv.
if [[ -x "${JETBOT_DIR}/scripts/jetson_telemetry.py" ]]; then
  python3 ${JETBOT_DIR}/scripts/jetson_telemetry.py \
      --out /tmp/jetson_timing.csv \
      --node-pattern scovox_mapping_node &
  TELEM_PID=$!
  trap "kill ${TELEM_PID} 2>/dev/null || true" EXIT
fi

exec ros2 launch scovox_mapping scenenet_jetbot.launch.py "$@"
