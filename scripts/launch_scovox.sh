#!/usr/bin/env bash
# Launch the SCovox mapping node on the RAW Ouster cloud, IN THE SCOVOX CONTAINER.
#
# A localizer (e.g. GLIM LiDAR-IMU SLAM, running in a separate container) owns the
# TF tree (map -> odom -> imu -> os_lidar) and provides the per-scan pose; SCovox
# subscribes to the full-resolution /ouster/points + /imu/data over ROS 2 DDS and
# builds the occupancy map online. It deskews each scan natively (gyro rotation)
# and voxel-downsamples it (downsample_voxel_size) to suppress the vertical smear
# without GLIM's recall-costing cloud downsample -- see config/ + the README.
#
# Manual:  docker compose exec scovox bash scripts/launch_scovox.sh raw
set -e
MODE="${1:-raw}"

source /opt/ros/jazzy/setup.bash
source /scovox/install/setup.bash

ROOT=/scovox
CFG="$ROOT/config"

case "$MODE" in
  raw)
    echo "[scovox] scovox_node <- /ouster/points (native deskew + downsample)  (params=$CFG/scovox_lidar_raw_deskew.yaml)"
    exec ros2 launch scovox_mapping lidar_mapping.launch.py \
      params_file:="$CFG/scovox_lidar_raw_deskew.yaml" \
      pointcloud_topic:=/ouster/points use_sim_time:=true
    ;;
  *) echo "usage: launch_scovox.sh raw"; exit 1 ;;
esac
