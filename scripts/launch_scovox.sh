#!/usr/bin/env bash
# Launch the SCovox mapping node on the RAW Ouster cloud, IN THE SCOVOX CONTAINER.
#
# A separate localizer (e.g. GLIM) owns the TF tree (map -> odom -> imu ->
# os_lidar) and the per-scan pose. SCovox reads /ouster/points and /imu/data,
# deskews each scan and voxel-downsamples it.
# (notes: launch-scovox-localizer-split)
#
# Manual:  docker compose exec scovox bash scripts/launch_scovox.sh raw
# Moved comments: docs/code_notes/scovox_code_notes.md
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
