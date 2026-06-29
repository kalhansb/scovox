#!/usr/bin/env bash
# Launch the SCovox mapping node(s) for a GLIM-driven experiment, IN THE SCOVOX
# CONTAINER. GLIM SLAM runs in the separate glim_loc container and publishes the
# TF tree (map->odom->imu->os_lidar) + the deskewed cloud /glim_ros/points; these
# nodes consume them over ROS 2 DDS (both containers are host-net + ipc host +
# ROS_DOMAIN_ID=0, so they share one DDS graph).
#
# Normally invoked by the host orchestrator ../run_glim_experiment.sh, which
# backgrounds it (`docker compose exec -d scovox ...`) and tears it down with
# `docker compose stop scovox` once the GLIM-side driver has captured the map.
# Manual:  docker compose exec scovox bash scripts/glim/launch_scovox.sh <mode>
#
# Mode (node name must match the salvage calls in the glim-side drivers):
#   map | odom | viz | verify | smoke   single node `scovox_node` from
#                                        config/glim/scovox_lidar_glim.yaml, fed /glim_ros/points
set -e
MODE="${1:?usage: launch_scovox.sh <map|odom|viz|verify|smoke>}"

source /opt/ros/jazzy/setup.bash
source /scovox/install/setup.bash

ROOT=/scovox
CFG="$ROOT/config/glim"

PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

launch_single() {
  echo "[scovox] single node 'scovox_node' <- /glim_ros/points  (params=$CFG/scovox_lidar_glim.yaml)"
  ros2 launch scovox_mapping lidar_mapping.launch.py \
    params_file:="$CFG/scovox_lidar_glim.yaml" \
    pointcloud_topic:=/glim_ros/points use_sim_time:=true
}

case "$MODE" in
  map|odom|viz|verify|smoke) launch_single ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac
