# Single-robot LiDAR occupancy mapping

Run the scovox LiDAR mapping stack on **one** robot: a localizer, the occupancy
mapper, and a merger. One robot is just a fleet of one — the merger fuses its own
stream and its output is the (trivially global) planner map. This is also the
per-robot building block for the fleet setup in
[distributed_mapping_lidar.md](distributed_mapping_lidar.md).

Everything runs inside the repos' Docker containers; nothing is installed on the host.

## Prerequisites

Build the workspace, an Ouster publishing `/ouster/points` (RELIABLE QoS) + an IMU
on `/imu/data`, and the `base_link → os_lidar` / `imu` extrinsics. See
[Prerequisites](distributed_mapping_lidar.md#prerequisites-each-robot) and
[Config files](distributed_mapping_lidar.md#config-files-identical-on-every-robot)
in the distributed runbook.

## Run

Three terminals / background jobs:

```bash
# ── 1. Localization tree (hmr_localisation container) ─────────────────────────
cd ws/src/hmr_localisation
docker compose exec -e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET ros \
  bash /ws/scripts/run_localization_live.sh
# wait for "active. TF tree: map -> odom -> base_link -> {os_lidar, imu}"

# ── 2. Mapping (scovox container; run each command in its own exec) ───────────
cd ../scovox
E="docker compose exec -e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET scovox"
S="source /opt/ros/jazzy/setup.bash && source /scovox/install/setup.bash"

# identity bridge  map -> r1_map  (the mapper integrates in a per-robot frame)
$E bash -lc "$S && ros2 run tf2_ros static_transform_publisher \
  --frame-id map --child-frame-id r1_map"

# merger FIRST (the delta stream is only sent once someone subscribes)
$E bash -lc "$S && ros2 run scovox_mapping dscovox_mapping_node --ros-args \
  -r __ns:=/robot1 \
  --params-file /scovox/src/scovox_mapping/config/dscovox_params.yaml"

# mapper: LiDAR-only base config + share overlay + the one per-robot frame
$E bash -lc "$S && ros2 run scovox_mapping scovox_mapping_node --ros-args \
  -r __ns:=/robot1 -r __node:=scovox_node \
  --params-file /scovox/config/scovox_lidar_geometric.yaml \
  --params-file /scovox/config/scovox_robot_share.yaml \
  -p integration_frame:=r1_map"
```

## Verify

```bash
# merger logs "dscovox_diag: sources=1 ... fused_voxels=…" every ~5 s
$E bash -lc "$S && ros2 topic hz /robot1/scovox_node/scovox_bin"   # ~2 Hz share cadence
```

The fused map for the planner is `/robot1/dscovox_node/scovox`; the viewable cloud
is `/robot1/dscovox_node/pointcloud`.

## Scaling to a fleet

Add more robots by giving each its own namespace + unique `rK_map` frame and
pointing every merger at all robots' bin streams — see
[distributed_mapping_lidar.md](distributed_mapping_lidar.md).
