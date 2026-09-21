# scovox — per-robot voxel mapping + delta merger (field setup)

`scovox_mapping_node` builds a per-robot probabilistic voxel map (Beta occupancy
evidence, optional Dirichlet semantics/TSDF) from a LiDAR `PointCloud2` stream.
It does **not** localise: each scan is placed with a pose read from TF
(`integration_frame` ← the cloud's frame), so a localizer must already be
publishing that transform; scans are gyro-deskewed from an IMU topic. In
`mode: "rolling"` the mapper also streams compressed binary map deltas
(`ScovoxMapBinary`); `dscovox_mapping_node` — the merger — subscribes to one
delta stream per robot, maintains the fused map, and republishes it as a
latched `ScovoxMap` topic. That fused topic is what the exploration planner
consumes.

```
/hesai/points ─┐
/imu/data ─────┤ (deskew)
TF map→sensor ─┴─> scovox_node ──/robot1/scovox_node/scovox_bin──> dscovox_node ──/robot1/dscovox_node/scovox──> explo_planner
  (localizer)      mode: rolling    (binary deltas, one per robot)   (merger)       (fused ScovoxMap, latched)
```

**The merger is required even with ONE robot.** The planner subscribes to the
merger's `/<robot>/dscovox_node/scovox` with transient-local durability; the
mapper's own `~/scovox` publisher is volatile, so the QoS never matches and the
planner can never consume the mapper directly. Without a running
`dscovox_node` the planner never leaves its startup state, logging
`Waiting to start: map=0 pose=... planning_map=...` every 5 s. That line is
the whole symptom — no node errors out and no topic reports the break.

## Requirements

- **ROS 2 Humble**, colcon workspace at `/home/jetsondevkit/hmr_explo/ws`.
  Packages: `scovox_core`, `scovox_msgs`, `scovox_mapping`.
- **LiDAR** publishing `sensor_msgs/PointCloud2` (`input_pointcloud_topic`).
  The subscription is best-effort by default, matching typical driver QoS.
- **IMU** publishing `sensor_msgs/Imu` (`imu_topic`) — gyro only, used for
  intra-scan deskew (`deskew_mode: "auto" | "on" | "off"`).
- **A pose source via TF**: something (e.g. the hmr_localisation stack) must
  publish, at scan timestamps, **both** `integration_frame` → the cloud's
  `frame_id` (where the points land) and `integration_frame` → `base_frame`
  (the ray origin). Either lookup failing drops the whole scan, so a missing
  `base_frame` edge leaves a healthy-looking node with a permanently empty
  map. scovox consumes these transforms; it never produces one.
- Every mapper and the merger must be **built with the same `K_TOP`**
  (compile-time semantic top-K in `scovox_core`); a mismatch fails binary
  deserialization at the merger.

## Setup (once)

```bash
cd /home/jetsondevkit/hmr_explo/ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to scovox_mapping \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Run

`scovox_mapping/config/bunker_jetson.yaml` is the worked field example (Hesai
LiDAR, external map-frame localizer). Note the package subdirectory: there is
also an unrelated `scovox/config/` at the repo root that does not hold it.
Start the mapper — **the namespace and node-name remaps are not optional**
(see Notes):

```bash
source /home/jetsondevkit/hmr_explo/ws/install/setup.bash
CFG=$(ros2 pkg prefix scovox_mapping)/share/scovox_mapping/config

ros2 run scovox_mapping scovox_mapping_node --ros-args \
  -r __ns:=/robot1 -r __node:=scovox_node \
  --params-file $CFG/bunker_jetson.yaml
```

That file already carries `input_pointcloud_topic: /hesai/points`,
`imu_topic: /imu/data`, `base_frame: base_link` and `integration_frame: map`;
add `-p` overrides only where your robot differs.

Start the merger in the same graph:

```bash
ros2 run scovox_mapping dscovox_mapping_node --ros-args \
  -r __ns:=/robot1 -r __node:=dscovox_node \
  -p "input_topics:=['/robot1/scovox_node/scovox_bin']" \
  -p map_frame:=map \
  -p pointcloud_min_interval_s:=0.5
```

For bag replay add `-p use_sim_time:=true` to both. Verify the chain:

```bash
ros2 topic hz /robot1/scovox_node/scovox_bin        # deltas flowing
ros2 topic info -v /robot1/dscovox_node/scovox      # fused map, TRANSIENT_LOCAL
```

`ros2 launch scovox_mapping dscovox_single_robot.launch.py` brings up the same
mapper+merger pair correctly namespaced — use it as the wiring reference when
the delta topic does not appear in your own bring-up. Its defaults target
Ouster bag replay, not bunker (`cloud_topic:=/ouster/points`,
`base_frame:=os_lidar`, `use_sim_time:=true`, `max_range:=40.0`), so pass the
matching launch arguments before running it on hardware.

## Notes

- **`mode: "rolling"` is mandatory in this stack.** It is the only mode that
  creates the `~/scovox_bin` (`ScovoxMapBinary`) publisher, the sole input
  `dscovox_node` subscribes to. Under `mode: "persistent"` that publisher does
  not exist, so the merger receives nothing and publishes nothing. The mapper
  warns once at startup, naming the topic that is missing; downstream there is
  only the planner's 5 s `Waiting to start` line. `bunker_jetson.yaml` ships
  `mode: "rolling"`. The voxel grid itself is fully persistent in both modes
  (no pruning). This is not a single-vs-multi-robot switch: it changes the
  wire format, and separately makes `planning_map` publish a robot-centred
  crop of `plan_window_size_m` instead of the fixed envelope (inert while
  `publish_planning_map` is false).
- **Namespace + node name build the share topic.** The delta topic is
  `<ns>/<node>/scovox_bin` (from `scovox_topic: "~/scovox"` + `_bin`), hence
  `-r __ns:=/robot1 -r __node:=scovox_node`. The merger's defaults expect
  exactly `/robot1/scovox_node/scovox_bin` and `/robot2/scovox_node/scovox_bin`
  (`input_topic_1`/`input_topic_2`, used when `input_topics` is unset). A
  mapper left in the root namespace publishes `/scovox_node/scovox_bin` and
  the merger matches nothing. The mapper logs "No subscriber on … scovox_bin"
  every 30 s when this is miswired. Deltas produced while nothing is attached
  are discarded rather than buffered, but starting the merger late costs no
  map: a new subscriber triggers a full snapshot of the whole grid on connect.
- **What to change for a new robot** (edit a copy of `bunker_jetson.yaml`):
  `input_pointcloud_topic`, `imu_topic`, `base_frame` (ray-origin frame),
  `integration_frame`/`map_frame` (the frame your localizer actually
  publishes — with a map-frame localizer and no odom link, `integration_frame`
  must be `map`), `resolution` (keep `downsample_voxel_size` equal to it),
  `min_range`/`max_range` (cost vs. coverage), and `deskew_ref_frac` (0.5 when
  the localizer registers the raw sweep; 0.0 if it deskews itself). Leave the
  rest at the file's values.
- **Two robots**: run a second mapper with `-r __ns:=/robot2
  -r __node:=scovox_node`; every robot runs its own merger listing all peers,
  e.g. `input_topics:=['/robot1/scovox_node/scovox_bin','/robot2/scovox_node/scovox_bin']`
  (`scovox_mapping/config/dscovox_params.yaml` is the fleet-wide template).
- **Vertical share band**: `share_roi_z_min`/`share_roi_z_max` on mapper (wire
  filter) and merger (ingest clip) crop what is *shared*, not the local map;
  min >= max (the 0/0 default) disables. If set, the band must be a superset
  of the planner's `roi_min_z`/`roi_max_z`.
- `share_rate_hz > 0` moves delta publishing onto a timer and coalesces
  repeated writes of a voxel into one wire record; `bunker_jetson.yaml` ships
  2.0. `<= 0` publishes after every scan.
