# SCovox & DScovox — user manual

A single, clear entry point for building maps with **SCovox** (the per-robot
voxel mapper) and **DScovox** (the multi-robot fusion / merger). It tells you
what the two pieces are, how to bring them up, what to watch, and how to fix the
handful of things that usually go wrong. For deep dives it points at the focused
runbooks rather than repeating them.

**Contents**

1. [What SCovox and DScovox are](#1-what-scovox-and-dscovox-are)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [Prerequisites](#3-prerequisites)
4. [Quick start](#4-quick-start)
5. [Topics & services](#5-topics--services)
6. [Configuration reference](#6-configuration-reference)
7. [Verifying a healthy map](#7-verifying-a-healthy-map)
8. [Troubleshooting](#8-troubleshooting)
9. [Bandwidth tuning](#9-bandwidth-tuning)
10. [Where to go deeper](#10-where-to-go-deeper)

---

## 1. What SCovox and DScovox are

**SCovox** ("Semantic Conjugate Voxel") is a per-robot occupancy mapper. It
consumes a LiDAR cloud (optionally plus RGB-D segmentation) and TF, and builds a
voxel grid on a split per-voxel substrate: a Beta–Bernoulli **occupancy** counter
alongside a Dirichlet over `{top-K classes, OTHER}` **semantics**. It runs on an
edge device and reports calibrated uncertainty. The ROS 2 node is
`scovox_mapping_node`.

**DScovox** ("distributed SCovox") is the fusion layer. Each robot's mapper emits
a small, LZ4-compressed **delta stream** (`~/scovox_bin`); a `dscovox_mapping_node`
merger subscribes to **every** robot's stream and fuses them into one
co-registered global map. There is **no central server** — every robot runs its
own merger and ends up with its own copy of the same global map. One robot is
just a fleet of one: the merger fuses its single stream and its output is the
(trivially global) map the exploration planner reads.

Two sensor paths, selected by one parameter:

| Path | `fuse_lidar_rgbd` | Builds | Base config |
|------|-------------------|--------|-------------|
| **LiDAR-only (geometric)** | `false` | occupancy + free space, no semantics | [`scovox_lidar_geometric.yaml`](../config/scovox_lidar_geometric.yaml) |
| **Fused LiDAR + RGB-D** | `true` | occupancy (LiDAR) + semantics (RGB-D) | [`scovox_fused_lidar_rgbd.yaml`](../config/scovox_fused_lidar_rgbd.yaml) |

If you only have a LiDAR, use the geometric path — everything below works the
same, semantics just stay empty.

---

## 2. Architecture at a glance

Each robot runs the **same three programs**:

```
                 robot 1                                robot 2
 ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
 │ Localizer  (NDT vs shared gt_map)│   │ Localizer  (NDT vs shared gt_map)│
 │   map → odom → base_link → sensor│   │   map → odom → base_link → sensor│
 │ SCovox mapper  (mode: rolling)   │   │ SCovox mapper  (mode: rolling)   │
 │   → /robot1/…/scovox_bin ────────┼──→                                   │
 │                             ←────┼───── /robot2/…/scovox_bin            │
 │ DScovox merger (fuses BOTH)      │   │ DScovox merger (fuses BOTH)      │
 │   = robot 1's GLOBAL map         │   │   = robot 2's GLOBAL map         │
 └──────────────────────────────────┘   └──────────────────────────────────┘
```

1. **Localizer** — [hmr_localisation](https://github.com/kalhansb/hmr_localisation)
   runs NDT against a **shared** ground-truth map (`gt_map/gt_map_us050.pcd`). This
   gives every robot the same global `map` frame, so **no robot-to-robot pose
   estimation is needed** — the shared frame is the whole trick that makes the
   merge spatially correct.
2. **SCovox mapper** (`scovox_mapping_node`, `mode: rolling`) — integrates each
   scan into a local voxel map and publishes the `~/scovox_bin` delta stream.
3. **DScovox merger** (`dscovox_mapping_node`) — subscribes to all robots' delta
   streams and fuses them. Its output is that robot's copy of the global map.

### Two rules that make fusion work

1. **Each robot must map in a UNIQUE frame** (`r1_map`, `r2_map`, … or
   `bunker_map`, `curt_map`), bridged to `map` by an identity static TF. The delta
   stream is tagged with this frame (`header.frame_id`) and the merger keys each
   robot's data by it. If two robots both map in `map`, their streams collapse into
   one source and overwrite each other where the maps overlap.
2. **Start the merger before the mapper.** The delta stream is subscriber-gated —
   a mapper with no listener throws its deltas away. (A late merger still catches
   up: the mapper re-sends a full snapshot whenever a new subscriber appears.)

Robots do **not** need to start at the same time. A late merger gets a fresh
snapshot; a late mapper just starts contributing when it comes up.

---

## 3. Prerequisites

Everything runs inside the repos' Docker containers; nothing is installed on the
host.

1. **Workspace built.** Clone recursively and bring up both containers:
   ```bash
   git clone --recursive https://github.com/kalhansb/hmr_explo.git
   cd hmr_explo/ws/src/hmr_localisation && docker compose build && docker compose up -d
   cd ../scovox && docker compose build && docker compose up -d
   ```
   Rebuild SCovox after editing anything under `src/` — `ros2 launch` resolves from
   `install/`, not `src/`:
   ```bash
   docker compose -f scovox/compose.yaml exec scovox bash -lc \
     'source /opt/ros/jazzy/setup.bash && cd /scovox && colcon build --packages-select scovox_mapping'
   ```
   Release builds are mandatory — debug builds inflate timing ~3–4×.
2. **Sensors.** A LiDAR driver publishing a `PointCloud2` (e.g. `/ouster/points`)
   with **RELIABLE QoS** — the localizer subscribes RELIABLE and a best-effort
   publisher never connects — plus an IMU (e.g. `/imu/data`) used for per-scan
   motion deskew.
3. **Extrinsics / TF.** The `base_link → sensor` and `base_link → imu` transforms.
   The map-test robot's are baked into hmr_localisation's `run_localization_live.sh`;
   re-measure for a different platform.
4. **Network** (real fleet). All robots on one LAN with the same `ROS_DOMAIN_ID`.
   The compose files pin DDS discovery to loopback, so any `docker compose exec`
   that must talk across machines needs `-e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`.

> **Bag replay vs live hardware.** Use `use_sim_time:=true` when driving from a
> recorded bag (nodes idle until `/clock`), and `use_sim_time:=false` on a live
> robot.

---

## 4. Quick start

Three ways in, easiest first. All commands run inside the containers with the
workspace sourced (`source /opt/ros/jazzy/setup.bash; source /scovox/install/setup.bash`).

### 4a. Single robot from a bag (fastest end-to-end check)

Start order matters: the localizer and SCovox come up first (they idle on
`/clock`), the **bag plays last**. Full walkthrough:
[dscovox_single_robot_run.md](dscovox_single_robot_run.md).

```bash
# 1. Localizer (leave running) — NDT map→odom + EKF, vs the shared gt_map.
#    (see the runbook for the full command)

# 2. SCovox rolling mapper + DScovox merger (leave running):
ros2 launch scovox_mapping dscovox_single_robot.launch.py \
    robot:=robot1 cloud_topic:=/ouster/points base_frame:=os_lidar use_sim_time:=true

# 3. Play the bag LAST (starts /clock + streams data):
ros2 bag play <bag_dir> --clock --rate 0.5 \
    --topics /ouster/points /imu/data /tf /tf_static
```

`dscovox_single_robot.launch.py` starts a rolling mapper **and** a merger (so the
subscriber gate is open), pins the topic to `/robot1/scovox_node/scovox_bin`, and
auto-publishes the identity `map → integration_frame` static TF when needed. It
doubles as the **known-good reference**: if the bin stream flows here but not in
your setup, the difference is your config.

### 4b. Multi-robot from bags (single-host fusion demo)

Two independently-recorded robots (`bunker`, `curt`) each localize against the
**same** gt_map, build a rolling map in their **own** frame, and one merger fuses
both. Full walkthrough: [dscovox_multi_robot_run.md](dscovox_multi_robot_run.md).

```bash
# SCovox mappers (one per robot, unique frames) + one merger:
ros2 launch scovox_mapping dscovox_multi_robot.launch.py use_sim_time:=true
```

Then bring up both localizers, and **play the bags sequentially** (earlier
recording first) under one advancing clock — the merger persists each robot's
contribution, so a "late" robot just adds its region. Watch for the merger to
reach `dscovox_diag: sources=2`.

### 4c. Live robot / real fleet

The production topology — per-robot localizer → identity static TF → merger →
mapper, no central server. Set each robot up identically, changing only the
per-robot index:

| per-robot value       | robot 1     | robot 2     |
|-----------------------|-------------|-------------|
| namespace (`__ns`)    | `/robot1`   | `/robot2`   |
| `integration_frame`   | `r1_map`    | `r2_map`    |
| static-TF child frame | `r1_map`    | `r2_map`    |

Mapper bring-up (LiDAR-only) = base config + share overlay + the one per-robot
frame:

```bash
# identity bridge  map → r1_map
ros2 run tf2_ros static_transform_publisher --frame-id map --child-frame-id r1_map

# merger FIRST (opens the subscriber gate)
ros2 run scovox_mapping dscovox_mapping_node --ros-args -r __ns:=/robot1 \
    --params-file /scovox/src/scovox_mapping/config/dscovox_params.yaml

# then the rolling mapper
ros2 run scovox_mapping scovox_mapping_node --ros-args \
    -r __ns:=/robot1 -r __node:=scovox_node \
    --params-file /scovox/config/scovox_lidar_geometric.yaml \
    --params-file /scovox/config/scovox_robot_share.yaml \
    -p integration_frame:=r1_map
```

For the fused (semantic) path, swap the base config to
`scovox_fused_lidar_rgbd.yaml` and feed the seg pipeline's RGB-D/segmentation
topics. On **every** robot, extend `input_topics` in `dscovox_params.yaml` to list
all robots' bin topics. Full runbooks:
[distributed_mapping_lidar.md](distributed_mapping_lidar.md) (LiDAR-only) and
[distributed_mapping.md](distributed_mapping.md) (fused/semantic).

---

## 5. Topics & services

Topic paths are `~`-relative to the node, so they depend on the node's namespace
and name. With `-r __ns:=/robot1 -r __node:=scovox_node` the mapper's
`~/scovox_bin` is `/robot1/scovox_node/scovox_bin`.

### SCovox mapper (`scovox_mapping_node`)

| Direction | Topic (param) | Type | Notes |
|-----------|---------------|------|-------|
| in  | `input_pointcloud_topic` | `PointCloud2` | LiDAR cloud. **Empty ⇒ RGB-D mode** (no LiDAR subscription — a common silent trap). |
| in  | `imu_topic` | `Imu` | gyro source for per-scan deskew |
| out | `~/scovox_bin` | `ScovoxMapBinary` | LZ4 delta stream. **Only exists in `mode: rolling`.** |

### DScovox merger (`dscovox_mapping_node`)

| Direction | Topic / service (param) | Type | Notes |
|-----------|-------------------------|------|-------|
| in  | `input_topics` (list) | `ScovoxMapBinary` | every robot's bin stream, fleet-wide |
| out | `~/scovox` (`scovox_topic`) | `ScovoxMap` | latched fused map — **the exploration planner's input** |
| out | `~/pointcloud` (`pointcloud_topic`) | `PointCloud2` | viewable fused cloud (RViz) |
| srv | `~/get_region` | `GetRegion` | per-voxel occupancy + class evidence in an AABB |
| srv | `~/get_occupancy_grid` | `GetOccupancyGrid` | 2D/3D occupancy grid projection |

Concrete defaults from the two launch files:

- **Single-robot** (`dscovox_single_robot.launch.py`, merger under `/<robot>`):
  fused map `/robot1/dscovox_node/scovox`, cloud `/robot1/dscovox_node/pointcloud`.
- **Multi-robot** (`dscovox_multi_robot.launch.py`, one root-namespace merger):
  cloud remapped to `/dscovox_mapping/pointcloud`.

---

## 6. Configuration reference

### Config files (load in order; later `--params-file` wins)

| File | Role |
|------|------|
| [`scovox_lidar_geometric.yaml`](../config/scovox_lidar_geometric.yaml) | LiDAR-only mapper **base** (Beta occupancy, full-ray carve, in-node deskew; `fuse_lidar_rgbd: false`, `mode: persistent`) |
| [`scovox_fused_lidar_rgbd.yaml`](../config/scovox_fused_lidar_rgbd.yaml) | Fused mapper **base** (LiDAR owns occupancy, RGB-D adds semantics) |
| [`scovox_robot_share.yaml`](../config/scovox_robot_share.yaml) | **Share overlay** — flips to `mode: rolling` (creates the bin stream) + low-bandwidth controls + z-band. Load **on top of** a base. |
| [`dscovox_params.yaml`](../src/scovox_mapping/config/dscovox_params.yaml) | Merger config. `input_topics` lists the fleet's bin topics — **extend it as the fleet grows.** |
| [`scovox_bin_min.yaml`](../src/scovox_mapping/config/scovox_bin_min.yaml) | Minimal params that **guarantee** the bin stream — a copy-paste starting point for debugging. |

> **Run the identical base config on every robot.** The merger pins the map's
> `num_classes` / Dirichlet prior from the first stream it sees and drops any
> source whose prior differs (`prior mismatch … dropping frame`). One config
> fleet-wide keeps them matched.

### Key mapper parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mode` | `persistent` | `rolling` creates the `~/scovox_bin` publisher — **required to share.** |
| `fuse_lidar_rgbd` | `false` | `true` adds the RGB-D semantic path. |
| `integration_frame` | `map` | Frame the map is built in; **must be unique per robot** in a fleet. |
| `base_frame` | `os_lidar` | LiDAR ray-origin (sensor) frame. |
| `resolution` | `0.10` | Voxel size (m). |
| `w_occ` / `w_free` | `8.0` / `4.0` | Beta occupancy evidence weights (hit / clear-air). |
| `carve_band` | `-1.0` | Full-ray free-space carve (planner needs complete free space); positive = carve only near surfaces (cheaper). |
| `min_range` / `max_range` | `1.0` / `20–40` | Per-scan integration range; **raise `max_range` to widen the map's radial footprint.** |
| `deskew_mode` | `auto` | `auto\|on\|off`; `off` silences the harmless IMU-not-ready note. |
| `downsample_voxel_size` | `0.10` | Per-scan sensor-frame downsample before integration (keeps dense LiDAR real-time). |
| `tf_require_exact` | `true` | Require TF at the scan's exact stamp; **do not relax** — the stale-pose fallback mis-places whole scans. |
| `share_change_gate` | `true` | Re-send a voxel only when it changed (bandwidth). |
| `share_rate_hz` | `2.0` | Coalesce deltas into a timer publish (`0.0` = legacy per-scan). |
| `share_roi_z_min/max` | `-0.5 / 2.0` | Vertical share band (map frame); `min ≥ max` disables. |

### Key merger parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `input_topics` | `[…scovox_bin]` | Fleet-wide list of bin streams to fuse. |
| `semantic_top_k` | `2` | Must match the senders' compile-time `K_TOP`. |
| `publish_rate_hz` | `1.0` | Fused `~/scovox` (planner) + viz cadence. |
| `share_roi_z_min/max` | `-0.5 / 2.0` | Receive-side clip; **keep in sync with the sender band.** |

> **Z-band coherence.** The share band must be a **superset** of the planner band
> (`roi_min_z`/`roi_max_z` in explo_planner's `exploration_params.yaml`) and in
> sync across the mapper (`scovox_robot_share.yaml`) and merger
> (`dscovox_params.yaml`).

---

## 7. Verifying a healthy map

```bash
# mapper is emitting deltas (~2 Hz with the share overlay):
ros2 topic hz /robot1/scovox_node/scovox_bin

# fused map + viewable cloud are flowing:
ros2 topic hz /robot1/dscovox_node/pointcloud
```

**Success** = the merger console (it runs `output=screen`) prints, about every
5 s:

```text
dscovox_diag: sources=N src_voxels=… fused_voxels=…
```

- `sources` = number of robots whose stream has arrived (`1` per robot).
- `fused_voxels > 0` = data arrived and fused.
- In a genuine multi-robot merge, `src_voxels > fused_voxels` (the maps overlap =
  co-registered), and the robots' `fused_voxels` counts converge toward each other.

Cross-robot spot check — query one robot's merger for a region only a *peer* has
visited; a populated response can only have crossed the wire:

```bash
ros2 service call /robot1/dscovox_node/get_region scovox_msgs/srv/GetRegion \
  '{min_corner: {x: <x0>, y: <y0>, z: -0.5}, max_corner: {x: <x1>, y: <y1>, z: 2.0}}'
```

**Validate without robots.** `run_mapshare_experiment.sh` (in the hmr_explo
workspace) replays a bag through the exact N-mapper/N-merger topology and ends
with `FUSION VERIFY: PASS/FAIL`. LiDAR-only arm:
`LIDAR_ONLY=1 ./run_mapshare_experiment.sh 120`.

---

## 8. Troubleshooting

Almost every "no map" / "no `scovox_bin`" report is one of **three gates** on the
delta stream. Full step-by-step diagnosis:
[scovox_bin_manual_bringup.md](scovox_bin_manual_bringup.md).

| Symptom | Gate | Fix |
|---------|------|-----|
| `scovox_bin` **absent** from `ros2 topic list` | 1 — publisher exists only in `mode: rolling` | `-p mode:=rolling` (or load the share overlay). Also check `-r __ns:=/robotK -r __node:=scovox_node`. |
| Topic **present** but `topic hz` **silent** | 2 — publish is subscriber-gated | Start the merger (or `ros2 topic echo`) — a fresh subscriber triggers a full snapshot. |
| `DROPPING scan` / `no exact-stamp TF` logs; hz empty | 3 — TF missing at scan stamp | Supply `map→odom→base_link→sensor` (localizer up + bag/robot statics). **Do not** relax `tf_require_exact`. |

Other common issues:

| Symptom | Cause → fix |
|---------|-------------|
| Merger stuck at `sources=0` | Mapper started before the merger subscribed (deltas drained), or the bin topic isn't visible on the peer → restart the mapper; for a fleet check DDS discovery (`ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, same `ROS_DOMAIN_ID`, same subnet). |
| Localizer gets no scans | The LiDAR driver publishes **BEST_EFFORT** → switch it to RELIABLE. |
| Robots' maps flicker / overwrite each other | They share one `integration_frame` → give each its own `rK_map` + identity static TF (rule 1). |
| `no map ← rK_map transform yet … deferring publish` | That robot's own identity static TF isn't up → start it; the mapper holds deltas and retries. |
| `prior mismatch … dropping frame` | A robot runs a different config → use the **identical** base config fleet-wide. |
| Empty map despite a cloud | `input_pointcloud_topic` is empty → node is in RGB-D mode; set your cloud topic. |
| NDT sits at "Configuring end" (never "Activating end") | Lifecycle handoff missed → `ros2 lifecycle set /lidar_localization activate`. |

> **Rig-specific gotcha.** Do not trust a LiDAR-embedded IMU blind. On the Unitree
> Go1 bag, `/hesai/imu`'s gyro was corrupt and fed garbage rotation into the
> deskew (flinging points under the floor); the fix was `imu_topic:=/imu/data`
> (the body IMU). Verify a new rig's gyro before wiring it to the deskew.

---

## 9. Bandwidth tuning

The share is a strict subset of the fused pipeline, so it's cheap. Measured
per-robot on the map-test-2 stream (0.1 m voxels, 14 classes, top-2 evidence):

| Configuration | Wire cost |
|---------------|-----------|
| Legacy per-scan publish | 32.8 Mbps |
| + change gate | 20.7 Mbps |
| + change gate + 2 Hz coalescing + z-band | **4.9 Mbps** |

LiDAR-only carries **no semantic records**, so it's cheaper still (wire cost per
delta ≈ `28 + 20·N_beta` bytes). The three knobs live in
[`scovox_robot_share.yaml`](../config/scovox_robot_share.yaml): `share_change_gate`,
`share_rate_hz`, and the z-band `share_roi_z_min/max` — kept in sync with the
merger and a superset of the planner band (see [§6](#6-configuration-reference)).
Full measurements: `map_share_bandwidth_experiment.md` in the hmr_explo workspace.

---

## 10. Where to go deeper

| Doc | Use it for |
|-----|-----------|
| [single_robot_lidar.md](single_robot_lidar.md) | Live single-robot LiDAR mapping, the per-robot building block. |
| [dscovox_single_robot_run.md](dscovox_single_robot_run.md) | Bag → localizer → SCovox → RViz, one robot, step by step. |
| [dscovox_multi_robot_run.md](dscovox_multi_robot_run.md) | Two bags → two localizers → merger, single-host fusion demo. |
| [distributed_mapping_lidar.md](distributed_mapping_lidar.md) | Real multi-robot fleet, **LiDAR-only**. |
| [distributed_mapping.md](distributed_mapping.md) | Real multi-robot fleet, **fused LiDAR + RGB-D semantics**. |
| [scovox_bin_manual_bringup.md](scovox_bin_manual_bringup.md) | The three-gate diagnosis when `scovox_bin` is missing. |
| [publish_scovox_bin_from_bag.md](publish_scovox_bin_from_bag.md) | Driving the bin stream from a recorded bag. |
| [`../README.md`](../README.md) | Package layout, build, and the theory behind the substrate. |
</content>
</invoke>
