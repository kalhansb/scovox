# SCovox

Semantic Conjugate Voxel mapping on a split per-voxel substrate — a
Beta–Bernoulli occupancy counter alongside a Dirichlet over
`{top-K classes, OTHER}` semantic classes — for semantic occupancy mapping with
calibrated uncertainty, multi-robot fusion, and edge-device runtime.

## Layout

```
src/
  scovox_core/      Voxel types, Beta/Dirichlet semantics, ray casting (C++17, zero ROS deps)
  scovox_mapping/   ROS 2 nodes: scovox_mapping_node (single robot), dscovox_mapping_node (fusion)
  scovox_msgs/      v5 binary-map wire format + query service definitions
  seg_pipeline/     Online RGB-D segmentation node (feeds SCovox seg_topic; own GPU container)

docker/             Standalone ROS 2 Jazzy build/test image + build_and_test.sh
compose.yaml        Persistent dev / run container (ROS 2 Jazzy)
config/             Param sets: raw-cloud smear cure, fused LiDAR+RGB-D, robot-share
                    overlay, and the HMR_Explo exploration experiment (planner + RViz)
scripts/            Launch helper for the raw-cloud config (see below)

docs/
  design/unified_dirichlet_design_2026_05_13.md   Unified per-voxel Dirichlet design
  occupancy_prior.md                              Why the symmetric Beta(1,1) occupancy prior
```

## Build

Target: ROS 2 Jazzy, Ubuntu 24.04, a C++17 compiler (GCC 13). The pinned
toolchain lives in [`docker/Dockerfile`](docker/Dockerfile)
(`osrf/ros:jazzy-desktop`). The three C++ packages build standalone — independent
of the rest of the HMR_Explo workspace they're checked out into. The Python
`seg_pipeline` package builds here too (`ros2 run seg_pipeline seg_node`), but its
node needs GPU/ML deps and runs in its own container — see
[`src/seg_pipeline/README.md`](src/seg_pipeline/README.md).

### Docker (recommended)

The image bakes in every build/test dependency via rosdep over the package
manifests (no network needed at build time); the workspace is bind-mounted, so
`build/`, `install/`, and `log/` land in the repo (git-ignored).

```bash
# One-shot: build the image, then colcon build + test in a throwaway container.
./docker/build_and_test.sh
./docker/build_and_test.sh --shell      # ...or drop into a dev shell instead

# Persistent dev / run container (exec into it repeatedly).
docker compose up -d --build
docker compose exec scovox bash -lc 'colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release'
docker compose exec scovox bash -lc 'colcon test && colcon test-result --all'
```

### Native colcon

```bash
mkdir -p ~/scovox_ws/src && cd ~/scovox_ws/src
git clone https://github.com/kalhansb/scovox.git .
cd ..
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Release builds are mandatory — debug builds inflate timing numbers by ~3–4× and
bias any wall-clock measurement. (The CMake files already default to Release when
no build type is given.)

Runtime dependencies: Eigen 3.4+, lz4, standard ROS 2 stack.

## Running on a robot

Both nodes consume live sensor topics + TF — no datasets needed. Use
`use_sim_time:=false` on hardware.

**Docker:** run everything below inside the compose container on the robot
(`docker compose up -d`, then `docker compose exec scovox bash`). It uses host
networking, but [`compose.yaml`](compose.yaml) pins DDS discovery to loopback —
add `-e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET` to `exec` when sensors or
teammate robots publish from other machines.

**TF required** (from your odometry/SLAM): `integration_frame → sensor_frame`
(the cloud/depth `frame_id`) and `integration_frame → base_frame` (robot pose).
Multi-robot fusion also needs `map_frame → <bin frame_id>` per peer (identity
in the recommended setup below).

### Single robot

LiDAR, geometry only:

```bash
ros2 launch scovox_mapping lidar_mapping.launch.py pointcloud_topic:=/your/lidar/points
```

Tune frames/ranges in [`config/lidar_mapping.yaml`](src/scovox_mapping/config/lidar_mapping.yaml).
TSDF surface integration is off by default (`enable_tsdf: false`) — occupancy
only; set `enable_tsdf: true` to also build the dense fused surface.
For semantics, give SCovox a class per voxel via a `semantic_label` field on the
`PointCloud2`, or a colored segmentation image (`seg_topic`) — RGB-D path is wired
in [`scovox_single_robot.launch.py`](src/scovox_mapping/launch/scovox_single_robot.launch.py).

**Recommended LiDAR setup** — map the full-resolution sensor cloud with native
deskew + downsample so the occupancy map stays surface-thin (no vertical smear)
without discarding points; an external localizer (e.g. GLIM) supplies the pose:

* **Params:** [`config/scovox_lidar_raw_deskew.yaml`](config/scovox_lidar_raw_deskew.yaml)
  — raw `/ouster/points`, `base_frame: os_lidar`, `integration_frame: odom`,
  `deskew_mode: auto`, `downsample_voxel_size: 0.5`. The file's header comments
  carry the full rationale.
* **Launch helper (in the SCovox container):**
  [`scripts/launch_scovox.sh raw`](scripts/launch_scovox.sh) starts
  `scovox_node` from that config against `/ouster/points`.

See [Raw-cloud occupancy mapping](#raw-cloud-occupancy-mapping-vertical-smear-cure)
for the problem this solves.

### Multi-robot mapping (SCovox + DSCovox)

> Start-to-finish real-robot runbook (single robot and two robots, incl.
> localizer bring-up, verification, and troubleshooting):
> [`docs/distributed_mapping.md`](docs/distributed_mapping.md).

One `scovox_mapping_node` per robot (`mode: rolling`, namespaced) publishes an
LZ4 `ScovoxMapBinary` delta stream on `/<robot>/scovox_node/scovox_bin`; every
robot also runs its own `dscovox_mapping_node` fusing ALL peers' streams —
there is no central merger.

**TF contract:** each robot runs the sibling `hmr_localisation` NDT localizer
(in the HMR_Explo workspace) against the same ground-truth map, giving it
`map → odom → base_link` locally — the fleet shares one global `map` frame.
Each robot's mapper then integrates in a **per-robot frame** (`r1_map`,
`r2_map`, …) bridged to `map` by an identity static TF. The unique frame is
**mandatory**: the bin stream's `header.frame_id` is the `integration_frame`,
and the merger keys its per-source grids by that frame_id — two robots sharing
one frame_id collapse into a single source grid and overwrite each other where
their maps overlap. The static TF is one latched `/tf_static` sample that
peers' mergers cache on first sight; there is no per-scan cross-robot `/tf`.

Per robot, inside the container (shown for robot 1 — substitute the
namespace, frame, and log names; **start the merger before the mapper**: the
bin publish is subscriber-gated, deltas are drained while nobody listens):

```bash
# Identity bridge map -> r1_map (see TF contract above).
ros2 run tf2_ros static_transform_publisher \
  --frame-id map --child-frame-id r1_map &

# Merger — fleet-wide bin-topic list lives in dscovox_params.yaml input_topics.
ros2 run scovox_mapping dscovox_mapping_node --ros-args -r __ns:=/robot1 \
  --params-file /scovox/src/scovox_mapping/config/dscovox_params.yaml &

# Mapper — base sensor config + the real-robot share overlay (later file
# wins): rolling mode, use_sim_time=false, and the low-bandwidth share
# controls (change gate + 2 Hz coalescing + z-band; measured 32.8 -> 4.9 Mbps
# per robot). integration_frame is the one per-robot param (see above).
ros2 run scovox_mapping scovox_mapping_node --ros-args \
  -r __ns:=/robot1 -r __node:=scovox_node \
  --params-file /scovox/config/scovox_fused_lidar_rgbd.yaml \
  --params-file /scovox/config/scovox_robot_share.yaml \
  -p integration_frame:=r1_map
```

The fused latched map comes out on `/<robot>/dscovox_node/scovox` — the
planner input consumed by
[explo_planner](https://github.com/kalhansb/explo_planner). Keep the share
z-band in sync across [`scovox_robot_share.yaml`](config/scovox_robot_share.yaml),
[`dscovox_params.yaml`](src/scovox_mapping/config/dscovox_params.yaml), and
explo_planner's `exploration_params.yaml` (shared band ⊇ planner band) — each
file's comments cross-reference the others.

For simulation/bag use,
[`scovox_multi_robot.launch.py`](src/scovox_mapping/launch/scovox_multi_robot.launch.py)
wires the same topology (edit its `robots` list / `input_topics`). To drive the
`~/scovox_bin` delta stream straight from a recorded rosbag's LiDAR (and why it
stays silent by default — persistent mode, subscriber-gating, and the bag having
no `map`/`odom` TF), see
[`docs/publish_scovox_bin_from_bag.md`](docs/publish_scovox_bin_from_bag.md).

### Exploration experiment (HMR_Explo)

The bag-driven exploration experiment in the parent
[HMR_Explo](https://github.com/kalhansb/hmr_explo) workspace
(`ws/src/run_explo_experiment.sh`) runs explo_planner and RViz inside this
repo's container, so both of its config files live here and are read live
through the `/scovox` bind mount — host edits apply on the next run, no
rebuild or copy:

* [`config/exploration_fused_bag.yaml`](config/exploration_fused_bag.yaml) —
  explo_planner param set: terrain-relative 3D mode, straight-line candidate
  costs (`require_planning_map: false`, no 2D planning_map — the file header
  carries the rationale).
* [`config/explo_experiment.rviz`](config/explo_experiment.rviz) — RViz
  layout for the run (semantic cloud, candidate arrows, selected goal).

### Interfaces

* `scovox_mapping_node` (node name `scovox_node`) — **in:** LiDAR `PointCloud2` *or* depth + camera_info + seg;
  **out:** `~/pointcloud`, `~/scovox`, `~/scovox_bin` (rolling mode), `~/tsdf_pointcloud` (if `enable_tsdf`), `~/planning_map` (if `publish_planning_map`);
  **service:** `~/extract_mesh` (`ExtractMesh`).
* `dscovox_mapping_node` (node name `dscovox_node`) — **in:** each robot's `~/scovox_bin` (`input_topics`);
  **out:** fused `~/pointcloud`, latched fused `~/scovox` (planner input);
  **services:** `~/get_region` (`GetRegion`), `~/get_occupancy_grid` (`GetOccupancyGrid`).

Full parameter reference: [`config/default_params.yaml`](src/scovox_mapping/config/default_params.yaml).

## Raw-cloud occupancy mapping (vertical-smear cure)

Beyond the normal use above, this repo carries a config + helper for mapping the
**raw, full-resolution Ouster cloud** (`/ouster/points`) while an
external localizer (e.g. GLIM LiDAR-IMU SLAM) owns the TF tree
(`map → odom → imu → os_lidar`) and supplies the per-scan pose. SCovox subscribes
to `/ouster/points` + `/imu/data` over ROS 2 DDS and builds the occupancy map
online — no prior map needed.

**The problem.** Integrating the raw full-res cloud accumulates a solid *vertical
smear*: each XY column fills many occupied z-cells where a clean surface should be
thin. The surface returns are noisy (pose accumulation, range, and incidence
angle), and dense full-res sampling keeps filling the *tails* of each column's
z-distribution. Feeding the localizer's own pre-downsampled cloud removes the
smear but throws away most of the points (a recall loss).

**The cure** (replicated natively, so we keep the points). SCovox preprocesses
each scan itself:

1. **Native per-point gyro deskew** — rotates each point to scan-start using the
   IMU (`deskew_mode`). Small for a gently-moving platform, kept for completeness.
2. **Uniform voxel-grid downsample** (`downsample_voxel_size`, the dominant lever)
   — collapses each scan to one centroid per voxel before integration, so it stops
   over-filling the noisy-surface tails. This reproduces the thinning a
   LiDAR-SLAM preprocessor does, but *inside* SCovox on the full-res cloud, so the
   map keeps far more density (recall) than consuming the downsampled SLAM cloud.

Larger `downsample_voxel_size` thins the smear further but trims the horizontal
footprint; the default (`0.5`) is the knee — it suppresses the smear while keeping
the surface dense. `0.0` disables downsampling.

Run it as the **Recommended LiDAR setup** under [Running on a robot](#single-robot)
— [`config/scovox_lidar_raw_deskew.yaml`](config/scovox_lidar_raw_deskew.yaml)
via [`scripts/launch_scovox.sh raw`](scripts/launch_scovox.sh). The node
logs `ds=out/in` per scan (kept vs raw points) so you can confirm the downsample is
active.

## Citation

If you use this code, please cite the paper (BibTeX block to be added
once the camera-ready DOI is assigned).

## License

Apache-2.0.  See [LICENSE](LICENSE).
