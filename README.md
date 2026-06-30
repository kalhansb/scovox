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
config/             Raw-cloud occupancy config with the smear-suppression cure (see below)
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

**TF required** (from your odometry/SLAM): `integration_frame → sensor_frame`
(the cloud/depth `frame_id`) and `integration_frame → base_frame` (robot pose).
Multi-robot fusion also needs `map_frame → <robot>/odom` per robot.

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

### Multi-robot (SCovox + DSCovox)

One `scovox_mapping_node` per robot (`mode:=rolling`, namespaced) publishes
`/<robot>/scovox_node/scovox_bin`; one `dscovox_mapping_node` fuses them:

```bash
ros2 launch scovox_mapping scovox_multi_robot.launch.py
```

Edit the `robots` list / `input_topics` in [`scovox_multi_robot.launch.py`](src/scovox_mapping/launch/scovox_multi_robot.launch.py).

### Interfaces

* `scovox_mapping_node` (node name `scovox_node`) — **in:** LiDAR `PointCloud2` *or* depth + camera_info + seg;
  **out:** `~/pointcloud`, `~/scovox`, `~/scovox_bin` (rolling mode), `~/tsdf_pointcloud` (if `enable_tsdf`), `~/planning_map` (if `publish_planning_map`);
  **service:** `~/extract_mesh` (`ExtractMesh`).
* `dscovox_mapping_node` (node name `dscovox_node`) — **in:** each robot's `~/scovox_bin` (`input_topics`);
  **out:** fused `~/pointcloud`; **services:** `~/get_region` (`GetRegion`), `~/get_occupancy_grid` (`GetOccupancyGrid`).

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
