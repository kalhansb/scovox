# SCovox

Semantic Conjugate Voxel mapping with a unified per-voxel Dirichlet
over `{top-K classes, OTHER, FREE}` — semantic occupancy mapping with
calibrated uncertainty, multi-robot fusion, and edge-device runtime.

## Layout

```
src/
  scovox_core/      Voxel types, Beta/Dirichlet semantics, ray casting (C++17, zero ROS deps)
  scovox_mapping/   ROS 2 nodes: scovox_mapping_node (single robot), dscovox_mapping_node (fusion)
  scovox_msgs/      v5 binary-map wire format + query service definitions

docker/             Standalone ROS 2 Jazzy build/test image + build_and_test.sh
compose.yaml        Persistent dev / run container (ROS 2 Jazzy)
config/glim/        SCovox config for the GLIM-driven experiment (see below)
scripts/glim/       GLIM experiment launch helper (see below)

docs/
  design/unified_dirichlet_design_2026_05_13.md   Unified per-voxel Dirichlet design
  occupancy_prior.md                              Why the symmetric Beta(1,1) occupancy prior
```

## Build

Target: ROS 2 Jazzy, Ubuntu 24.04, a C++17 compiler (GCC 13). The pinned
toolchain lives in [`docker/Dockerfile`](docker/Dockerfile)
(`osrf/ros:jazzy-desktop`). The three packages build standalone — independent of
the rest of the HMR_Explo workspace they're checked out into.

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

## GLIM-driven occupancy mapping (separate experiment)

Beyond the normal use above, this repo carries the config + a helper to run the
SCovox mapping node driven by **GLIM LiDAR-IMU SLAM**, which runs in a *separate*
container ([`../glim_localisation`](../glim_localisation)). GLIM owns the TF tree
(`map → odom → imu → os_lidar`) and publishes a deskewed cloud `/glim_ros/points`;
the SCovox node here subscribes to both over ROS 2 DDS (both containers are
host-net + `ipc host` + `ROS_DOMAIN_ID=0`, so they share one DDS graph) and builds
the occupancy map online. No prior `gt_map` is needed — GLIM builds its own
map + trajectory from scratch.

* **Params:** [`config/glim/scovox_lidar_glim.yaml`](config/glim/scovox_lidar_glim.yaml)
  — the single SCovox config, fed `/glim_ros/points` with `base_frame: imu`. It
  currently integrates in GLIM's `integration_frame: odom` (smooth, but drifts —
  no loop closure); the file's header comments explain switching to the `map` frame.
* **Launch helper (this container):**
  [`scripts/glim/launch_scovox.sh <mode>`](scripts/glim/launch_scovox.sh) starts
  `scovox_node` from that config. You normally don't call it directly.
* **Run it (from the host):**
  [`../run_glim_experiment.sh <mode>`](../run_glim_experiment.sh) brings up both
  containers, starts the SCovox node here, runs the matching GLIM-side driver,
  then stops the SCovox node once the driver has captured the map.

`<mode>` is one of:

| mode   | GLIM-side driver          | what it does                                                   |
|--------|---------------------------|---------------------------------------------------------------|
| `map`  | `run_glim_scovox.sh`      | headless run, unsuffixed results                              |
| `odom` | `run_glim_scovox_odom.sh` | headless "Experiment A" (odom-frame), `_odom`-suffixed results |
| `viz`  | `run_glim_scovox_viz.sh`  | like `map`, plus a live RViz window                          |

All three launch the same `scovox_node`; the mode selects the GLIM-side driver
(output naming + map/odom intent). Outputs land in
[`../glim_localisation/output/`](../glim_localisation/output/):
`scovox_map[_odom].npy`, `path_glim[_odom].csv`, `glim_map[_odom].pcd`.

## Citation

If you use this code, please cite the paper (BibTeX block to be added
once the camera-ready DOI is assigned).

## License

Apache-2.0.  See [LICENSE](LICENSE).
