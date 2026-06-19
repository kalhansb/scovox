# SCovox

Semantic Conjugate Voxel mapping with a unified per-voxel Dirichlet
over `{top-K classes, OTHER, FREE}` — semantic occupancy mapping with
calibrated uncertainty, multi-robot fusion, and edge-device runtime.

## Layout

```
src/
  scovox_core/      Voxel types, Dirichlet semantics, ray casting (C++17, zero ROS deps)
  scovox_mapping/   ROS 2 nodes: scovox_mapping_node, dscovox_mapping_node
  scovox_msgs/      BinaryMap v3 wire format + service definitions

docs/
  design/unified_dirichlet_design_2026_05_13.md   v3 Dirichlet design doc
```

## Build

ROS 2 Humble (or newer), Ubuntu 22.04, GCC ≥ 11.

```bash
mkdir -p ~/scovox_ws/src && cd ~/scovox_ws/src
git clone https://github.com/<owner>/scovox.git .
cd ..
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Release builds are mandatory — debug builds inflate timing numbers
by ~3–4× and bias any wall-clock measurement.

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

* `scovox_mapping_node` — **in:** LiDAR `PointCloud2` *or* depth + camera_info + seg;
  **out:** `~/pointcloud`, `~/tsdf_pointcloud`, `~/scovox`, `~/planning_map`, `~/scovox_bin` (rolling).
* `dscovox_mapping_node` — **in:** each robot's `~/scovox_bin` (`input_topics`);
  **out:** fused `~/pointcloud`, `~/scovox`, `~/planning_map`, `~/frontier_centroids`.

Full parameter reference: [`config/default_params.yaml`](src/scovox_mapping/config/default_params.yaml).

## Citation

If you use this code, please cite the paper (BibTeX block to be added
once the camera-ready DOI is assigned).

## License

Apache-2.0.  See [LICENSE](LICENSE).
