# SCovox

Semantic Conjugate Voxel mapping with a unified per-voxel Dirichlet
over `{top-K classes, OTHER, FREE}` — semantic occupancy mapping with
calibrated uncertainty, multi-robot fusion, and edge-device runtime.

This repository accompanies the paper *"SCovox: Hit-Driven Semantic
Voxel Mapping with Unified-Dirichlet Fusion."*  It contains the
code and configurations used to produce every number in the paper's
results section and the Jetson Nano edge-feasibility experiment.

## Headline results

| Result | Value | Source |
|--------|-------|--------|
| K_TOP Pareto knee (per-voxel memory) | K=2 matches K=19 ±0.001 mIoU, 6.2× less memory | [STEP8 §2 / Phase 1](docs/STEP8_EXPERIMENT_RESULTS_2026_05_14.md) |
| SCovox vs SLIM-VDB, SceneNet h2h (n=13, sdf_trunc=0.10) | Δ mIoU **+0.092 ± 0.016**, 13/13 wins | [STEP8 §5 / Phase 4](docs/STEP8_EXPERIMENT_RESULTS_2026_05_14.md) |
| SCovox vs SLIM-VDB, KITTI h2h (n=5, PolarSeg soft) | Δ mIoU **+0.096 ± 0.018**, 5/5 wins | [STEP8 §5 / Phase 4](docs/STEP8_EXPERIMENT_RESULTS_2026_05_14.md) |
| Two-robot Dirichlet fusion (SceneNet, n=13) | Δ mIoU **+0.071**, F1@5cm +0.013, Chamfer −1.4 cm | [STEP8 §4 / Phase 3](docs/STEP8_EXPERIMENT_RESULTS_2026_05_14.md) |
| Jetson Nano 4 GB edge runtime | 0.42 Hz sustained, 1.22 Hz integrate-only | [jetbot_experiment_2026_05_21.md](docs/jetbot_experiment_2026_05_21.md) |

The gap-mechanism analysis (SceneNet sweep over SLIM-VDB's
`sdf_trunc`) and per-voxel memory ratios are in the same Step 8 doc.

## Layout

```
src/
  scovox_core/      Voxel types, Dirichlet semantics, ray casting (C++17, zero ROS deps)
  scovox_mapping/   ROS 2 nodes: scovox_mapping_node, dscovox_mapping_node
  scovox_msgs/      BinaryMap v3 wire format + service definitions

docs/
  STEP8_EXPERIMENT_RESULTS_2026_05_14.md   Full Step-8 results write-up
  jetbot_experiment_2026_05_21.md          Jetson Nano edge-feasibility report
  NEW_EXPERIMENT_PLAN.md                   Phase 0–7 protocol the results follow
  METHODS.md                               Math + per-voxel storage layout
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

> Paper-reproduction tooling (`scovox_eval/`) is not in this deployment tree; measured results are under [`docs/`](docs/).

## Determinism and run-to-run variance

* SCovox is bit-deterministic across x86_64 and aarch64 when no DDS
  messages are dropped (verified — 44,808 / 66,197 voxels exact
  match between desktop and Jetson on SceneNet 0_223).
* SceneNet two-robot fusion has higher run-to-run variance than the
  single-robot KITTI pipeline; published Phase 3 deltas average
  over 13 trajectories at 0.25 Hz replay (4 Hz introduces
  `message_filters` drop bias).

## Citation

If you use this code, please cite the paper (BibTeX block to be added
once the camera-ready DOI is assigned).

## License

Apache-2.0.  See [LICENSE](LICENSE).
