# Two-robot dscovox map merge — manual run (2 bags → 2 localizers → 2 mappers → merger → RViz)

The multi-robot sibling of [dscovox_single_robot_run.md](dscovox_single_robot_run.md).
Two independently-recorded robots (`bunker`, `curt`) each localize against the
**same** ground-truth map, each build a rolling scovox map in their **own** frame,
and a single `dscovox` merger fuses both bin streams into one co-registered map on
`/dscovox_mapping/pointcloud`.

Because both robots run NDT against the **same** `gt_map_us050.pcd`, they share one
global `map` frame — no robot-to-robot pose estimation. That shared frame is the
whole trick that makes the merge spatially correct.

## What these two bags are

| | **bunker** | **curt** |
|---|---|---|
| LiDAR topic / frame | `/hesai/points` / `hesai_lidar` | `/ouster/points` / `os_lidar` |
| LiDAR QoS in bag | reliable | **best_effort** (needs override) |
| IMU topic / frame | `/imu/data` / `imu` | `/curt/imu/data` / `imu_curt` |
| base frame | `base_link` | `base_link_curt` |
| extrinsics (base→sensor/imu) | in bag `/tf_static` (URDF) | in bag `/tf_static` |
| `/tf` in bag | empty | wheel joints only (no odom) |
| scovox integration frame | `bunker_map` | `curt_map` |
| NDT odom frame | `odom` | `odom_curt` |

The two robots' frame names are **disjoint**, so they coexist in one ROS graph
with no TF collisions. Play each bag with **`/tf_static` + sensors only** — do NOT
play `/tf` (curt's is wheels; we own the odom→base chain).

**Verified (2026-07-16):** both robots self-localize against `gt_map_us050` from
the config's default near-origin seed — aligned-scan-vs-gt_map fitness ≈ **0.23 m**
(bunker) and **0.39 m** median (curt). **No custom initial pose is needed.** With
both playing, the merger reaches `sources=2` with ~10 M fused voxels, and
`src_voxels > fused_voxels` (the two maps overlap = genuinely co-registered).
See `docs/img/coop_*_loc_vs_gtmap.png` and `docs/img/coop_merged_dscovox_over_gtmap.png`.

All commands run from `ws/src/`.

```bash
cd ~/Projects/hmr_explo_ws/hmr_explo/ws/src
```

## 0. Bags in place + containers up + rebuild scovox

Put both bag directories under `hmr_explo/bags/` (mounted `/ws/bags` in the
localizer container, `/scovox/bags` in scovox). The download of the two zips was
incomplete (missing `.mcap` segments); after replacing them, if `ros2 bag play`
complains about missing files listed in `metadata.yaml`, drop the bad entries by
reindexing the segments actually present:

```bash
# per bag, if segments are missing/renamed:
cd /ws/bags/<bag_dir> && mv metadata.yaml metadata.yaml.orig
ros2 bag reindex . -s mcap        # regenerates metadata from the .mcap files present
```

```bash
docker compose -f hmr_localisation/compose.yaml up -d
docker compose -f scovox/compose.yaml up -d

docker compose -f scovox/compose.yaml exec scovox bash -lc '
  source /opt/ros/jazzy/setup.bash
  cd /scovox && colcon build --packages-select scovox_mapping
'
```

## 1. Both localizers (NDT vs the shared gt_map) — leave running

One launch brings up both robots' NDT stacks, namespaced (`/bunker`, `/curt`),
each publishing `map → <robot>/odom` plus an identity `<robot>/odom → <base>`
('noekf' baseline — `map→odom` carries the pose). See
[coop_multi_robot_localization.launch.py](../../hmr_localisation/launch/coop_multi_robot_localization.launch.py).

```bash
docker compose -f hmr_localisation/compose.yaml exec ros bash -lc '
  source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=/ws/config/fastdds_shm.xml
  ros2 launch /ws/launch/coop_multi_robot_localization.launch.py use_sim_time:=true
'
```

Wait until **both** `bunker.lidar_localization` and `curt.lidar_localization` log
`Activating end` (each loads `gt_map_us050.pcd`, ~196 k pts, in well under a second).

## 2. scovox mappers + merger — leave running

Two rolling mappers (unique integration frames `bunker_map`/`curt_map`, each
identity-linked to `map`) feeding one merger that fuses both bin streams. See
[dscovox_multi_robot.launch.py](../src/scovox_mapping/launch/dscovox_multi_robot.launch.py).

```bash
docker compose -f scovox/compose.yaml exec scovox bash -lc '
  source /opt/ros/jazzy/setup.bash; source /scovox/install/setup.bash
  ros2 launch scovox_mapping dscovox_multi_robot.launch.py use_sim_time:=true
'
```

Map-extent knobs (same as single-robot; defaults opened up — z-band off,
`max_range 40`):

```bash
... dscovox_multi_robot.launch.py ... max_range:=60.0
... dscovox_multi_robot.launch.py ... share_roi_z_min:=-0.5 share_roi_z_max:=3.0
```

## 3. Play the bags LAST — SEQUENTIALLY (bunker, then curt)

The two bags were recorded ~76 min apart (16:53 vs 18:09), so they **cannot share
one sim `/clock` simultaneously**. Play them one after the other under the single
advancing clock — the merger persists each robot's contribution, so a "late" robot
just adds its region (the distributed design explicitly allows non-simultaneous
robots). **Always play forward in time** (bunker first, since it is the earlier
recording); replaying a bag rewinds `/clock` and freezes the sim-time nodes.

**bunker** (reliable cloud, no QoS override):

```bash
docker compose -f hmr_localisation/compose.yaml exec ros bash -lc '
  source /opt/ros/jazzy/setup.bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=/ws/config/fastdds_shm.xml
  ros2 bag play /ws/bags/2026_07_06_16_53_31__bunker_kalhan_coop_ \
    --clock --topics /hesai/points /imu/data /tf_static
'
```

**curt** (Ouster cloud is best_effort in the bag → override to reliable, else NDT
and the mapper never receive scans):

```bash
docker compose -f hmr_localisation/compose.yaml exec ros bash -lc '
  source /opt/ros/jazzy/setup.bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=/ws/config/fastdds_shm.xml
  ros2 bag play /ws/bags/2026_07_06_18_09_02__curt_kalhan_coop_ \
    --clock --qos-profile-overrides-path /ws/config/qos_reliable_ouster.yaml \
    --topics /ouster/points /curt/imu/data /tf_static
'
```

(Prefer simultaneous, cooperative-feel replay? Re-record one bag with its stamps
offset so both share a timeline, then play both — one with `--clock`, the other
without. Sequential is simpler and sufficient to build the merged map.)

## 4. RViz (inside the scovox container, hardware GL)

```bash
# on the HOST, once per login:
xhost +local:root
```

```bash
docker compose -f scovox/compose.yaml exec scovox bash -lc '
  source /opt/ros/jazzy/setup.bash; source /scovox/install/setup.bash
  export DISPLAY="${DISPLAY:-:1}"
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  rviz2 -d /scovox/src/scovox_mapping/config/scovox_bin_smoketest.rviz \
    --ros-args -p use_sim_time:=true
'
```

Add a PointCloud2 display on `/dscovox_mapping/pointcloud` (Fixed Frame = `map`)
to watch the fused two-robot map. Software-GL fallback: see the single-robot doc.

## Verify

```bash
# both robots localizing:
docker compose -f hmr_localisation/compose.yaml exec ros bash -lc '
  source /opt/ros/jazzy/setup.bash
  ros2 topic hz /bunker/pcl_pose /curt/pcl_pose'

# both bin streams + fused map:
docker compose -f scovox/compose.yaml exec scovox bash -lc '
  source /opt/ros/jazzy/setup.bash; source /scovox/install/setup.bash
  ros2 topic hz /bunker/scovox_node/scovox_bin /curt/scovox_node/scovox_bin
  ros2 topic hz /dscovox_mapping/pointcloud'
```

**Success** = the merger console (step 2, `output=screen`) reaches
`dscovox_diag: sources=2 ... fused_voxels>0` (source 1 after bunker plays,
source 2 after curt plays). `sources=0` / `DROPPING scan` = the
`map→<robot>/odom→<base>→<sensor>` TF isn't resolving — check the localizer is up
and, for curt, that the QoS override is in place (a best_effort `/ouster/points`
never reaches the reliable subscribers). See
[scovox_bin_manual_bringup.md](scovox_bin_manual_bringup.md) for the three-gate
diagnosis and [distributed_mapping_lidar.md](distributed_mapping_lidar.md) for the
real (per-machine) fleet topology this single-host replay stands in for.
