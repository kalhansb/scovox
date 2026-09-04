# Single-robot dscovox map — manual run (bag → localizer → scovox → RViz)

All commands run from `ws/src/`. Start order matters: localizer and scovox come
up first (they run `use_sim_time` and idle until `/clock`), the **bag plays last**.

```bash
cd ~/Projects/hmr_explo_ws/hmr_explo/ws/src
```

## 0. Containers up + rebuild scovox

Rebuild is mandatory after editing anything in `src/` — `ros2 launch` resolves
from `install/`, not `src/`.

```bash
docker compose -f hmr_localisation/compose.yaml up -d
docker compose -f scovox/compose.yaml up -d

docker compose -f scovox/compose.yaml exec scovox bash -lc '
  source /opt/ros/jazzy/setup.bash
  cd /scovox && colcon build --packages-select scovox_mapping
'
```

## 1. Localizer (EKF + NDT map→odom + imu_link→imu alias) — leave running

```bash
docker compose -f hmr_localisation/compose.yaml exec ros bash -lc '
  source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=/ws/config/fastdds_shm.xml
  ros2 launch /ws/launch/ekf_odom.launch.py use_sim_time:=true &
  ros2 launch lidar_localization_ros2 lidar_localization.launch.py \
    localization_param_dir:=/ws/config/gt_ouster_ndt_tree_fused.yaml \
    cloud_topic:=/ouster/points imu_topic:=/imu/data use_sim_time:=true \
    global_frame_id:=map odom_frame_id:=odom base_frame_id:=base_link \
    use_imu_preintegration:=true imu_preintegration_use_base_frame_transform:=true \
    publish_lidar_tf:=false publish_imu_tf:=false &
  ros2 run tf2_ros static_transform_publisher \
    --x 0.0 --y 0.0 --z 0.0 --qx 0.0 --qy 0.0 --qz 0.0 --qw 1.0 \
    --frame-id imu_link --child-frame-id imu &
  wait
'
```

Wait until NDT loads its map (log settles on `Activating end`) before step 3.

## 2. scovox single-robot (rolling mapper + merger) — leave running

```bash
docker compose -f scovox/compose.yaml exec scovox bash -lc '
  source /opt/ros/jazzy/setup.bash; source /scovox/install/setup.bash
  ros2 launch scovox_mapping dscovox_single_robot.launch.py \
    robot:=robot1 cloud_topic:=/ouster/points base_frame:=os_lidar use_sim_time:=true
'
```

Map-extent knobs (defaults are opened up: z-band off, `max_range 40`):

```bash
# widen the radial footprint
... dscovox_single_robot.launch.py ... max_range:=60.0
# put a vertical clip back
... dscovox_single_robot.launch.py ... share_roi_z_min:=-0.5 share_roi_z_max:=3.0
```

## 3. Play the bag LAST (starts /clock + streams data)

```bash
docker compose -f hmr_localisation/compose.yaml exec ros bash -lc '
  source /opt/ros/jazzy/setup.bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=/ws/config/fastdds_shm.xml
  ros2 bag play /ws/bags/2026_06_19_18_19_06__kalhan-map-test-2_ \
    --clock --rate 0.5 --topics /ouster/points /imu/data /tf /tf_static
'
```

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

Software-GL fallback (if RViz crashes on GL):

```bash
docker compose -f scovox/compose.yaml exec scovox bash -lc '
  source /opt/ros/jazzy/setup.bash; source /scovox/install/setup.bash
  export DISPLAY="${DISPLAY:-:1}"
  export LIBGL_ALWAYS_SOFTWARE=1
  unset __GLX_VENDOR_LIBRARY_NAME __NV_PRIME_RENDER_OFFLOAD
  rviz2 -d /scovox/src/scovox_mapping/config/scovox_bin_smoketest.rviz \
    --ros-args -p use_sim_time:=true
'
```

## Verify

```bash
docker compose -f scovox/compose.yaml exec scovox bash -lc '
  source /opt/ros/jazzy/setup.bash; source /scovox/install/setup.bash
  ros2 topic hz /robot1/dscovox_node/pointcloud
'
```

Success = the merger (step 2 console) reaches `dscovox_diag: sources=1 fused_voxels>0`
and `topic hz` ticks. `sources=0` / `DROPPING scan` = the `map→odom→base_link→os_lidar`
TF isn't resolving (localizer/bag side, GATE 3).
