# =======================================================================
# scovox_bin smoke test — minimal KNOWN-GOOD bring-up of the delta stream.
# =======================================================================
# One rolling-mode mapper + one merger, correctly namespaced so the topic path
# is exactly `/<robot>/scovox_node/scovox_bin` (what dscovox subscribes to).
# Use it as a reference when a mapper is NOT publishing scovox_bin: if the topic
# flows HERE but not in your setup, the difference is your config (almost always
# mode != rolling, or a namespace/node-name mismatch). See
# docs/scovox_bin_manual_bringup.md for the full step-by-step diagnosis.
#
# This launch provides ONLY the scovox side. YOU supply, in the same ROS graph
# (same ROS_DOMAIN_ID / container network):
#   * the sensor stream       — cloud on `cloud_topic` (+ IMU on `imu_topic`)
#   * the TF tree             — map -> odom -> base_link -> <base_frame>
#     (a localizer + the bag/robot statics). Without it every scan is dropped
#     and the map — hence the bin stream — stays empty (GATE 3).
#
# Examples (run inside the scovox container after `colcon build`):
#   ros2 launch scovox_mapping dscovox.launch.py
#   ros2 launch scovox_mapping dscovox.launch.py \
#       robot:=robot1 cloud_topic:=/ouster/points base_frame:=os_lidar
#   ros2 launch scovox_mapping dscovox.launch.py with_merger:=false
#
# Verify (another shell in the container):
#   ros2 topic list | grep scovox_bin
#   ros2 topic hz   /robot1/scovox_node/scovox_bin
# =======================================================================
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Deployment-specific knobs. Defaults reproduce the LiDAR-only distributed
# experiment (Ouster cloud + os_lidar origin, sim time for bag replay).
_ARGS = {
    "robot": ("robot1", "Namespace + node-name pin: topic becomes /<robot>/scovox_node/scovox_bin"),
    "cloud_topic": ("/ouster/points", "Input LiDAR PointCloud2 topic (EMPTY = RGB-D mode — no bin)"),
    "imu_topic": ("/imu/data", "IMU topic for deskew"),
    "base_frame": ("os_lidar", "LiDAR ray-origin frame (sensor origin)"),
    "integration_frame": ("map", "Frame the map is built in; == bin header.frame_id (unique per robot in a fleet)"),
    "map_frame": ("map", "Global/world frame"),
    "use_sim_time": ("true", "Use /clock (true for bag replay, false for a live robot)"),
    "deskew_mode": ("auto", "auto|on|off — 'off' silences the IMU-not-ready note"),
    "with_merger": ("true", "Also start a dscovox merger so the subscriber gate (GATE 2) opens"),
}


def _bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "on")


def launch_setup(context, *args, **kwargs):
    # Resolve every arg to a plain string once, then build nodes with native
    # Python types — far simpler than threading substitutions into list params.
    g = {k: LaunchConfiguration(k).perform(context) for k in _ARGS}
    robot = g["robot"]
    use_sim_time = _bool(g["use_sim_time"])
    bin_topic = f"/{robot}/scovox_node/scovox_bin"

    mapper = Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot,
        name="scovox_node",              # topic path depends on BOTH ns and name
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "mode": "rolling",           # GATE 1 — the one line that makes the publisher exist
            "fuse_lidar_rgbd": False,
            "input_pointcloud_topic": g["cloud_topic"],
            "imu_topic": g["imu_topic"],
            "base_frame": g["base_frame"],
            "integration_frame": g["integration_frame"],
            "map_frame": g["map_frame"],
            "deskew_mode": g["deskew_mode"],
            # LiDAR occupancy defaults (mirror config/scovox_lidar_geometric.yaml)
            "resolution": 0.10,
            "w_occ": 8.0, "w_free": 4.0, "carve_band": -1.0,
            "min_range": 1.0, "max_range": 20.0,
            "enable_tsdf": False,
            # Low-bandwidth share controls
            "share_change_gate": True,
            "share_rate_hz": 2.0,
            "share_roi_z_min": -0.5, "share_roi_z_max": 2.0,
            # Exact-stamp TF; never fall back to a stale Time(0) pose (GATE 3)
            "tf_require_exact": True,
            "tf_lookup_timeout_sec": 1.0,
        }],
    )

    # The merger opens the subscriber gate (GATE 2): with 0 subscribers the
    # mapper drains its touched voxels and emits nothing. It also verifies the
    # end-to-end wire path (a running merger proves QoS + K_TOP match).
    merger = Node(
        package="scovox_mapping",
        executable="dscovox_mapping_node",
        namespace=robot,
        name="dscovox_node",
        output="screen",
        condition=IfCondition(g["with_merger"]),
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topics": [bin_topic],
            "map_frame": g["map_frame"],
            "share_roi_z_min": -0.5, "share_roi_z_max": 2.0,  # keep in sync with the sender
            "pointcloud_min_interval_s": 0.5,
        }],
    )

    nodes = [mapper, merger]

    # If the map is built in a frame other than map_frame, bridge it with an
    # identity static TF so the merger can fold this source back into map_frame.
    if g["integration_frame"] != g["map_frame"]:
        nodes.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_integration_frame",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                "--frame-id", g["map_frame"], "--child-frame-id", g["integration_frame"],
            ],
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=d, description=desc)
         for k, (d, desc) in _ARGS.items()]
        + [OpaqueFunction(function=launch_setup)]
    )
