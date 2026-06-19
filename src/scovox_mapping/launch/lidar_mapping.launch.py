"""SCovox LiDAR mapping — geometry / occupancy only (no semantics).

Integrates a LiDAR PointCloud2 stream into a Beta-occupancy voxel map (TSDF
off by default; see below). The input cloud carries no `semantic_label`
field, so scovox_node runs purely geometrically — no Dirichlet / top-K
semantic fusion.

Parameters are loaded from config/lidar_mapping.yaml; a couple of common
ones are exposed as launch arguments.

TSDF surface integration is off by default here — pure occupancy mapping.
It is not a launch argument; edit `enable_tsdf: true` in config/lidar_mapping.yaml
to also build the dense fused surface and publish ~/tsdf_pointcloud.

Outputs (under /scovox_node):
  ~/pointcloud        coloured occupancy point cloud
  ~/scovox            full ScovoxMap (occupancy + uncertainty)
  ~/tsdf_pointcloud   TSDF surface point cloud (only when enable_tsdf: true)
  ~/planning_map      2-D occupancy grid for navigation (off by default)

Usage:
  ros2 launch scovox_mapping lidar_mapping.launch.py
  ros2 launch scovox_mapping lidar_mapping.launch.py pointcloud_topic:=/lidar/points
  ros2 launch scovox_mapping lidar_mapping.launch.py use_sim_time:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    use_sim_time = LaunchConfiguration("use_sim_time")

    scovox_node = Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        name="scovox_node",
        output="screen",
        parameters=[
            params_file,
            {
                # Launch-arg overrides take precedence over the YAML file.
                "use_sim_time": use_sim_time,
                "input_pointcloud_topic": pointcloud_topic,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("scovox_mapping"), "config", "lidar_mapping.yaml",
            ]),
            description="YAML parameter file for scovox_node.",
        ),
        DeclareLaunchArgument(
            "pointcloud_topic",
            default_value="/velodyne_points",
            description="Input LiDAR PointCloud2 topic (overrides the config value).",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use /clock simulation time (set true when replaying a bag with sim time).",
        ),
        scovox_node,
    ])
