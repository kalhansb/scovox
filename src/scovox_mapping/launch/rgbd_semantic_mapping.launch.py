"""SCovox RGB-D semantic mapping — outdoor, fed by the seg_pipeline node.

Integrates a depth image + colored segmentation image into a Beta-occupancy +
Dirichlet-semantic voxel map, in the LiDAR `map` frame (so it fuses with the
LiDAR map). Inputs come from src/seg_pipeline's online seg node, which publishes
depth/seg/camera_info re-stamped to one timestamp and re-framed to a body frame.

Parameters: config/rgbd_semantic_mapping.yaml (palette/num_classes must match
src/seg_pipeline/outdoor_palette.py).

Prereqs running alongside this node (shared DDS graph, host net + ROS_DOMAIN_ID):
  * ros2 bag play <bag> --clock
  * a LiDAR localizer publishing  map -> ... -> base_link  (NOT map->os_lidar)
  * src/seg_pipeline online seg node  (./run_seg.sh)

Outputs (under /scovox_node):
  ~/pointcloud   coloured semantic-occupancy point cloud (in `map`)
  ~/scovox       full ScovoxMap (occupancy + uncertainty)

Usage:
  ros2 launch scovox_mapping rgbd_semantic_mapping.launch.py use_sim_time:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
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
                # Force RGB-D mode regardless of the YAML (empty = depth+seg path).
                "input_pointcloud_topic": "",
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("scovox_mapping"), "config", "rgbd_semantic_mapping.yaml",
            ]),
            description="YAML parameter file for scovox_node (RGB-D semantic).",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use /clock simulation time (true when replaying a bag).",
        ),
        scovox_node,
    ])
