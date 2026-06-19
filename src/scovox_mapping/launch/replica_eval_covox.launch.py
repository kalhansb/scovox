"""CoVox: Beta occupancy only, no semantics on Replica.

Ablation variant: tests whether semantics affect occupancy quality.
Same SCovox node but max_semantic_classes=0 disables all semantic processing.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_setup(context):
    from launch_ros.actions import Node

    robot_name = context.launch_configurations["robot_name"]
    resolution = float(context.launch_configurations["resolution"])
    min_occ = float(context.launch_configurations["occupancy_vis_threshold"])
    stride = int(context.launch_configurations["stride"])

    node = Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot_name,
        name="scovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "dataset_mode": True,
            "resolution": resolution,
            "w_free": 1.0, "w_occ": 2.0,
            # No semantics
            "max_semantic_classes": 0,
            "semantic_mode": "dirichlet",
            "range_decay_length": -1.0, "min_range": 0.1, "max_range": 8.0,
            "grazing_angle_threshold": -1.0,
            "transient_decay_rate": 0.0,
            "base_frame": f"{robot_name}/base_link",
            "integration_frame": f"{robot_name}/odom",
            "map_frame": "map",
            "depth_topic": "rgbd_camera_depth_image",
            "depth_info_topic": "rgbd_camera_info",
            "seg_topic": "segmentation/colored",
            "stride": int(stride), "min_depth": 0.1, "max_depth": 8.0,
            "trace_no_return_rays": False,
            "mode": "persistent", "submap_max_distance": 999.0, "robot_id": robot_name,
            "publish_pointcloud": True, "pointcloud_topic": "~/pointcloud",
            "scovox_topic": "~/scovox", "occupancy_vis_threshold": min_occ,
            "scovox_publish_rate": 1.0, "publish_planning_map": False,
        }],
    )
    return [node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_name", default_value="atlas"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("occupancy_vis_threshold", default_value="0.5"),
        DeclareLaunchArgument("stride", default_value="1"),
        OpaqueFunction(function=_launch_setup),
    ])
