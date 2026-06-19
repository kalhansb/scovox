"""SCovox mapping node configured for flatforest simulation evaluation.

8 semantic classes: ground, deciduous_tree, conifer, rock, undergrowth,
structure, wall, human.  RGB-D camera (320x240, 60 deg HFOV, 8m max range).

Usage:
  ros2 launch scovox_mapping flatforest_eval.launch.py
  ros2 launch scovox_mapping flatforest_eval.launch.py resolution:=0.10
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
    w_occ = float(context.launch_configurations["w_occ"])
    w_free = float(context.launch_configurations["w_free"])

    scovox_node = Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot_name,
        name="scovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "dataset_mode": True,

            # Map model
            "resolution": resolution,
            "w_free": w_free,
            "w_occ": w_occ,

            # Semantics — 8 classes for flatforest
            "kappa0": 2.0,
            "semantic_top_k": 4,
            "semantic_occ_gate": 0.6,
            "semantic_mode": "dirichlet",
            "max_semantic_classes": 8,

            # Evidence

            # Range — RGB-D camera, max 8m
            "range_decay_length": -1.0,
            "min_range": 0.1,
            "max_range": 8.0,

            # Angle weighting
            "grazing_angle_threshold": -1.0,

            # No dynamic classes
            "transient_decay_rate": 0.0,

            # Frames
            "base_frame": f"{robot_name}/base_link",
            "integration_frame": f"{robot_name}/odom",
            "map_frame": "map",

            # Topics
            "depth_topic": "rgbd_camera_depth_image",
            "depth_info_topic": "rgbd_camera_info",
            "seg_topic": "segmentation/colored",

            # Depth processing
            "stride": int(stride),
            "min_depth": 0.1,
            "max_depth": 8.0,
            "trace_no_return_rays": False,

            # Persistent mode — no submaps for offline eval
            "mode": "persistent",
            "submap_max_distance": 999.0,
            "robot_id": robot_name,

            # Output
            "publish_pointcloud": True,
            "pointcloud_topic": "~/pointcloud",
            "scovox_topic": "~/scovox",
            "occupancy_vis_threshold": min_occ,
            "scovox_publish_rate": 1.0,
            "publish_planning_map": False,
            "planning_map_topic": "~/planning_map",
            "planning_map_resolution": 0.20,
            "planning_map_size_m": 100.0,
            "planning_map_origin_x": -50.0,
            "planning_map_origin_y": -50.0,
            "planning_map_min_z": -0.5,
            "planning_map_max_z": 5.0,
            "planning_map_inflation_m": 0.0,

            # Flatforest 8-class semantic color map
            # Packed keys: (R<<16 | G<<8 | B), must match replay node CATEGORY_COLORS
            "semantic_color_map_keys": [
                10018698,   # 1  ground       (152,223,138)
                2263842,    # 2  deciduous    ( 34,139, 34)
                25600,      # 3  conifer      (  0,100,  0)
                8421504,    # 4  rock         (128,128,128)
                7048739,    # 5  undergrowth  (107,142, 35)
                12357519,   # 6  structure    (188,143,143)
                11454440,   # 7  wall         (174,199,232)
                16711680,   # 8  human        (255,  0,  0)
                0,          # 0  unknown
            ],
            "semantic_color_map_classes": [
                1, 2, 3, 4, 5, 6, 7, 8, 0,
            ],
        }],
    )

    return [scovox_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_name", default_value="atlas"),
        DeclareLaunchArgument("resolution", default_value="0.10"),
        DeclareLaunchArgument("occupancy_vis_threshold", default_value="0.5"),
        DeclareLaunchArgument("stride", default_value="1"),
        DeclareLaunchArgument("w_occ", default_value="6.0"),
        DeclareLaunchArgument("w_free", default_value="1.0"),
        OpaqueFunction(function=_launch_setup),
    ])
