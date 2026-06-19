"""SCovox-MV: Beta occupancy + Majority Vote semantics on Replica.

Ablation variant: replaces Dirichlet with simple majority voting.
Same as replica_eval.launch.py but semantic_mode=majority_vote.
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
            "kappa0": 2.0, "semantic_top_k": 10,
            "semantic_occ_gate": 0.6,
            "semantic_mode": "majority_vote",
            "max_semantic_classes": 19,
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
            "semantic_color_map_keys": [
                11452392, 10018698, 16750742, 12956885, 12885140,
                16744206, 9196107, 14907330, 9726909, 16759672,
                14408589, 13092807, 12369186, 10410725, 1556175,
                2924588, 14034216, 8355796, 0,
            ],
            "semantic_color_map_classes": [
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 0,
            ],
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
