"""SCovox two-robot fusion eval for SceneNet RGB-D (Step 8 follow-up,
NEW_EXPERIMENT_PLAN.md Phase 3).

Brings up:
  - scovox_node A (namespace=robot_a, rolling mode → publishes scovox_bin)
  - scovox_node B (namespace=robot_b, rolling mode → publishes scovox_bin)
  - dscovox_node consuming both binaries → fused pointcloud

Mirrors replica_eval_fusion.launch.py with SceneNet-specific knobs
(14-class NYUv2 colour map, 5cm voxels, max_range=10m, w_occ=6.0).
The replay node (scenenet_replay_node.py) is launched separately by
the orchestrator with distinct robot_name + start_frame for the
trajectory split (typically A=[0..200), B=[100..300) for the 50%-
overlap split per NEW_EXPERIMENT_PLAN.md).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction


# Single source of truth for the SceneNet 14-class NYUv2 colour LUT.
# Keys are (R<<16 | G<<8 | B); must match scenenet_replay_node.py emissions
# AND the scenenet_eval.launch.py colour map (the single-robot path).
_NYU14_COLOR_KEYS = [
    0, 255, 15292720, 55808, 9765104, 14610712, 16764622,
    57573, 6981836, 7675177, 15737835, 42908, 16354048, 14804418,
]
_NYU14_COLOR_CLASSES = list(range(14))


def _make_scovox(robot_name, resolution,
                 use_split=False, share_tsdf=False, fused_walker=True,
                 wire_format="v2", num_classes=14, dirichlet_prior=0.01,
                 semantic_mode="dirichlet"):
    from launch_ros.actions import Node
    return Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot_name,
        name="scovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "dataset_mode": True,
            "mode": "rolling",
            "submap_max_distance": 9999.0,
            "resolution": resolution,
            "w_free": 1.0, "w_occ": 6.0,
            "kappa0": 2.0,
            "semantic_top_k": 10,
            "semantic_occ_gate": 0.6,
            "semantic_mode": semantic_mode,
            "max_semantic_classes": 14,
            "range_decay_length": -1.0,
            "min_range": 0.1, "max_range": 10.0,
            "grazing_angle_threshold": -1.0,
            "transient_decay_rate": 0.0,
            "base_frame": f"{robot_name}/base_link",
            "integration_frame": f"{robot_name}/odom",
            "map_frame": "map",
            "depth_topic": "rgbd_camera_depth_image",
            "depth_info_topic": "rgbd_camera_info",
            "seg_topic": "segmentation/colored",
            "stride": 1, "min_depth": 0.1, "max_depth": 10.0,
            "trace_no_return_rays": False,
            "robot_id": robot_name,
            "publish_pointcloud": True,
            "pointcloud_topic": "~/pointcloud",
            "scovox_topic": "~/scovox",
            "occupancy_vis_threshold": 0.5,
            "scovox_publish_rate": 1.0,
            "publish_planning_map": False,
            "semantic_color_map_keys":    _NYU14_COLOR_KEYS,
            "semantic_color_map_classes": _NYU14_COLOR_CLASSES,
            # Split-grid + v3 wire toggles. Both robots MUST share
            # (num_classes, dirichlet_prior) — dscovox pins from the
            # first frame and rejects mismatched followups.
            "use_split": use_split,
            "share_tsdf": share_tsdf,
            "fused_walker": fused_walker,
            "wire_format": wire_format,
            "num_classes": num_classes,
            "dirichlet_prior": dirichlet_prior,
        }],
    )


def _launch_setup(context):
    from launch_ros.actions import Node
    robot_a = context.launch_configurations["robot_a"]
    robot_b = context.launch_configurations["robot_b"]
    resolution = float(context.launch_configurations["resolution"])
    use_split = context.launch_configurations.get("use_split", "false").lower() in ("true", "1", "yes")
    share_tsdf = context.launch_configurations.get("share_tsdf", "false").lower() in ("true", "1", "yes")
    fused_walker = context.launch_configurations.get("fused_walker", "true").lower() in ("true", "1", "yes")
    wire_format = context.launch_configurations.get("wire_format", "v2")
    num_classes = int(context.launch_configurations.get("num_classes", "14"))
    dirichlet_prior = float(context.launch_configurations.get("dirichlet_prior", "0.01"))
    semantic_mode = context.launch_configurations.get("semantic_mode", "dirichlet")

    scovox_a = _make_scovox(robot_a, resolution, use_split, share_tsdf, fused_walker,
                            wire_format, num_classes, dirichlet_prior, semantic_mode)
    scovox_b = _make_scovox(robot_b, resolution, use_split, share_tsdf, fused_walker,
                            wire_format, num_classes, dirichlet_prior, semantic_mode)

    # Identity static TFs map → <robot>/odom — same shape as Replica fusion.
    static_a = Node(package="tf2_ros", executable="static_transform_publisher",
        name=f"static_map_to_{robot_a}_odom", output="log",
        arguments=["0", "0", "0", "0", "0", "0", "map", f"{robot_a}/odom"])
    static_b = Node(package="tf2_ros", executable="static_transform_publisher",
        name=f"static_map_to_{robot_b}_odom", output="log",
        arguments=["0", "0", "0", "0", "0", "0", "map", f"{robot_b}/odom"])

    dscovox_node = Node(
        package="scovox_mapping", executable="dscovox_mapping_node",
        name="dscovox_node", output="screen",
        parameters=[{
            "use_sim_time": False,
            "input_topics": [
                f"/{robot_a}/scovox_node/scovox_bin",
                f"/{robot_b}/scovox_node/scovox_bin",
            ],
            "pointcloud_topic": "~/pointcloud",
            "resolution": resolution,
            "publish_planning_map": False,
            "use_split": use_split,
            "share_tsdf": share_tsdf,
            # MUST match the two robot scovox_nodes' wire_format. dscovox
            # onBinaryMapV3 pins (num_classes, alpha_0) on the first frame.
            "wire_format": wire_format,
        }],
    )
    return [static_a, static_b, scovox_a, scovox_b, dscovox_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_a", default_value="robotA"),
        DeclareLaunchArgument("robot_b", default_value="robotB"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("semantic_mode", default_value="dirichlet"),
        DeclareLaunchArgument("use_split", default_value="false"),
        DeclareLaunchArgument("share_tsdf", default_value="false"),
        DeclareLaunchArgument("fused_walker", default_value="true"),
        DeclareLaunchArgument("wire_format", default_value="v2",
            description="Step 8: v2=SemBeta-projected, v3=SemDir-native. All 3 nodes share."),
        DeclareLaunchArgument("num_classes", default_value="14",
            description="Step 8: SceneNet NYU14. Both robots and dscovox pin the same value."),
        DeclareLaunchArgument("dirichlet_prior", default_value="0.01"),
        OpaqueFunction(function=_launch_setup),
    ])
