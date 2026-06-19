"""SCovox mapping node configured for SceneNet RGB-D evaluation.

Matches SLIM-VDB protocol: 5cm voxels, 14 NYUv2 classes, 320x240 RGB-D.

Usage:
  ros2 launch scovox_mapping scenenet_eval.launch.py
  ros2 launch scovox_mapping scenenet_eval.launch.py resolution:=0.10 semantic_mode:=majority_vote
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_setup(context):
    from launch_ros.actions import Node

    robot_name = context.launch_configurations["robot_name"]
    resolution = float(context.launch_configurations["resolution"])
    semantic_mode = context.launch_configurations["semantic_mode"]
    # Step 9 (D2/D7) — split-grid v2 toggles. Defaults preserve legacy v1.
    use_split_arg = context.launch_configurations.get("use_split", "false").lower() in ("true", "1", "yes")
    share_tsdf_arg = context.launch_configurations.get("share_tsdf", "false").lower() in ("true", "1", "yes")
    # iter6 single-DDA fused ray walker (default true to match production).
    fused_walker_arg = context.launch_configurations.get("fused_walker", "true").lower() in ("true", "1", "yes")
    # Publish-time occupancy gate — used by Phase 2.5 to vary the labelling
    # envelope. Was hardcoded 0.5 pre-2026-05-14; now plumbed through
    # context so phase2_5_gate_threshold_sweep.sh can integrate at 0.0
    # and see the unfiltered grid.
    occ_vis_arg = float(context.launch_configurations.get("occupancy_vis_threshold", "0.5"))
    # Phase 2.5-v2 — integration-time admission gate (distinct from
    # the publish-time occupancy_vis_threshold above).
    dirichlet_min_p_occ_arg = float(context.launch_configurations.get("dirichlet_min_p_occ", "0.5"))

    scovox_node = Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot_name,
        name="scovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "dataset_mode": True,

            # Map model — match SLIM-VDB: 5cm voxels
            "resolution": resolution,
            "w_free": 1.0,
            "w_occ": 6.0,

            # Semantics — 14 SceneNet NYUv2 classes
            "kappa0": 2.0,
            "semantic_top_k": 10,
            "semantic_occ_gate": 0.6,
            "semantic_mode": semantic_mode,
            "max_semantic_classes": 14,

            # Evidence

            # Range — SceneNet indoor, match SLIM-VDB max_range=10
            "range_decay_length": -1.0,
            "min_range": 0.1,
            "max_range": 10.0,

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
            "stride": 1,
            "min_depth": 0.1,
            "max_depth": 10.0,
            "trace_no_return_rays": False,

            # Persistent mode — no submaps for offline eval
            "mode": "persistent",
            "submap_max_distance": 999.0,
            "robot_id": robot_name,

            # Output
            "publish_pointcloud": True,
            "pointcloud_topic": "~/pointcloud",
            "scovox_topic": "~/scovox",
            "occupancy_vis_threshold": occ_vis_arg,
            "dirichlet_min_p_occ": dirichlet_min_p_occ_arg,
            "scovox_publish_rate": 1.0,
            "publish_planning_map": False,

            # SceneNet 14-class semantic color map
            # Keys: (R<<16 | G<<8 | B), must match scenenet_replay_node.py
            "semantic_color_map_keys": [
                0,         # 0  Unknown    (0,0,0)
                255,       # 1  Bed        (0,0,255)
                15292720,  # 2  Books      (233,89,48)
                55808,     # 3  Ceiling    (0,218,0)
                9765104,   # 4  Chair      (149,0,240)
                14610712,  # 5  Floor      (222,241,24)
                16764622,  # 6  Furniture  (255,206,206)
                57573,     # 7  Objects    (0,224,229)
                6981836,   # 8  Picture    (106,136,204)
                7675177,   # 9  Sofa       (117,29,41)
                15737835,  # 10 Table      (240,35,235)
                42908,     # 11 TV         (0,167,156)
                16354048,  # 12 Wall       (249,139,0)
                14804418,  # 13 Window     (225,229,194)
            ],
            "semantic_color_map_classes": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            ],

            # Split-grid v2 toggles (Step 9 / D2 / D7).
            "use_split": use_split_arg,
            "share_tsdf": share_tsdf_arg,
            # iter6 single-DDA fused walker (production default).
            "fused_walker": fused_walker_arg,
        }],
    )

    return [scovox_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_name", default_value="atlas"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("semantic_mode", default_value="dirichlet",
                              description="dirichlet | majority_vote | naive"),
        DeclareLaunchArgument("use_split", default_value="false",
                              description="Step 9: route integration through ScovoxMapSplit "
                                          "(TsdfMap + SemBetaMap) instead of the legacy fused scovox::Map."),
        DeclareLaunchArgument("share_tsdf", default_value="false",
                              description="Step 9: when use_split=true, controls whether v2 binary "
                                          "frames carry the TSDF stream (true: 57 B/voxel) or just "
                                          "SemBeta deltas (false: 37 B/voxel — production default)."),
        DeclareLaunchArgument("occupancy_vis_threshold", default_value="0.5",
                              description="Publish-time gate on p_occ; voxels with "
                                          "p_occ < threshold are filtered before the "
                                          "PointCloud2 emit. Phase 2.5 ablation sweeps this."),
        DeclareLaunchArgument("dirichlet_min_p_occ", default_value="0.5",
                              description="Phase 2.5-v2 ablation: integration-time admission "
                                          "gate inside SemDirMap::applyHitUpdate. Below this "
                                          "p_occ_post the per-class dirichletUpdate is skipped "
                                          "(mass routes to OTHER). 0.0 = label every hit."),
        DeclareLaunchArgument("fused_walker", default_value="true",
                              description="iter6 single-DDA fused ray walker (production default)."),
        OpaqueFunction(function=_launch_setup),
    ])
