"""SCovox mapping node configured for Replica dataset evaluation.

Usage:
  ros2 launch scovox_mapping replica_eval.launch.py
  ros2 launch scovox_mapping replica_eval.launch.py resolution:=0.10 occupancy_vis_threshold:=0.5
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
    sem_occ_gate = float(context.launch_configurations["semantic_occ_gate"])
    range_decay_length_arg = float(context.launch_configurations["range_decay_length"])
    kappa0_arg = float(context.launch_configurations["kappa0"])
    carve_skip_arg = float(context.launch_configurations["carve_skip_occ_threshold"])
    evsat_arg = int(float(context.launch_configurations["evidence_saturation"]))
    dir_min_p_occ_arg = float(context.launch_configurations["dirichlet_min_p_occ"])
    carve_band_arg = float(context.launch_configurations["carve_band"])
    sdf_trunc_voxels_arg = int(context.launch_configurations["sdf_trunc_voxels"])
    tsdf_space_carving_arg = context.launch_configurations["tsdf_space_carving"].lower() in ("true", "1", "yes")
    band_only_integration_arg = context.launch_configurations["band_only_integration"].lower() in ("true", "1", "yes")
    topk_probs_dir_arg = context.launch_configurations.get("topk_probs_dir", "")
    semantic_mode_arg = context.launch_configurations.get("semantic_mode", "dirichlet")
    eviction_stats_csv_arg = context.launch_configurations.get("eviction_stats_csv", "")
    # Step 9 (D2/D7) — split-grid v2 toggles. use_split=true switches the
    # node from legacy fused scovox::Map to ScovoxMapSplit (TsdfMap +
    # SemBetaMap). share_tsdf=false (the v2 default) emits SemBeta-only
    # frames on ~/scovox_bin (37 B/voxel); share_tsdf=true emits the dual
    # stream (+54%). Defaults preserve the legacy v1 path verbatim.
    use_split_arg = context.launch_configurations.get("use_split", "false").lower() in ("true", "1", "yes")
    share_tsdf_arg = context.launch_configurations.get("share_tsdf", "false").lower() in ("true", "1", "yes")
    # Step 12.10 (2026-05-09) — fused single-DDA ray walker.
    fused_walker_arg = context.launch_configurations.get("fused_walker", "true").lower() in ("true", "1", "yes")

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

            # Semantics — 19 classes for Replica
            "kappa0": kappa0_arg,
            "semantic_top_k": 10,
            "semantic_occ_gate": sem_occ_gate,
            "carve_skip_occ_threshold": carve_skip_arg,
            "evidence_saturation": evsat_arg,
            "dirichlet_min_p_occ": dir_min_p_occ_arg,
            "semantic_mode": semantic_mode_arg,
            "max_semantic_classes": 19,

            # Evidence

            # Range — Replica indoor scenes, max ~8m
            "range_decay_length": range_decay_length_arg,
            "min_range": 0.1,
            "max_range": 8.0,

            # Angle weighting
            "grazing_angle_threshold": -1.0,

            # No dynamic classes for Replica
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
            # carve_band: -1 = full-ray fused integration (correct TSDF band).
            # 0 = endpoint-only. >0 = partial-ray (legacy 0.1 default).
            "carve_band": carve_band_arg,

            # TSDF integration controls
            "sdf_trunc_voxels": sdf_trunc_voxels_arg,
            "tsdf_space_carving": tsdf_space_carving_arg,
            "band_only_integration": band_only_integration_arg,

            # Split-grid v2 toggles (Step 9 / D2 / D7).
            "use_split": use_split_arg,
            "share_tsdf": share_tsdf_arg,
            "fused_walker": fused_walker_arg,

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
            "planning_map_size_m": 40.0,
            "planning_map_origin_x": -20.0,
            "planning_map_origin_y": -20.0,
            "planning_map_min_z": -0.5,
            "planning_map_max_z": 3.0,
            "planning_map_inflation_m": 0.0,

            # Replica 18-class semantic color map.
            # Keys computed at load time as (R<<16 | G<<8 | B) to guarantee
            # they match the replay-node's CATEGORY_COLORS exactly (prior
            # hand-entered values were wrong for wall/door/table/bed/rug,
            # which mapped those classes to "unknown" at integration time).
            "semantic_color_map_keys": [
                (174<<16) | (199<<8) | 232,  # 1  wall
                (152<<16) | (223<<8) | 138,  # 2  floor
                (255<<16) | (152<<8) | 150,  # 3  ceiling
                (197<<16) | (176<<8) | 213,  # 4  door
                (196<<16) | (156<<8) | 148,  # 5  window
                (255<<16) | (127<<8) |  14,  # 6  chair
                (140<<16) | ( 86<<8) |  75,  # 7  table
                (227<<16) | (119<<8) | 194,  # 8  sofa
                (148<<16) | (103<<8) | 189,  # 9  bed
                (255<<16) | (187<<8) | 120,  # 10 cushion
                (219<<16) | (219<<8) | 141,  # 11 lamp
                (199<<16) | (199<<8) | 199,  # 12 cabinet
                (188<<16) | (189<<8) |  34,  # 13 blinds
                (158<<16) | (218<<8) | 229,  # 14 book
                ( 23<<16) | (190<<8) | 207,  # 15 picture
                ( 44<<16) | (160<<8) |  44,  # 16 plant
                (214<<16) | ( 39<<8) |  40,  # 17 rug
                (127<<16) | (127<<8) | 212,  # 18 pillar
                0,                           # 0  unknown
            ],
            "semantic_color_map_classes": [
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 0,
            ],
            # Soft-prob ablation: when set, scovox_node ignores the
            # color-keyed RGB segmentation and reads per-pixel top-K probs
            # from "<dir>/<frame:06d>.topk" (uint16 H,W,K + uint8 ids/probs).
            "topk_probs_dir": topk_probs_dir_arg,
            "eviction_stats_csv": eviction_stats_csv_arg,
        }],
    )

    return [scovox_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_name", default_value="atlas"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("occupancy_vis_threshold", default_value="0.5"),
        DeclareLaunchArgument("stride", default_value="1"),
        DeclareLaunchArgument("w_occ", default_value="6.0"),
        DeclareLaunchArgument("w_free", default_value="1.0"),
        DeclareLaunchArgument("semantic_occ_gate", default_value="0.6",
                              description="Minimum gate value to admit semantic update; set 0.0 for SCovox-NG ablation."),
        DeclareLaunchArgument("range_decay_length", default_value="-1.0",
                              description="Range-decay length in m; -1 disables. Replica default off."),
        DeclareLaunchArgument("kappa0", default_value="2.0",
                              description="Base Dirichlet weight scalar."),
        DeclareLaunchArgument("carve_skip_occ_threshold", default_value="0.7",
                              description="Stop free-carve at voxels with p_occ above this. 0.7 is the OctoMap-style default."),
        DeclareLaunchArgument("evidence_saturation", default_value="1000",
                              description="Cap on (a_occ, a_free, sem_cnt). 0 disables. Default 1000 reproduces OLD-pipeline mIoU."),
        DeclareLaunchArgument("dirichlet_min_p_occ", default_value="0.5",
                              description="Skip Dirichlet update when p_occ below this. 0 disables. Default 0.5 reproduces OLD-pipeline mIoU."),
        DeclareLaunchArgument("carve_band", default_value="0.1",
                              description="-1=full-ray TSDF, 0=endpoint-only, >0=partial-ray."),
        DeclareLaunchArgument("sdf_trunc_voxels", default_value="3"),
        DeclareLaunchArgument("tsdf_space_carving", default_value="false"),
        DeclareLaunchArgument("band_only_integration", default_value="false",
                              description="True confines fused walk to [hit-trunc, hit+trunc] (skips full-ray Beta-free)."),
        DeclareLaunchArgument("topk_probs_dir", default_value="",
                              description="Soft-prob ablation: directory of <frame>.topk flat-binary blobs."),
        DeclareLaunchArgument("semantic_mode", default_value="dirichlet",
                              description="Component ablation: dirichlet | majority_vote | naive."),
        DeclareLaunchArgument("eviction_stats_csv", default_value="",
                              description="E5.2 — append per-frame sparse_add branch counters to this CSV."),
        DeclareLaunchArgument("use_split", default_value="false",
                              description="Step 9: route integration through ScovoxMapSplit "
                                          "(TsdfMap + SemBetaMap) instead of the legacy fused scovox::Map. "
                                          "Default false preserves the v1 wire-format path."),
        DeclareLaunchArgument("share_tsdf", default_value="false",
                              description="Step 9: when use_split=true, controls whether v2 binary "
                                          "frames carry the TSDF stream (true: 57 B/voxel) or just "
                                          "SemBeta deltas (false: 37 B/voxel — production default; "
                                          "each robot keeps its own local TSDF for mesh extraction)."),
        DeclareLaunchArgument("fused_walker", default_value="true",
                              description="Step 12.10 (2026-05-09): when use_split=true, hit rays "
                                          "go through a single Bresenham DDA over the union of the "
                                          "TSDF band and SemBeta carve range with per-voxel dispatch "
                                          "into both grids. Default true (production); set false to "
                                          "fall back to the legacy two-DDA split path for A/B parity."),
        OpaqueFunction(function=_launch_setup),
    ])
