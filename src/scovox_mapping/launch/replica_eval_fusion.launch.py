"""SCovox two-robot fusion eval for Replica (E2.1).

Brings up:
  - scovox_node A (namespace=robotA, rolling mode → publishes scovox_bin)
  - scovox_node B (namespace=robotB, rolling mode → publishes scovox_bin)
  - dscovox_node consuming both binaries → fused pointcloud

submap_max_distance is set to 9999 m so rolling mode never actually splits;
the only behavioural difference vs persistent is that binaries are published
(persistent mode skips that path entirely — see scovox_node.cpp:217).

The replay node is launched separately by the orchestrator script with
distinct robot_name + start_frame for the two-robot trajectory split.

Usage:
  ros2 launch scovox_mapping replica_eval_fusion.launch.py topk_probs_dir:=/path/to/topk
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def _make_scovox(robot_name, resolution, topk_dir,
                 use_split=False, share_tsdf=False, fused_walker=True,
                 wire_format="v2", num_classes=14, dirichlet_prior=0.01):
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
            # Rolling mode enables ScovoxMapBinary publication; submap_max_distance
            # at 9999 m means no actual rolling happens for indoor Replica.
            "mode": "rolling",
            "submap_max_distance": 9999.0,

            "resolution": resolution,
            "w_free": 1.0,
            "w_occ": 2.0,

            "kappa0": 2.0,
            "semantic_top_k": 10,
            "semantic_occ_gate": 0.0,
            "carve_skip_occ_threshold": 0.7,
            "evidence_saturation": 1000,
            "dirichlet_min_p_occ": 0.5,
            "semantic_mode": "dirichlet",
            "max_semantic_classes": 19,
            "semantic_min_confidence": 0.1,

            "range_decay_length": -1.0,
            "min_range": 0.1,
            "max_range": 8.0,
            "grazing_angle_threshold": -1.0,
            "transient_decay_rate": 0.0,

            "base_frame": f"{robot_name}/base_link",
            "integration_frame": f"{robot_name}/odom",
            "map_frame": "map",

            "depth_topic": "rgbd_camera_depth_image",
            "depth_info_topic": "rgbd_camera_info",
            "seg_topic": "segmentation/colored",

            "stride": 1,
            "min_depth": 0.1,
            "max_depth": 8.0,
            "trace_no_return_rays": False,
            "carve_band": 0.1,
            "sdf_trunc_voxels": 3,
            "tsdf_space_carving": False,
            "band_only_integration": False,

            "robot_id": robot_name,

            "publish_pointcloud": True,
            "pointcloud_topic": "~/pointcloud",
            "scovox_topic": "~/scovox",
            "occupancy_vis_threshold": 0.5,
            "scovox_publish_rate": 1.0,
            "publish_planning_map": False,

            "semantic_color_map_keys": [
                (174 << 16) | (199 << 8) | 232,
                (152 << 16) | (223 << 8) | 138,
                (255 << 16) | (152 << 8) | 150,
                (197 << 16) | (176 << 8) | 213,
                (196 << 16) | (156 << 8) | 148,
                (255 << 16) | (127 << 8) | 14,
                (140 << 16) | (86 << 8) | 75,
                (227 << 16) | (119 << 8) | 194,
                (148 << 16) | (103 << 8) | 189,
                (255 << 16) | (187 << 8) | 120,
                (219 << 16) | (219 << 8) | 141,
                (199 << 16) | (199 << 8) | 199,
                (188 << 16) | (189 << 8) | 34,
                (158 << 16) | (218 << 8) | 229,
                (23 << 16) | (190 << 8) | 207,
                (44 << 16) | (160 << 8) | 44,
                (214 << 16) | (39 << 8) | 40,
                (127 << 16) | (127 << 8) | 212,
                0,
            ],
            "semantic_color_map_classes": [
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 0,
            ],
            "topk_probs_dir": topk_dir,
            # Step 9 / D10 — fusion smoke flag-through. use_split propagates
            # to both robot scovox_nodes AND the dscovox parent (constructed
            # below). share_tsdf=false keeps the wire SemBeta-only.
            "use_split": use_split,
            "share_tsdf": share_tsdf,
            # Step 12.10 — fused single-DDA ray walker (production default
            # since 2026-05-09). Override per-launch via fused_walker:=false
            # for legacy two-DDA A/B comparisons.
            "fused_walker": fused_walker,
            # Step 8 — v3 wire format opt-in. v3 = SemDir-native frames;
            # v2 = SemBeta-projected (current production). num_classes /
            # dirichlet_prior are SemDir priors used by the integration
            # substrate AND embedded in the v3 frame header. The two
            # robot scovox_nodes MUST share these or the dscovox receiver
            # will reject the second robot's frames with a prior-mismatch
            # warning. Replica defaults: C=14 (NYU13) / α_0=0.01.
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
    topk_dir = context.launch_configurations.get("topk_probs_dir", "")
    use_split = context.launch_configurations.get("use_split", "false").lower() in ("true", "1", "yes")
    share_tsdf = context.launch_configurations.get("share_tsdf", "false").lower() in ("true", "1", "yes")
    fused_walker = context.launch_configurations.get("fused_walker", "true").lower() in ("true", "1", "yes")
    wire_format = context.launch_configurations.get("wire_format", "v2")
    num_classes = int(context.launch_configurations.get("num_classes", "14"))
    dirichlet_prior = float(context.launch_configurations.get("dirichlet_prior", "0.01"))

    scovox_a = _make_scovox(robot_a, resolution, topk_dir, use_split, share_tsdf, fused_walker,
                            wire_format, num_classes, dirichlet_prior)
    scovox_b = _make_scovox(robot_b, resolution, topk_dir, use_split, share_tsdf, fused_walker,
                            wire_format, num_classes, dirichlet_prior)

    # Identity TFs map -> <robot>/odom so dscovox can lookup source->map.
    # Replica poses are camera-to-world in the mesh native frame and both robots
    # share that world, so the per-robot odom frames coincide with the global
    # map frame. Without these, dscovox rejects every binary because the
    # source->map TF lookup fails.
    static_a = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"static_map_to_{robot_a}_odom",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "map", f"{robot_a}/odom"],
    )
    static_b = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"static_map_to_{robot_b}_odom",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "map", f"{robot_b}/odom"],
    )

    dscovox_node = Node(
        package="scovox_mapping",
        executable="dscovox_mapping_node",
        name="dscovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "input_topics": [
                f"/{robot_a}/scovox_node/scovox_bin",
                f"/{robot_b}/scovox_node/scovox_bin",
            ],
            "pointcloud_topic": "~/pointcloud",
            "resolution": resolution,
            "publish_planning_map": False,
            # Step 9 / D9 — propagate use_split to dscovox parent. When
            # use_split=true, dscovox_node accepts only v2 envelopes
            # (msg->version==2) and routes through onBinaryMapV2 +
            # publishPointCloudV2. share_tsdf is sender-side only;
            # receiver tolerates both modes via tsdf_count=0 elision.
            "use_split": use_split,
            "share_tsdf": share_tsdf,
            # Step 8 — receiver-side wire-format selector. MUST match the
            # two robot scovox_nodes' wire_format. dscovox onBinaryMapV3
            # also pins (num_classes, alpha_0) on the first frame so the
            # priors don't drift between the integration-time and
            # consensus-time Dirichlets.
            "wire_format": wire_format,
        }],
    )

    return [static_a, static_b, scovox_a, scovox_b, dscovox_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_a", default_value="robotA"),
        DeclareLaunchArgument("robot_b", default_value="robotB"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("topk_probs_dir", default_value=""),
        DeclareLaunchArgument("use_split", default_value="false",
                              description="Step 9 / D10: propagate split-grid v2 mode to all "
                                          "three nodes (both robot scovox_nodes + dscovox parent). "
                                          "Step-11 fusion smoke launches with use_split:=true."),
        DeclareLaunchArgument("share_tsdf", default_value="false",
                              description="Step 9: when use_split=true, controls TSDF wire "
                                          "stream (false: SemBeta-only 37 B/voxel; true: dual 57 B/voxel)."),
        DeclareLaunchArgument("wire_format", default_value="v2",
                              description="Step 8: inter-robot wire format. v2 = SemBeta-"
                                          "projected (current production); v3 = SemDir-native "
                                          "(20 B/voxel + header priors). All 3 nodes (robot A, "
                                          "robot B, dscovox) MUST use the same format."),
        DeclareLaunchArgument("num_classes", default_value="14",
                              description="Step 8 (use_split=true only): dataset class count C. "
                                          "Replica default 14 (NYU13). Both robots and dscovox "
                                          "must pin the same value."),
        DeclareLaunchArgument("dirichlet_prior", default_value="0.01",
                              description="Step 8 (use_split=true only): symmetric Dirichlet "
                                          "prior α_0. Ship default 0.01 matches defaultSemDirVoxel."),
        DeclareLaunchArgument("fused_walker", default_value="true",
                              description="Step 12.10 (2026-05-09, production default): when "
                                          "use_split=true, hit rays go through a single Bresenham "
                                          "DDA over the union of the TSDF band and SemBeta carve "
                                          "range with per-voxel dispatch into both grids. Default "
                                          "true (iter6 production); set false for legacy two-DDA "
                                          "A/B parity."),
        OpaqueFunction(function=_launch_setup),
    ])
