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
    # Step 9 (D2/D7) — split-grid v2 toggles. The split substrate itself is
    # always active in the node; the old use_split selector was removed as
    # dead (never declared by the node).
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
    # A9 inverse-sensor-model weights. Were hardcoded 6.0/1.0 below; plumbed
    # through context (as semantickitti_eval.launch.py already does) so the A9
    # sweep can vary them per cell. Defaults reproduce the previous constants
    # exactly, so every run that does not pass them is bit-identical to before.
    w_occ_arg = float(context.launch_configurations.get("w_occ", "6.0"))
    w_free_arg = float(context.launch_configurations.get("w_free", "1.0"))
    # Soft-prob input: per-frame image .topk blobs (H,W,C) from a 2D segmenter.
    # When non-empty, scovox_node reads per-pixel class distributions instead of
    # the one-hot seg colour. Frame index = depth-stamp low 16 bits.
    topk_probs_dir_arg = context.launch_configurations.get("topk_probs_dir", "")
    # E1 uncertainty capture (mirrors semantickitti_eval.launch.py): rolling mode
    # enables the ScovoxMapBinary publisher (bin_pub_ exists only when
    # mode==rolling); share_rate_hz>0 gives a timer-owned binary publish so a
    # snapshot fires when a capture subscriber connects AFTER replay (with no
    # subscriber the timer is a cheap no-op, so replay recv stays 300/300).
    # scovox_publish_rate is overridable so the E1 runner can slow the pointcloud
    # republish timer down for offline capture. Defaults preserve the paper runs.
    map_mode_arg = context.launch_configurations.get("map_mode", "persistent")
    share_rate_hz_arg = float(context.launch_configurations.get("share_rate_hz", "0.0"))
    scovox_publish_rate_arg = float(context.launch_configurations.get("scovox_publish_rate", "1.0"))
    # E4 measured-memory hook: emits the periodic [memSplit] (+ [memGate]) grid
    # MB lines. Diagnostic-only walk of the grid, off unless a run asks for it.
    log_mem_usage_arg = context.launch_configurations.get(
        "log_mem_usage", "false").lower() in ("true", "1", "yes")

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
            "w_free": w_free_arg,
            "w_occ": w_occ_arg,

            # Semantics — 14 SceneNet NYUv2 classes
            "kappa0": 2.0,
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

            # Mode: persistent (paper offline eval) or rolling (E1 binary capture).
            "mode": map_mode_arg,
            "share_rate_hz": share_rate_hz_arg,
            "log_mem_usage": log_mem_usage_arg,
            "robot_id": robot_name,

            # Output
            "publish_pointcloud": True,
            "pointcloud_topic": "~/pointcloud",
            "scovox_topic": "~/scovox",
            "occupancy_vis_threshold": occ_vis_arg,
            "dirichlet_min_p_occ": dirichlet_min_p_occ_arg,
            "scovox_publish_rate": scovox_publish_rate_arg,
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
            "share_tsdf": share_tsdf_arg,
            # iter6 single-DDA fused walker (production default).
            "fused_walker": fused_walker_arg,
            # Soft-prob per-pixel input (empty = one-hot seg colour path).
            "topk_probs_dir": topk_probs_dir_arg,
            # SceneNN mechanism round. Defaults reproduce the shipped path
            # exactly; semantic_evict_by_confidence additionally needs a
            # -DSCOVOX_TRACK_QMAX=1 build or the node warns and ignores it.
            "semantic_topk_trunc": int(
                context.launch_configurations.get("semantic_topk_trunc", "0")),
            "semantic_evict_by_confidence":
                context.launch_configurations.get(
                    "semantic_evict_by_confidence", "false").lower() == "true",
            "semantic_spread_radius": float(
                context.launch_configurations.get("semantic_spread_radius", "0.0")),
            # Perf-attribution knobs (SLIM-VDB frame-time comparison). Every
            # default below is the value the node already used, so an unset
            # launch is byte-identical; they exist so the cost of the full-ray
            # carve, the TSDF band and the carve staging can be measured
            # SEPARATELY instead of inferred from a single frame_ms.
            "carve_band": float(
                context.launch_configurations.get("carve_band", "-1.0")),
            "sdf_trunc_voxels": int(
                context.launch_configurations.get("sdf_trunc_voxels", "3")),
            "batch_free_carve":
                context.launch_configurations.get(
                    "batch_free_carve", "true").lower() == "true",
        }],
    )

    # publishBinaryMap is TF-gated: it looks up map <- integration_frame at
    # publish time and defers (never publishes) until the transform exists.
    # The replay node only broadcasts odom -> base -> camera, so pin the
    # map <- odom identity here (same pattern as scenenet_eval_fusion).
    static_map_tf = Node(package="tf2_ros", executable="static_transform_publisher",
        name=f"static_map_to_{robot_name}_odom", output="log",
        arguments=["0", "0", "0", "0", "0", "0", "map", f"{robot_name}/odom"])

    return [scovox_node, static_map_tf]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_name", default_value="atlas"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("semantic_mode", default_value="dirichlet",
                              description="dirichlet | majority_vote | naive"),
        DeclareLaunchArgument("share_tsdf", default_value="false",
                              description="Step 9: controls whether v2 binary "
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
        DeclareLaunchArgument("w_occ", default_value="6.0",
                              description="A9: Beta hit weight (a_occ = 1 + w_occ*H). Default "
                                          "reproduces the pre-A9 hardcoded constant."),
        DeclareLaunchArgument("w_free", default_value="1.0",
                              description="A9: Beta carve weight (a_free = 1 + w_free*F). Default "
                                          "reproduces the pre-A9 hardcoded constant."),
        DeclareLaunchArgument("fused_walker", default_value="true",
                              description="iter6 single-DDA fused ray walker (production default)."),
        DeclareLaunchArgument("topk_probs_dir", default_value="",
                              description="Soft-prob input: dir of per-frame image .topk "
                                          "blobs (H,W,C). Empty = one-hot seg colour path."),
        DeclareLaunchArgument("semantic_topk_trunc", default_value="0",
                              description="Keep only the N most probable classes per "
                                          "observation; dropped mass becomes OTHER. "
                                          "0 (default) = full distribution."),
        DeclareLaunchArgument("semantic_evict_by_confidence", default_value="false",
                              description="Evict the weakest slot on incoming class "
                                          "PROBABILITY rather than accumulated evidence. "
                                          "Needs a -DSCOVOX_TRACK_QMAX=1 build."),
        DeclareLaunchArgument("semantic_spread_radius", default_value="0.0",
                              description="Metres. >0 spreads Stream B class evidence over "
                                          "already-occupied voxels within this radius "
                                          "(Melkumyan-Ramos kernel); Stream A occupancy "
                                          "still commits at the endpoint only."),
        DeclareLaunchArgument("carve_band", default_value="-1.0",
                              description="Perf attribution: metres of free-space "
                                          "carve before the surface. -1 (default) = "
                                          "full ray, origin to hit."),
        DeclareLaunchArgument("sdf_trunc_voxels", default_value="3",
                              description="Perf attribution: TSDF truncation in "
                                          "voxels. 0 disables the band update."),
        DeclareLaunchArgument("batch_free_carve", default_value="true",
                              description="Perf attribution: stage carves in a hash "
                                          "map and write once per voxel per scan."),
        DeclareLaunchArgument("map_mode", default_value="persistent",
                              description="E1: 'rolling' enables the ScovoxMapBinary "
                                          "publisher (a_occ/a_free/Dir snapshot capture); "
                                          "'persistent' (default) = paper offline eval."),
        DeclareLaunchArgument("share_rate_hz", default_value="0.0",
                              description="E1: >0 gives a timer-owned binary publish so a "
                                          "post-replay capture subscriber triggers a full "
                                          "snapshot. 0 (default) = legacy inline publish."),
        DeclareLaunchArgument("scovox_publish_rate", default_value="1.0",
                              description="Pointcloud republish timer rate (Hz). Lowered by "
                                          "the E1 runner for offline capture; default 1.0 "
                                          "preserves the paper runs."),
        DeclareLaunchArgument("log_mem_usage", default_value="false",
                              description="E4: emit the periodic [memSplit]/[memGate] grid-MB "
                                          "lines (diagnostic full-grid walk). Off by default."),
        OpaqueFunction(function=_launch_setup),
    ])
