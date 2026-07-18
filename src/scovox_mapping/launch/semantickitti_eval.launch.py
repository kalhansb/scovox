"""SCovox mapping node configured for SemanticKITTI evaluation (LiDAR input).

Usage:
  ros2 launch scovox_mapping semantickitti_eval.launch.py
  ros2 launch scovox_mapping semantickitti_eval.launch.py resolution:=0.10 semantic_mode:=majority_vote
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_setup(context):
    from launch_ros.actions import Node

    robot_name = context.launch_configurations["robot_name"]
    resolution = float(context.launch_configurations["resolution"])
    semantic_mode = context.launch_configurations["semantic_mode"]
    min_occ = float(context.launch_configurations["occupancy_vis_threshold"])
    carve_band = float(context.launch_configurations["carve_band"])
    sem_occ_gate = float(context.launch_configurations["semantic_occ_gate"])
    range_decay_length_arg = float(context.launch_configurations["range_decay_length"])
    w_occ_arg = float(context.launch_configurations["w_occ"])
    w_free_arg = float(context.launch_configurations["w_free"])
    kappa0_arg = float(context.launch_configurations["kappa0"])
    sdf_trunc_voxels_arg = int(context.launch_configurations["sdf_trunc_voxels"])
    tsdf_space_carving_arg = context.launch_configurations["tsdf_space_carving"].lower() in ("true", "1", "yes")
    band_only_integration_arg = context.launch_configurations["band_only_integration"].lower() in ("true", "1", "yes")
    topk_probs_dir_arg = context.launch_configurations.get("topk_probs_dir", "")
    eviction_stats_csv_arg = context.launch_configurations.get("eviction_stats_csv", "")
    # Step 9 (D2/D7) — split-grid v2 toggles (see replica_eval.launch.py).
    use_split_arg = context.launch_configurations.get("use_split", "false").lower() in ("true", "1", "yes")
    share_tsdf_arg = context.launch_configurations.get("share_tsdf", "false").lower() in ("true", "1", "yes")
    # 2026-05-09 audit hook — when non-empty, scovox_node writes a flat
    # binary snapshot of TsdfMap on every periodic memlog tick (see
    # tools/tsdf_parity_test.py).
    tsdf_dump_path_arg = context.launch_configurations.get("tsdf_dump_path", "")
    # Step 12.10 (2026-05-09) — fused single-DDA ray walker.
    fused_walker_arg = context.launch_configurations.get("fused_walker", "true").lower() in ("true", "1", "yes")
    # Step 8 SemDir priors (use_split=true only). SemanticKITTI uses C=20
    # (REPLAY layout: 0=unlabeled + 19 semantic + lane-marking at id 15).
    num_classes_arg     = int(context.launch_configurations.get("num_classes", "20"))
    dirichlet_prior_arg = float(context.launch_configurations.get("dirichlet_prior", "0.01"))
    wire_format_arg     = context.launch_configurations.get("wire_format", "v2")
    # Phase 2.5-v2 — integration-time admission gate. Below this p_occ_post
    # the per-class dirichletUpdate inside SemDirMap::applyHitUpdate is
    # skipped (mass routes to alpha_other / OTHER). 0.0 = label every hit
    # regardless of occupancy posterior (SLIM-VDB-like envelope).
    dirichlet_min_p_occ_arg = float(context.launch_configurations.get("dirichlet_min_p_occ", "0.5"))
    # scovox-repo eval-harness extras (passed by phase0/phase1/phase2 scripts).
    # scovox_node declares carve_skip_occ_threshold + evidence_saturation; it
    # ignores semantic_min_confidence (no matching param) as a harmless override.
    carve_skip_occ_arg = float(context.launch_configurations.get("carve_skip_occ_threshold", "0.4"))
    # node declares evidence_saturation as an integer (dp(..., 1000)); pass an int
    evidence_saturation_arg = int(float(context.launch_configurations.get("evidence_saturation", "1000")))
    semantic_min_confidence_arg = float(context.launch_configurations.get("semantic_min_confidence", "0.1"))
    # E1 uncertainty capture: rolling mode enables the ScovoxMapBinary publisher
    # (bin_pub_ is created only when mode==rolling); share_rate_hz>0 gives a
    # timer-owned binary publish so a snapshot fires when a capture subscriber
    # connects AFTER replay (with no subscriber the timer is a cheap no-op, so
    # replay recv stays 100/100). Defaults preserve the paper persistent runs.
    map_mode_arg = context.launch_configurations.get("map_mode", "persistent")
    share_rate_hz_arg = float(context.launch_configurations.get("share_rate_hz", "0.0"))

    scovox_node = Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot_name,
        name="scovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "dataset_mode": True,

            # PointCloud2 input (LiDAR mode)
            "input_pointcloud_topic": f"/{robot_name}/velodyne_points",
            # Reliable input sub: the replay publishes reliable, and large KITTI
            # clouds lose UDP fragments on a best-effort link (small rmem_max),
            # dropping ~60% of frames. Reliable retransmits -> full delivery.
            "input_reliable_qos": True,

            # Map model — leaf_bits=1 for sparse outdoor LiDAR (2×2×2 blocks)
            "resolution": resolution,
            "inner_bits": 2,
            "leaf_bits": 1,
            "w_free": w_free_arg,
            "w_occ": w_occ_arg,
            # Free-space carving: 0=endpoint only, >0=band before hit, <0=full ray
            "carve_band": carve_band,

            # TSDF integration controls
            "sdf_trunc_voxels": sdf_trunc_voxels_arg,
            "tsdf_space_carving": tsdf_space_carving_arg,
            "band_only_integration": band_only_integration_arg,

            # Split-grid v2 toggles (Step 9 / D2 / D7).
            "use_split": use_split_arg,
            "share_tsdf": share_tsdf_arg,
            "fused_walker": fused_walker_arg,
            "tsdf_dump_path": tsdf_dump_path_arg,
            # SemDir priors — only consumed when use_split=true. KITTI default
            # is C=20 (REPLAY-layout class id range) to match the topk blob.
            "num_classes": num_classes_arg,
            "dirichlet_prior": dirichlet_prior_arg,
            "dirichlet_min_p_occ": dirichlet_min_p_occ_arg,
            # Wire-format selector (use_split=true only). Default v2 keeps
            # backward compat with dscovox_node's onBinaryMapV2 receiver.
            # Set v3 once the v3 receiver lands (Step 8 follow-up).
            "wire_format": wire_format_arg,

            # Semantics — 20 classes for SemanticKITTI (0=unlabeled, 1-19=semantic)
            "kappa0": kappa0_arg,
            "semantic_top_k": 10,
            "semantic_occ_gate": sem_occ_gate,
            "semantic_mode": semantic_mode,
            "max_semantic_classes": 20,

            # Evidence

            # Range — match SLIM-VDB KITTI config for apples-to-apples comparison
            # (slim_vdb/examples/cpp/config/kitti.yaml uses min=5.0, max=30.0).
            # Pre-2026-05-11 value was min=1.0; flipped to 5.0 to eliminate
            # the only meaningful KITTI head-to-head config asymmetry.
            "range_decay_length": range_decay_length_arg,
            "min_range": 5.0,
            "max_range": 30.0,

            # No angle weighting for LiDAR
            "grazing_angle_threshold": -1.0,

            # Frames
            "base_frame": f"{robot_name}/base_link",
            "integration_frame": f"{robot_name}/odom",
            "map_frame": "map",

            # Depth topics unused in pointcloud mode, but must be declared
            "depth_topic": "unused",
            "depth_info_topic": "unused",
            "seg_topic": "unused",

            "stride": 1,
            "min_depth": 1.0,
            "max_depth": 80.0,
            "trace_no_return_rays": False,
            # KITTI vehicle moves ~1m/frame at 0.5Hz replay; default 0.5m
            # threshold causes the startup TF-stability gate to reset every
            # frame. Raise so integration starts on frame 2.
            "startup_tf_jump_threshold": 10.0,
            "startup_tf_stable_sec": 0.0,
            # The scovox node adds a runtime TF-divergence gate (default
            # runtime_tf_jump_threshold=1.0 m) that the paper node lacked. KITTI's
            # ~1.2 m/frame motion exceeds it, so after frame 2 stabilizes every
            # subsequent frame is flagged as "localization diverged" and gated
            # forever -> empty map. The replay feeds exact GT poses (no real
            # divergence to guard against), so disable the runtime gate and raise
            # the threshold well above the per-frame vehicle step.
            "runtime_tf_gate": False,
            "runtime_tf_jump_threshold": 20.0,

            # Mode: persistent (paper offline eval) or rolling (E1 binary capture).
            "mode": map_mode_arg,
            "share_rate_hz": share_rate_hz_arg,
            "submap_max_distance": 999.0,
            "robot_id": robot_name,

            # Output
            "publish_pointcloud": True,
            "pointcloud_topic": "~/pointcloud",
            "scovox_topic": "~/scovox",
            "occupancy_vis_threshold": min_occ,
            # The publish timer holds a shared map lock and walks the whole map
            # every tick; integration needs the unique lock, so a fast timer on a
            # growing 600k-voxel map starves the queue-depth-1 cloud subscription
            # and most replay frames are dropped (recv ~22/100 at 1 Hz). Slow the
            # timer right down for offline eval — we only need one publish at the
            # end for the npz capture. Also drop the TSDF cloud republisher (unused
            # by the eval, another full-grid walk per tick).
            "scovox_publish_rate": 0.2,
            "publish_tsdf_pointcloud": False,
            "publish_planning_map": False,

            # No semantic color map needed — labels come as integers from PointCloud2
            "semantic_color_map_keys": [0],
            "semantic_color_map_classes": [0],
            # Soft-prob ablation: when set to a non-empty path, scovox_node
            # ignores the PointCloud2 semantic_label field and reads the
            # per-frame top-K (id, prob) table from "<dir>/<frame:06d>.topk"
            # instead. The replay node must be launched with
            # soft_prob_passthrough:=true so cloud point ordering matches.
            "topk_probs_dir": topk_probs_dir_arg,
            "eviction_stats_csv": eviction_stats_csv_arg,

            # scovox-repo eval-harness extras.
            "carve_skip_occ_threshold": carve_skip_occ_arg,
            "evidence_saturation": evidence_saturation_arg,
            "semantic_min_confidence": semantic_min_confidence_arg,
        }],
    )

    return [scovox_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_name", default_value="atlas"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("occupancy_vis_threshold", default_value="0.5"),
        DeclareLaunchArgument("semantic_mode", default_value="dirichlet"),
        DeclareLaunchArgument("carve_band", default_value="0.1"),
        DeclareLaunchArgument("semantic_occ_gate", default_value="0.6",
                              description="Minimum gate value; 0.0 = NG ablation."),
        DeclareLaunchArgument("range_decay_length", default_value="50.0",
                              description="Range-decay length in m; -1 disables = NR ablation."),
        DeclareLaunchArgument("w_occ", default_value="6.0"),
        DeclareLaunchArgument("w_free", default_value="1.0"),
        DeclareLaunchArgument("kappa0", default_value="2.0"),
        DeclareLaunchArgument("sdf_trunc_voxels", default_value="3",
                              description="TSDF truncation in voxels; 0 disables TSDF."),
        DeclareLaunchArgument("tsdf_space_carving", default_value="false"),
        DeclareLaunchArgument("band_only_integration", default_value="false",
                              description="True confines fused walk to [hit-trunc, hit+trunc] (skips full-ray Beta-free)."),
        DeclareLaunchArgument("topk_probs_dir", default_value="",
                              description="Soft-prob ablation: directory of <frame>.topk flat-binary blobs."),
        DeclareLaunchArgument("eviction_stats_csv", default_value="",
                              description="E5.2 — append per-frame sparse_add branch counters to this CSV."),
        DeclareLaunchArgument("use_split", default_value="false",
                              description="Step 9: route integration through ScovoxMapSplit "
                                          "(TsdfMap + SemBetaMap) instead of the legacy fused scovox::Map. "
                                          "Default false preserves the v1 wire-format path."),
        DeclareLaunchArgument("share_tsdf", default_value="false",
                              description="Step 9: when use_split=true, controls whether v2 binary "
                                          "frames carry the TSDF stream (true: 57 B/voxel) or just "
                                          "SemBeta deltas (false: 37 B/voxel — production default)."),
        DeclareLaunchArgument("tsdf_dump_path", default_value="",
                              description="2026-05-09 audit hook: when non-empty (split mode), "
                                          "scovox_node overwrites this path with a flat binary "
                                          "snapshot of TsdfMap on every periodic memlog tick. Used "
                                          "by tools/tsdf_parity_test.py to compare voxel sets "
                                          "against SLIM-VDB after a Tr_inv frame conversion."),
        DeclareLaunchArgument("wire_format", default_value="v2",
                              description="Step 8 (use_split=true only): inter-robot wire format. "
                                          "v2 = SemBeta-projected (current production, 37 B/voxel). "
                                          "v3 = SemDir-native (20 B/voxel + header priors). Set v3 "
                                          "after dscovox_node lands its onBinaryMapV3 receiver."),
        DeclareLaunchArgument("num_classes", default_value="20",
                              description="Step 8 (use_split=true only): dataset class count C. "
                                          "SemanticKITTI default 20 (REPLAY layout). Sets OTHER-prior "
                                          "(C−K_TOP)·α_0 and the v3 wire-format header."),
        DeclareLaunchArgument("dirichlet_prior", default_value="0.01",
                              description="Step 8 (use_split=true only): symmetric Dirichlet prior α_0. "
                                          "Ship default 0.01 matches `defaultSemDirVoxel`. "
                                          "1/(C+1) recovers Jeffreys for ablation."),
        DeclareLaunchArgument("dirichlet_min_p_occ", default_value="0.5",
                              description="Phase 2.5-v2 ablation: integration-time admission "
                                          "gate inside SemDirMap::applyHitUpdate. Below this "
                                          "p_occ_post the per-class dirichletUpdate is skipped "
                                          "(mass routes to OTHER). 0.0 = label every hit (the "
                                          "controlled ablation behind the SceneNet head-to-head "
                                          "gap-mechanism finding)."),
        DeclareLaunchArgument("fused_walker", default_value="true",
                              description="Step 12.10 (2026-05-09): when use_split=true, hit rays "
                                          "go through a single Bresenham DDA over the union of the "
                                          "TSDF band and SemBeta carve range with per-voxel dispatch "
                                          "into both grids. Default true (production); set false to "
                                          "fall back to the legacy two-DDA split path for A/B parity."),
        # scovox-repo eval-harness extras (phase0/phase1/phase2 pass these).
        DeclareLaunchArgument("carve_skip_occ_threshold", default_value="0.4"),
        DeclareLaunchArgument("evidence_saturation", default_value="1000.0"),
        DeclareLaunchArgument("semantic_min_confidence", default_value="0.1"),
        DeclareLaunchArgument("map_mode", default_value="persistent",
                              description="E1: 'rolling' enables the ScovoxMapBinary "
                                          "publisher (a_occ/a_free/Dir snapshot capture); "
                                          "'persistent' (default) = paper offline eval."),
        DeclareLaunchArgument("share_rate_hz", default_value="0.0",
                              description="E1: >0 gives a timer-owned binary publish so a "
                                          "post-replay capture subscriber triggers a full "
                                          "snapshot. 0 (default) = legacy inline publish."),
        OpaqueFunction(function=_launch_setup),
    ])
