"""SCovox mapping node configured for SceneNN (HKUST) evaluation.

SceneNN is real Asus Xtion capture at 640x480, annotated with the NYU-40
class set, so this differs from scenenet_eval.launch.py in three ways:

  * 41 semantic classes (0 = unknown, 1..40 = NYU-40) instead of 14, with the
    palette read from the dataset's own nyu_color.xml so the launch and
    scenenn_replay_node.py cannot drift apart.
  * Depth range clamped to the Asus Xtion's usable band (0.4-4.0 m) rather
    than SceneNet's synthetic 0.1-10 m. Xtion returns nothing beyond ~4 m and
    its stereo error grows quadratically, so integrating past that just
    smears the surface across voxels.
  * 5 cm voxels by default, matching the SLIM-VDB indoor protocol the other
    eval launches use.

Usage (nyu_colormap is required — there is no default path):
  ros2 launch scovox_mapping scenenn_eval.launch.py \
      nyu_colormap:=<scenenn_root>/nyu_color.xml
  ros2 launch scovox_mapping scenenn_eval.launch.py \
      nyu_colormap:=<scenenn_root>/nyu_color.xml resolution:=0.10 semantic_mode:=majority_vote
"""

import re
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

NYU_NUM_CLASSES = 41


def _load_palette(xml_path):
    """nyu_color.xml -> (keys, classes) for SCovox's colour->class lookup.

    Fails loudly on a missing/unparseable file. (This replaced a silent
    generated-palette fallback: the old default pointed into another
    machine's home directory, so on any other box the launch came up with a
    synthetic palette and mislabelled everything without a single warning.)
    """
    if not xml_path:
        raise RuntimeError(
            "scenenn_eval.launch.py: nyu_colormap was not set. Pass "
            "nyu_colormap:=<scenenn_root>/nyu_color.xml — it must be the same "
            "file the replay node reads, or colours and classes drift apart.")
    if not Path(xml_path).exists():
        raise RuntimeError(
            f"scenenn_eval.launch.py: nyu_colormap file not found: {xml_path}")
    text = Path(xml_path).read_text()
    keys, classes = [], []
    seen = {}
    for m in re.finditer(r'<class\s+id="(\d+)"[^>]*color="(\d+)\s+(\d+)\s+(\d+)"', text):
        cid = int(m.group(1))
        if not (0 <= cid < NYU_NUM_CLASSES):
            continue
        rgb = (int(m.group(2)) << 16) | (int(m.group(3)) << 8) | int(m.group(4))
        # Duplicate colours would make the class lookup ambiguous; first wins.
        if rgb in seen:
            continue
        seen[rgb] = cid
        keys.append(rgb)
        classes.append(cid)
    if not keys:
        raise RuntimeError(
            f"scenenn_eval.launch.py: no <class id=... color=...> entries "
            f"parsed from {xml_path} — wrong file?")
    return keys, classes


def _launch_setup(context):
    from launch_ros.actions import Node

    cfg = context.launch_configurations
    robot_name = cfg["robot_name"]
    resolution = float(cfg["resolution"])
    semantic_mode = cfg["semantic_mode"]
    keys, classes = _load_palette(cfg.get("nyu_colormap", ""))

    def flag(name, default):
        return cfg.get(name, default).lower() in ("true", "1", "yes")

    scovox_node = Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot_name,
        name="scovox_node",
        output="screen",
        # The node logs one INFO line per integrated frame. That formatting, the
        # /rosout publish and the stdout write all sit INSIDE the frame callback
        # but OUTSIDE the frame_ms window (scovox_node.cpp:1233 takes t_end
        # before them), so they cost throughput without showing up in frame_ms.
        # `log_level:=warn` suppresses them for timing runs.
        # `ros_arguments`, NOT `arguments`: launch_ros builds its own `--ros-args`
        # block (remaps + params file) and a hand-rolled one passed via
        # `arguments` is dropped without warning -- the node then runs at INFO
        # while the log says the arm was quiet.
        ros_arguments=["--log-level", cfg.get("log_level", "info")],
        parameters=[{
            "use_sim_time": False,
            "dataset_mode": True,

            "resolution": resolution,
            "w_free": float(cfg.get("w_free", "1.0")),
            "w_occ": float(cfg.get("w_occ", "6.0")),

            # Semantics — NYU-40 (+ unknown)
            "kappa0": 2.0,
            "semantic_occ_gate": 0.6,
            "semantic_mode": semantic_mode,
            "max_semantic_classes": NYU_NUM_CLASSES,
            "num_classes": NYU_NUM_CLASSES,

            # Range — Asus Xtion usable band
            "range_decay_length": -1.0,
            "min_range": float(cfg.get("min_range", "0.4")),
            "max_range": float(cfg.get("max_range", "4.0")),
            "grazing_angle_threshold": -1.0,
            "transient_decay_rate": 0.0,

            "base_frame": f"{robot_name}/base_link",
            "integration_frame": f"{robot_name}/odom",
            "map_frame": "map",

            "depth_topic": "rgbd_camera_depth_image",
            "depth_info_topic": "rgbd_camera_info",
            "seg_topic": "segmentation/colored",

            "stride": int(cfg.get("stride", "1")),
            "min_depth": float(cfg.get("min_range", "0.4")),
            "max_depth": float(cfg.get("max_range", "4.0")),
            "trace_no_return_rays": False,

            "mode": cfg.get("map_mode", "persistent"),
            "share_rate_hz": float(cfg.get("share_rate_hz", "0.0")),
            "log_mem_usage": flag("log_mem_usage", "false"),
            "robot_id": robot_name,

            "publish_pointcloud": True,
            "pointcloud_topic": "~/pointcloud",
            "scovox_topic": "~/scovox",
            "occupancy_vis_threshold": float(cfg.get("occupancy_vis_threshold", "0.5")),
            "dirichlet_min_p_occ": float(cfg.get("dirichlet_min_p_occ", "0.5")),
            "scovox_publish_rate": float(cfg.get("scovox_publish_rate", "1.0")),
            "publish_planning_map": False,

            "semantic_color_map_keys": keys,
            "semantic_color_map_classes": classes,

            "share_tsdf": flag("share_tsdf", "false"),
            "fused_walker": flag("fused_walker", "true"),
            "topk_probs_dir": cfg.get("topk_probs_dir", ""),
            "enable_tsdf": flag("enable_tsdf", "true"),
            "carve_band": float(cfg.get("carve_band", "-1.0")),
            "sdf_trunc_voxels": int(cfg.get("sdf_trunc_voxels", "3")),
            "batch_free_carve": flag("batch_free_carve", "true"),
            # Bonxai block geometry. Coordinates are a pure function of
            # `resolution` (VoxelGrid::posToCoord), so these change only how
            # voxels are packed into leaf/inner nodes -- never a voxel's
            # identity or value. That makes them the one set of performance
            # knobs here that cannot degrade the map, unlike stride/carve_band.
            "leaf_bits": int(cfg.get("leaf_bits", "3")),
            "inner_bits": int(cfg.get("inner_bits", "2")),
            "dir_leaf_bits": int(cfg.get("dir_leaf_bits", "2")),
        }],
    )

    # publishBinaryMap is TF-gated on map <- integration_frame; the replay node
    # only broadcasts odom -> base -> camera, so pin the identity here.
    static_map_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name=f"static_map_to_{robot_name}_odom", output="log",
        arguments=["0", "0", "0", "0", "0", "0", "map", f"{robot_name}/odom"])

    return [scovox_node, static_map_tf]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_name", default_value="atlas"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("semantic_mode", default_value="dirichlet",
                              description="dirichlet | majority_vote | naive"),
        DeclareLaunchArgument("nyu_colormap", default_value="",
                              description="REQUIRED: SceneNN nyu_color.xml; must match the "
                                          "replay node's. Empty (the default) fails at launch."),
        DeclareLaunchArgument("max_range", default_value="4.0"),
        # Asus Xtion usable band is 0.4-4.0 m; tightening either end drops rays
        # rather than making each ray cheaper, so it trades map coverage for
        # speed the same way stride and carve_band do.
        DeclareLaunchArgument("min_range", default_value="0.4"),
        # SAMPLING: a 5 cm voxel spans 0.05*fx/z px, so at fx=544 and the
        # 1.5-3 m range SceneNN actually observes, samples per voxel are:
        #     z      span     stride1  stride2  stride4
        #    1.5m   18.1px      328       82       20
        #    3.0m    9.1px       83       21        5
        # stride 1 is ~300 correlated pixels per voxel, which saturates the
        # Beta posterior on what is effectively one surface observation and
        # makes the uncertainty output meaningless (same argument as
        # start_scovox.sh on the robot). stride 2 is 4x cheaper and still
        # oversamples; drop to 4 if frame_ms matters more than thin structures.
        DeclareLaunchArgument("stride", default_value="2"),
        DeclareLaunchArgument("map_mode", default_value="persistent"),
        DeclareLaunchArgument("enable_tsdf", default_value="true"),
        DeclareLaunchArgument("share_tsdf", default_value="false"),
        DeclareLaunchArgument("fused_walker", default_value="true"),
        DeclareLaunchArgument("occupancy_vis_threshold", default_value="0.5"),
        DeclareLaunchArgument("dirichlet_min_p_occ", default_value="0.5"),
        DeclareLaunchArgument("scovox_publish_rate", default_value="1.0"),
        DeclareLaunchArgument("share_rate_hz", default_value="0.0"),
        DeclareLaunchArgument("log_mem_usage", default_value="false"),
        DeclareLaunchArgument("topk_probs_dir", default_value=""),
        DeclareLaunchArgument("w_free", default_value="1.0"),
        DeclareLaunchArgument("w_occ", default_value="6.0"),
        DeclareLaunchArgument("carve_band", default_value="-1.0"),
        DeclareLaunchArgument("sdf_trunc_voxels", default_value="3"),
        DeclareLaunchArgument("batch_free_carve", default_value="true"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("leaf_bits", default_value="3"),
        DeclareLaunchArgument("inner_bits", default_value="2"),
        DeclareLaunchArgument("dir_leaf_bits", default_value="2"),
        OpaqueFunction(function=_launch_setup),
    ])
