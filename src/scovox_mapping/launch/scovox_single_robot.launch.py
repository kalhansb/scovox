# Moved comments: doc/scovox_mapping_code_notes.md
"""Single-robot SCovox launch file — parameter audit reference.

Launch with:
  ros2 launch scovox_mapping scovox_single_robot.launch.py robot_name:=atlas

Only parameters the nodes actually declare are set here (see
declare_parameter calls in scovox_node.cpp / dscovox_node.cpp — undeclared
keys in a launch dict are silently ignored by ROS 2).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    robot_name = LaunchConfiguration("robot_name")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # -- SCovoxNode ----------------------------------------------------------------
    # AUDIT NOTE: 54 params total.  Groups marked [KEEP] / [CANDIDATE FOR REMOVAL]
    # to reach the <= 20 target.
    scovox_node = Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot_name,
        name="scovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,

            # -- Map model (4) [KEEP] --------------------------------------------------
            # resolution       double  0.10   m       Voxel side length
            # w_free           double  1.0    -        Beta free-hit weight
            # w_occ            double  2.0    -        Beta occupied-hit weight
            "resolution": 0.10,
            "w_free":     1.0,
            "w_occ":      2.0,

            # -- Semantics (4) [KEEP] -------------------------------------------------
            # kappa0: Dirichlet base pseudo-count per hit. Per-voxel top-K width
            # is the compile-time K_TOP (scovox_core voxel.hpp), not a
            # parameter. (notes: single-launch-semantics-params)
            "kappa0":                   2.0,
            "semantic_occ_gate":        0.5,

            # -- Evidence weights (3) --------------------------------------------------
            # w_occ / w_free control pseudo-count increment per observation.
            "w_occ":                    2.0,
            "w_free":                   1.0,

            # -- Evidence saturation (1) -----------------------------------------------
            # evidence_saturation  int  1000  counts  Per-voxel cap on pseudo-count
            # totals (Beta a_occ+a_free, Dir other+sum(cnt)); the proportional
            # rescale preserves p_occ / class ratios. 0 disables.
            "evidence_saturation": 1000,

            # -- Range weighting (3) ---------------------------------------------------
            # range_decay_length (m): distance at which the weight decays to
            # ~37%; <= 0 disables. min_range/max_range (m): returns closer or
            # farther are discarded. (notes: single-launch-range-params)
            "range_decay_length": -1.0,
            "min_range":           0.3,
            "max_range":           10.0,

            # -- Angle weighting (1) --------------------------------------------------
            # grazing_angle_threshold <= 0 disables angle weighting.
            # grazing_angle_threshold  double  -1.0   -    cos(t) below which weight ramps down (<=0 disables)
            "grazing_angle_threshold": -1.0,

            # -- Transient / dynamic layer (3) ----------------------------------------
            # max_semantic_classes is the total label space (per-voxel top-K is
            # compile-time K_TOP). dynamic_classes routes argmax classes to the
            # transient decaying grid; leave it unset to keep it off, as []
            # breaks ROS 2 type inference.
            # (notes: single-launch-transient-params)
            "max_semantic_classes": 10,
            "transient_decay_rate": 0.8,

            # -- Input frames & topics (7) [KEEP -- robot-specific] --------------------
            # Base, integration and global map frames; depth, CameraInfo and
            # segmentation topics; depth pixel stride (px) and clip range (m);
            # trace_no_return_rays carves free space for no-return pixels.
            # (notes: single-launch-frames-topics-params)
            "base_frame":          ["", robot_name, "/base_link"],
            "integration_frame":   ["", robot_name, "/odom"],
            "map_frame":           "map",
            "depth_topic":         "rgbd_camera_depth_image",
            "depth_info_topic":    "rgbd_camera_info",
            "seg_topic":           "segmentation/colored",
            "stride":              1,
            "min_depth":           0.1,
            "max_depth":           10.0,
            "trace_no_return_rays": False,

            # -- Mode / identity (2) [KEEP] -------------------------------------------
            # mode: rolling publishes ScovoxMapBinary snapshots and a rolling
            # planning_map crop; persistent publishes no binary. robot_id is
            # informational. (notes: single-launch-mode-params)
            "mode":                "rolling",
            "robot_id":            robot_name,

            # -- Output / visualisation (11) [CANDIDATE: planning map -> separate node] -
            # Outputs: coloured occupancy pointcloud, full ScovoxMap
            # (a_occ/a_free) at scovox_publish_rate Hz, min P(occ) for outputs,
            # and a 2-D OccupancyGrid for planners (cell, side, origin, z band,
            # inflation in m). (notes: single-launch-output-params)
            "publish_pointcloud":       True,
            "pointcloud_topic":         "~/pointcloud",
            "scovox_topic":             "~/scovox",
            "occupancy_vis_threshold":       0.7,
            "scovox_publish_rate":      1.0,
            "publish_planning_map":     True,
            "planning_map_topic":       "~/planning_map",
            "planning_map_resolution":  0.20,
            "planning_map_size_m":      80.0,
            "planning_map_origin_x":   -40.0,
            "planning_map_origin_y":   -40.0,
            "planning_map_min_z":       0.10,
            "planning_map_max_z":       1.0,
            "planning_map_inflation_m": 0.0,
        }],
    )

    # -- DSCovoxNode ---------------------------------------------------------------
    # Only parameters dscovox_mapping_node declares are set here; undeclared
    # keys are silently ignored. Consensus constants live in scovox_core;
    # per-voxel top-K width is the compile-time K_TOP.
    # (notes: single-launch-dscovox-declared-params)
    dscovox_node = Node(
        package="scovox_mapping",
        executable="dscovox_mapping_node",
        name="dscovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,

            # -- Input -----------------------------------------------------------------
            # input_topics  string[]  []  -  Binary scovox topics from each robot
            "input_topics": [["", robot_name, "/scovox_node/scovox_bin"]],

            # -- Output ----------------------------------------------------------------
            # Fused pointcloud topic, fused-map frame, min P(occ) for outputs,
            # and fused-map publish rate (Hz).
            # (notes: single-launch-dscovox-output-params)
            "pointcloud_topic":         "/dscovox_mapping/pointcloud",
            "map_frame":                "map",
            "occupancy_vis_threshold":  0.7,
            "semantic_occ_gate":        0.6,
            "publish_rate_hz":          1.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_name",
            description="Robot namespace (e.g. atlas, rama, ravana)",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time (/clock)",
        ),
        scovox_node,
        dscovox_node,
    ])
