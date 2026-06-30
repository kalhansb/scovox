"""Single-robot SCovox launch file — parameter audit reference.

Launch with:
  ros2 launch scovox_mapping scovox_single_robot.launch.py robot_name:=atlas

Current param counts (post Phase-0 tasks 0.1-0.7):
  SCovoxNode  : 54  (target <= 20)
  DSCovoxNode : 26  (target <= 14)
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
            # kappa0                   double  2.0   -     Dirichlet base pseudo-count per hit
            # semantic_top_k           int     10    -     Max named classes stored per voxel
            # semantic_occ_gate        double  0.5   -     Hard p_occ threshold for semantic updates
            "kappa0":                   2.0,
            "semantic_top_k":           10,
            "semantic_occ_gate":        0.5,

            # -- Evidence weights (3) --------------------------------------------------
            # w_occ / w_free control pseudo-count increment per observation.
            "w_occ":                    2.0,
            "w_free":                   1.0,

            # -- Evidence saturation ---------------------------------------------------

            # -- Range weighting (3) ---------------------------------------------------
            # range_decay_length <= 0 disables range weighting.
            # range_decay_length   double  -1.0   m    Distance at which w decays to ~37% (<=0 disables)
            # min_range            double  0.3    m    Discard returns closer than this
            # max_range            double  10.0   m    Discard returns farther than this
            "range_decay_length": -1.0,
            "min_range":           0.3,
            "max_range":           10.0,

            # -- Angle weighting (1) --------------------------------------------------
            # grazing_angle_threshold <= 0 disables angle weighting.
            # grazing_angle_threshold  double  -1.0   -    cos(t) below which weight ramps down (<=0 disables)
            "grazing_angle_threshold": -1.0,

            # -- Transient / dynamic layer (2) ----------------------------------------
            # max_semantic_classes is redundant with semantic_top_k and K_TOP
            # in most usage; consider removing in favour of semantic_top_k alone.
            # max_semantic_classes  int            10   -    Total class count (label space)
            # transient_decay_rate  double         0.8  -    Per-frame decay for dynamic voxels
            "max_semantic_classes": 10,
            "transient_decay_rate": 0.8,

            # -- Input frames & topics (7) [KEEP -- robot-specific] --------------------
            # base_frame          string  "base_link"  -   Robot base frame
            # integration_frame   string  "odom"       -   Frame for map integration
            # map_frame           string  "map"        -   Global map frame
            # depth_topic         string  ...          -   Depth image topic
            # depth_info_topic    string  ...          -   CameraInfo topic for depth
            # seg_topic           string  ...          -   Semantic segmentation topic
            # stride              int     1            px  Pixel stride when subsampling depth
            # min_depth           double  0.1          m   Depth clip minimum
            # max_depth           double  10.0         m   Depth clip maximum
            # trace_no_return_rays bool   false        -   Carve free space for no-return pixels
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
            # mode      string  "rolling"  -   "rolling" (publishes ScovoxMapBinary
            #                                   snapshots, rolling planning_map crop)
            #                                   | "persistent" (no binary)
            # robot_id  string  ""         -   Robot identifier (informational)
            "mode":                "rolling",
            "robot_id":            robot_name,

            # -- Output / visualisation (11) [CANDIDATE: planning map -> separate node] -
            # The 8 planning_map_* params could be moved to a dedicated
            # planning-map server node, leaving just publish_pointcloud,
            # occupancy_vis_threshold, and scovox_publish_rate here.
            # publish_pointcloud        bool    true    -    Publish coloured occupancy pointcloud
            # pointcloud_topic          string  ~/pc    -    Topic for pointcloud output
            # scovox_topic              string  ~/sec   -    Topic for full ScovoxMap (a_occ/a_free)
            # occupancy_vis_threshold        double  0.7     -    Minimum P(occ) to include in outputs
            # scovox_publish_rate       double  1.0     Hz   Rate of full ScovoxMap publication
            # publish_planning_map      bool    true    -    Publish 2-D OccupancyGrid for planners
            # planning_map_topic        string  ~/pm    -    Topic for planning OccupancyGrid
            # planning_map_resolution   double  0.20    m    Grid cell size
            # planning_map_size_m       double  80.0    m    Square grid side length
            # planning_map_origin_x     double  -40.0   m    Grid origin X (world frame)
            # planning_map_origin_y     double  -40.0   m    Grid origin Y (world frame)
            # planning_map_min_z        double  -1.0    m    Min voxel Z projected into grid
            # planning_map_max_z        double   2.0    m    Max voxel Z projected into grid
            # planning_map_inflation_m  double  0.0     m    Obstacle inflation radius
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
    # AUDIT NOTE: 26 params total.
    # Target <= 14.
    dscovox_node = Node(
        package="scovox_mapping",
        executable="dscovox_mapping_node",
        name="dscovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,

            # -- Input (1) [KEEP] ------------------------------------------------------
            # input_topics  string[]  []  -  Binary scovox topics from each robot
            "input_topics": [["", robot_name, "/scovox_node/scovox_bin"]],

            # -- Output (5) [KEEP 2, CANDIDATE: planning map -> separate node] ---------
            # Same 8-param planning_map block as SCovoxNode -- prime candidate for
            # extraction to a shared planning-map server.
            # pointcloud_topic   string  ~/pointcloud  -    Fused pointcloud output
            # map_frame          string  "map"         -    Frame for fused map
            # occupancy_vis_threshold double  0.7           -    Min P(occ) for outputs
            # publish_rate_hz    double  5.0           Hz   Fused map publish rate
            # publish_planning_map      bool    false  -    Publish merged planning grid
            # planning_map_topic        string  ~/pm   -
            # planning_map_resolution   double  0.20   m
            # planning_map_size_m       double  80.0   m
            # planning_map_origin_x     double  -40.0  m
            # planning_map_origin_y     double  -40.0  m
            # planning_map_min_z        double  -1.0   m
            # planning_map_max_z        double   2.0   m
            # planning_map_inflation_m  double  0.0    m
            "pointcloud_topic":         "/dscovox_mapping/pointcloud",
            "map_frame":                "map",
            "occupancy_vis_threshold":       0.7,
            "semantic_occ_gate":     0.6,
            "publish_rate_hz":          5.0,
            "publish_planning_map":     False,
            "planning_map_topic":       "~/planning_map",
            "planning_map_resolution":  0.20,
            "planning_map_size_m":      80.0,
            "planning_map_origin_x":   -40.0,
            "planning_map_origin_y":   -40.0,
            "planning_map_min_z":      -1.0,
            "planning_map_max_z":       2.0,
            "planning_map_inflation_m": 0.0,

            # -- Pose correction (3) [CANDIDATE FOR REMOVAL] ---------------------------
            # pose_source = "tf" means these are never used; remove until the
            # pose-correction path is actually implemented.
            # pose_source              string  "tf"   -    "tf" | "topic"
            # correction_check_hz      double  1.0    Hz   Rate to check for pose corrections
            # correction_pos_threshold double  0.05   m    Min position delta to trigger correction
            # correction_rot_threshold double  0.05   rad  Min rotation delta to trigger correction
            "pose_source":              "tf",
            "correction_check_hz":      1.0,
            "correction_pos_threshold": 0.05,
            "correction_rot_threshold": 0.05,

            # -- Consensus fusion (6) [KEEP -- core algorithm] -------------------------
            # epsilon_w   double  1e-3  -    Evidence floor (prevents divide-by-zero)
            # Lsat        double  3.0   -    Evidence saturation level for fusion
            # k_conflict  double  2.0   -    Conflict penalty scale factor
            # epsilon_sem double  1e-3  -    Semantic evidence floor
            # lambda_sem  double  5.0   -    Semantic consensus weight
            "epsilon_w":   1e-3,
            "Lsat":        3.0,
            "k_conflict":  2.0,
            "epsilon_sem": 1e-3,
            "lambda_sem":  5.0,

            # -- Semantics (1) [KEEP] -------------------------------------------------
            # semantic_top_k  int  10  -  Max named classes per fused voxel
            "semantic_top_k": 10,
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
