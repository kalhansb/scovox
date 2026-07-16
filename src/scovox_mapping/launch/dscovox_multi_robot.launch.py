# =======================================================================
# dscovox — MULTI-ROBOT dscovox map bring-up (N rolling mappers + 1 merger).
# =======================================================================
# The two-robot sibling of dscovox_single_robot.launch.py. Each robot runs its
# own rolling-mode scovox mapper (creating its `~/scovox_bin` LZ4 delta stream)
# in a UNIQUE integration frame; a single dscovox merger subscribes to EVERY
# robot's bin stream and reconstructs the fused, multi-robot `dscovox` map on
# `/dscovox_mapping/pointcloud`. This is the on-one-host replay analog of the
# real distributed fleet in docs/distributed_mapping_lidar.md.
#
# The two rules that make fusion work (see docs/distributed_mapping_lidar.md):
#   1. Each robot maps in a UNIQUE frame (bunker_map, curt_map, ...), bridged to
#      `map` by an identity static TF (published here). The bin stream is tagged
#      with this frame and the merger keys each source by it. Two robots sharing
#      one integration frame collapse into one source and overwrite each other.
#   2. The merger starts alongside the mappers so the subscriber gate (GATE 2)
#      is open — a mapper with no subscriber drains its deltas and emits nothing.
#
# This launch provides ONLY the scovox side. YOU supply, in the SAME ROS graph,
# for EACH robot (see docs/dscovox_multi_robot_run.md):
#   * the sensor stream  — cloud on <cloud_topic> (+ IMU on <imu_topic>)
#   * a COMPLETE TF tree — map -> <robot>/odom -> <robot base> -> <base_frame>
#     from a per-robot hmr_localisation NDT stack localizing against the SAME
#     gt_map (so every robot shares one global `map`) + the bag's /tf_static.
#     Without it every scan is dropped and the map — hence the bin stream — stays
#     empty (GATE 3).
#
# Defaults target the two "kalhan_coop" bags (bunker = Hesai, curt = Ouster);
# their sensor topics and base frames differ, so each robot is described by its
# own entry in _ROBOTS below. Override the map-extent knobs on the command line.
#
# Examples (inside the scovox container after `colcon build`):
#   ros2 launch scovox_mapping dscovox_multi_robot.launch.py
#   ros2 launch scovox_mapping dscovox_multi_robot.launch.py max_range:=60.0
#   ros2 launch scovox_mapping dscovox_multi_robot.launch.py \
#       share_roi_z_min:=-0.5 share_roi_z_max:=3.0
#
# Verify (another shell in the container):
#   ros2 topic hz /bunker/scovox_node/scovox_bin /curt/scovox_node/scovox_bin
#   ros2 topic hz /dscovox_mapping/pointcloud
#   # the merger console (output=screen) should reach: dscovox_diag: sources=2 ...
# =======================================================================
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ---- Per-robot deployment table --------------------------------------------
# One entry per robot. Everything that DIFFERS between the two coop robots
# (namespace, sensor topics, sensor origin frame, unique integration frame)
# lives here; the shared occupancy/share knobs are the launch args below.
#   * name              : ROS namespace + keys the bin topic /<name>/scovox_node/scovox_bin
#   * cloud_topic       : that robot's raw LiDAR PointCloud2
#   * imu_topic         : that robot's IMU (for in-node deskew)
#   * base_frame        : the cloud's header.frame_id (LiDAR ray origin)
#   * integration_frame : UNIQUE per robot; identity-linked to map (rule 1)
_ROBOTS = [
    {
        "name": "bunker",
        "cloud_topic": "/hesai/points",
        "imu_topic": "/imu/data",
        "base_frame": "hesai_lidar",
        "integration_frame": "bunker_map",
    },
    {
        "name": "curt",
        "cloud_topic": "/ouster/points",
        "imu_topic": "/curt/imu/data",
        "base_frame": "os_lidar",
        "integration_frame": "curt_map",
    },
]


# Shared knobs (mirror dscovox_single_robot.launch.py). Map-extent defaults are
# opened up (z-band off, max_range 40) so the fused map isn't clipped; tighten
# share_roi_z_*/max_range to match the low-bandwidth distributed experiment.
_ARGS = {
    "map_frame": ("map", "Global/world frame shared by every robot (NDT vs the same gt_map)"),
    "use_sim_time": ("true", "Use /clock (true for bag replay)"),
    "deskew_mode": ("auto", "auto|on|off — 'off' silences the IMU-not-ready note"),
    "min_range": ("1.0", "Per-scan min integration range (m)"),
    "max_range": ("40.0", "Per-scan max integration range (m) — RAISE to widen the footprint"),
    "share_roi_z_min": ("0.0", "Vertical share-band lower bound (m, map frame); wired to mapper+merger"),
    "share_roi_z_max": ("0.0", "Vertical share-band upper bound (m). min>=max (0/0) = band OFF"),
    "resolution": ("0.10", "Voxel size (m)"),
}


def launch_setup(context, *args, **kwargs):
    g = {k: LaunchConfiguration(k).perform(context) for k in _ARGS}
    use_sim_time = str(g["use_sim_time"]).lower() in ("1", "true", "yes", "on")
    map_frame = g["map_frame"]

    nodes = []
    bin_topics = []
    for r in _ROBOTS:
        bin_topic = f"/{r['name']}/scovox_node/scovox_bin"
        bin_topics.append(bin_topic)

        # Rolling-mode mapper: builds this robot's occupancy map in its UNIQUE
        # integration_frame and publishes the LZ4 delta stream (~/scovox_bin).
        nodes.append(Node(
            package="scovox_mapping",
            executable="scovox_mapping_node",
            namespace=r["name"],
            name="scovox_node",              # topic path depends on BOTH ns and name
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "mode": "rolling",           # GATE 1 — creates the bin publisher
                "fuse_lidar_rgbd": False,
                "input_pointcloud_topic": r["cloud_topic"],
                "imu_topic": r["imu_topic"],
                "base_frame": r["base_frame"],
                "integration_frame": r["integration_frame"],
                "map_frame": map_frame,
                "deskew_mode": g["deskew_mode"],
                # LiDAR occupancy defaults (mirror scovox_lidar_geometric.yaml)
                "resolution": float(g["resolution"]),
                "w_occ": 8.0, "w_free": 4.0, "carve_band": -1.0,
                "min_range": float(g["min_range"]), "max_range": float(g["max_range"]),
                "enable_tsdf": False,
                # Low-bandwidth share controls
                "share_change_gate": True,
                "share_rate_hz": 2.0,
                "share_roi_z_min": float(g["share_roi_z_min"]),
                "share_roi_z_max": float(g["share_roi_z_max"]),
                # Exact-stamp TF; never integrate at a stale pose (GATE 3)
                "tf_require_exact": True,
                "tf_lookup_timeout_sec": 1.0,
            }],
        ))

        # Identity map -> integration_frame bridge (rule 1): lets the merger fold
        # this source back into the common map frame. One latched static sample.
        nodes.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"map_to_{r['integration_frame']}",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                "--frame-id", map_frame, "--child-frame-id", r["integration_frame"],
            ],
        ))

    # Single shared merger fusing EVERY robot's bin stream -> the multi-robot
    # dscovox map. (The real fleet runs one merger per robot; a single merger is
    # enough to reconstruct/visualize the merged map on one host.)
    nodes.append(Node(
        package="scovox_mapping",
        executable="dscovox_mapping_node",
        name="dscovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topics": bin_topics,
            "map_frame": map_frame,
            "pointcloud_topic": "/dscovox_mapping/pointcloud",
            "share_roi_z_min": float(g["share_roi_z_min"]),   # keep in sync with senders
            "share_roi_z_max": float(g["share_roi_z_max"]),
            "pointcloud_min_interval_s": 0.5,
        }],
    ))

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=d, description=desc)
         for k, (d, desc) in _ARGS.items()]
        + [OpaqueFunction(function=launch_setup)]
    )
