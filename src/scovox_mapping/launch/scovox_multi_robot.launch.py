from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def make_scovox_node(robot_name: str, use_sim_time):
    # Use RGBD depth + semantic colored segmentation image.
    return Node(
        package="scovox_mapping",
        executable="scovox_mapping_node",
        namespace=robot_name,
        name="scovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "depth_topic": "rgbd_camera_depth_image",
            "depth_info_topic": "rgbd_camera_info",
            "seg_topic": "segmentation/colored",
            "integration_frame": f"{robot_name}/odom",
            "base_frame": f"{robot_name}/base_link",
            "robot_id": robot_name,
            "scovox_topic": "~/scovox",
            "pointcloud_topic": "~/pointcloud",
            "publish_planning_map": True,
            "planning_map_topic": "~/planning_map",
            "planning_map_resolution": 0.20,
            "planning_map_size_m": 80.0,
            "planning_map_origin_x": -40.0,
            "planning_map_origin_y": -40.0,
            "planning_map_min_z": 0.10,
            "planning_map_max_z": 1.0,
        }],
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    robots = ["atlas", "rama", "ravana"]

    scovox_nodes = [make_scovox_node(robot, use_sim_time) for robot in robots]

    # Each scovox_node publishes binary map on /<robot>/scovox_node/scovox_bin
    map_inputs = [f"/{robot}/scovox_node/scovox_bin" for robot in robots]

    dscovox_mapping_node = Node(
        package="scovox_mapping",
        executable="dscovox_mapping_node",
        name="dscovox_node",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topics": map_inputs,
            "pointcloud_topic": "/dscovox_mapping/pointcloud",
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time (/clock)",
        ),
        *scovox_nodes,
        dscovox_mapping_node,
    ])
