#!/usr/bin/env python3
"""ROS2 node that replays Gazebo TrajectoryRecorder output into SCovox.

Reads .npy frames + poses.json produced by the TrajectoryRecorder Gazebo plugin
and publishes them as standard ROS2 messages that SCovoxNode subscribes to:
  - depth_image       (sensor_msgs/Image, 32FC1 in meters)
  - camera_info       (sensor_msgs/CameraInfo, pinhole intrinsics)
  - segmentation      (sensor_msgs/Image, RGB8 semantic colors)
  - TF: odom -> base_link -> camera_link  (from GT poses)

Key difference from replica_replay_node: poses are body-to-world (Gazebo
convention: X=forward, Y=left, Z=up), NOT optical-to-world (NICE-SLAM).
The TF rotation is published as R directly — SCovoxNode applies kR internally
to get R @ kR @ p_optical = R @ p_body, which is correct.

Usage:
    python3 -m scovox_eval.gazebo_replay_node --ros-args \
        -p dataset_path:=/path/to/flatforest_rendered \
        -p rate_hz:=2.0 \
        -p robot_name:=atlas
"""

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as TimeMsg

from .replica_replay_node import CATEGORY_COLORS


def _rotation_matrix_to_quaternion(R: np.ndarray):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [w, x, y, z]


class GazeboReplayNode(Node):
    def __init__(self):
        super().__init__("gazebo_replay_node")

        # Parameters
        self.declare_parameter("dataset_path", "")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("robot_name", "atlas")
        self.declare_parameter("loop", False)

        dataset_path = self.get_parameter("dataset_path").value
        self.rate_hz = self.get_parameter("rate_hz").value
        self.robot_name = self.get_parameter("robot_name").value
        self.loop = self.get_parameter("loop").value

        if not dataset_path:
            self.get_logger().fatal("dataset_path parameter is required")
            raise SystemExit(1)

        self.dataset = Path(dataset_path)
        self._load_dataset()

        # Frame ID strings
        self.odom_frame = f"{self.robot_name}/odom"
        self.base_frame = f"{self.robot_name}/base_link"
        self.camera_frame = f"{self.robot_name}/camera_link"

        # Publishers — RELIABLE QoS so no messages are dropped
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
        )

        self.depth_pub = self.create_publisher(
            Image,
            f"/{self.robot_name}/rgbd_camera_depth_image",
            reliable_qos,
        )
        self.seg_pub = self.create_publisher(
            Image,
            f"/{self.robot_name}/segmentation/colored",
            reliable_qos,
        )
        self.info_pub = self.create_publisher(
            CameraInfo,
            f"/{self.robot_name}/rgbd_camera_info",
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        # Playback state
        self.frame_idx = 0
        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self._publish_frame)

        self.get_logger().info(
            f"Gazebo replay: {len(self.poses)} frames at {self.rate_hz} Hz "
            f"from {self.dataset}"
        )

    def _load_dataset(self):
        """Load poses.json, camera.json, and semantic_class_map.json."""
        with open(self.dataset / "poses.json") as f:
            self.poses = json.load(f)

        with open(self.dataset / "camera.json") as f:
            self.camera = json.load(f)

        # Load semantic class map if available
        sem_map_path = self.dataset / "semantic_class_map.json"
        self.sem_class_map = {}
        if sem_map_path.exists():
            with open(sem_map_path) as f:
                raw = json.load(f)
            for inst_id_str, info in raw.items():
                cat_name = info.get("category_name", "").lower()
                for key, color in CATEGORY_COLORS.items():
                    if key in cat_name:
                        self.sem_class_map[int(inst_id_str)] = color
                        break

        self.get_logger().info(
            f"Loaded {len(self.poses)} poses, "
            f"{len(self.sem_class_map)} semantic mappings"
        )

    def _stamp_from_index(self, idx: int) -> TimeMsg:
        """Use current wall-clock time, encode frame index in low bits of nanosec."""
        now = self.get_clock().now()
        msg = now.to_msg()
        msg.nanosec = (msg.nanosec & 0xFFFF0000) | (idx & 0xFFFF)
        return msg

    def _publish_frame(self):
        if self.frame_idx >= len(self.poses):
            if self.loop:
                self.frame_idx = 0
                self.get_logger().info("Looping back to frame 0")
            else:
                self.get_logger().info("Replay complete.")
                self.timer.cancel()
                raise SystemExit(0)

        idx = self.frame_idx
        pose = self.poses[idx]

        depth = np.load(self.dataset / "depth" / f"{idx:06d}.npy")
        semantic = np.load(self.dataset / "semantic" / f"{idx:06d}.npy")

        stamp = self._stamp_from_index(idx)

        depth_msg = self._make_depth_msg(depth, stamp)
        seg_msg = self._make_seg_msg(semantic, stamp)

        assert depth_msg.header.stamp == seg_msg.header.stamp

        # 1. Publish TF first
        self._publish_tf(pose, stamp)

        # 2. Publish CameraInfo
        self._publish_camera_info(stamp)

        # 3. Publish depth and seg back-to-back
        self.depth_pub.publish(depth_msg)
        self.seg_pub.publish(seg_msg)

        pos = pose["position"]
        self.get_logger().info(
            f"Published frame {idx}/{len(self.poses)} "
            f"stamp={stamp.sec}.{stamp.nanosec:09d} "
            f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        )

        self.frame_idx += 1

    def _publish_tf(self, pose, stamp):
        """Publish odom -> base_link -> camera_link.

        Gazebo TrajectoryRecorder poses are body-to-world:
          - Position: camera world position
          - Rotation: body frame (X-fwd, Y-left, Z-up) to world frame

        SCovoxNode applies kR internally (optical->body), so the TF rotation
        is published as R directly:
          SCovoxNode computes: R_tf @ kR @ p_optical = R @ p_body = p_world  ✓
        """
        pos = [float(v) for v in pose["position"]]
        quat = [float(v) for v in pose["rotation_quat_wxyz"]]  # [w, x, y, z]

        # Publish R directly as TF rotation — no kR^T correction needed
        # because poses are body-to-world, not optical-to-world.
        t_base = TransformStamped()
        t_base.header.stamp = stamp
        t_base.header.frame_id = self.odom_frame
        t_base.child_frame_id = self.base_frame
        t_base.transform.translation.x = pos[0]
        t_base.transform.translation.y = pos[1]
        t_base.transform.translation.z = pos[2]
        t_base.transform.rotation.w = quat[0]
        t_base.transform.rotation.x = quat[1]
        t_base.transform.rotation.y = quat[2]
        t_base.transform.rotation.z = quat[3]

        # Camera link identity — pose is already the camera
        t_cam = TransformStamped()
        t_cam.header.stamp = stamp
        t_cam.header.frame_id = self.base_frame
        t_cam.child_frame_id = self.camera_frame
        t_cam.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform([t_base, t_cam])

    def _publish_camera_info(self, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.width = self.camera["width"]
        msg.height = self.camera["height"]
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        fx = self.camera["fx"]
        fy = self.camera["fy"]
        cx = self.camera["cx"]
        cy = self.camera["cy"]

        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.info_pub.publish(msg)

    def _make_depth_msg(self, depth: np.ndarray, stamp) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.height, msg.width = depth.shape
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = msg.width * 4
        msg.data = depth.astype(np.float32).tobytes()
        return msg

    def _make_seg_msg(self, semantic: np.ndarray, stamp) -> Image:
        h, w = semantic.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for inst_id, color in self.sem_class_map.items():
            mask = semantic == inst_id
            if mask.any():
                rgb[mask] = color

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = rgb.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = GazeboReplayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
