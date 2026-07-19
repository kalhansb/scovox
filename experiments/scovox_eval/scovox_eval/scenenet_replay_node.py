#!/usr/bin/env python3
"""ROS2 node that replays SceneNet RGB-D frames into SCovox.

Reads depth PNGs + semantic label PNGs + poses.txt from the SLIM-VDB
SceneNet data layout and publishes standard ROS2 messages:
  - depth_image       (sensor_msgs/Image, 32FC1 in meters)
  - camera_info       (sensor_msgs/CameraInfo, pinhole intrinsics)
  - segmentation      (sensor_msgs/Image, RGB8 semantic colors)
  - TF: odom -> base_link -> camera_link  (from GT poses)

SceneNet data layout (SLIM-VDB format):
  data_root/train/<seq>/depth/NNNN.png     — uint16, Euclidean ray length in mm
  data_root/train/<seq>/photo/NNNN.jpg     — RGB 320x240
  data_root/train/<seq>/ground_truth_labels/NNNN.png  — uint8 class IDs (0-13)
  data_root/train/<seq>/prediction/NNNN.png           — uint8 predicted class IDs
  data_root/train/<seq>/poses.txt          — 300 lines, 12 floats per line (3x4 cam-to-world)
  data_root/train/<seq>/intrinsics.txt     — fx, fy, cx, cy

Usage:
    ros2 run scovox_eval scenenet_replay \
        --ros-args -p data_root:=/path/to/scenenet \
                   -p sequence:=2 \
                   -p rate_hz:=10.0 \
                   -p use_gt_labels:=true
"""

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


# SceneNet 14 NYUv2 classes -> distinct RGB colors for SCovox semantic input.
# Class 0 = Unknown (black).
# These must match semantic_color_map_keys/classes in the SceneNet launch config.
SCENENET_CLASS_COLORS = {
    0:  (  0,   0,   0),  # Unknown
    1:  (  0,   0, 255),  # Bed
    2:  (233,  89,  48),  # Books
    3:  (  0, 218,   0),  # Ceiling
    4:  (149,   0, 240),  # Chair
    5:  (222, 241,  24),  # Floor
    6:  (255, 206, 206),  # Furniture
    7:  (  0, 224, 229),  # Objects
    8:  (106, 136, 204),  # Picture
    9:  (117,  29,  41),  # Sofa
    10: (240,  35, 235),  # Table
    11: (  0, 167, 156),  # TV
    12: (249, 139,   0),  # Wall
    13: (225, 229, 194),  # Window
}


class SceneNetReplayNode(Node):
    def __init__(self):
        super().__init__("scenenet_replay_node")

        # Parameters
        self.declare_parameter("data_root", "")
        # Declared as string (not int) so we can pass e.g. "0_223" for val
        # trajectories. Pre-iter6 train cells used int IDs ("2") — both work.
        self.declare_parameter("sequence", "2")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("robot_name", "atlas")
        self.declare_parameter("use_gt_labels", True)
        # Step 8 / NEW_EXPERIMENT_PLAN Phase 3 — trajectory split for
        # multi-robot fusion. start_frame defaults to 0 (full sequence,
        # matches pre-Step-8 behaviour byte-for-byte); n_scans defaults
        # to -1 (no cap). For fusion: robot A → start=0 n=200,
        # robot B → start=100 n=200  (50% overlap convention).
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("n_scans", -1)

        data_root = self.get_parameter("data_root").value
        sequence = str(self.get_parameter("sequence").value)
        self.rate_hz = self.get_parameter("rate_hz").value
        self.robot_name = self.get_parameter("robot_name").value
        self.use_gt_labels = self.get_parameter("use_gt_labels").value
        self.start_frame = int(self.get_parameter("start_frame").value)
        self.n_scans = int(self.get_parameter("n_scans").value)

        if not data_root:
            self.get_logger().fatal("data_root parameter is required")
            raise SystemExit(1)

        self.seq_dir = Path(data_root) / "train" / str(sequence)
        if not self.seq_dir.exists():
            self.get_logger().fatal(f"Sequence directory not found: {self.seq_dir}")
            raise SystemExit(1)

        self._load_dataset()

        # Frame IDs
        self.odom_frame = f"{self.robot_name}/odom"
        self.base_frame = f"{self.robot_name}/base_link"
        self.camera_frame = f"{self.robot_name}/camera_link"

        # Publishers
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

        # Playback
        self.frame_idx = max(0, self.start_frame)
        # `stop_frame` is the exclusive upper bound. -1 (default) means
        # "play to the end of the sequence" — preserves legacy behaviour.
        if self.n_scans > 0:
            self.stop_frame = min(len(self.depth_files),
                                  self.frame_idx + self.n_scans)
        else:
            self.stop_frame = len(self.depth_files)
        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self._publish_frame)

        label_src = "GT" if self.use_gt_labels else "prediction"
        self.get_logger().info(
            f"SceneNet replay: frames [{self.frame_idx}, {self.stop_frame}) of "
            f"{len(self.depth_files)} @ {self.rate_hz} Hz "
            f"from {self.seq_dir} (labels: {label_src})"
        )

    def _load_dataset(self):
        """Load intrinsics, poses, and file lists."""
        # --- Intrinsics ---
        self.intrinsics = {}
        with open(self.seq_dir / "intrinsics.txt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, val = line.split(":")
                self.intrinsics[key.strip()] = float(val.strip())
        self.fx = self.intrinsics["fx"]
        self.fy = self.intrinsics["fy"]
        self.cx = self.intrinsics["cx"]
        self.cy = self.intrinsics["cy"]
        self.img_w = 320
        self.img_h = 240

        # --- Poses (3x4 camera-to-world matrices) ---
        self.poses = []
        with open(self.seq_dir / "poses.txt") as f:
            for line in f:
                vals = [float(v) for v in line.strip().split()]
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                self.poses.append(T)

        # --- File lists ---
        depth_dir = self.seq_dir / "depth"
        self.depth_files = sorted(depth_dir.glob("*.png"))

        if self.use_gt_labels:
            label_dir = self.seq_dir / "ground_truth_labels"
        else:
            label_dir = self.seq_dir / "prediction"
        self.label_files = sorted(label_dir.glob("*.png"))

        assert len(self.depth_files) == len(self.label_files) == len(self.poses), (
            f"Mismatch: {len(self.depth_files)} depth, "
            f"{len(self.label_files)} labels, {len(self.poses)} poses"
        )

        # Pre-compute per-pixel z-conversion factor.
        # SceneNet depth is Euclidean ray length d. Z-depth = d / sqrt(xn^2 + yn^2 + 1)
        # where xn = (v - cx)/fx, yn = (u - cy)/fy.
        u_coords = np.arange(self.img_h).reshape(-1, 1)  # rows
        v_coords = np.arange(self.img_w).reshape(1, -1)  # cols
        x_norm = (v_coords - self.cx) / self.fx
        y_norm = (u_coords - self.cy) / self.fy
        self._ray_to_z = (1.0 / np.sqrt(x_norm**2 + y_norm**2 + 1.0)).astype(np.float32)

        self.get_logger().info(
            f"Loaded {len(self.poses)} poses, "
            f"intrinsics: fx={self.fx:.2f} fy={self.fy:.2f} cx={self.cx:.1f} cy={self.cy:.1f}"
        )

    def _stamp_from_index(self, idx: int) -> TimeMsg:
        """Use current clock time, encode frame index in low bits."""
        now = self.get_clock().now()
        msg = now.to_msg()
        msg.nanosec = (msg.nanosec & 0xFFFF0000) | (idx & 0xFFFF)
        return msg

    def _publish_frame(self):
        # `stop_frame` is min(start + n_scans, len) and trims both the
        # legacy "play-to-end" behaviour and the new bounded-window mode
        # into a single termination predicate.
        if self.frame_idx >= self.stop_frame:
            self.get_logger().info(
                f"SceneNet replay complete (last frame={self.frame_idx - 1}, "
                f"stop={self.stop_frame})."
            )
            self.timer.cancel()
            raise SystemExit(0)

        idx = self.frame_idx

        # Load depth (uint16 mm Euclidean) -> float32 Z-depth metres
        from PIL import Image as PILImage
        depth_raw = np.array(PILImage.open(str(self.depth_files[idx])))
        depth_m = depth_raw.astype(np.float32) / 1000.0  # mm -> m (Euclidean)
        depth_z = depth_m * self._ray_to_z  # convert to Z-depth

        # Load semantic labels
        labels = np.array(PILImage.open(str(self.label_files[idx])))

        stamp = self._stamp_from_index(idx)

        depth_msg = self._make_depth_msg(depth_z, stamp)
        seg_msg = self._make_seg_msg(labels, stamp)

        # 1. TF
        self._publish_tf(self.poses[idx], stamp)
        # 2. CameraInfo
        self._publish_camera_info(stamp)
        # 3. Depth + Seg
        self.depth_pub.publish(depth_msg)
        self.seg_pub.publish(seg_msg)

        t = self.poses[idx][:3, 3]
        if idx % 25 == 0:
            self.get_logger().info(
                f"Frame {idx}/{len(self.poses)} "
                f"pos=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})"
            )

        self.frame_idx += 1

    def _publish_tf(self, T_cam2world: np.ndarray, stamp):
        """Publish odom -> base_link -> camera_link.

        SceneNet poses are camera-to-world. Same as replica_replay_node
        camera_poses=True mode: we apply kR^T so that SCovoxNode's internal
        kR multiplication recovers the original camera-to-world rotation.
        """
        R = T_cam2world[:3, :3]
        t = T_cam2world[:3, 3]

        # kR converts optical (Z-fwd, X-right, Y-down) -> body (X-fwd, Y-left, Z-up)
        kR = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        R_tf = R @ kR.T
        q_tf = _rotation_matrix_to_quaternion(R_tf)

        t_base = TransformStamped()
        t_base.header.stamp = stamp
        t_base.header.frame_id = self.odom_frame
        t_base.child_frame_id = self.base_frame
        t_base.transform.translation.x = float(t[0])
        t_base.transform.translation.y = float(t[1])
        t_base.transform.translation.z = float(t[2])
        t_base.transform.rotation.w = q_tf[0]
        t_base.transform.rotation.x = q_tf[1]
        t_base.transform.rotation.y = q_tf[2]
        t_base.transform.rotation.z = q_tf[3]

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
        msg.width = self.img_w
        msg.height = self.img_h
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
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

    def _make_seg_msg(self, labels: np.ndarray, stamp) -> Image:
        h, w = labels.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in SCENENET_CLASS_COLORS.items():
            mask = labels == cls_id
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


def main(args=None):
    rclpy.init(args=args)
    node = SceneNetReplayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
