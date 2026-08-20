#!/usr/bin/env python3
"""ROS2 node that replays SceneNN (HKUST) RGB-D frames into SCovox.

Reads the frame folders produced by SceneNN's `playback` tool plus the
dataset's trajectory.log, and publishes the same message contract as
scenenet_replay_node.py:
  - depth_image   (sensor_msgs/Image, 32FC1 metres)
  - camera_info   (sensor_msgs/CameraInfo, pinhole intrinsics from asus.ini)
  - segmentation  (sensor_msgs/Image, RGB8 NYU-40 colours)
  - TF: odom -> base_link -> camera_link  (from trajectory.log)

Layout:
  <scene>/frames/depth/depth%05d.png   uint16 mm, 640x480, 1-indexed
  <scene>/frames/image/image%05d.png   RGB8
  <scene>/labels/label%05d.png         uint8 nyu_class, from scenenn_make_labels.py
  <scene>/trajectory.log               Redwood .log, 4x4 camera-to-world
  asus.ini                             fx/fy/cx/cy

Differences from the SceneNet path, both verified on scene 016 rather than
assumed (see scenenn_check_relpose.py):
  * OpenNI depth is already perspective Z-depth, so there is NO ray-length
    -> Z rescaling. Applying SceneNet's correction here would shrink depth
    by up to 8% at the image corners.
  * The extracted depth PNG needs no vertical flip, and the optical frame is
    the standard (x right, y down, z forward) one. The relative-pose check
    scores that combination at 1.3 cm against 2-7 cm for every alternative.
  * trajectory.log has fewer poses (1300) than playback emits frames (1364),
    so replay stops at min(frames, poses).

Usage:
    python3 -m scovox_eval.scenenn_replay_node --ros-args \
        -p scene_dir:=/home/jetsondevkit/datasets/scenenn/016 \
        -p intrinsics:=/home/jetsondevkit/datasets/scenenn/asus.ini \
        -p rate_hz:=10.0
"""

import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as TimeMsg

try:
    from scovox_eval.scenenn_data import (
        load_intrinsics, load_trajectory, load_depth_m, load_nyu_colormap, NYU_NUM_CLASSES,
    )
except ImportError:  # running straight from the scripts dir
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scovox_eval.scenenn_data import (
        load_intrinsics, load_trajectory, load_depth_m, load_nyu_colormap, NYU_NUM_CLASSES,
    )


class SceneNNReplayNode(Node):
    def __init__(self):
        super().__init__("scenenn_replay_node")

        self.declare_parameter("scene_dir", "")
        self.declare_parameter("intrinsics", "")
        self.declare_parameter("nyu_colormap", "")
        self.declare_parameter("labels_dir", "")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("robot_name", "atlas")
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("n_scans", -1)
        # depth_files[i] pairs with poses[i + pose_offset]. 0 and 1 score within
        # 0.04 cm of each other on scene 016; 0 is the documented convention.
        self.declare_parameter("pose_offset", 0)

        scene_dir = self.get_parameter("scene_dir").value
        if not scene_dir:
            self.get_logger().fatal("scene_dir parameter is required")
            raise SystemExit(1)
        self.scene = Path(scene_dir)

        intr = self.get_parameter("intrinsics").value or str(self.scene.parent / "asus.ini")
        cmap = self.get_parameter("nyu_colormap").value or str(self.scene.parent / "nyu_color.xml")
        labels_dir = self.get_parameter("labels_dir").value or str(self.scene / "labels")

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.robot_name = self.get_parameter("robot_name").value
        self.start_frame = int(self.get_parameter("start_frame").value)
        self.n_scans = int(self.get_parameter("n_scans").value)
        self.pose_offset = int(self.get_parameter("pose_offset").value)

        self.K = load_intrinsics(intr)
        self.poses = load_trajectory(self.scene / "trajectory.log")
        self.depth_files = sorted((self.scene / "frames" / "depth").glob("depth*.png"))
        self.label_files = sorted(Path(labels_dir).glob("label*.png")) if Path(labels_dir).is_dir() else []
        self.palette = load_nyu_colormap(cmap) if Path(cmap).exists() else np.zeros((NYU_NUM_CLASSES, 3), np.uint8)

        if not self.depth_files:
            self.get_logger().fatal(f"no depth frames under {self.scene / 'frames' / 'depth'}")
            raise SystemExit(1)

        # Poses are the binding constraint: playback emits more frames than
        # trajectory.log covers, and integrating a frame against a pose that
        # does not exist would silently drop or misplace it.
        self.n_frames = min(len(self.depth_files), len(self.poses) - self.pose_offset)
        if self.label_files:
            self.n_frames = min(self.n_frames, len(self.label_files))
        else:
            self.get_logger().warn(
                f"no label PNGs in {labels_dir} -- publishing 'unknown' seg; "
                "run scenenn_make_labels.py for semantics")

        self.odom_frame = f"{self.robot_name}/odom"
        self.base_frame = f"{self.robot_name}/base_link"
        self.camera_frame = f"{self.robot_name}/camera_link"

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1000)
        self.depth_pub = self.create_publisher(Image, f"/{self.robot_name}/rgbd_camera_depth_image", qos)
        self.seg_pub = self.create_publisher(Image, f"/{self.robot_name}/segmentation/colored", qos)
        self.info_pub = self.create_publisher(CameraInfo, f"/{self.robot_name}/rgbd_camera_info", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.frame_idx = max(0, self.start_frame)
        self._warmup_ticks = None  # None = still waiting for subscriber match
        self.stop_frame = (min(self.n_frames, self.frame_idx + self.n_scans)
                           if self.n_scans > 0 else self.n_frames)
        self._seg_unknown = None

        self.timer = self.create_timer(1.0 / self.rate_hz, self._publish_frame)
        self.get_logger().info(
            f"SceneNN replay {self.scene.name}: frames [{self.frame_idx}, {self.stop_frame}) "
            f"of {len(self.depth_files)} extracted / {len(self.poses)} poses @ {self.rate_hz} Hz "
            f"| fx={self.K['fx']:.2f} {self.K['width']}x{self.K['height']} "
            f"| labels: {'yes' if self.label_files else 'NO'}")

    def _stamp_from_index(self, idx):
        msg = self.get_clock().now().to_msg()
        msg.nanosec = (msg.nanosec & 0xFFFF0000) | (idx & 0xFFFF)
        return msg

    def _publish_frame(self):
        # Hold playback until every image subscriber has matched, then two more
        # seconds of grace for /tf. Without this, frames published during DDS
        # discovery are silently lost and the COUNT lost varies run to run —
        # which makes any cross-run map comparison (bit-identity A/Bs) read as
        # a diff even when the mapping code is unchanged.
        if self._warmup_ticks is None:
            if (self.depth_pub.get_subscription_count() > 0
                    and self.seg_pub.get_subscription_count() > 0
                    and self.info_pub.get_subscription_count() > 0):
                self._warmup_ticks = int(2.0 * self.rate_hz)
            return
        if self._warmup_ticks > 0:
            self._warmup_ticks -= 1
            return

        if self.frame_idx >= self.stop_frame:
            self.get_logger().info(f"SceneNN replay complete (last frame={self.frame_idx - 1}).")
            self.timer.cancel()
            raise SystemExit(0)

        idx = self.frame_idx
        # OpenNI depth is already Z-depth; no flip, no ray->Z rescale.
        depth = load_depth_m(self.depth_files[idx], flip_vertical=False)
        stamp = self._stamp_from_index(idx)

        self._publish_tf(self.poses[idx + self.pose_offset], stamp)
        self._publish_camera_info(stamp)
        self.depth_pub.publish(self._make_depth_msg(depth, stamp))
        self.seg_pub.publish(self._make_seg_msg(idx, depth.shape, stamp))

        if idx % 100 == 0:
            t = self.poses[idx + self.pose_offset][:3, 3]
            self.get_logger().info(
                f"Frame {idx}/{self.stop_frame} pos=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
        self.frame_idx += 1

    def _publish_tf(self, T_cam2world, stamp):
        R, t = T_cam2world[:3, :3], T_cam2world[:3, 3]
        # SCovoxNode multiplies the looked-up rotation by kR (optical->body),
        # so pre-apply kR^T here and the two cancel, exactly as the SceneNet
        # and Replica camera-pose paths do.
        kR = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        q = _rotation_matrix_to_quaternion(R @ kR.T)

        tb = TransformStamped()
        tb.header.stamp = stamp
        tb.header.frame_id = self.odom_frame
        tb.child_frame_id = self.base_frame
        tb.transform.translation.x = float(t[0])
        tb.transform.translation.y = float(t[1])
        tb.transform.translation.z = float(t[2])
        tb.transform.rotation.w, tb.transform.rotation.x = q[0], q[1]
        tb.transform.rotation.y, tb.transform.rotation.z = q[2], q[3]

        tc = TransformStamped()
        tc.header.stamp = stamp
        tc.header.frame_id = self.base_frame
        tc.child_frame_id = self.camera_frame
        tc.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform([tb, tc])

    def _publish_camera_info(self, stamp):
        m = CameraInfo()
        m.header.stamp = stamp
        m.header.frame_id = self.camera_frame
        m.width, m.height = self.K["width"], self.K["height"]
        m.distortion_model = "plumb_bob"
        m.d = [0.0] * 5
        fx, fy, cx, cy = self.K["fx"], self.K["fy"], self.K["cx"], self.K["cy"]
        m.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        m.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(m)

    def _make_depth_msg(self, depth, stamp):
        m = Image()
        m.header.stamp = stamp
        m.header.frame_id = self.camera_frame
        m.height, m.width = depth.shape
        m.encoding = "32FC1"
        m.is_bigendian = 0
        m.step = m.width * 4
        m.data = depth.astype(np.float32).tobytes()
        return m

    def _make_seg_msg(self, idx, shape, stamp):
        h, w = shape
        if self.label_files:
            from PIL import Image as PILImage
            labels = np.array(PILImage.open(str(self.label_files[idx])))
            rgb = self.palette[np.clip(labels, 0, NYU_NUM_CLASSES - 1)]
        else:
            if self._seg_unknown is None:
                self._seg_unknown = np.zeros((h, w, 3), dtype=np.uint8)
            rgb = self._seg_unknown

        m = Image()
        m.header.stamp = stamp
        m.header.frame_id = self.camera_frame
        m.height, m.width = h, w
        m.encoding = "rgb8"
        m.is_bigendian = 0
        m.step = w * 3
        m.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        return m


def _rotation_matrix_to_quaternion(R):
    """3x3 rotation -> quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]


def main(args=None):
    rclpy.init(args=args)
    node = SceneNNReplayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
