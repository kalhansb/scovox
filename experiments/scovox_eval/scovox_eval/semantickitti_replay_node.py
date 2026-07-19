#!/usr/bin/env python3
"""ROS2 node that replays SemanticKITTI sequences into SCovox.

Reads Velodyne .bin point clouds + .label semantic annotations + poses
and publishes them as:
  - PointCloud2 with fields: x, y, z, intensity, semantic_label
  - TF: {robot}/odom -> {robot}/base_link (from GT poses)

SemanticKITTI format:
  - velodyne/*.bin: N×4 float32 (x, y, z, remission)
  - labels/*.label: N uint32 (lower 16 bits = semantic ID, upper 16 = instance)
  - poses/XX.txt: one 3×4 row-major transform per line (cam0-to-world)
  - calib.txt: Tr = velodyne-to-cam0 rigid transform

Usage:
    ros2 run scovox_eval semantickitti_replay \
        --ros-args -p dataset_path:=/path/to/semantickitti/dataset \
                   -p sequence:=08 \
                   -p rate_hz:=10.0 \
                   -p robot_name:=atlas
"""

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as TimeMsg

# SemanticKITTI 34 raw IDs -> 20 evaluation classes (0=unlabeled, 1-19=semantic)
# Standard mapping from semantic-kitti-api
LEARNING_MAP = {
    0: 0,       # unlabeled
    1: 0,       # outlier
    10: 1,      # car
    11: 2,      # bicycle
    13: 5,      # bus
    15: 3,      # motorcycle
    16: 5,      # on-rails
    18: 4,      # truck
    20: 5,      # other-vehicle
    30: 6,      # person
    31: 7,      # bicyclist
    32: 8,      # motorcyclist
    40: 9,      # road
    44: 10,     # parking
    48: 11,     # sidewalk
    49: 12,     # other-ground
    50: 13,     # building
    51: 14,     # fence
    52: 0,      # other-structure
    60: 15,     # lane-marking
    70: 16,     # vegetation
    71: 17,     # trunk
    72: 18,     # terrain
    80: 19,     # pole
    81: 0,      # traffic-sign
    99: 0,      # other-object
    252: 1,     # moving-car
    253: 7,     # moving-bicyclist
    254: 6,     # moving-person
    255: 8,     # moving-motorcyclist
    256: 5,     # moving-on-rails
    257: 5,     # moving-bus
    258: 4,     # moving-truck
    259: 5,     # moving-other-vehicle
}

# Build a fast lookup table (max raw ID is 259)
_LABEL_LUT = np.zeros(260, dtype=np.uint16)
for raw_id, mapped_id in LEARNING_MAP.items():
    _LABEL_LUT[raw_id] = mapped_id


class SemanticKITTIReplayNode(Node):
    def __init__(self):
        super().__init__("semantickitti_replay_node")

        self.declare_parameter("dataset_path", "")
        self.declare_parameter("sequence", 8)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("robot_name", "atlas")
        self.declare_parameter("loop", False)
        self.declare_parameter("max_range", 80.0)
        self.declare_parameter("min_range", 1.0)
        # Labels subdir (relative to sequences/<seq>/). Override to "predictions"
        # to feed PolarSeg / other network predictions instead of GT labels.
        self.declare_parameter("labels_subdir", "labels")
        # Max number of scans to replay (paper protocol: 100 for SLIM-VDB parity).
        # -1 means "use all available scans".
        self.declare_parameter("n_scans", -1)
        # Soft-prob passthrough: when True, skip the range mask so the
        # PointCloud2 publishes points in raw .bin order. scovox_node
        # then indexes into the matching .topk file by point index. The
        # C++ side still applies the same range filter, so the integration
        # results match the masked path bitwise.
        self.declare_parameter("soft_prob_passthrough", False)

        dataset_path = self.get_parameter("dataset_path").value
        self.sequence = f"{int(self.get_parameter('sequence').value):02d}"
        self.rate_hz = self.get_parameter("rate_hz").value
        self.robot_name = self.get_parameter("robot_name").value
        self.loop = self.get_parameter("loop").value
        self.max_range = self.get_parameter("max_range").value
        self.min_range = self.get_parameter("min_range").value
        self.labels_subdir = self.get_parameter("labels_subdir").value
        self.soft_prob_passthrough = self.get_parameter("soft_prob_passthrough").value
        self.n_scans_cap = int(self.get_parameter("n_scans").value)

        if not dataset_path:
            self.get_logger().fatal("dataset_path parameter is required")
            raise SystemExit(1)

        self.dataset = Path(dataset_path)
        self._load_dataset()

        # Frame IDs
        self.odom_frame = f"{self.robot_name}/odom"
        self.base_frame = f"{self.robot_name}/base_link"
        self.sensor_frame = f"{self.robot_name}/velodyne_link"

        # Publisher — RELIABLE QoS so no messages are dropped
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
        )

        self.pc_pub = self.create_publisher(
            PointCloud2,
            f"/{self.robot_name}/velodyne_points",
            reliable_qos,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        # Playback state
        self.frame_idx = 0
        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self._publish_frame)

        self.get_logger().info(
            f"SemanticKITTI replay: seq {self.sequence}, "
            f"{self.n_scans} frames at {self.rate_hz} Hz"
        )

    def _load_dataset(self):
        """Load poses, calibration, and enumerate scan files."""
        seq_dir = self.dataset / "sequences" / self.sequence
        self.velo_dir = seq_dir / "velodyne"
        self.label_dir = seq_dir / self.labels_subdir

        if not self.label_dir.is_dir():
            self.get_logger().fatal(
                f"labels_subdir '{self.labels_subdir}' not found under {seq_dir}")
            raise SystemExit(2)

        # Calibration: extract Tr (velodyne -> cam0)
        calib_file = seq_dir / "calib.txt"
        self.Tr = np.eye(4, dtype=np.float64)
        with open(calib_file) as f:
            for line in f:
                if line.startswith("Tr:"):
                    vals = [float(x) for x in line.strip().split()[1:]]
                    self.Tr[:3, :] = np.array(vals).reshape(3, 4)
                    break

        # Poses: cam0-to-world, one 3×4 row-major per line. Support both the
        # per-sequence location (sequences/<seq>/poses.txt) and the legacy
        # dataset-root location (poses/<seq>.txt).
        per_seq_poses = seq_dir / "poses.txt"
        legacy_poses = self.dataset / "poses" / f"{self.sequence}.txt"
        poses_file = per_seq_poses if per_seq_poses.exists() else legacy_poses
        self.poses = []
        with open(poses_file) as f:
            for line in f:
                vals = [float(x) for x in line.strip().split()]
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals).reshape(3, 4)
                self.poses.append(T)

        # Precompute velodyne-to-world for each frame: T_world_velo = T_world_cam0 @ Tr
        self.T_world_velo = [pose @ self.Tr for pose in self.poses]

        # Enumerate scans
        self.scan_files = sorted(self.velo_dir.glob("*.bin"))
        self.n_scans = len(self.scan_files)

        if self.n_scans != len(self.poses):
            self.get_logger().warn(
                f"Scan count ({self.n_scans}) != pose count ({len(self.poses)})"
            )
            self.n_scans = min(self.n_scans, len(self.poses))
        if self.n_scans_cap > 0:
            self.n_scans = min(self.n_scans, self.n_scans_cap)
            self.get_logger().info(f"n_scans capped to {self.n_scans}")

        self.get_logger().info(
            f"Loaded {self.n_scans} scans, Tr shape {self.Tr.shape}"
        )

    def _stamp_from_index(self, idx: int) -> TimeMsg:
        """Encode frame index in low bits of nanosec (same pattern as replica replay)."""
        now = self.get_clock().now()
        msg = now.to_msg()
        msg.nanosec = (msg.nanosec & 0xFFFF0000) | (idx & 0xFFFF)
        return msg

    def _publish_frame(self):
        if self.frame_idx >= self.n_scans:
            if self.loop:
                self.frame_idx = 0
                self.get_logger().info("Looping back to frame 0")
            else:
                self.get_logger().info("Replay complete.")
                self.timer.cancel()
                raise SystemExit(0)

        idx = self.frame_idx
        stem = f"{idx:06d}"

        # Load velodyne points (N, 4): x, y, z, remission
        points = np.fromfile(self.velo_dir / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)

        # Load labels (N,) uint32: lower 16 = semantic, upper 16 = instance
        labels_raw = np.fromfile(self.label_dir / f"{stem}.label", dtype=np.uint32)
        semantic_ids = (labels_raw & 0xFFFF).astype(np.uint16)

        # Remap to evaluation classes
        # Clip to LUT range, map unknown IDs to 0
        safe_ids = np.minimum(semantic_ids, 259)
        mapped_labels = _LABEL_LUT[safe_ids]

        # Range filter — skipped in soft-prob passthrough so the cloud point
        # ordering matches the raw .bin/.topk row ordering (scovox_node will
        # apply the same range filter on the C++ side).
        if not getattr(self, "soft_prob_passthrough", False):
            ranges = np.linalg.norm(points[:, :3], axis=1)
            mask = (ranges >= self.min_range) & (ranges <= self.max_range)
            points = points[mask]
            mapped_labels = mapped_labels[mask]

        stamp = self._stamp_from_index(idx)

        # 1. Publish TF
        self._publish_tf(idx, stamp)

        # 2. Publish PointCloud2
        pc_msg = self._make_pointcloud2(points, mapped_labels, stamp)
        self.pc_pub.publish(pc_msg)

        T = self.T_world_velo[idx]
        pos = T[:3, 3]
        self.get_logger().info(
            f"Published frame {idx}/{self.n_scans} "
            f"pts={len(points)} stamp={stamp.sec}.{stamp.nanosec:09d} "
            f"pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        )

        self.frame_idx += 1

    def _publish_tf(self, idx: int, stamp):
        """Publish odom -> base_link and base_link -> velodyne_link."""
        T = self.T_world_velo[idx]
        R = T[:3, :3]
        t = T[:3, 3]
        q = _rotation_matrix_to_quaternion(R)

        # odom -> base_link = velodyne pose in world
        t_base = TransformStamped()
        t_base.header.stamp = stamp
        t_base.header.frame_id = self.odom_frame
        t_base.child_frame_id = self.base_frame
        t_base.transform.translation.x = t[0]
        t_base.transform.translation.y = t[1]
        t_base.transform.translation.z = t[2]
        t_base.transform.rotation.w = q[0]
        t_base.transform.rotation.x = q[1]
        t_base.transform.rotation.y = q[2]
        t_base.transform.rotation.z = q[3]

        # base_link -> velodyne_link = identity (sensor at base)
        t_velo = TransformStamped()
        t_velo.header.stamp = stamp
        t_velo.header.frame_id = self.base_frame
        t_velo.child_frame_id = self.sensor_frame
        t_velo.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform([t_base, t_velo])

    def _make_pointcloud2(self, points: np.ndarray, labels: np.ndarray, stamp) -> PointCloud2:
        """Build PointCloud2 with fields: x, y, z, intensity, semantic_label."""
        n = len(points)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="semantic_label", offset=16, datatype=PointField.UINT16, count=1),
        ]
        # Use structured numpy array for fast serialization
        # 4 floats (16 bytes) + 1 uint16 (2 bytes) = 18 bytes, pad to 20 for alignment
        point_step = 20
        fields.append(
            PointField(name="_pad", offset=18, datatype=PointField.UINT16, count=1),
        )
        dt = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
            ("semantic_label", "<u2"), ("_pad", "<u2"),
        ])
        arr = np.empty(n, dtype=dt)
        arr["x"] = points[:, 0]
        arr["y"] = points[:, 1]
        arr["z"] = points[:, 2]
        arr["intensity"] = points[:, 3]
        arr["semantic_label"] = labels
        arr["_pad"] = 0

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.sensor_frame
        msg.height = 1
        msg.width = n
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * n
        msg.data = arr.tobytes()
        msg.is_dense = True
        return msg


def _rotation_matrix_to_quaternion(R: np.ndarray):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    import math
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
    node = SemanticKITTIReplayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
