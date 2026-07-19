#!/usr/bin/env python3
"""ROS2 node that replays pre-rendered Replica frames into SCovox.

Supports two dataset formats (auto-detected):

  NEW (replica_niceslam, from render_replica_dataset.py):
    depth/000000.png     uint16 PNG (meters * 6553.5)
    semantic/000000.png  uint16 PNG (per-pixel object_id)
    poses.txt            one flattened 4x4 cam-to-world per line
    info_semantic.json   object_id -> class_name mapping
    cam_params.json      in parent directory

  OLD (replica_rendered, from niceslam_to_scovox.py):
    depth/000000.npy     float32 (meters)
    semantic/000000.npy  int32 (instance_id)
    poses.json           list of {position, rotation_quat_wxyz, T_world_camera}
    camera.json          per-scene intrinsics
    semantic_class_map.json  instance_id -> category info

Publishes:
  - depth_image       (sensor_msgs/Image, 32FC1 in meters)
  - camera_info       (sensor_msgs/CameraInfo, pinhole intrinsics)
  - segmentation      (sensor_msgs/Image, RGB8 semantic colors)
  - TF: odom -> camera_link  (from GT poses)

Usage:
    ros2 run scovox_eval replica_replay \
        --ros-args -p dataset_path:=/path/to/replica_niceslam/room0 \
                   -p rate_hz:=10.0 \
                   -p robot_name:=atlas
"""

import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as TimeMsg


# Map category names to distinct RGB colors for SCovox semantic input.
# These colors must match semantic_color_map_keys/classes in the SCovox launch config.
# Class 0 = unknown (black). Unmapped categories also get black.
#
# Format: "category_substring": (R, G, B) — matched case-insensitively.
# The RGB value is packed as (R<<16 | G<<8 | B) for the SCovox color map key.
#
# Replica indoor categories          R     G     B
CATEGORY_COLORS = {
    "wall":       (174,  199,  232),  # light blue (shared with flatforest)
    "floor":      (152,  223,  138),  # light green
    "ceiling":    (255,  152,  150),  # salmon
    "door":       (197,  176,  213),  # lavender
    "window":     (196,  156,  148),  # dusty rose
    "chair":      (255,  127,   14),  # orange
    "table":      (140,   86,   75),  # brown
    "sofa":       (227,  119,  194),  # pink
    "bed":        (148,  103,  189),  # purple
    "cushion":    (255,  187,  120),  # peach (common in Replica)
    "lamp":       (219,  219,  141),  # khaki
    "cabinet":    (199,  199,  199),  # silver
    "blinds":     (188,  189,   34),  # olive
    "book":       (158,  218,  229),  # sky blue
    "picture":    ( 23,  190,  207),  # teal
    "plant":      ( 44,  160,   44),  # dark green (matches indoor-plant too)
    "rug":        (214,   39,   40),  # red
    "pillar":     (127,  127,  212),  # periwinkle
    # Flatforest / simulation categories
    "ground":     (152,  223,  138),  # light green (same as floor)
    "deciduous":  ( 34,  139,   34),  # forest green
    "conifer":    (  0,  100,    0),  # dark green
    "rock":       (128,  128,  128),  # gray
    "undergrowth":(107,  142,   35),  # olive drab
    "structure":  (188,  143,  143),  # rosy brown
    "human":      (255,    0,    0),  # red
}


class ReplicaReplayNode(Node):
    def __init__(self):
        super().__init__("replica_replay_node")

        # Parameters
        self.declare_parameter("dataset_path", "")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("robot_name", "atlas")
        self.declare_parameter("loop", False)
        self.declare_parameter("camera_poses", False)  # True if poses are camera-to-world (NICE-SLAM format)
        # Semantic label subfolder. Override to use predicted labels instead of
        # the default GT labels. E.g. "semantic_gt_fixed", "semantic_m2f_ade".
        self.declare_parameter("semantic_subdir", "semantic")
        # Max frames to replay (-1 = all).
        self.declare_parameter("n_scans", -1)
        # First frame index to publish (0 = from beginning). Lets a second replay
        # start mid-trajectory for two-robot fan-out (E2.1). The published frame_id
        # in the timestamp's low 16 bits stays absolute, so scovox topk lookups by
        # frame_id still resolve to the original on-disk file index.
        self.declare_parameter("start_frame", 0)

        dataset_path = self.get_parameter("dataset_path").value
        self.rate_hz = self.get_parameter("rate_hz").value
        self.robot_name = self.get_parameter("robot_name").value
        self.loop = self.get_parameter("loop").value
        self.camera_poses = self.get_parameter("camera_poses").value
        self.semantic_subdir = self.get_parameter("semantic_subdir").value
        self.n_scans_cap = int(self.get_parameter("n_scans").value)
        self.start_frame = int(self.get_parameter("start_frame").value)

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
        self.frame_idx = self.start_frame
        self.start_time = self.get_clock().now()
        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self._publish_frame)

        self.get_logger().info(
            f"Replica replay: {len(self.poses)} frames at {self.rate_hz} Hz "
            f"from {self.dataset}"
        )

    def _load_dataset(self):
        """Load poses, camera intrinsics, and semantic mapping. Auto-detect format."""
        # Detect format: new has poses.txt, old has poses.json
        self.new_format = (self.dataset / "poses.txt").exists()

        if self.new_format:
            self._load_new_format()
        else:
            self._load_old_format()

        # start_frame + n_scans → window [start_frame, end_frame_excl). Don't
        # slice self.poses — keep absolute indexing so file paths and the
        # frame_id encoded in the timestamp both stay aligned with on-disk
        # depth/semantic/topk filenames.
        if self.start_frame < 0 or self.start_frame >= len(self.poses):
            self.get_logger().fatal(
                f"start_frame={self.start_frame} out of range [0,{len(self.poses)})")
            raise SystemExit(1)
        if self.n_scans_cap > 0:
            self.end_frame_excl = min(len(self.poses), self.start_frame + self.n_scans_cap)
        else:
            self.end_frame_excl = len(self.poses)
        self.get_logger().info(
            f"replay window: frames [{self.start_frame},{self.end_frame_excl}) "
            f"of {len(self.poses)} total")

        self.get_logger().info(
            f"Loaded {len(self.poses)} poses, "
            f"{len(self.sem_class_map)} semantic mappings "
            f"(format={'new-png' if self.new_format else 'old-npy'}, "
            f"semantic_subdir='{self.semantic_subdir}')"
        )

    def _load_new_format(self):
        """Load replica_niceslam format (PNG depth/semantic, poses.txt, info_semantic.json)."""
        # Poses: one flattened 4x4 per line
        self.poses = []
        with open(self.dataset / "poses.txt") as f:
            for line in f:
                vals = [float(v) for v in line.strip().split()]
                if len(vals) == 16:
                    T = np.array(vals).reshape(4, 4)
                    pos = T[:3, 3].tolist()
                    quat_wxyz = _rotation_matrix_to_quaternion(T[:3, :3])
                    self.poses.append({
                        "position": pos,
                        "rotation_quat_wxyz": quat_wxyz,
                        "T_world_camera": T.tolist(),
                    })

        # Camera params: look in current dir first, then parent
        cam_path = self.dataset / "cam_params.json"
        if not cam_path.exists():
            cam_path = self.dataset.parent / "cam_params.json"
        with open(cam_path) as f:
            cam = json.load(f)["camera"]
        self.camera = {
            "fx": cam["fx"], "fy": cam["fy"],
            "cx": cam["cx"], "cy": cam["cy"],
            "width": cam["w"], "height": cam["h"],
        }
        self._depth_scale = cam.get("scale", 6553.5)

        # Semantic mapping: build pixel-value → RGB color lookup.
        # When semantic_subdir == "semantic", pixel values are object/instance IDs
        # and we map through info_semantic.json objects[] (obj_id → class_name → color).
        # When semantic_subdir is anything else (e.g. "semantic_gt_fixed",
        # "semantic_m2f_ade"), pixel values are CLASS IDs directly and we map
        # through info_semantic.json classes[] (class_id → class_name → color).
        self.sem_class_map = {}
        info_path = self.dataset / "info_semantic.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            # Match longer SCovox category names first so e.g. "indoor-plant"
            # hits "plant" instead of "door", and "wall-cabinet" hits "cabinet"
            # instead of "wall". Substring match is kept for robustness.
            sorted_cats = sorted(CATEGORY_COLORS.items(), key=lambda kv: -len(kv[0]))
            if self.semantic_subdir == "semantic":
                # Original format: object/instance IDs
                for obj in info.get("objects", []):
                    obj_id = obj["id"]
                    class_name = obj.get("class_name", "").lower()
                    for key, color in sorted_cats:
                        if key in class_name:
                            self.sem_class_map[obj_id] = color
                            break
            elif "ade" in self.semantic_subdir:
                # Mask2Former ADE-150 predictions. Pixel values are ADE class
                # ID + 1 (so 0 stays as void). Use the bundled ADE-150 label
                # names rather than Replica's classes list.
                ade_path = Path(__file__).resolve().parent.parent / "scripts" / "ade150_labels.json"
                if not ade_path.exists():
                    # Fallback: bundled next to this node.
                    ade_path = Path(__file__).resolve().parent / "ade150_labels.json"
                with open(ade_path) as f:
                    ade = json.load(f)  # {"0":"wall","1":"building",...}
                for aid_str, name in ade.items():
                    aid = int(aid_str)
                    stored = aid + 1  # m2f PNG shift
                    lname = name.lower()
                    parts = [p.strip() for p in lname.replace(";", ",").split(",")]
                    for p in parts:
                        matched = False
                        for key, color in sorted_cats:
                            if key in p:
                                self.sem_class_map[stored] = color
                                matched = True
                                break
                        if matched:
                            break
            else:
                # Class-ID-direct format (semantic_gt_fixed, etc.): Replica class IDs.
                for cls in info.get("classes", []):
                    cls_id = cls["id"]
                    class_name = cls.get("name", "").lower()
                    for key, color in sorted_cats:
                        if key in class_name:
                            self.sem_class_map[cls_id] = color
                            break

    def _load_old_format(self):
        """Load replica_rendered format (NPY depth/semantic, poses.json, camera.json)."""
        with open(self.dataset / "poses.json") as f:
            self.poses = json.load(f)

        with open(self.dataset / "camera.json") as f:
            self.camera = json.load(f)
        self._depth_scale = None  # not needed, depth is float32

        self.sem_class_map = {}
        sem_map_path = self.dataset / "semantic_class_map.json"
        if sem_map_path.exists():
            with open(sem_map_path) as f:
                raw = json.load(f)
            for inst_id_str, info in raw.items():
                cat_name = info.get("category_name", "").lower()
                for key, color in CATEGORY_COLORS.items():
                    if key in cat_name:
                        self.sem_class_map[int(inst_id_str)] = color
                        break

    def _stamp_from_index(self, idx: int) -> TimeMsg:
        """Use current wall-clock time, encode frame index in low bits of nanosec."""
        now = self.get_clock().now()
        msg = now.to_msg()
        # Encode frame index in lowest 16 bits of nanosec for debugging.
        # This preserves timestamp precision to ~65µs which is fine.
        msg.nanosec = (msg.nanosec & 0xFFFF0000) | (idx & 0xFFFF)
        return msg

    def _publish_frame(self):
        if self.frame_idx >= self.end_frame_excl:
            if self.loop:
                self.frame_idx = self.start_frame
                self.get_logger().info(f"Looping back to frame {self.start_frame}")
            else:
                self.get_logger().info("Replay complete.")
                self.timer.cancel()
                raise SystemExit(0)

        idx = self.frame_idx
        pose = self.poses[idx]

        # Pre-load both images BEFORE creating the timestamp
        # so file I/O doesn't create a gap between publishes
        if self.new_format:
            depth_raw = cv2.imread(
                str(self.dataset / "depth" / f"{idx:06d}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            depth = depth_raw.astype(np.float32) / self._depth_scale
            semantic = cv2.imread(
                str(self.dataset / self.semantic_subdir / f"{idx:06d}.png"),
                cv2.IMREAD_UNCHANGED,
            ).astype(np.int32)
        else:
            depth = np.load(self.dataset / "depth" / f"{idx:06d}.npy")
            semantic = np.load(self.dataset / self.semantic_subdir / f"{idx:06d}.npy")

        # Single timestamp for all messages in this frame
        stamp = self._stamp_from_index(idx)

        # Build both Image messages before publishing either
        depth_msg = self._make_depth_msg(depth, stamp)
        seg_msg = self._make_seg_msg(semantic, stamp)

        # Verify stamps are identical
        assert depth_msg.header.stamp == seg_msg.header.stamp

        # 1. Publish TF first (needed for lookupTransform in SCovoxNode)
        self._publish_tf(pose, stamp)

        # 2. Publish CameraInfo
        self._publish_camera_info(stamp)

        # 3. Publish depth and seg back-to-back with no gap
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

        Two modes:
        - camera_poses=False (default): pose is Habitat agent pose (Y-up).
          base_link gets agent position (converted to ROS Z-up),
          camera_link offset by 1.5m Z for sensor height.
        - camera_poses=True (NICE-SLAM format): pose is camera-to-world
          in the mesh's native frame. Published as-is (no coordinate conversion).
          The map will be in the mesh's native frame — GT comparison uses raw mesh.
        """
        pos = [float(v) for v in pose["position"]]
        quat = [float(v) for v in pose["rotation_quat_wxyz"]]  # [w, x, y, z]

        # Convert quaternion to rotation matrix
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])

        if self.camera_poses:
            # NICE-SLAM: camera-to-world in mesh native frame.
            # Publish directly — no coordinate conversion.
            # SCovoxNode's kR converts optical→body; the TF rotation
            # must undo kR and apply the camera-to-world rotation.
            # kR converts optical (Z-fwd,X-right,Y-down) → body (X-fwd,Y-left,Z-up)
            # We need: T_world = T_cam2world * kR^(-1) * p_optical
            # So TF rotation = R_cam2world * kR^T  (since kR is orthogonal)
            # But SCovoxNode does: T_oo.linear() = TF_rotation * kR
            # So: T_oo = R_cam2world * kR^T * kR = R_cam2world ✓
            # Therefore: TF_rotation = R_cam2world * kR^T
            kR = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=np.float64)
            R_tf = R @ kR.T
            q_tf = _rotation_matrix_to_quaternion(R_tf)

            t_base = TransformStamped()
            t_base.header.stamp = stamp
            t_base.header.frame_id = self.odom_frame
            t_base.child_frame_id = self.base_frame
            t_base.transform.translation.x = pos[0]
            t_base.transform.translation.y = pos[1]
            t_base.transform.translation.z = pos[2]
            t_base.transform.rotation.w = q_tf[0]
            t_base.transform.rotation.x = q_tf[1]
            t_base.transform.rotation.y = q_tf[2]
            t_base.transform.rotation.z = q_tf[3]

            # Camera link identity — pose is already the camera
            t_cam = TransformStamped()
            t_cam.header.stamp = stamp
            t_cam.header.frame_id = self.base_frame
            t_cam.child_frame_id = self.camera_frame
            t_cam.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform([t_base, t_cam])
        else:
            # Habitat agent pose — convert Y-up → ROS Z-up
            R_conv = np.array([
                [0, 0, -1],
                [-1, 0, 0],
                [0, 1, 0],
            ], dtype=np.float64)

            R_ros = R_conv @ R @ R_conv.T
            q_ros = _rotation_matrix_to_quaternion(R_ros)

            t_base = TransformStamped()
            t_base.header.stamp = stamp
            t_base.header.frame_id = self.odom_frame
            t_base.child_frame_id = self.base_frame
            t_base.transform.translation.x = -pos[2]
            t_base.transform.translation.y = -pos[0]
            t_base.transform.translation.z = pos[1]
            t_base.transform.rotation.w = q_ros[0]
            t_base.transform.rotation.x = q_ros[1]
            t_base.transform.rotation.y = q_ros[2]
            t_base.transform.rotation.z = q_ros[3]

            t_cam = TransformStamped()
            t_cam.header.stamp = stamp
            t_cam.header.frame_id = self.base_frame
            t_cam.child_frame_id = self.camera_frame
            t_cam.transform.translation.z = 1.5
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
        """Build depth Image message (32FC1, meters)."""
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
        """Build segmentation Image message (RGB8)."""
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
    node = ReplicaReplayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
