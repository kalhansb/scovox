#!/usr/bin/env python3
"""Render SCovox PointCloud2 snapshots to per-frame PNGs (and optional PLY).

Example (run while scovox_node + semantickitti_replay_node are active):
  python3 -m scovox_eval.render_pointcloud_frames \
    --topic /atlas/scovox_node/pointcloud \
    --sync-topic /atlas/velodyne_points \
    --out-dir /tmp/scovox_seq08_frames \
    --every-n 1 --max-frames 200

Notes:
  - Uses Open3D offscreen rendering for PNG output. If Open3D is missing,
    use --ply-only to write colored PLYs instead.
  - The sync topic is optional; when provided, its header stamp low 16 bits
    are used as the frame id for naming outputs (matches SemKITTI replay).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Prefer CPU rendering for headless servers.
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")

try:
    import open3d as o3d
    from open3d.visualization import rendering
    _HAVE_O3D = True
except Exception:
    o3d = None
    rendering = None
    _HAVE_O3D = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField


_DTYPE_MAP = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def _field_dtype(field: PointField, is_bigendian: bool) -> np.dtype:
    base = np.dtype(_DTYPE_MAP.get(field.datatype, np.uint8))
    base = base.newbyteorder(">" if is_bigendian else "<")
    if field.count and field.count > 1:
        return np.dtype((base, field.count))
    return base


def _pc2_to_structured(msg: PointCloud2) -> np.ndarray:
    if msg.point_step <= 0:
        raise ValueError("PointCloud2 has invalid point_step")

    names = []
    formats = []
    offsets = []
    for f in msg.fields:
        names.append(f.name)
        formats.append(_field_dtype(f, msg.is_bigendian))
        offsets.append(f.offset)

    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": msg.point_step,
        }
    )
    count = int(msg.width * msg.height)
    return np.frombuffer(msg.data, dtype=dtype, count=count)


def _extract_xyz_rgb_from_structured(data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    for k in ("x", "y", "z"):
        if k not in data.dtype.names:
            raise ValueError(f"PointCloud2 missing field '{k}'")
    points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32, copy=False)

    colors = None
    if "rgb" in data.dtype.names:
        rgb = data["rgb"]
        if rgb.dtype != np.uint32:
            rgb = rgb.view(np.uint32)
        r = (rgb >> 16) & 0xFF
        g = (rgb >> 8) & 0xFF
        b = rgb & 0xFF
        colors = np.column_stack([r, g, b]).astype(np.float32) / 255.0
    return points, colors


def _write_ply_ascii(path: Path, points: np.ndarray, colors: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        else:
            c = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            for p, col in zip(points, c):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {col[0]} {col[1]} {col[2]}\n")


class _Open3DRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        point_size: float,
        background: Tuple[float, float, float, float],
        lookat: Optional[np.ndarray],
        eye: Optional[np.ndarray],
        up: np.ndarray,
        dynamic_view: bool,
        eye_scale: float,
    ) -> None:
        if not _HAVE_O3D or rendering is None:
            raise RuntimeError("Open3D rendering is unavailable. Install open3d or use --ply-only.")
        self._renderer = rendering.OffscreenRenderer(width, height)
        self._renderer.scene.set_background(background)
        self._mat = rendering.MaterialRecord()
        self._mat.shader = "defaultUnlit"
        self._mat.point_size = float(point_size)
        self._lookat = lookat
        self._eye = eye
        self._up = up
        self._dynamic_view = dynamic_view
        self._eye_scale = eye_scale
        self._view_locked = False

    def _resolve_view(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._lookat is not None and self._eye is not None:
            return self._lookat, self._eye, self._up

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = np.max(bbox.get_extent())
        if extent <= 0:
            extent = 1.0
        lookat = center
        eye = center + np.array([1.2, -1.2, 0.6]) * extent * self._eye_scale
        return lookat, eye, self._up

    def render(self, points: np.ndarray, colors: Optional[np.ndarray], out_path: Path) -> None:
        self._renderer.scene.clear_geometry()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        self._renderer.scene.add_geometry("pcd", pcd, self._mat)

        if self._dynamic_view or not self._view_locked:
            lookat, eye, up = self._resolve_view(points)
            self._renderer.scene.camera.look_at(lookat, eye, up)
            if not self._dynamic_view:
                self._lookat, self._eye, self._up = lookat, eye, up
                self._view_locked = True

        img = self._renderer.render_to_image()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_image(str(out_path), img)


class PointCloudFrameRenderer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("scovox_pointcloud_frame_renderer")
        self._out_dir = Path(args.out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._write_ply = args.write_ply or args.ply_only
        self._ply_only = args.ply_only
        self._every_n = max(1, int(args.every_n))
        self._max_frames = int(args.max_frames)
        self._max_points = int(args.max_points)
        self._point_stride = max(1, int(args.point_stride))
        self._voxel_size = float(args.voxel_size)
        self._saved = 0
        self._received = 0
        self._last_sync_id: Optional[int] = None
        self._last_saved_id: Optional[int] = None

        self._renderer = None
        if not self._ply_only:
            self._renderer = _Open3DRenderer(
                width=int(args.width),
                height=int(args.height),
                point_size=float(args.point_size),
                background=(args.bg[0], args.bg[1], args.bg[2], 1.0),
                lookat=np.array(args.lookat) if args.lookat else None,
                eye=np.array(args.eye) if args.eye else None,
                up=np.array(args.up),
                dynamic_view=bool(args.dynamic_view),
                eye_scale=float(args.eye_scale),
            )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST

        self._pc_sub = self.create_subscription(PointCloud2, args.topic, self._on_cloud, qos)
        self._sync_sub = None
        if args.sync_topic:
            self._sync_sub = self.create_subscription(PointCloud2, args.sync_topic, self._on_sync, qos)

        self.get_logger().info(
            f"Rendering from {args.topic} -> {self._out_dir} (every_n={self._every_n}, max_frames={self._max_frames})"
        )

    def _on_sync(self, msg: PointCloud2) -> None:
        self._last_sync_id = int(msg.header.stamp.nanosec & 0xFFFF)

    def _downsample(self, points: np.ndarray, colors: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._voxel_size <= 0 or not _HAVE_O3D:
            return points, colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(self._voxel_size)
        return np.asarray(pcd.points), np.asarray(pcd.colors) if colors is not None else None

    def _on_cloud(self, msg: PointCloud2) -> None:
        self._received += 1
        frame_id = self._last_sync_id if self._last_sync_id is not None else (self._received - 1)

        if frame_id == self._last_saved_id:
            return
        if (frame_id % self._every_n) != 0:
            return

        try:
            data = _pc2_to_structured(msg)
            if self._point_stride > 1:
                data = data[:: self._point_stride]
            if self._max_points > 0 and data.shape[0] > self._max_points:
                idx = np.random.choice(data.shape[0], self._max_points, replace=False)
                data = data[idx]
            points, colors = _extract_xyz_rgb_from_structured(data)
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse PointCloud2: {exc}")
            return

        if points.size == 0:
            return

        points, colors = self._downsample(points, colors)

        stem = f"frame_{frame_id:06d}"
        if self._write_ply:
            ply_path = self._out_dir / f"{stem}.ply"
            _write_ply_ascii(ply_path, points, colors)

        if self._renderer is not None:
            png_path = self._out_dir / f"{stem}.png"
            self._renderer.render(points, colors, png_path)

        self._last_saved_id = frame_id
        self._saved += 1
        if self._saved % 10 == 0:
            self.get_logger().info(f"Saved {self._saved} frames (last={stem})")

        if self._max_frames > 0 and self._saved >= self._max_frames:
            self.get_logger().info("Reached max_frames; shutting down.")
            rclpy.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description="Render SCovox PointCloud2 frames to images")
    ap.add_argument("--topic", default="/atlas/scovox_node/pointcloud", help="PointCloud2 topic to render")
    ap.add_argument("--sync-topic", default="/atlas/velodyne_points",
                    help="Optional PointCloud2 topic used for frame ids (SemKITTI replay)")
    ap.add_argument("--out-dir", default="scovox_frames", help="Output directory for frames")
    ap.add_argument("--max-frames", type=int, default=-1, help="Maximum number of frames to save")
    ap.add_argument("--every-n", type=int, default=1, help="Save every Nth frame")
    ap.add_argument("--max-points", type=int, default=0, help="Randomly sample to this many points per frame")
    ap.add_argument("--point-stride", type=int, default=1, help="Keep every Nth point before sampling")
    ap.add_argument("--voxel-size", type=float, default=0.0, help="Voxel downsample size (m)")
    ap.add_argument("--width", type=int, default=1280, help="Output image width")
    ap.add_argument("--height", type=int, default=720, help="Output image height")
    ap.add_argument("--point-size", type=float, default=2.0, help="Point size in pixels")
    ap.add_argument("--dynamic-view", action="store_true", help="Recompute camera per frame")
    ap.add_argument("--eye", nargs=3, type=float, default=None, help="Camera eye position (x y z)")
    ap.add_argument("--lookat", nargs=3, type=float, default=None, help="Camera look-at position (x y z)")
    ap.add_argument("--up", nargs=3, type=float, default=[0.0, 0.0, 1.0], help="Camera up vector")
    ap.add_argument("--eye-scale", type=float, default=2.0, help="Scale for auto camera distance")
    ap.add_argument("--bg", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Background color (r g b)")
    ap.add_argument("--write-ply", action="store_true", help="Also save PLY for each frame")
    ap.add_argument("--ply-only", action="store_true", help="Save only PLY (no PNG) if Open3D is missing")
    args = ap.parse_args()

    if not args.ply_only and not _HAVE_O3D:
        raise SystemExit("Open3D is required for PNG rendering. Install open3d or use --ply-only.")

    rclpy.init()
    node = PointCloudFrameRenderer(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
