#!/usr/bin/env python3
"""Build voxelized GT NPZ from SceneNet RGB-D frames.

Projects depth + GT semantic labels into 3D using camera intrinsics and
GT poses, voxelizes at the target resolution, and saves the result as
an NPZ file for mIoU evaluation.

SceneNet depth is Euclidean ray length in uint16 millimetres.
SceneNet labels are uint8 NYUv2 class IDs (0-13).

Usage:
    python scenenet_build_gt.py /path/to/scenenet 2 /path/to/output/gt.npz
    python scenenet_build_gt.py /path/to/scenenet 2 gt.npz --resolution 0.05
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_intrinsics(seq_dir: Path):
    """Load fx, fy, cx, cy from intrinsics.txt."""
    intrinsics = {}
    with open(seq_dir / "intrinsics.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(":")
            intrinsics[key.strip()] = float(val.strip())
    return intrinsics


def load_poses(seq_dir: Path):
    """Load 3x4 camera-to-world poses."""
    poses = []
    with open(seq_dir / "poses.txt") as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def project_depth_to_3d(depth_path: str, label_path: str,
                        fx, fy, cx, cy, min_range, max_range):
    """Project one depth+label frame to 3D points in camera frame.

    Returns (points_Nx3, labels_N) in camera coordinates.
    """
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    labels = np.array(Image.open(label_path)).astype(np.uint8)
    h, w = depth_raw.shape

    # Depth in metres (Euclidean ray length)
    depth_m = depth_raw / 1000.0

    # Pixel grid
    u_coords = np.arange(h).reshape(-1, 1)
    v_coords = np.arange(w).reshape(1, -1)
    x_norm = (v_coords - cx) / fx
    y_norm = (u_coords - cy) / fy

    # Convert Euclidean to Z-depth, then to 3D
    norm_sq = x_norm**2 + y_norm**2 + 1.0
    z = depth_m / np.sqrt(norm_sq)
    x = x_norm * z
    y = y_norm * z

    # Range filter
    valid = (depth_m > min_range) & (depth_m < max_range) & (depth_m > 0)
    # Also skip unknown class
    valid &= (labels > 0)

    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    lab = labels[valid]

    return points.astype(np.float64), lab


def main():
    parser = argparse.ArgumentParser(description="Build SceneNet GT voxel NPZ")
    parser.add_argument("data_root", help="Root of SceneNet data")
    parser.add_argument("sequence", help="Sequence ID (e.g. 2)")
    parser.add_argument("output", help="Output NPZ path")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Voxel resolution in metres (default 0.05)")
    parser.add_argument("--min-range", type=float, default=0.1)
    parser.add_argument("--max-range", type=float, default=10.0)
    args = parser.parse_args()

    seq_dir = Path(args.data_root) / "train" / str(args.sequence)
    intrinsics = load_intrinsics(seq_dir)
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    poses = load_poses(seq_dir)

    depth_files = sorted((seq_dir / "depth").glob("*.png"))
    label_files = sorted((seq_dir / "ground_truth_labels").glob("*.png"))
    assert len(depth_files) == len(label_files) == len(poses)

    inv_res = 1.0 / args.resolution

    # Running voxel map: key (int64) -> best label via majority vote
    # For small indoor scenes this fits in memory easily
    voxel_counts = {}  # key -> {label: count}

    for i, (df, lf, T) in enumerate(zip(depth_files, label_files, poses)):
        points_cam, labels = project_depth_to_3d(
            str(df), str(lf), fx, fy, cx, cy, args.min_range, args.max_range
        )

        if len(points_cam) == 0:
            continue

        # Transform to world frame
        ones = np.ones((len(points_cam), 1))
        pts_h = np.hstack([points_cam, ones])  # Nx4
        pts_world = (T @ pts_h.T).T[:, :3]

        # Voxelize
        coords = np.floor(pts_world * inv_res).astype(np.int32)
        keys = (
            (coords[:, 0].astype(np.int64) & 0xFFFFFF) << 40 |
            (coords[:, 1].astype(np.int64) & 0xFFFFF) << 20 |
            (coords[:, 2].astype(np.int64) & 0xFFFFF)
        )

        for k, lab in zip(keys, labels):
            k = int(k)
            if k not in voxel_counts:
                voxel_counts[k] = {}
            lbl = int(lab)
            voxel_counts[k][lbl] = voxel_counts[k].get(lbl, 0) + 1

        if i % 25 == 0:
            print(f"  Frame {i}/{len(depth_files)}: "
                  f"{len(points_cam)} points, {len(voxel_counts)} total voxels")

    # Extract majority label per voxel
    n = len(voxel_counts)
    gt_keys = np.zeros(n, dtype=np.int64)
    gt_labels = np.zeros(n, dtype=np.uint8)

    for idx, (k, counts) in enumerate(voxel_counts.items()):
        gt_keys[idx] = k
        gt_labels[idx] = max(counts, key=counts.get)

    # Unpack keys to XYZ
    x = ((gt_keys >> 40) & 0xFFFFFF).astype(np.int32)
    y = ((gt_keys >> 20) & 0xFFFFF).astype(np.int32)
    z = (gt_keys & 0xFFFFF).astype(np.int32)

    # Sign extension for negative coords
    x[x >= 0x800000] -= 0x1000000
    y[y >= 0x80000] -= 0x100000
    z[z >= 0x80000] -= 0x100000

    gt_coords = np.stack([x, y, z], axis=-1).astype(np.float64) * args.resolution

    np.savez_compressed(
        args.output,
        coords=gt_coords.astype(np.float32),
        labels=gt_labels,
        keys=gt_keys,
        resolution=args.resolution,
        num_frames=len(depth_files),
        sequence=args.sequence,
    )

    print(f"\nGT saved: {args.output}")
    print(f"  {n} voxels at {args.resolution}m resolution")
    unique, counts = np.unique(gt_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  class {u:2d}: {c:8d} voxels")


if __name__ == "__main__":
    main()
