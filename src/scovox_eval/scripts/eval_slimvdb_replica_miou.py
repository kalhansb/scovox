#!/usr/bin/env python3
"""Voxel-wise mIoU for SLIM-VDB on Replica (GT-oracle condition).

Builds a GT voxel grid by projecting depth + semantic_gt_fixed PNGs through GT
poses at the map resolution (5 cm). Compares against SLIM-VDB's argmax voxels
from voxels.bin. Same methodology as KITTI: majority-class per GT voxel,
per-class IoU excluding class 0 (void), mean IoU.

Usage:
    python eval_slimvdb_replica_miou.py \
        --replica_root data/replica_niceslam \
        --voxels_dir third_party_sw/slim_vdb/outputs/replica_gt \
        --voxel_size 0.05 --n_frames 2000 --stride 10
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_voxels_bin(path: Path):
    raw = np.fromfile(path, dtype=np.uint8)
    rec = raw.view(np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "<i4")]))
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    return xyz, rec["c"].astype(np.int32)


def voxel_key(xyz, vs):
    ix = np.floor(xyz[:, 0] / vs).astype(np.int64)
    iy = np.floor(xyz[:, 1] / vs).astype(np.int64)
    iz = np.floor(xyz[:, 2] / vs).astype(np.int64)
    mask = (1 << 24) - 1
    return ((ix & mask) << 48) | ((iy & mask) << 24) | (iz & mask)


def read_poses(poses_file):
    poses = []
    with open(poses_file) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            if len(vals) == 16:
                poses.append(np.array(vals, dtype=np.float64).reshape(4, 4))
    return poses


def read_cam_params(root):
    cp = root / "cam_params.json"
    if not cp.exists():
        cp = root.parent / "cam_params.json"
    cam = json.load(open(cp))["camera"]
    return cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["w"], cam["h"], cam["scale"]


def build_gt_voxels(scene_dir: Path, voxel_size: float,
                    n_frames: int, stride: int) -> dict:
    """Project depth + GT semantics into 3D, voxelize, majority-vote per voxel."""
    root = scene_dir.parent
    fx, fy, cx, cy, W, H, depth_scale = read_cam_params(root)
    poses = read_poses(scene_dir / "poses.txt")
    sem_dir = scene_dir / "semantic_gt_fixed"
    depth_dir = scene_dir / "depth"

    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    n = min(n_frames, len(poses))
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u_grid.flatten().astype(np.float32)
    v_flat = v_grid.flatten().astype(np.float32)

    for i in range(0, n, stride):
        depth_img = cv2.imread(str(depth_dir / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        sem_img = cv2.imread(str(sem_dir / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth_img is None or sem_img is None:
            continue
        d = depth_img.flatten().astype(np.float32) / depth_scale
        s = sem_img.flatten().astype(np.int32)

        valid = (d > 0.01) & (d < 8.0) & (s > 0)
        d, s = d[valid], s[valid]
        u, v = u_flat[valid], v_flat[valid]

        x = (u - cx) / fx * d
        y = (v - cy) / fy * d
        z = d
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
        T = poses[i].astype(np.float32)
        pts_world = (T @ pts_cam.T).T[:, :3]

        keys = voxel_key(pts_world, voxel_size)
        for k, c in zip(keys, s):
            counts[int(k)][int(c)] += 1

    voxel2class = {k: max(d.items(), key=lambda kv: kv[1])[0] for k, d in counts.items()}
    return voxel2class


def compute_miou(pred_voxels, gt_voxels, num_classes, class_names=None):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_keys = set(pred_voxels.keys()) | set(gt_voxels.keys())
    for k in all_keys:
        g = gt_voxels.get(k, 0)
        p = pred_voxels.get(k, 0)
        if g < num_classes and p < num_classes:
            cm[g, p] += 1
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    finite_nonzero = [iou[i] for i in range(1, num_classes) if np.isfinite(iou[i])]
    miou = float(np.mean(finite_nonzero)) if finite_nonzero else float("nan")
    return {"miou": miou, "n_pred": len(pred_voxels), "n_gt": len(gt_voxels),
            "n_intersect": sum(1 for k in pred_voxels if k in gt_voxels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--voxels_dir", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+",
                    default=["room0", "room1", "room2",
                             "office0", "office1", "office2", "office3", "office4"])
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=10,
                    help="Process every Nth frame for GT voxelization (speed tradeoff)")
    ap.add_argument("--num_classes", type=int, default=102)
    args = ap.parse_args()

    print(f"[mIoU] Replica GT, voxel={args.voxel_size}m, stride={args.stride}, "
          f"scenes={args.scenes}")

    results = []
    for scene in args.scenes:
        vbin = args.voxels_dir / scene / "voxels.bin"
        if not vbin.exists():
            print(f"  {scene}: MISSING voxels.bin"); continue
        xyz, cls = load_voxels_bin(vbin)
        keys = voxel_key(xyz, args.voxel_size)
        pred = {int(k): int(c) for k, c in zip(keys, cls)}

        scene_dir = args.replica_root / scene
        print(f"  {scene}: building GT voxels (stride={args.stride}) …", flush=True)
        gt = build_gt_voxels(scene_dir, args.voxel_size, args.n_frames, args.stride)

        res = compute_miou(pred, gt, args.num_classes)
        print(f"    pred={res['n_pred']}  gt={res['n_gt']}  intersect={res['n_intersect']}  "
              f"mIoU={res['miou']:.4f}")
        results.append((scene, res))

    if results:
        miou_vals = [r["miou"] for _, r in results]
        print(f"\n=== Replica GT mIoU = {np.mean(miou_vals):.4f} ± {np.std(miou_vals):.4f} ===")


if __name__ == "__main__":
    main()
