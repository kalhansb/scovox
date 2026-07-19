#!/usr/bin/env python3
"""B6 preliminary: count distinct semantic classes proposed per voxel.

Question: is K_TOP=2 enough to cover the typical voxel? For each voxel that
gets observed, this script counts how many *distinct* mapped class IDs the
classifier proposes across all observations. If ~95%+ of frequently-observed
voxels have <=2 distinct classes, K_TOP=2 covers nearly all real cases. If a
significant tail has 3+, the K_TOP=2 cap is forcing slot rotation.

Anchors:
  - KITTI seq08, PolarSeg predictions, 100 scans, voxel resolution 0.10 m,
    learning-map (20 classes; 0 = unlabeled).
  - Replica room0, m2f ADE150 predictions, 2000 frames, voxel resolution
    0.05 m, ADE150 -> scovox18 LUT (18 classes; 0 = unknown).

Output: per-anchor histogram of distinct-class counts, plus the same
stratified by per-voxel observation count.

This is purely offline: no ROS, no scovox runtime. We voxelize at the same
resolution scovox uses and count distinct classes that *would* be proposed
to a Dirichlet update, after the same dataset->scovox label mapping the
runtime applies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ----- Paths ---------------------------------------------------------------

WS = Path.home() / "projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG = WS / "src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT = WS / "data/semantickitti/dataset"
REPLICA_ROOT = WS / "data/replica_niceslam"

# ----- KITTI label mapping (raw -> learning, 0..19) -----------------------

LEARNING_MAP = {
    0: 0, 1: 0,
    10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8,
    40: 9, 44: 10, 48: 11, 49: 12,
    50: 13, 51: 14, 52: 0,
    60: 15, 70: 16, 71: 17, 72: 18, 80: 19, 81: 0, 99: 0,
    252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}
_KITTI_LUT = np.zeros(260, dtype=np.uint8)
for raw, mapped in LEARNING_MAP.items():
    _KITTI_LUT[raw] = mapped


# ----- Voxel-key packing ---------------------------------------------------

def voxelize_pack(points: np.ndarray, res: float) -> np.ndarray:
    """Pack int voxel coords into uint64 keys.

    Coord range +-2^20 cells (~1 km at 1 mm; with 0.05 m it's ~52 km box —
    plenty). Apply a bias so coords fit unsigned.
    """
    BIAS = 1 << 20
    ijk = np.floor(points / res).astype(np.int64) + BIAS
    if (ijk < 0).any() or (ijk >= (1 << 21)).any():
        raise RuntimeError("voxel coord overflow — increase BIAS or check data")
    keys = (ijk[:, 0].astype(np.uint64) << 42) \
         | (ijk[:, 1].astype(np.uint64) << 21) \
         | ijk[:, 2].astype(np.uint64)
    return keys


# ----- KITTI loader --------------------------------------------------------

def load_kitti_calib(seq_dir: Path):
    Tr = None
    for line in (seq_dir / "calib.txt").read_text().splitlines():
        if line.startswith("Tr:"):
            vals = np.array([float(x) for x in line.split()[1:]], dtype=np.float64)
            Tr = np.eye(4, dtype=np.float64)
            Tr[:3, :4] = vals.reshape(3, 4)
    if Tr is None:
        raise RuntimeError("Tr not found in calib.txt")
    return Tr


def kitti_frame_obs(seq_dir: Path, frame_idx: int, Tr: np.ndarray, pose_cam: np.ndarray,
                    res: float, max_range: float = 30.0, min_range: float = 1.0):
    """Return (keys uint64, classes uint8) for one scan, mapped + voxelized."""
    bin_path = seq_dir / "velodyne" / f"{frame_idx:06d}.bin"
    lab_path = seq_dir / "predictions" / f"{frame_idx:06d}.label"
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    lab = np.fromfile(lab_path, dtype=np.uint32) & 0xFFFF
    sem = _KITTI_LUT[np.minimum(lab, 259)]

    # Range filter in sensor frame.
    r = np.linalg.norm(pts, axis=1)
    keep = (r >= min_range) & (r <= max_range) & (sem > 0)
    pts = pts[keep]
    sem = sem[keep]
    if pts.size == 0:
        return np.empty(0, np.uint64), np.empty(0, np.uint8)

    # velodyne -> cam0 -> world
    Tw = pose_cam @ Tr
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_w = (Tw @ pts_h.T).T[:, :3]

    keys = voxelize_pack(pts_w, res)
    return keys, sem.astype(np.uint8)


def run_kitti(seq: str = "08", res: float = 0.10, n_scans: int = 100):
    seq_dir = KITTI_ROOT / "sequences" / seq
    Tr = load_kitti_calib(seq_dir)
    poses = np.loadtxt(seq_dir / "poses.txt").reshape(-1, 3, 4)
    poses4 = np.tile(np.eye(4, dtype=np.float64), (poses.shape[0], 1, 1))
    poses4[:, :3, :4] = poses

    print(f"[kitti seq{seq}] loading {n_scans} scans, res={res} m")
    all_keys = []
    all_classes = []
    t0 = time.time()
    for i in range(n_scans):
        keys, sem = kitti_frame_obs(seq_dir, i, Tr, poses4[i], res)
        all_keys.append(keys)
        all_classes.append(sem)
        if (i + 1) % 25 == 0:
            print(f"  scan {i+1}/{n_scans}  ({time.time()-t0:.1f}s)")
    keys = np.concatenate(all_keys)
    classes = np.concatenate(all_classes)
    print(f"[kitti] {len(keys):,} obs across {n_scans} scans  ({time.time()-t0:.1f}s)")
    return keys, classes


# ----- Replica loader ------------------------------------------------------

def load_replica_lut() -> np.ndarray:
    p = EVAL_PKG / "scripts/ade150_to_scovox18.npz"
    return np.load(p)["lut"]  # uint8, len 151, output 0..17


def replica_frame_obs(scene_dir: Path, frame_idx: int, K: np.ndarray, pose: np.ndarray,
                      lut: np.ndarray, res: float,
                      pixel_stride: int = 2, max_depth: float = 6.0):
    """Backproject one Replica frame; return (keys, classes)."""
    import cv2
    depth_path = scene_dir / "depth" / f"{frame_idx:06d}.png"
    sem_path = scene_dir / "semantic_m2f_ade" / f"{frame_idx:06d}.png"
    if not depth_path.exists() or not sem_path.exists():
        return np.empty(0, np.uint64), np.empty(0, np.uint8)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 6553.5
    sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED).astype(np.uint8)
    depth = depth[::pixel_stride, ::pixel_stride]
    sem = sem[::pixel_stride, ::pixel_stride]
    H, W = depth.shape
    fx = K[0, 0] / pixel_stride
    fy = K[1, 1] / pixel_stride
    cx = K[0, 2] / pixel_stride
    cy = K[1, 2] / pixel_stride

    valid = (depth > 0.05) & (depth < max_depth)
    ys, xs = np.where(valid)
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    cls_raw = sem[ys, xs]
    cls_mapped = lut[np.minimum(cls_raw, 150)]
    keep = cls_mapped > 0
    pts_cam = pts_cam[keep]
    cls_mapped = cls_mapped[keep]
    if pts_cam.size == 0:
        return np.empty(0, np.uint64), np.empty(0, np.uint8)

    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_w = (pose @ pts_h.T).T[:, :3]
    keys = voxelize_pack(pts_w, res)
    return keys, cls_mapped.astype(np.uint8)


def run_replica(scene: str = "room0", res: float = 0.05, n_frames: int = 2000,
                pixel_stride: int = 2, frame_stride: int = 1):
    scene_dir = REPLICA_ROOT / scene
    cam = json.loads((REPLICA_ROOT / "cam_params.json").read_text())["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]],
                  [0, cam["fy"], cam["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    poses = np.loadtxt(scene_dir / "poses.txt").reshape(-1, 4, 4)
    n_avail = min(n_frames, poses.shape[0])
    lut = load_replica_lut()

    print(f"[replica {scene}] {n_avail} frames stride={frame_stride} pix_stride={pixel_stride} res={res} m")
    all_keys, all_cls = [], []
    t0 = time.time()
    for i in range(0, n_avail, frame_stride):
        keys, cls = replica_frame_obs(scene_dir, i, K, poses[i], lut, res,
                                      pixel_stride=pixel_stride)
        all_keys.append(keys)
        all_cls.append(cls)
        if ((i // frame_stride) + 1) % 200 == 0:
            print(f"  frame {i+1}/{n_avail}  ({time.time()-t0:.1f}s, accum {sum(len(k) for k in all_keys):,} obs)")
    keys = np.concatenate(all_keys) if all_keys else np.empty(0, np.uint64)
    classes = np.concatenate(all_cls) if all_cls else np.empty(0, np.uint8)
    print(f"[replica] {len(keys):,} obs across {n_avail//frame_stride} frames  ({time.time()-t0:.1f}s)")
    return keys, classes


# ----- Aggregation ---------------------------------------------------------

def aggregate(keys: np.ndarray, classes: np.ndarray, n_classes_max: int):
    """Per voxel: bitmask of distinct class IDs + observation count."""
    bits = np.uint32(1) << classes.astype(np.uint32)  # bit per class
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    bits_s = bits[order]

    unique_keys, first_idx = np.unique(keys_s, return_index=True)
    # bitwise-OR reduce within groups
    masks = np.bitwise_or.reduceat(bits_s, first_idx)
    next_idx = np.append(first_idx[1:], len(bits_s))
    counts = (next_idx - first_idx).astype(np.int64)

    distinct = np.zeros(len(masks), dtype=np.uint8)
    for c in range(1, n_classes_max + 1):
        distinct += ((masks >> c) & 1).astype(np.uint8)
    return distinct, counts


def report(name: str, distinct: np.ndarray, counts: np.ndarray):
    n = len(distinct)
    print(f"\n=== {name} ===")
    print(f"total voxels: {n:,}")
    print(f"obs/voxel: mean={counts.mean():.1f} median={np.median(counts):.0f} "
          f"p99={np.percentile(counts, 99):.0f} max={counts.max()}")

    def hist(mask):
        sub = distinct[mask]
        if len(sub) == 0:
            return None
        return {
            "n": len(sub),
            "1": int((sub == 1).sum()),
            "2": int((sub == 2).sum()),
            "3": int((sub == 3).sum()),
            "4": int((sub == 4).sum()),
            "5+": int((sub >= 5).sum()),
        }

    strata = [
        ("all", np.ones(n, dtype=bool)),
        (">=5 obs", counts >= 5),
        (">=20 obs", counts >= 20),
        (">=100 obs", counts >= 100),
    ]
    print(f"\n{'stratum':<12} {'n':>10}  {'%1cls':>7} {'%2cls':>7} {'%3cls':>7} {'%4cls':>7} {'%5+cls':>7}  {'<=2cls':>7}")
    for label, mask in strata:
        h = hist(mask)
        if h is None:
            continue
        nh = h["n"]
        p1 = 100 * h["1"] / nh
        p2 = 100 * h["2"] / nh
        p3 = 100 * h["3"] / nh
        p4 = 100 * h["4"] / nh
        p5 = 100 * h["5+"] / nh
        le2 = p1 + p2
        print(f"{label:<12} {nh:>10,}  {p1:>6.2f}% {p2:>6.2f}% {p3:>6.2f}% {p4:>6.2f}% {p5:>6.2f}%  {le2:>6.2f}%")


# ----- CLI -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anchor", choices=["kitti", "replica", "both"], nargs="?", default="both")
    ap.add_argument("--kitti-seq", default="08")
    ap.add_argument("--kitti-scans", type=int, default=100)
    ap.add_argument("--kitti-res", type=float, default=0.10)
    ap.add_argument("--replica-scene", default="room0")
    ap.add_argument("--replica-frames", type=int, default=2000)
    ap.add_argument("--replica-res", type=float, default=0.05)
    ap.add_argument("--replica-pix-stride", type=int, default=2)
    ap.add_argument("--replica-frame-stride", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="optional .npz to write per-voxel arrays for follow-up analysis")
    args = ap.parse_args()

    out_blobs = {}

    if args.anchor in ("kitti", "both"):
        keys, classes = run_kitti(seq=args.kitti_seq, res=args.kitti_res,
                                  n_scans=args.kitti_scans)
        d, c = aggregate(keys, classes, n_classes_max=19)
        report(f"KITTI seq{args.kitti_seq} polarseg, res={args.kitti_res} m, {args.kitti_scans} scans",
               d, c)
        out_blobs["kitti_distinct"] = d
        out_blobs["kitti_counts"] = c

    if args.anchor in ("replica", "both"):
        keys, classes = run_replica(scene=args.replica_scene, res=args.replica_res,
                                    n_frames=args.replica_frames,
                                    pixel_stride=args.replica_pix_stride,
                                    frame_stride=args.replica_frame_stride)
        d, c = aggregate(keys, classes, n_classes_max=17)
        report(f"Replica {args.replica_scene} m2f-ADE->scovox18, res={args.replica_res} m, "
               f"{args.replica_frames} frames, pix_stride={args.replica_pix_stride}",
               d, c)
        out_blobs["replica_distinct"] = d
        out_blobs["replica_counts"] = c

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **out_blobs)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
