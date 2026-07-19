#!/usr/bin/env python3
"""E1 occupancy reference for SceneNet RGB-D (UNCERTAINTY_PLAN.md §0), the
depth-camera analogue of build_occ_reference.py (LiDAR).

Occupied ref = voxels with >=1 depth endpoint (hit).
Free ref     = voxels traversed by >=m rays with ZERO endpoint hits, inside the
               range gate. Never-observed voxels excluded.

Frame parity with scovox is the whole game. The SceneNet replay node
(scenenet_replay_node.py) publishes the GT pose as odom->base_link with rotation
R@kR.T; scovox re-applies its internal kR so a point p in the OPTICAL camera
frame (Z-fwd, X-right, Y-down) maps to world = R@p + t, i.e. exactly the raw
3x4 cam-to-world pose. SceneNet depth is Euclidean ray length, so the optical
endpoint is p = depth_m * unit_ray with unit_ray = (xn, yn, 1)/||.||,
xn=(col-cx)/fx, yn=(row-cy)/fy. Voxelization coord = floor(world/res) and the
range gate [min,max] match the node (min_depth=0.1, max_depth=10.0), so the
reference and the captured map cover the same observed set.

Output npz: coord (int32 Nx3), hit (uint32), free (uint32) — identical schema to
the LiDAR reference, so score_e1.py consumes it unchanged.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# reuse the exact vectorized Amanatides-Woo DDA + key packing from the LiDAR ref
from build_occ_reference import dda_free_keys, merge_counts, pack, unpack  # noqa: E402

CHUNK = 40000  # rays per DDA chunk (memory cap)


def load_intrinsics(seq_dir: Path):
    d = {}
    with open(seq_dir / "intrinsics.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(":")
            d[k.strip()] = float(v.strip())
    return d["fx"], d["fy"], d["cx"], d["cy"]


def load_poses(seq_dir: Path):
    """3x4 camera-to-world matrices (same parse as the replay node)."""
    poses = []
    with open(seq_dir / "poses.txt") as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            if not vals:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/kalhan/Projects/scovox_ws/data/scenenet_val_layout")
    ap.add_argument("--sequence", required=True, help="e.g. 0_223")
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--min-range", type=float, default=0.1)
    ap.add_argument("--max-range", type=float, default=10.0)
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1, help="pixel subsample stride")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    sd = Path(args.data_root) / "train" / args.sequence
    fx, fy, cx, cy = load_intrinsics(sd)
    poses = load_poses(sd)
    depth_files = sorted((sd / "depth").glob("*.png"))
    n = min(len(depth_files), len(poses), args.n_frames)
    res = args.resolution
    st = max(1, args.stride)
    print(f"seq {args.sequence}: {n} frames res={res} gate[{args.min_range},{args.max_range}] "
          f"stride={st} K=({fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f})")

    # precompute per-pixel unit ray directions in the optical frame (constant
    # across frames): dir = (xn, yn, 1)/||.||, xn=(col-cx)/fx, yn=(row-cy)/fy
    W, H = 320, 240
    cols = np.arange(0, W, st)
    rows = np.arange(0, H, st)
    vv, uu = np.meshgrid(cols, rows)  # vv=col, uu=row
    xn = (vv.astype(np.float64) - cx) / fx
    yn = (uu.astype(np.float64) - cy) / fy
    dirs = np.stack([xn, yn, np.ones_like(xn)], axis=-1)          # (h,w,3)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)          # unit optical rays
    dirs = dirs.reshape(-1, 3)                                    # (P,3)

    fkey, fcnt, hkey, hcnt = [], [], [], []
    for i in range(n):
        depth_raw = np.array(Image.open(str(depth_files[i])))[::st, ::st]
        depth_m = depth_raw.astype(np.float64).ravel() / 1000.0  # mm -> m (Euclidean)
        valid = (depth_m >= args.min_range) & (depth_m <= args.max_range)
        if not valid.any():
            continue
        d = depth_m[valid]
        p_opt = dirs[valid] * d[:, None]                          # optical-frame points
        T = poses[i]
        R, t = T[:3, :3], T[:3, 3]
        E = (R @ p_opt.T).T + t                                   # world endpoints
        O = t                                                    # camera centre (world)

        hk, hc = np.unique(pack(np.floor(E / res).astype(np.int64)), return_counts=True)
        hkey.append(hk); hcnt.append(hc)

        scan_free = []
        for s in range(0, len(E), CHUNK):
            scan_free.append(dda_free_keys(O, E[s:s + CHUNK], res))
        fk = np.concatenate(scan_free) if scan_free else np.zeros(0, np.int64)
        if len(fk):
            uk, uc = np.unique(fk, return_counts=True)
            fkey.append(uk); fcnt.append(uc)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  frame {i+1}/{n}: rays={len(E)} distinct_free_voxels={sum(len(c) for c in fkey):,}")

    fk_u, fk_c = merge_counts(fkey, fcnt)
    hk_u, hk_c = merge_counts(hkey, hcnt)

    allk = np.union1d(fk_u, hk_u)
    free = np.zeros(len(allk), np.uint32); hit = np.zeros(len(allk), np.uint32)
    free[np.searchsorted(allk, fk_u)] = fk_c
    hit[np.searchsorted(allk, hk_u)] = hk_c
    coord = unpack(allk)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, coord=coord, hit=hit, free=free,
                        resolution=np.float32(res))
    occ = int((hit >= 1).sum())
    fre = int(((hit == 0) & (free >= 3)).sum())
    print(f"observed voxels={len(allk):,}  occupied(hit>=1)={occ:,}  "
          f"free(hit==0 & free>=3)={fre:,}  -> {args.output}")


if __name__ == "__main__":
    main()
