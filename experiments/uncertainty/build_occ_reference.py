#!/usr/bin/env python3
"""E1 occupancy reference for SemanticKITTI (UNCERTAINTY_PLAN.md §0).

Occupied ref = voxels with >=1 LiDAR endpoint (hit).
Free ref     = voxels traversed by >=m rays with ZERO endpoint hits, inside the
               range gate. Never-observed voxels excluded.

Rays are cast from the GT sensor pose to each endpoint via an exact vectorized
Amanatides-Woo voxel traversal. Frame + voxelization MATCH scovox: world =
poses[i] @ Tr (velodyne->cam/world), coord = floor(pos/res); range gate [5,30] m
matches the node's min_range/max_range so the reference and the captured map
cover the same observed set.

Output npz: coord (int32 Nx3), hit (uint32), free (uint32). Classification and
the m-sensitivity sweep happen in the scorer.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

# reuse the exact loaders/LUT from the GT builder for frame consistency
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kitti"))
from build_gt import load_calib, load_poses  # noqa: E402

K_MAX = 700          # max voxel steps per ray (30 m / 0.1 m * sqrt(3) ~ 520)
CHUNK = 40000        # rays per DDA chunk (memory cap)


def pack(coords):
    """Pack int voxel coords -> int64 key (matches build_gt bit scheme)."""
    x = coords[:, 0].astype(np.int64); y = coords[:, 1].astype(np.int64); z = coords[:, 2].astype(np.int64)
    return ((x & 0xFFFFFF) << 40) | ((y & 0xFFFFF) << 20) | (z & 0xFFFFF)


def unpack(keys):
    ix = ((keys >> 40) & 0xFFFFFF).astype(np.int64)
    iy = ((keys >> 20) & 0xFFFFF).astype(np.int64)
    iz = (keys & 0xFFFFF).astype(np.int64)
    ix = np.where(ix >= 0x800000, ix - 0x1000000, ix)
    iy = np.where(iy >= 0x80000, iy - 0x100000, iy)
    iz = np.where(iz >= 0x80000, iz - 0x100000, iz)
    return np.stack([ix, iy, iz], axis=1).astype(np.int32)


def dda_free_keys(O, E, res):
    """Vectorized Amanatides-Woo. O: (3,) world origin. E: (R,3) endpoints.
    Returns int64 keys of all traversed voxels (v0 .. v1 exclusive), one entry
    per (ray, voxel). Endpoint voxels are NOT included (they're hits)."""
    R = E.shape[0]
    inv = 1.0 / res
    v = np.floor(O * inv).astype(np.int64)[None, :].repeat(R, axis=0)  # current voxel (R,3)
    v1 = np.floor(E * inv).astype(np.int64)                            # endpoint voxel
    D = E - O[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.sign(D).astype(np.int64)
        # world coord of the next boundary crossed on each axis
        nb = (v + (step > 0)) * res
        tMax = np.where(D != 0, (nb - O[None, :]) / D, np.inf)
        tDelta = np.where(D != 0, res / np.abs(D), np.inf)
    active = np.ones(R, bool)
    # ray degenerate (O==E voxel) -> nothing to traverse
    active &= ~np.all(v == v1, axis=1)
    out = []
    for _ in range(K_MAX):
        if not active.any():
            break
        # record current voxel of active rays that haven't reached the endpoint
        rec = active & ~np.all(v == v1, axis=1)
        if rec.any():
            out.append(pack(v[rec]))
        # rays whose current voxel IS the endpoint are done (don't record as free)
        active &= ~np.all(v == v1, axis=1)
        if not active.any():
            break
        a = np.argmin(tMax, axis=1)                 # axis to advance (R,)
        ai = np.arange(R)
        adv = active & (tMax[ai, a] <= 1.0 + 1e-9)  # still within the ray segment
        v[ai[adv], a[adv]] += step[ai[adv], a[adv]]
        tMax[ai[adv], a[adv]] += tDelta[ai[adv], a[adv]]
        active &= adv                                # overshoot (tMax>1) -> stop
    if not out:
        return np.zeros(0, np.int64)
    return np.concatenate(out)


def merge_counts(key_chunks, cnt_chunks):
    """Lists of per-scan (unique_keys, counts) -> global (unique_keys, summed)."""
    if not key_chunks:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    allk = np.concatenate(key_chunks)
    allc = np.concatenate(cnt_chunks).astype(np.int64)
    order = np.argsort(allk, kind="stable")
    allk, allc = allk[order], allc[order]
    uk, start = np.unique(allk, return_index=True)
    summed = np.add.reduceat(allc, start)
    return uk, summed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="/home/kalhan/Projects/scovox_ws/data/semantickitti/dataset")
    ap.add_argument("--sequence", type=int, required=True)
    ap.add_argument("--resolution", type=float, default=0.10)
    ap.add_argument("--min-range", type=float, default=5.0)
    ap.add_argument("--max-range", type=float, default=30.0)
    ap.add_argument("--n-scans", type=int, default=100)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    seq = f"{args.sequence:02d}"
    sd = Path(args.dataset_path) / "sequences" / seq
    Tr = load_calib(sd / "calib.txt")
    poses = load_poses(sd / "poses.txt")
    scans = sorted((sd / "velodyne").glob("*.bin"))
    n = min(len(scans), len(poses), args.n_scans)
    res = args.resolution
    print(f"seq {seq}: {n} scans res={res} gate[{args.min_range},{args.max_range}]")

    fkey, fcnt, hkey, hcnt = [], [], [], []
    for i in range(n):
        pts = np.fromfile(scans[i], dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        rng = np.linalg.norm(pts, axis=1)
        pts = pts[(rng >= args.min_range) & (rng <= args.max_range)]
        if len(pts) == 0:
            continue
        T = poses[i] @ Tr
        O = T[:3, 3]
        E = (T[:3, :3] @ pts.T).T + T[:3, 3]
        hk, hc = np.unique(pack(np.floor(E / res).astype(np.int64)), return_counts=True)
        hkey.append(hk); hcnt.append(hc)
        # DDA free traversal, chunked over rays; collapse to per-scan (key,count)
        scan_free = []
        for s in range(0, len(E), CHUNK):
            scan_free.append(dda_free_keys(O, E[s:s + CHUNK], res))
        fk = np.concatenate(scan_free) if scan_free else np.zeros(0, np.int64)
        if len(fk):
            uk, uc = np.unique(fk, return_counts=True)   # per-scan ray count per voxel
            fkey.append(uk); fcnt.append(uc)
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  scan {i+1}/{n}: rays={len(E)} distinct_free_voxels={sum(len(c) for c in fkey):,}")

    fk_u, fk_c = merge_counts(fkey, fcnt)
    hk_u, hk_c = merge_counts(hkey, hcnt)

    # union of all observed voxels
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
