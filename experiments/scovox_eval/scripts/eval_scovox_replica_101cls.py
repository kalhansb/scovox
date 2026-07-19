#!/usr/bin/env python3
"""SCovox Replica mIoU in the native Replica 101-class space.

Same GT construction pipeline as `eval_slimvdb_replica_miou.py` (project depth
+ semantic_gt_fixed PNGs through GT poses at the map's voxel resolution, then
majority-vote per voxel). Scores SCovox's `semantic_class` field in the NPZ
directly — no 18-class collapse — so the number is directly comparable to
SLIM-VDB's 101-class mIoU.

Usage:
    python eval_scovox_replica_101cls.py \
        --replica_root data/replica_niceslam \
        --npz_root src/robot_sw/distributed_mapping/scovox_eval/results/replica_gt_5cm \
        --voxel_size 0.05 --n_frames 2000 --stride 10
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_slimvdb_replica_miou import (
    build_gt_voxels, voxel_key, compute_miou,
)


def load_scovox_pred_voxels_101(npz_path: Path, voxel_size: float):
    d = np.load(npz_path)
    xyz = d["points"].astype(np.float32)
    cls = d["semantic_class"].astype(np.int32)   # native Replica class IDs
    keys = voxel_key(xyz, voxel_size)
    # In case SCovox saved duplicates at the same cell, keep first observed.
    seen = {}
    for k, c in zip(keys, cls):
        ki = int(k)
        if ki not in seen:
            seen[ki] = int(c)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--npz_root", type=Path, required=True,
                    help="Directory with <scene>/scovox.npz")
    ap.add_argument("--scenes", nargs="+",
                    default=["room0","room1","room2","office0","office1","office2","office3","office4"])
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--num_classes", type=int, default=102,
                    help="Replica has 101 classes; +1 for the void/0 label.")
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    print(f"[mIoU] SCovox on Replica (101-cls, native), voxel={args.voxel_size} m, "
          f"stride={args.stride}, scenes={args.scenes}")
    for scene in args.scenes:
        npz = args.npz_root / scene / "scovox.npz"
        if not npz.exists():
            print(f"  [skip] no npz {npz}"); continue
        pred = load_scovox_pred_voxels_101(npz, args.voxel_size)
        gt = build_gt_voxels(args.replica_root / scene, args.voxel_size,
                             args.n_frames, args.stride)
        m = compute_miou(pred, gt, args.num_classes)
        print(f"  {scene:<8s}  pred={m['n_pred']:>8d}  gt={m['n_gt']:>8d}  "
              f"∩={m['n_intersect']:>8d}  mIoU={m['miou']:.4f}")
        rows.append((scene, m['miou'], m['n_pred'], m['n_gt'], m['n_intersect']))

    if rows:
        vals = np.array([r[1] for r in rows])
        print(f"\n=== SCovox Replica mIoU (101-cls) = {vals.mean():.4f} ± {vals.std():.4f} ===")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w") as f:
            f.write("scene,miou,n_pred,n_gt,n_intersect\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
