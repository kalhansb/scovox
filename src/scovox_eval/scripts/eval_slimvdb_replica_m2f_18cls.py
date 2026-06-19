#!/usr/bin/env python3
"""SLIM-VDB Replica m2f mIoU in the collapsed 18-class SCovox space.

SLIM-VDB's voxels.bin for the m2f condition stores the raw ADE-150 class ID
written at integration time (since SLIM-VDB consumed the m2f ADE PNGs as a
generic integer label). We apply an ADE-150 → SCovox-18 LUT (built from
substring matching ADE class names against CATEGORY_COLORS keys) to fairly
compare against SCovox's 18-class GT.

The same ADE LUT is used by SCovox m2f ingestion (replica_replay_node with
semantic_subdir starting with `semantic_m2f_ade`).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_slimvdb_replica_miou import load_voxels_bin, voxel_key, compute_miou
from eval_scovox_replica import SCOVOX_CATS, build_gt_scovox_voxels


def load_ade_lut(lut_path: Path) -> np.ndarray:
    """Return LUT: ade_stored_id (0..150) → scovox_18 id (0..18)."""
    d = np.load(lut_path, allow_pickle=True)
    return d["lut"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--voxels_dir", type=Path, required=True,
                    help="Dir with <scene>/voxels.bin (SLIM-VDB m2f output).")
    ap.add_argument("--scenes", nargs="+",
                    default=["room0", "room1", "room2",
                             "office0", "office1", "office2", "office3", "office4"])
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--num_classes", type=int, default=19)  # 0..18
    ap.add_argument("--ade_lut",
                    default=str(Path(__file__).resolve().parent / "ade150_to_scovox18.npz"))
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args()

    lut = load_ade_lut(Path(args.ade_lut))
    print(f"[mIoU] SLIM-VDB m2f on Replica (18-cls via ADE LUT), "
          f"voxel={args.voxel_size} m, stride={args.stride}")

    rows = []
    for scene in args.scenes:
        vbin = args.voxels_dir / scene / "voxels.bin"
        if not vbin.exists():
            print(f"  [skip] no {vbin}"); continue

        xyz, cls_ade = load_voxels_bin(vbin)
        cls_ade = np.clip(cls_ade, 0, lut.size - 1)
        cls_18 = lut[cls_ade]  # 0..18

        keys = voxel_key(xyz, args.voxel_size)
        # One label per voxel (most frequent in cell)
        from collections import defaultdict, Counter
        bucket = defaultdict(list)
        for k, c in zip(keys, cls_18):
            bucket[int(k)].append(int(c))
        pred = {k: Counter(cs).most_common(1)[0][0] for k, cs in bucket.items()}

        gt = build_gt_scovox_voxels(args.replica_root / scene,
                                    args.voxel_size,
                                    args.n_frames, args.stride)

        res = compute_miou(pred, gt, args.num_classes)
        rows.append((scene, res["miou"], res["n_intersect"]))
        print(f"  [{scene}] mIoU={res['miou']:.4f}  "
              f"n_pred={res['n_pred']:>6d}  n_gt={res['n_gt']:>6d}  "
              f"n_match={res['n_intersect']:>6d}")

    if rows:
        mious = [r[1] for r in rows]
        print(f"\n  Mean mIoU over {len(rows)} scenes: {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    if args.out_csv and rows:
        with open(args.out_csv, "w") as f:
            f.write("scene,miou,n_voxels\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]:.6f},{r[2]}\n")
        print(f"  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
