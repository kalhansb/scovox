#!/usr/bin/env python3
"""SLIM-VDB Replica mIoU in the collapsed 18-class SCovox space.

Mirrors SLIM-VDB's SceneNet protocol (bin fine-grained predictions into a
smaller canonical class set). Uses the same 101→18 LUT that SCovox applies
internally at integration time, then computes mIoU against GT built in the
same 18-class space. Directly comparable to SCovox's 18-class mIoU.
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_slimvdb_replica_miou import load_voxels_bin, voxel_key, compute_miou
from eval_scovox_replica import (
    SCOVOX_CATS, class_id_remap, build_gt_scovox_voxels,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--voxels_dir", type=Path, required=True,
                    help="Dir with <scene>/voxels.bin (SLIM-VDB output).")
    ap.add_argument("--scenes", nargs="+",
                    default=["room0","room1","room2","office0","office1","office2","office3","office4"])
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--num_classes", type=int, default=19)  # 0..18
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    print(f"[mIoU] SLIM-VDB on Replica (18-cls collapsed), voxel={args.voxel_size} m, "
          f"stride={args.stride}, scenes={args.scenes}")

    for scene in args.scenes:
        vbin = args.voxels_dir / scene / "voxels.bin"
        if not vbin.exists():
            print(f"  [skip] no {vbin}"); continue

        # Load 101-class voxels, collapse to 18-class via the same LUT SCovox uses.
        xyz, cls_101 = load_voxels_bin(vbin)
        info = json.load(open(args.replica_root / scene / "info_semantic.json"))
        lut = class_id_remap(info)
        cls_101 = np.clip(cls_101, 0, lut.size - 1)
        cls_18 = lut[cls_101]  # 0..18

        keys = voxel_key(xyz, args.voxel_size)
        # If multiple input voxels collapse to the same 5 cm cell with different
        # 18-class ids (boundary effect), majority-vote.
        counts = defaultdict(lambda: defaultdict(int))
        for k, c in zip(keys, cls_18):
            if c > 0:  # skip unmapped (→ 0)
                counts[int(k)][int(c)] += 1
        pred = {k: max(d.items(), key=lambda kv: kv[1])[0] for k, d in counts.items()}

        # GT in same 18-class space.
        gt = build_gt_scovox_voxels(args.replica_root / scene,
                                    args.voxel_size, args.n_frames, args.stride)
        m = compute_miou(pred, gt, args.num_classes)
        print(f"  {scene:<8s}  pred={m['n_pred']:>8d}  gt={m['n_gt']:>8d}  "
              f"∩={m['n_intersect']:>8d}  mIoU={m['miou']:.4f}")
        rows.append((scene, m['miou'], m['n_pred'], m['n_gt'], m['n_intersect']))

    if rows:
        vals = np.array([r[1] for r in rows])
        print(f"\n=== SLIM-VDB Replica mIoU (18-cls, collapsed) = "
              f"{vals.mean():.4f} ± {vals.std():.4f} ===")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w") as f:
            f.write("scene,miou,n_pred,n_gt,n_intersect\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
