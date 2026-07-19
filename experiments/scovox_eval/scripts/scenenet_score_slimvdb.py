#!/usr/bin/env python3
"""Strict bucket-IoU evaluator for SLIM-VDB SceneNet runs.

Mirror of scenenet_compute_metrics.py but reads SLIM-VDB's `voxels.bin` flat
binary format (16-byte records: float32 x, y, z, int32 argmax_class) emitted
by build-scenenet/bin/vdb_to_voxels.

Same union-set bucket-IoU protocol as the SCovox scorer:
- Voxelise SLIM-VDB world-coords to int64 keys at GT resolution (5 cm)
- Union with GT keys, missing voxels on either side → class 0 (Unknown)
- Per-class TP/FP/FN over [0..13], ignore class 0 from mIoU mean
- Matches scenenet_compute_metrics.py byte-for-byte on union/intersect/IoU

Usage:
  python3 scenenet_score_slimvdb.py \
      --voxels_root third_party_sw/slim_vdb/outputs/scenenet_val \
      --gt_root     data/scenenet/val_preprocessed \
      --out_csv     third_party_sw/slim_vdb/outputs/scenenet_val/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


NYU_13_CLASSES = [
    (0,  "Unknown"),
    (1,  "Bed"),     (2,  "Books"),   (3,  "Ceiling"), (4,  "Chair"),
    (5,  "Floor"),   (6,  "Furniture"),(7,  "Objects"),(8,  "Picture"),
    (9,  "Sofa"),    (10, "Table"),   (11, "TV"),     (12, "Wall"),
    (13, "Window"),
]
NUM_CLASSES = 14


def _voxel_keys(coords: np.ndarray, resolution: float) -> np.ndarray:
    """Pack (x,y,z) into int64 keys — same scheme as scenenet_build_gt.py."""
    inv_res = 1.0 / resolution
    c = np.floor(coords * inv_res).astype(np.int64)
    k = ((c[:, 0] & 0xFFFFFF) << 40
         | (c[:, 1] & 0xFFFFF) << 20
         |  (c[:, 2] & 0xFFFFF))
    return k


def _load_voxels_bin(path: Path):
    """SLIM-VDB vdb_to_voxels.cpp record: (f32 x, f32 y, f32 z, i32 cls), 16 B."""
    raw = np.fromfile(path, dtype=np.uint8)
    rec = raw.view(np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "<i4")]))
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
    cls = rec["c"].astype(np.int32)
    return xyz, cls


def _per_class_iou(gt_keys, gt_labels, pred_keys, pred_labels,
                   num_classes=NUM_CLASSES):
    all_keys = np.union1d(gt_keys, pred_keys)
    n = len(all_keys)

    gt_lookup = {int(k): int(l) for k, l in zip(gt_keys, gt_labels)}
    pr_lookup = {int(k): int(l) for k, l in zip(pred_keys, pred_labels)}

    gt = np.zeros(n, dtype=np.int32)
    pr = np.zeros(n, dtype=np.int32)
    for i, k in enumerate(all_keys):
        ki = int(k)
        gt[i] = gt_lookup.get(ki, 0)
        pr[i] = pr_lookup.get(ki, 0)

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        tp[c] = np.sum((gt == c) & (pr == c))
        fp[c] = np.sum((gt != c) & (pr == c))
        fn[c] = np.sum((gt == c) & (pr != c))

    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), 0.0)
    return iou, tp, fp, fn, n, int(np.sum((gt > 0) & (gt == pr)))


def score_one(gt_npz: Path, voxels_bin: Path, resolution=None,
              ignore_classes=(0,), verbose=False):
    gt = np.load(gt_npz)
    if resolution is None:
        resolution = float(gt["resolution"])

    gt_keys = gt["keys"].astype(np.int64)
    gt_labels = gt["labels"].astype(np.int32)

    pred_xyz, pred_labels = _load_voxels_bin(voxels_bin)
    pred_keys = _voxel_keys(pred_xyz, resolution)

    # Multiple SLIM-VDB voxels can hash to the same coarse 5 cm GT key when its
    # internal grid is finer or when DDA writes neighbours. Keep first.
    if len(np.unique(pred_keys)) != len(pred_keys):
        u, idx = np.unique(pred_keys, return_index=True)
        pred_keys = u
        pred_labels = pred_labels[idx]

    iou, tp, fp, fn, n_union, n_intersect = _per_class_iou(
        gt_keys, gt_labels, pred_keys, pred_labels)

    ignore_mask = np.zeros(NUM_CLASSES, dtype=bool)
    for c in ignore_classes:
        ignore_mask[c] = True
    has_data = (tp + fp + fn) > 0
    mask = has_data & ~ignore_mask
    miou = float(iou[mask].mean()) if mask.any() else 0.0

    if verbose:
        print(f"\ngt={gt_npz}  pred={voxels_bin}")
        print(f"  resolution: {resolution} m   union voxels: {n_union}   "
              f"intersection: {n_intersect}")
        print(f"  mIoU = {miou:.4f}  (over {mask.sum()} classes, "
              f"ignoring {list(ignore_classes)})")
        for c, name in NYU_13_CLASSES:
            flag = ("  [ignored]" if c in ignore_classes
                    else ("  [absent]" if not has_data[c] else ""))
            print(f"    {c:2d} {name:10s}: {iou[c]:.4f}  "
                  f"tp={tp[c]:>6d} fp={fp[c]:>6d} fn={fn[c]:>6d}{flag}")

    return {
        "miou": miou,
        "iou": iou.tolist(),
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "n_union": int(n_union),
        "n_intersect": int(n_intersect),
        "n_classes_in_mean": int(mask.sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voxels_root", required=True,
                   help="<root>/<seq>/voxels.bin layout")
    p.add_argument("--gt_root", required=True,
                   help="<root>/<seq>/gt_5cm.npz layout (same as SCovox scorer)")
    p.add_argument("--resolution", type=float, default=None,
                   help="default: read from GT NPZ")
    p.add_argument("--ignore_classes", default="0")
    p.add_argument("--print_per_class", action="store_true")
    p.add_argument("--out_csv")
    args = p.parse_args()

    ignore = tuple(int(x) for x in args.ignore_classes.split(",") if x.strip())

    voxels_root = Path(args.voxels_root)
    gt_root = Path(args.gt_root)
    seqs = sorted([d.name for d in voxels_root.iterdir() if d.is_dir()])

    print(f"scoring {len(seqs)} SLIM-VDB cells under {voxels_root}")
    rows = []
    for seq in seqs:
        voxels_bin = voxels_root / seq / "voxels.bin"
        gt_npz = gt_root / seq / "gt_5cm.npz"
        if not voxels_bin.exists() or not gt_npz.exists():
            print(f"  SKIP {seq}: voxels={voxels_bin.exists()} gt={gt_npz.exists()}")
            continue
        r = score_one(gt_npz, voxels_bin, resolution=args.resolution,
                      ignore_classes=ignore, verbose=args.print_per_class)
        rows.append({"seq": seq, **r})
        if not args.print_per_class:
            print(f"  {seq:12s} mIoU = {r['miou']:.4f}   "
                  f"({r['n_classes_in_mean']} classes, {r['n_union']} union)")

    if rows:
        mean = float(np.mean([r["miou"] for r in rows]))
        std = float(np.std([r["miou"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0
        se = std / np.sqrt(len(rows)) if len(rows) > 1 else 0.0
        print(f"\n=== SLIM-VDB SceneNet across {len(rows)} cells: "
              f"mean mIoU = {mean:.4f} ± {std:.4f}  (SE {se:.4f}) ===")

    if args.out_csv and rows:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            header = ["seq", "miou", "n_classes_in_mean", "n_union", "n_intersect"]
            header += [f"iou_{c}_{name.lower()}" for c, name in NYU_13_CLASSES]
            w.writerow(header)
            for r in rows:
                row = [r["seq"], f'{r["miou"]:.6f}', r["n_classes_in_mean"],
                       r["n_union"], r["n_intersect"]]
                row += [f'{x:.6f}' for x in r["iou"]]
                w.writerow(row)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
