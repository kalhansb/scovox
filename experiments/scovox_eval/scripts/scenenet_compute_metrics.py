#!/usr/bin/env python3
"""Strict bucket-IoU mIoU evaluator for SCovox SceneNet runs.

Models after scovox/scripts/eval_scovox_kitti_miou_fixed.py — the strict-protocol
evaluator that matches the published SLIM-VDB protocol (mIoU on `pred ∪ gt` voxel
keys; missing voxels in either map → class 0).

Inputs:
- GT NPZ from scenenet_build_gt.py: {coords, labels, keys, resolution, ...}
- SCovox NPZ from pointcloud_to_npz: {points (Nx3 float32 m), semantic_class (uint8)}

The SCovox `points` are already voxel-centres at the SCovox grid resolution. We
voxelise them at the GT NPZ's resolution (default 5 cm) using the same int-key
packing as scenenet_build_gt.py.

Usage:
  python3 scenenet_compute_metrics.py \
      --gt_npz data/scenenet/val_preprocessed/0_223/gt_5cm.npz \
      --pred_npz results/scenenet_005cm/0_223/scovox_dirichlet.npz \
      [--ignore_classes 0] [--print_per_class]

  python3 scenenet_compute_metrics.py \
      --batch_root results/scenenet_val_iter6 \
      --gt_root   data/scenenet/val_preprocessed \
      --variant scovox_dirichlet \
      --out_csv results/scenenet_val_iter6/summary.csv
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
    """Pack (x,y,z) into int64 keys with the same scheme as scenenet_build_gt.py."""
    inv_res = 1.0 / resolution
    c = np.floor(coords * inv_res).astype(np.int64)
    k = ((c[:, 0] & 0xFFFFFF) << 40
         | (c[:, 1] & 0xFFFFF) << 20
         |  (c[:, 2] & 0xFFFFF))
    return k


def _per_class_iou(gt_keys: np.ndarray, gt_labels: np.ndarray,
                   pred_keys: np.ndarray, pred_labels: np.ndarray,
                   num_classes: int = NUM_CLASSES):
    """Strict bucket-IoU on pred ∪ gt voxel keys.

    Voxels present in only one side get class 0 (Unknown) on the missing side.

    Returns: (per_class_iou ndarray, per_class_tp, per_class_fp, per_class_fn,
              n_voxels_in_union, n_voxels_in_intersection)
    """
    # Union of keys
    all_keys = np.union1d(gt_keys, pred_keys)
    n = len(all_keys)

    # Build dense per-key label arrays (0 = unknown / not-in-map)
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


def score_one(gt_npz: Path, pred_npz: Path, resolution: float | None = None,
              ignore_classes=(0,), verbose=True):
    gt = np.load(gt_npz)
    pred = np.load(pred_npz)

    if resolution is None:
        resolution = float(gt["resolution"])

    gt_keys = gt["keys"].astype(np.int64)
    gt_labels = gt["labels"].astype(np.int32)

    # SCovox NPZ → voxelise points to keys, take semantic_class as label
    if "points" not in pred.files or "semantic_class" not in pred.files:
        raise ValueError(f"{pred_npz} missing required keys 'points'/'semantic_class'; "
                         f"has {list(pred.files)}")
    pred_keys = _voxel_keys(pred["points"].astype(np.float64), resolution)
    pred_labels = pred["semantic_class"].astype(np.int32)

    # If multiple SCovox voxels map to the same GT key (shouldn't happen if SCovox res
    # ≤ GT res, but be defensive), keep first.
    if len(np.unique(pred_keys)) != len(pred_keys):
        u, idx = np.unique(pred_keys, return_index=True)
        pred_keys = u
        pred_labels = pred_labels[idx]

    iou, tp, fp, fn, n_union, n_intersect = _per_class_iou(
        gt_keys, gt_labels, pred_keys, pred_labels)

    ignore_mask = np.zeros(NUM_CLASSES, dtype=bool)
    for c in ignore_classes:
        ignore_mask[c] = True
    # Exclude classes that have no GT *and* no pred (denom == 0) from the mean
    has_data = (tp + fp + fn) > 0
    mask = has_data & ~ignore_mask
    miou = float(iou[mask].mean()) if mask.any() else 0.0

    if verbose:
        print(f"\ngt={gt_npz}  pred={pred_npz}")
        print(f"  resolution: {resolution} m   union voxels: {n_union}   "
              f"intersection: {n_intersect}")
        print(f"  mIoU = {miou:.4f}  (over {mask.sum()} classes, ignoring {list(ignore_classes)})")
        print(f"  per-class IoU:")
        for c, name in NYU_13_CLASSES:
            flag = ""
            if c in ignore_classes:
                flag = "  [ignored]"
            elif not has_data[c]:
                flag = "  [absent]"
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
    # Single-cell or batch mode
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred_npz", help="Single SCovox NPZ to score")
    g.add_argument("--batch_root",
                   help="Per-trajectory results dir, expects <root>/<seq>/<variant>.npz")
    p.add_argument("--gt_npz", help="Single GT NPZ (when --pred_npz)")
    p.add_argument("--gt_root", help="Per-trajectory GT root, expects <root>/<seq>/gt_5cm.npz")
    p.add_argument("--variant", default="scovox_dirichlet",
                   help="In batch mode, NPZ filename stem")
    p.add_argument("--resolution", type=float, default=None,
                   help="Voxel resolution; default = read from GT NPZ")
    p.add_argument("--ignore_classes", default="0",
                   help="Comma-separated classes excluded from mIoU mean (default 0=Unknown)")
    p.add_argument("--print_per_class", action="store_true",
                   help="Print per-class IoU for each cell")
    p.add_argument("--out_csv", help="Write per-cell mIoU + per-class IoU to CSV")
    args = p.parse_args()

    ignore = tuple(int(x) for x in args.ignore_classes.split(",") if x.strip())

    rows = []
    if args.pred_npz:
        if not args.gt_npz:
            p.error("--gt_npz required with --pred_npz")
        r = score_one(Path(args.gt_npz), Path(args.pred_npz),
                      resolution=args.resolution,
                      ignore_classes=ignore, verbose=True)
        rows.append({"seq": Path(args.pred_npz).parent.name, "variant": args.variant, **r})
    else:
        if not args.gt_root:
            p.error("--gt_root required with --batch_root")
        batch_root = Path(args.batch_root)
        gt_root = Path(args.gt_root)
        seqs = sorted([d.name for d in batch_root.iterdir() if d.is_dir()])
        print(f"batch scoring {len(seqs)} cells under {batch_root}")
        for seq in seqs:
            gt_npz = gt_root / seq / "gt_5cm.npz"
            pred_npz = batch_root / seq / f"{args.variant}.npz"
            if not gt_npz.exists() or not pred_npz.exists():
                print(f"  SKIP {seq}: gt={gt_npz.exists()} pred={pred_npz.exists()}")
                continue
            r = score_one(gt_npz, pred_npz, resolution=args.resolution,
                          ignore_classes=ignore, verbose=args.print_per_class)
            rows.append({"seq": seq, "variant": args.variant, **r})
            if not args.print_per_class:
                print(f"  {seq:12s} mIoU = {r['miou']:.4f}   "
                      f"({r['n_classes_in_mean']} classes, {r['n_union']} union voxels)")

        if rows:
            mean = float(np.mean([r["miou"] for r in rows]))
            print(f"\n=== {args.variant} across {len(rows)} cells: mean mIoU = {mean:.4f} ===")

    if args.out_csv and rows:
        # Per-cell wide CSV with per-class IoU columns
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            header = ["seq", "variant", "miou", "n_classes_in_mean",
                      "n_union", "n_intersect"]
            header += [f"iou_{c}_{name.lower()}" for c, name in NYU_13_CLASSES]
            w.writerow(header)
            for r in rows:
                row = [r["seq"], r["variant"], f'{r["miou"]:.6f}',
                       r["n_classes_in_mean"], r["n_union"], r["n_intersect"]]
                row += [f'{x:.6f}' for x in r["iou"]]
                w.writerow(row)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
