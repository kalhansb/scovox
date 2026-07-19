#!/usr/bin/env python3
"""Aggregate mIoU across KITTI seq 08 PolarSeg ablation cells."""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scovox_eval.metrics.miou import compute_miou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--results_root", type=Path, required=True)
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--max_dist", type=float, default=0.10)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    gt = np.load(args.gt)
    gt_pts = gt["points"]; gt_lbl = gt["semantic_class"].astype(int)
    tree = KDTree(gt_pts)
    print(f"GT voxels={len(gt_pts)}", flush=True)

    cells = sorted(d for d in args.results_root.iterdir() if d.is_dir())
    rows = []
    for cell_dir in cells:
        npz = cell_dir / "scovox.npz"
        if not npz.exists():
            print(f"[skip] {cell_dir.name}: no npz")
            continue
        pr = np.load(npz)
        pr_pts = pr["points"]; pr_lbl = pr["semantic_class"].astype(int)
        d, i = tree.query(pr_pts)
        m = d < args.max_dist
        matched_pr = pr_lbl[m]; matched_gt = gt_lbl[i[m]]
        r = compute_miou(matched_pr, matched_gt,
                         num_classes=args.num_classes, ignore_label=0)
        miou = r["miou"] if isinstance(r, dict) else float(r)
        rows.append({
            "cell": cell_dir.name,
            "miou": round(miou, 4),
            "n_pred": len(pr_pts),
            "n_matched": int(m.sum()),
        })
        print(f"{cell_dir.name:14s} mIoU={miou:.4f} pred={len(pr_pts)} matched={int(m.sum())}", flush=True)

    if args.csv:
        cols = ["cell", "miou", "n_pred", "n_matched"]
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\n[csv] {args.csv}")

    print("\n=== summary ===")
    base = next((r for r in rows if r["cell"] == "baseline"), None)
    if base:
        print(f"baseline mIoU={base['miou']:.4f}")
        for r in sorted(rows, key=lambda r: -r["miou"]):
            if r["cell"] == "baseline": continue
            d = r["miou"] - base["miou"]
            sign = "+" if d >= 0 else ""
            print(f"  {r['cell']:14s} mIoU={r['miou']:.4f} ({sign}{d:.4f})")


if __name__ == "__main__":
    main()
