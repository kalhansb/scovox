#!/usr/bin/env python3
"""Per-sequence + mean mIoU for scovox KITTI runs vs voxelized GT.

Matches each predicted voxel to the nearest GT voxel (KDTree, <=MAX_DIST) and
computes 19-class mIoU (ignore 0=unlabeled), exactly as the paper's
semantickitti_polarseg_miou.py does.

Usage: score_kitti.py <run_root> [--tag gt|polarseg]
  run_root/<seq>/scovox.npz  are scored against  results/kitti_gt/<seq>/gt.npz
"""
import argparse
import os
import sys
import numpy as np
from scipy.spatial import KDTree

WS = "/home/kalhan/Projects/scovox_ws"
sys.path.insert(0, os.path.join(WS, "experiments", "scovox_eval"))
from scovox_eval.metrics.miou import compute_miou  # noqa: E402

GT_ROOT = os.path.join(WS, "experiments", "results", "kitti_gt")
SEQS = ["06", "07", "08", "09", "10"]
NUM_CLASSES = 20
MAX_DIST = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", help="dir with <seq>/scovox.npz")
    ap.add_argument("--tag", default="scovox")
    ap.add_argument("--npz-name", default="scovox.npz")
    args = ap.parse_args()

    rows = {}
    for seq in SEQS:
        gt_file = os.path.join(GT_ROOT, seq, "gt.npz")
        pr_file = os.path.join(args.run_root, seq, args.npz_name)
        if not os.path.exists(gt_file):
            print(f"  MISS gt {seq}"); continue
        if not os.path.exists(pr_file):
            print(f"  MISS pred {seq}"); continue
        gt = np.load(gt_file)
        gt_pts, gt_lbl = gt["points"], gt["semantic_class"].astype(int)
        pr = np.load(pr_file)
        pr_pts, pr_lbl = pr["points"], pr["semantic_class"].astype(int)
        d, i = KDTree(gt_pts).query(pr_pts)
        m = d < MAX_DIST
        r = compute_miou(pr_lbl[m], gt_lbl[i[m]],
                         num_classes=NUM_CLASSES, ignore_label=0)
        rows[seq] = r["miou"]
        print(f"  seq {seq}: pred_vox={len(pr_pts):>8d} matched={int(m.sum()):>8d} "
              f"mIoU={r['miou']:.4f}")

    if rows:
        vals = list(rows.values())
        mean = float(np.mean(vals))
        print(f"\n  MEAN ({args.tag}) = {mean:.4f} +- {np.std(vals):.4f}  (n={len(vals)})")
        # write summary.csv next to run_root
        summ = os.path.join(args.run_root, "summary.csv")
        with open(summ, "w") as f:
            f.write("seq,miou\n")
            for s in SEQS:
                f.write(f"{s},{rows.get(s, '')}\n")
            f.write(f"mean,{mean:.4f}\n")
        print(f"  wrote {summ}")


if __name__ == "__main__":
    main()
