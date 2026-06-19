#!/usr/bin/env python3
"""Per-sequence mIoU for SCovox PolarSeg runs (seq 06-10, 10cm, 100 scans)."""
import os, sys, numpy as np
from scipy.spatial import KDTree

sys.path.insert(0, os.path.expanduser(
    "~/projects/HMR_Exploration_Experiment/hmr_exploration_ws/"
    "src/robot_sw/distributed_mapping/scovox_eval"))
from scovox_eval.metrics.miou import compute_miou

ROOT = os.path.expanduser(
    "~/projects/HMR_Exploration_Experiment/hmr_exploration_ws/"
    "src/robot_sw/distributed_mapping/scovox_eval/results/semantickitti_polarseg_10cm")
SEQS = ["06", "07", "08", "09", "10"]
VARIANTS = ["scovox", "scovox_mv", "scovox_np"]
NUM_CLASSES = 20
MAX_DIST = 0.10

per_seq = {}
for seq in SEQS:
    sdir = os.path.join(ROOT, seq)
    gt_file = os.path.join(sdir, "gt.npz")
    if not os.path.exists(gt_file):
        print(f"MISS GT {seq}"); continue
    gt = np.load(gt_file)
    gt_pts = gt["points"]
    gt_lbl = gt["semantic_class"].astype(int)
    tree = KDTree(gt_pts)
    print(f"\n=== seq {seq}  GT voxels={len(gt_pts)} ===")
    per_seq[seq] = {}
    for v in VARIANTS:
        p = os.path.join(sdir, f"{v}.npz")
        if not os.path.exists(p): print(f"  MISS {v}"); continue
        pr = np.load(p)
        pr_pts = pr["points"]; pr_lbl = pr["semantic_class"].astype(int)
        d, i = tree.query(pr_pts)
        m = d < MAX_DIST
        matched_pr = pr_lbl[m]; matched_gt = gt_lbl[i[m]]
        r = compute_miou(matched_pr, matched_gt,
                         num_classes=NUM_CLASSES, ignore_label=0)
        per_seq[seq][v] = r["miou"]
        print(f"  {v:11s}  pred_vox={len(pr_pts):>8d}  "
              f"matched={int(m.sum()):>8d}  mIoU={r['miou']:.4f}")

print("\n=== PER-SEQUENCE mIoU ===")
print(f"{'seq':<4} " + " ".join(f"{v:>10s}" for v in VARIANTS))
for seq in SEQS:
    if seq not in per_seq: continue
    print(f"{seq:<4} " + " ".join(f"{per_seq[seq].get(v, float('nan')):>10.4f}" for v in VARIANTS))

print("\n=== MEAN (06-10) ===")
for v in VARIANTS:
    vals = [per_seq[s][v] for s in SEQS if v in per_seq.get(s, {})]
    if vals:
        print(f"  {v:11s}  mIoU={np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")
