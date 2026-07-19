#!/usr/bin/env python3
"""Unknown-vs-ambiguous AUC on Exp 7 end-of-trial fused snapshots.

Uses the per-robot dscovox NPZ (a_occ, a_free, occupancy_prob, points in
<robot>/odom frame) and the flatforest voxel GT (world frame, gt_binary).
Points are transformed to world via the spawn pose from
exp7_start_configs.csv, then filtered to the GT z-slab before the AUC.

Target: SCovox AUC > 0.90 (the "figure that sells the paper").
"""
from __future__ import annotations
import csv, os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scovox_eval.metrics.unknown_vs_ambiguous import (
    compute_unknown_vs_ambiguous, beta_variance)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
EXP7_DIR = Path(__file__).resolve().parents[5] / "src/robot_sw/eig_exploration_planner/results"
CONFIG_CSV = Path(__file__).resolve().parents[5] / "src/robot_sw/eig_exploration_planner/scripts/exp7_start_configs.csv"
GT_FLAT = RESULTS_DIR / "flatforest_10cm" / "gt_voxels.npz"


def to_world(pts, spawn):
    x0, y0, yaw = spawn
    c, s = np.cos(yaw), np.sin(yaw)
    x = c * pts[:, 0] - s * pts[:, 1] + x0
    y = s * pts[:, 0] + c * pts[:, 1] + y0
    z = pts[:, 2]
    return np.column_stack([x, y, z]).astype(np.float32)


def load_configs():
    out = {}
    with open(CONFIG_CSV) as fh:
        for row in csv.reader(fh):
            if not row or row[0] in ("", "config_id") or row[0].startswith("#"):
                continue
            cid, ax, ay, ayaw, bx, by, byaw = row
            out[cid] = {
                "atlas":  (float(ax), float(ay), float(ayaw)),
                "bestla": (float(bx), float(by), float(byaw)),
            }
    return out


def run_one(npz_path: Path, spawn, gt_surface_pts, resolution=0.10,
            prob_lo=0.4, prob_hi=0.6):
    d = np.load(npz_path)
    pts = to_world(d["points"].astype(np.float32), spawn)
    zlo, zhi = gt_surface_pts[:, 2].min(), gt_surface_pts[:, 2].max()
    m = (pts[:, 2] >= zlo) & (pts[:, 2] <= zhi)
    if m.sum() == 0:
        return None
    pts = pts[m]
    occ = d["occupancy_prob"][m]
    a_occ = d["a_occ"][m]
    a_free = d["a_free"][m]
    return compute_unknown_vs_ambiguous(
        points=pts, occupancy_prob=occ, a_occ=a_occ, a_free=a_free,
        gt_surface_points=gt_surface_pts, resolution=resolution,
        prob_lo=prob_lo, prob_hi=prob_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planners", nargs="+",
                    default=["eig", "entropy", "frontier", "random"])
    ap.add_argument("--configs", nargs="+", default=["c1", "c2", "c3"])
    ap.add_argument("--world", default="flatforest",
                    help="Only flatforest has voxel GT at the moment")
    ap.add_argument("--resolution", type=float, default=0.10)
    ap.add_argument("--prob-lo", type=float, default=0.4)
    ap.add_argument("--prob-hi", type=float, default=0.6)
    ap.add_argument("--out", default=str(RESULTS_DIR / "exp7_unknown_vs_ambiguous.csv"))
    args = ap.parse_args()

    gt = np.load(GT_FLAT)
    gt_surface = gt["points"][gt["gt_binary"] > 0.5].astype(np.float32)
    print(f"GT surface points: {len(gt_surface)} from {GT_FLAT}")

    spawns = load_configs()
    rows = []
    for planner in args.planners:
        for cfg in args.configs:
            for robot in ("atlas", "bestla"):
                stem = f"exp7_{planner}_{args.world}_{cfg}"
                p = EXP7_DIR / f"{stem}_{robot}_dscovox.npz"
                if not p.exists():
                    print(f"  MISS {p.name}"); continue
                spawn = spawns[cfg][robot]
                r = run_one(p, spawn, gt_surface, args.resolution,
                            args.prob_lo, args.prob_hi)
                if r is None:
                    print(f"  {stem}_{robot}: no points in GT z-slab")
                    continue
                auc = r["auc"]
                print(f"  {stem}_{robot:<6s}  AUC={auc:.3f}  "
                      f"n_unknown={r['n_unknown']:>6d}  "
                      f"n_boundary={r['n_boundary']:>6d}  "
                      f"(uncertain band = {r['n_total_uncertain']})")
                rows.append((planner, cfg, robot, auc,
                             r['n_unknown'], r['n_boundary'],
                             r['n_total_uncertain']))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("planner,config,robot,auc,n_unknown,n_boundary,n_uncertain\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"\nSummary CSV: {out}")
    if rows:
        aucs = [r[3] for r in rows if not np.isnan(r[3])]
        print(f"AUC across {len(aucs)} robot-trials: "
              f"mean={np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
              f"min={np.min(aucs):.3f}  max={np.max(aucs):.3f}")


if __name__ == "__main__":
    main()
