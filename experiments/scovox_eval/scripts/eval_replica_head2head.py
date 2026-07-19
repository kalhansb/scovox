#!/usr/bin/env python3
"""Replica head-to-head geometric comparison: SCovox (.npz) and SLIM-VDB (.ply)
× {GT, m2f} conditions. Chamfer (L2) + F@5cm + precision + recall, unified
format so the 4 conditions are in one table.

Usage:
  eval_replica_head2head.py --replica_root data/replica_niceslam
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

WS = Path(__file__).resolve().parents[5]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from eval_slimvdb_replica_chamfer import load_ply_xyz, sample_mesh, chamfer_and_f


def load_scovox_npz(path: Path, occ_thresh: float = 0.5) -> np.ndarray:
    d = np.load(path)
    pts = d["points"]
    if "occupancy_prob" in d.files:
        pts = pts[d["occupancy_prob"] >= occ_thresh]
    return pts.astype(np.float64)


def eval_dir(label: str, runs_dir: Path, kind: str, replica_root: Path,
             gt_samples: int, sample_k: int | None) -> list[dict]:
    rows = []
    if not runs_dir.exists():
        print(f"[{label}] missing dir {runs_dir}"); return rows
    for scene_dir in sorted(runs_dir.iterdir()):
        if not scene_dir.is_dir(): continue
        scene = scene_dir.name
        if kind == "ply":
            cand = sorted(scene_dir.glob("*.ply"))
            if not cand: continue
            pred_path = cand[0]
            pred = load_ply_xyz(pred_path)
        elif kind == "npz":
            pred_path = scene_dir / "scovox.npz"
            if not pred_path.exists(): continue
            pred = load_scovox_npz(pred_path)
        else:
            raise ValueError(kind)

        gt_mesh = replica_root / scene / "mesh.ply"
        if not gt_mesh.exists():
            print(f"  [skip] no GT mesh {gt_mesh}"); continue
        try:
            gt = sample_mesh(gt_mesh, gt_samples)
        except Exception as e:
            print(f"  [err] {scene}: {e}"); continue

        m = chamfer_and_f(pred, gt, sample_k=sample_k)
        m.update({"condition": label, "scene": scene, "pred_path": str(pred_path)})
        print(f"  [{label:<14}] {scene:<8}  Chamfer={m['chamfer_l2_m']*100:5.2f}cm  "
              f"F@5={m['fscore_at_5cm']:.3f}  P={m['precision_at_5cm']:.3f}  "
              f"R={m['recall_at_5cm']:.3f}")
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, default=WS / "data" / "replica_niceslam")
    ap.add_argument("--scovox_gt",  type=Path, default=WS / "src/robot_sw/distributed_mapping/scovox_eval/results/replica_gt_5cm")
    ap.add_argument("--scovox_m2f", type=Path, default=WS / "src/robot_sw/distributed_mapping/scovox_eval/results/replica_m2f_5cm")
    ap.add_argument("--slim_gt",    type=Path, default=WS / "third_party_sw/slim_vdb/outputs/replica_gt")
    ap.add_argument("--slim_m2f",   type=Path, default=WS / "third_party_sw/slim_vdb/outputs/replica_m2f")
    ap.add_argument("--gt_points", type=int, default=200_000)
    ap.add_argument("--sample_k", type=int, default=500_000)
    ap.add_argument("--out_csv", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results" / "replica_head2head.csv")
    args = ap.parse_args()

    conditions = [
        ("scovox_gt",   args.scovox_gt,   "npz"),
        ("scovox_m2f",  args.scovox_m2f,  "npz"),
        ("slimvdb_gt",  args.slim_gt,     "ply"),
        ("slimvdb_m2f", args.slim_m2f,    "ply"),
    ]
    all_rows = []
    for label, d, kind in conditions:
        print(f"\n=== {label}  ({d}) ===")
        all_rows.extend(eval_dir(label, d, kind, args.replica_root,
                                  args.gt_points, args.sample_k))

    if all_rows:
        by = defaultdict(list)
        for r in all_rows: by[r["condition"]].append(r)
        print("\n=== SUMMARY (mean ± std across scenes) ===")
        print(f"{'condition':<14s} {'n':>2}  {'Chamfer(cm)':>14s}  {'F@5cm':>13s}  "
              f"{'P@5cm':>13s}  {'R@5cm':>13s}")
        for label, rs in by.items():
            ch = np.array([r["chamfer_l2_m"] for r in rs]) * 100
            fs = np.array([r["fscore_at_5cm"] for r in rs])
            p  = np.array([r["precision_at_5cm"] for r in rs])
            r_ = np.array([r["recall_at_5cm"] for r in rs])
            print(f"{label:<14s} {len(rs):>2}  {ch.mean():5.2f} ± {ch.std():5.2f}    "
                  f"{fs.mean():.3f} ± {fs.std():.3f}    "
                  f"{p.mean():.3f} ± {p.std():.3f}    "
                  f"{r_.mean():.3f} ± {r_.std():.3f}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_rows for k in r.keys()})
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows: w.writerow(r)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
