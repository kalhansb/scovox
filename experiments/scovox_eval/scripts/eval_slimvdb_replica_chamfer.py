#!/usr/bin/env python3
"""L2 Chamfer distance + F-score between SLIM-VDB's exported `.ply` and the
Replica scene GT mesh. Paper-compatible with SLIM-VDB Table III (the paper
reports L2 Chamfer on final exported maps).

Metrics:
  * Bidirectional L2 Chamfer mean (pred→gt + gt→pred) / 2
  * F-score @ 5 cm  (precision + recall of points within 5 cm)
  * Per-run and per-condition aggregate

Usage:
    python eval_slimvdb_replica_chamfer.py \
        --replica_root data/replica_niceslam \
        --runs_m2f third_party_sw/slim_vdb/outputs/replica_m2f \
        --runs_gt  third_party_sw/slim_vdb/outputs/replica_gt \
        --gt_points 200000 \
        --out_csv results/slimvdb_replica_chamfer.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def load_ply_xyz(path: Path) -> np.ndarray:
    """Minimal ASCII PLY loader: returns (N,3) xyz."""
    lines = []
    header_done = False
    n_vertex = 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not header_done:
                if s.startswith("element vertex"):
                    n_vertex = int(s.split()[-1])
                if s == "end_header":
                    header_done = True
                continue
            lines.append(s)
    data = np.array([l.split()[:3] for l in lines[:n_vertex]], dtype=np.float64)
    return data


def sample_mesh(mesh_ply: Path, n_samples: int) -> np.ndarray:
    """Uniformly sample a surface by triangle-area weighting. Avoids trimesh dep."""
    # Fall back to trimesh if available (cleaner), else parse PLY manually.
    try:
        import trimesh
        mesh = trimesh.load(mesh_ply, process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
        return np.asarray(pts, dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(
            f"trimesh required to sample mesh surface; install `pip install trimesh` "
            f"(for {mesh_ply}): {exc}") from exc


def chamfer_and_f(pred: np.ndarray, gt: np.ndarray,
                  threshold_m: float = 0.05,
                  sample_k: int | None = None) -> dict:
    """Bidirectional NN distances, symmetric Chamfer L2, F-score."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    if sample_k is not None:
        if pred.shape[0] > sample_k:
            idx = rng.choice(pred.shape[0], sample_k, replace=False)
            pred = pred[idx]
        if gt.shape[0] > sample_k:
            idx = rng.choice(gt.shape[0], sample_k, replace=False)
            gt = gt[idx]

    tree_p = cKDTree(pred)
    tree_g = cKDTree(gt)
    d_p2g, _ = tree_g.query(pred, k=1)
    d_g2p, _ = tree_p.query(gt, k=1)

    cham_l2 = 0.5 * (d_p2g.mean() + d_g2p.mean())
    precision = float((d_p2g < threshold_m).mean())
    recall = float((d_g2p < threshold_m).mean())
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "chamfer_l2_m": float(cham_l2),
        "chamfer_pred_to_gt_m": float(d_p2g.mean()),
        "chamfer_gt_to_pred_m": float(d_g2p.mean()),
        "precision_at_5cm": precision,
        "recall_at_5cm": recall,
        "fscore_at_5cm": fscore,
        "n_pred": int(pred.shape[0]),
        "n_gt": int(gt.shape[0]),
    }


def eval_condition(label: str, runs_dir: Path, replica_root: Path, gt_samples: int,
                   sample_k: int | None) -> list[dict]:
    rows = []
    for scene_dir in sorted(runs_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        ply_candidates = sorted(scene_dir.glob("*.ply"))
        if not ply_candidates:
            print(f"  [skip] no .ply in {scene_dir}")
            continue
        pred_ply = ply_candidates[0]
        gt_mesh = replica_root / scene / "mesh.ply"
        if not gt_mesh.exists():
            print(f"  [skip] no GT mesh at {gt_mesh}")
            continue
        try:
            pred = load_ply_xyz(pred_ply)
            gt = sample_mesh(gt_mesh, gt_samples)
        except Exception as exc:
            print(f"  [err]  {scene}: {exc}")
            continue
        metrics = chamfer_and_f(pred, gt, sample_k=sample_k)
        metrics.update({"condition": label, "scene": scene, "pred_ply": str(pred_ply)})
        print(f"  [{label}] {scene:<8}  Chamfer={metrics['chamfer_l2_m']*100:6.2f} cm  "
              f"F@5cm={metrics['fscore_at_5cm']:.4f}  "
              f"npred={metrics['n_pred']} ngt={metrics['n_gt']}")
        rows.append(metrics)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--runs_m2f", type=Path, default=None,
                    help="e.g. third_party_sw/slim_vdb/outputs/replica_m2f")
    ap.add_argument("--runs_gt", type=Path, default=None,
                    help="e.g. third_party_sw/slim_vdb/outputs/replica_gt")
    ap.add_argument("--gt_points", type=int, default=200_000,
                    help="Number of points to sample from each GT mesh.")
    ap.add_argument("--sample_k", type=int, default=500_000,
                    help="Subsample both point clouds to this size before kd-tree NN.")
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args()

    all_rows = []
    for label, runs in [("m2f", args.runs_m2f), ("gt", args.runs_gt)]:
        if runs is None:
            continue
        if not runs.exists():
            print(f"[{label}] runs dir missing: {runs}")
            continue
        print(f"\n=== {label} condition: {runs} ===")
        all_rows.extend(eval_condition(label, runs, args.replica_root,
                                       args.gt_points, args.sample_k))

    if all_rows:
        # Aggregate per condition
        from collections import defaultdict
        by_cond = defaultdict(list)
        for r in all_rows:
            by_cond[r["condition"]].append(r)
        print("\n=== Summary (mean ± std across scenes) ===")
        print(f"{'condition':<10s} {'n':>3}  {'Chamfer(cm)':>12}  {'F@5cm':>10}  "
              f"{'precision':>10}  {'recall':>10}")
        for label, rows in by_cond.items():
            ch = np.array([r["chamfer_l2_m"] for r in rows]) * 100
            fs = np.array([r["fscore_at_5cm"] for r in rows])
            pr = np.array([r["precision_at_5cm"] for r in rows])
            rc = np.array([r["recall_at_5cm"] for r in rows])
            print(f"{label:<10s} {len(rows):>3}  "
                  f"{ch.mean():5.2f} ± {ch.std():5.2f}  "
                  f"{fs.mean():5.3f} ± {fs.std():5.3f}  "
                  f"{pr.mean():5.3f} ± {pr.std():5.3f}  "
                  f"{rc.mean():5.3f} ± {rc.std():5.3f}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in all_rows for k in r.keys()})
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nwrote per-scene CSV → {args.out_csv}")


if __name__ == "__main__":
    main()
