"""Voxel-grid F@5cm + Chamfer metrics for SceneNet fusion NPZs.

Unlike the Replica fusion scorer (eval_e21_fusion.py), SceneNet GT is a
voxel grid (gt_5cm.npz) not a continuous mesh — so the geometry metrics
are computed point-cloud-style between voxel centers.

Treats pred voxel centers (from NPZ 'points') and GT voxel centers (from
NPZ 'coords') as 3D point clouds. Uses scipy cKDTree for nearest-neighbor
queries.

Metrics:
* **F@5cm**: precision = fraction of pred points with a GT point within
  5 cm; recall = fraction of GT points with a pred point within 5 cm;
  F1 = 2·p·r / (p + r). At 5cm voxel resolution this is the natural
  "is this voxel near GT" question — strictly tighter than set
  intersection (which would also count exact-cell matches at any range).
* **Chamfer L2**: (mean(d_pred→gt) + mean(d_gt→pred)) / 2, in meters.

Usage:
  python3 scenenet_compute_geometry_metrics.py \
      --pred_npz fused.npz --gt_npz gt_5cm.npz --resolution 0.05

  # batch mode — iterates all subdirs of --fusion_root
  python3 scenenet_compute_geometry_metrics.py \
      --fusion_root results/phase3_scenenet_fusion_v3_2026_05_14 \
      --gt_root data/scenenet_val_layout/train \
      --out_csv geometry.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


F_AT_THRESHOLD_M = 0.05  # 5 cm — matches the SceneNet voxel resolution


def load_points(npz_path: Path, key_candidates: tuple[str, ...]) -> np.ndarray:
    """Load a (N,3) float array from an NPZ, trying multiple key names.

    Pred NPZs (from pointcloud_to_npz) use 'points'.
    GT NPZs (from scenenet_val_to_slimvdb_format) use 'coords'.
    """
    data = np.load(npz_path)
    for k in key_candidates:
        if k in data.files:
            arr = data[k]
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr.astype(np.float64)
    raise KeyError(f"{npz_path}: none of {key_candidates} present "
                   f"(keys = {list(data.files)})")


def f_at_threshold(pred_xyz: np.ndarray, gt_xyz: np.ndarray,
                   threshold_m: float = F_AT_THRESHOLD_M) -> tuple[float, float, float]:
    """Precision/recall/F1 at the threshold distance.

    Symmetric, point-cloud style — counts a pred point as 'matched' iff
    any GT point is within threshold (and vice versa for recall).
    """
    if pred_xyz.size == 0 or gt_xyz.size == 0:
        return 0.0, 0.0, 0.0
    gt_tree = cKDTree(gt_xyz)
    pred_tree = cKDTree(pred_xyz)
    d_p2g, _ = gt_tree.query(pred_xyz, k=1)
    d_g2p, _ = pred_tree.query(gt_xyz, k=1)
    precision = float(np.mean(d_p2g <= threshold_m))
    recall    = float(np.mean(d_g2p <= threshold_m))
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def chamfer_l2(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """Symmetric mean-Chamfer L2 in meters: (mean(d_p2g) + mean(d_g2p)) / 2."""
    if pred_xyz.size == 0 or gt_xyz.size == 0:
        return float("nan")
    gt_tree = cKDTree(gt_xyz)
    pred_tree = cKDTree(pred_xyz)
    d_p2g, _ = gt_tree.query(pred_xyz, k=1)
    d_g2p, _ = pred_tree.query(gt_xyz, k=1)
    return 0.5 * (float(np.mean(d_p2g)) + float(np.mean(d_g2p)))


def score_pair(pred_npz: Path, gt_npz: Path) -> dict:
    pred_xyz = load_points(pred_npz, ("points", "coords"))
    gt_xyz   = load_points(gt_npz,   ("coords", "points"))
    p, r, f1 = f_at_threshold(pred_xyz, gt_xyz)
    cham = chamfer_l2(pred_xyz, gt_xyz)
    return {
        "n_pred": len(pred_xyz),
        "n_gt": len(gt_xyz),
        "precision_5cm": p,
        "recall_5cm": r,
        "f1_5cm": f1,
        "chamfer_l2_m": cham,
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred_npz", type=Path, help="Single pred NPZ")
    g.add_argument("--fusion_root", type=Path,
                   help="Batch mode: <root>/<traj>/{solo_a,solo_b,fused}.npz")
    ap.add_argument("--gt_npz", type=Path, help="GT NPZ (single mode)")
    ap.add_argument("--gt_root", type=Path,
                    help="Batch mode: <root>/<traj>/gt_5cm.npz")
    ap.add_argument("--trajs", nargs="+", default=None,
                    help="Batch mode: subset of trajs (default: all subdirs)")
    ap.add_argument("--labels", nargs="+", default=["solo_a", "solo_b", "fused"],
                    help="Batch mode: NPZ basenames per traj")
    ap.add_argument("--resolution", type=float, default=0.05,
                    help="Voxel resolution (currently informational only)")
    ap.add_argument("--out_csv", type=Path,
                    help="Batch mode: write per-cell metrics to CSV")
    args = ap.parse_args()

    if args.pred_npz is not None:
        if args.gt_npz is None:
            sys.exit("--gt_npz required when --pred_npz is given")
        m = score_pair(args.pred_npz, args.gt_npz)
        print(f"n_pred={m['n_pred']} n_gt={m['n_gt']}")
        print(f"precision@5cm = {m['precision_5cm']:.4f}")
        print(f"recall@5cm    = {m['recall_5cm']:.4f}")
        print(f"F1@5cm        = {m['f1_5cm']:.4f}")
        print(f"chamfer_l2_m  = {m['chamfer_l2_m']:.4f}")
        return

    # Batch mode.
    if args.gt_root is None:
        sys.exit("--gt_root required when --fusion_root is given")
    trajs = args.trajs or sorted(
        p.name for p in args.fusion_root.iterdir() if p.is_dir()
    )
    rows = []
    for traj in trajs:
        gt_npz = args.gt_root / traj / "gt_5cm.npz"
        if not gt_npz.exists():
            print(f"[{traj}] SKIP — no GT at {gt_npz}", file=sys.stderr)
            continue
        for label in args.labels:
            pred_npz = args.fusion_root / traj / f"{label}.npz"
            if not pred_npz.exists() or pred_npz.stat().st_size < 1024:
                print(f"[{traj} {label}] SKIP — missing/empty pred", file=sys.stderr)
                continue
            m = score_pair(pred_npz, gt_npz)
            row = {"traj": traj, "label": label, **m}
            rows.append(row)
            print(f"[{traj} {label}] F1@5cm={m['f1_5cm']:.4f} "
                  f"chamfer={m['chamfer_l2_m']:.4f} "
                  f"(n_pred={m['n_pred']}, n_gt={m['n_gt']})")

    if args.out_csv is not None and rows:
        with args.out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
