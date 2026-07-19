#!/usr/bin/env python3
"""Aggregate mIoU + Chamfer across the Replica room0 m2f ablation cells.

Each cell lives under <results_root>/<cell>/scovox.npz. GT is built once
from the room0 ground-truth labels and reused across cells.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_slimvdb_replica_miou import compute_miou
from eval_slimvdb_replica_chamfer import chamfer_and_f, sample_mesh
from eval_scovox_replica import load_scovox_pred_voxels, build_gt_scovox_voxels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--results_root", type=Path, required=True)
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--num_classes", type=int, default=19)
    ap.add_argument("--gt_points", type=int, default=200_000)
    ap.add_argument("--sample_k", type=int, default=500_000)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    print(f"Building GT voxel grid for {args.scene}...", flush=True)
    gt = build_gt_scovox_voxels(args.replica_root / args.scene,
                                 args.voxel_size, args.n_frames, args.stride)
    print(f"  GT voxels: {len(gt)}", flush=True)

    mesh_path = args.replica_root / args.scene / "mesh.ply"
    gt_pts = sample_mesh(mesh_path, args.gt_points) if mesh_path.exists() else None

    cells = sorted(d for d in args.results_root.iterdir() if d.is_dir())
    rows = []
    for cell_dir in cells:
        npz = cell_dir / "scovox.npz"
        if not npz.exists():
            print(f"[skip] {cell_dir.name}: no npz")
            continue
        pred, pred_xyz = load_scovox_pred_voxels(npz, args.voxel_size)
        m = compute_miou(pred, gt, args.num_classes)
        row = {
            "cell": cell_dir.name,
            "n_pred": m["n_pred"],
            "n_gt": m["n_gt"],
            "n_intersect": m["n_intersect"],
            "miou": round(m["miou"], 4),
        }
        if gt_pts is not None and pred_xyz is not None and len(pred_xyz):
            cm = chamfer_and_f(pred_xyz.astype(np.float64), gt_pts,
                                sample_k=args.sample_k)
            row["chamfer_cm"] = round(cm["chamfer_l2_m"] * 100, 2)
            row["f_at_5cm"] = round(cm["fscore_at_5cm"], 4)
        rows.append(row)
        print(f"{cell_dir.name:14s} mIoU={row['miou']:.4f} "
              f"chamfer={row.get('chamfer_cm','--')}cm "
              f"F@5={row.get('f_at_5cm','--')} "
              f"pred={m['n_pred']}", flush=True)

    if args.csv:
        cols = ["cell", "miou", "chamfer_cm", "f_at_5cm", "n_pred", "n_gt", "n_intersect"]
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"\n[csv] wrote {args.csv}")

    print("\n=== summary ===")
    base = next((r for r in rows if r["cell"] == "baseline"), None)
    if base:
        print(f"baseline mIoU={base['miou']:.4f}")
        for r in sorted(rows, key=lambda r: -r["miou"]):
            if r["cell"] == "baseline":
                continue
            d = r["miou"] - base["miou"]
            sign = "+" if d >= 0 else ""
            print(f"  {r['cell']:14s} mIoU={r['miou']:.4f} ({sign}{d:.4f})")


if __name__ == "__main__":
    main()
