"""E2.1 evaluation — score solo_a / solo_b / fused per scene against GT.

Reuses load_scovox_pred_voxels + build_gt_scovox_voxels + compute_miou +
chamfer_and_f from the existing eval_scovox_replica.py module (no behaviour
divergence — just iterates over the 3 npz names instead of scovox.npz).
"""
import argparse, sys
from pathlib import Path
import numpy as np

# Reuse existing eval primitives.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_scovox_replica import (
    load_scovox_pred_voxels,
    build_gt_scovox_voxels,
)
from eval_slimvdb_replica_miou import compute_miou
from eval_slimvdb_replica_chamfer import chamfer_and_f, sample_mesh


SCENES_DEFAULT = [
    "room0", "room1", "room2",
    "office0", "office1", "office2", "office3", "office4",
]
NPZ_LABELS = ["solo_a", "solo_b", "fused"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--fusion_root", type=Path, required=True,
                    help="Root with <scene>/{solo_a,solo_b,fused}.npz")
    ap.add_argument("--scenes", nargs="+", default=SCENES_DEFAULT)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--num_classes", type=int, default=19)
    ap.add_argument("--gt_points", type=int, default=200_000)
    ap.add_argument("--sample_k", type=int, default=500_000)
    args = ap.parse_args()

    rows = []
    for scene in args.scenes:
        scene_dir = args.fusion_root / scene
        gt = build_gt_scovox_voxels(args.replica_root / scene,
                                     args.voxel_size, args.n_frames, args.stride)
        gt_mesh = args.replica_root / scene / "mesh.ply"
        gt_pts = sample_mesh(gt_mesh, args.gt_points) if gt_mesh.exists() else None

        for label in NPZ_LABELS:
            npz = scene_dir / f"{label}.npz"
            if not npz.exists():
                print(f"  [skip] {scene}/{label}: no npz")
                rows.append((scene, label, np.nan, np.nan, np.nan, 0))
                continue
            pred, pred_xyz = load_scovox_pred_voxels(npz, args.voxel_size)
            m = compute_miou(pred, gt, args.num_classes)
            f5 = np.nan
            chl2 = np.nan
            if gt_pts is not None:
                cm = chamfer_and_f(pred_xyz.astype(np.float64), gt_pts,
                                    sample_k=args.sample_k)
                f5 = cm["fscore_at_5cm"]
                chl2 = cm["chamfer_l2_m"]
            print(f"{scene:9s} {label:7s}  mIoU={m['miou']:.4f}  "
                  f"F@5cm={f5:.4f}  Cham={chl2*100:.2f}cm  vox={m['n_pred']}")
            rows.append((scene, label, m["miou"], f5, chl2, m["n_pred"]))

    print("\n=== E2.1 cross-scene aggregates ===")
    print(f"{'label':9s}  {'mIoU':>14s}  {'F@5cm':>14s}  {'Chamfer-cm':>14s}")
    for label in NPZ_LABELS:
        miou = [r[2] for r in rows if r[1] == label and not np.isnan(r[2])]
        f5s = [r[3] for r in rows if r[1] == label and not np.isnan(r[3])]
        chs = [r[4] for r in rows if r[1] == label and not np.isnan(r[4])]
        if miou:
            print(f"{label:9s}  "
                  f"{np.mean(miou):.4f} ± {np.std(miou):.3f}  "
                  f"{np.mean(f5s):.4f} ± {np.std(f5s):.3f}  "
                  f"{np.mean(chs)*100:.2f} ± {np.std(chs)*100:.2f}")

    out_csv = args.fusion_root / "e2_1_summary.csv"
    with open(out_csv, "w") as f:
        f.write("scene,label,miou,f_at_5cm,chamfer_l2_m,n_pred\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
