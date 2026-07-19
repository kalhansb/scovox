"""E3.2 component ablation scoring.

Replica room0 + KITTI seq08, variants {scovox, scovox_mv, scovox_np}.
For Replica: mIoU + F@5cm + Chamfer (full Replica eval primitives).
For KITTI: mIoU only (no GT mesh).
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_scovox_replica import load_scovox_pred_voxels, build_gt_scovox_voxels
from eval_slimvdb_replica_miou import compute_miou as compute_miou_replica
from eval_slimvdb_replica_chamfer import chamfer_and_f, sample_mesh
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scovox_eval.metrics.miou import compute_miou as compute_miou_kitti  # KDTree-based
from scipy.spatial import KDTree

VARIANTS = ["scovox", "scovox_mv", "scovox_np"]


def score_replica(replica_root, npz_dir, voxel_size=0.05, n_frames=2000, stride=10,
                  num_classes=19, gt_points=200_000, sample_k=500_000):
    gt = build_gt_scovox_voxels(replica_root / "room0", voxel_size, n_frames, stride)
    gt_pts = sample_mesh(replica_root / "room0" / "mesh.ply", gt_points)
    rows = []
    for v in VARIANTS:
        npz = npz_dir / f"replica_room0_{v}" / "scovox.npz"
        if not npz.exists():
            print(f"  [skip] replica/{v}: no npz")
            continue
        pred, pred_xyz = load_scovox_pred_voxels(npz, voxel_size)
        m = compute_miou_replica(pred, gt, num_classes)
        cm = chamfer_and_f(pred_xyz.astype(np.float64), gt_pts, sample_k=sample_k)
        print(f"replica room0  {v:12s}  mIoU={m['miou']:.4f}  "
              f"F@5cm={cm['fscore_at_5cm']:.4f}  "
              f"Cham={cm['chamfer_l2_m']*100:.2f}cm  "
              f"vox={m['n_pred']}")
        rows.append((v, m["miou"], cm["fscore_at_5cm"], cm["chamfer_l2_m"], m["n_pred"]))
    return rows


def score_kitti(kitti_gt_npz, npz_dir, max_dist=0.10, num_classes=20):
    """Match the K_TOP-table scoring methodology: pre-built GT npz + KDTree NN
    match within max_dist + per-class IoU with ignore_label=0."""
    gt = np.load(kitti_gt_npz)
    gt_pts = gt["points"]
    gt_lbl = gt["semantic_class"].astype(int)
    tree = KDTree(gt_pts)
    rows = []
    for v in VARIANTS:
        npz = npz_dir / f"kitti_seq08_{v}" / "scovox.npz"
        if not npz.exists():
            print(f"  [skip] kitti/{v}: no npz")
            continue
        pr = np.load(npz)
        pr_pts = pr["points"]
        pr_lbl = pr["semantic_class"].astype(int)
        d, i = tree.query(pr_pts)
        mask = d < max_dist
        matched_pr = pr_lbl[mask]
        matched_gt = gt_lbl[i[mask]]
        r = compute_miou_kitti(matched_pr, matched_gt,
                               num_classes=num_classes, ignore_label=0)
        miou = r["miou"] if isinstance(r, dict) else float(r)
        n_matched = int(mask.sum())
        print(f"kitti seq08    {v:12s}  mIoU={miou:.4f}  matched={n_matched}/{len(pr_pts)}")
        rows.append((v, miou, n_matched))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--kitti_gt_npz", type=Path, required=True,
                    help="Pre-built KITTI seq08 GT npz (e.g. results/semantickitti_polarseg_10cm/08/gt.npz)")
    ap.add_argument("--npz_root", type=Path, required=True)
    args = ap.parse_args()

    print("\n=== Replica room0 (5 cm, soft M2F) ===")
    r_rows = score_replica(args.replica_root, args.npz_root)

    print("\n=== KITTI seq08 (10 cm, soft PolarSeg) ===")
    k_rows = score_kitti(args.kitti_gt_npz, args.npz_root)

    out_csv = args.npz_root / "e3_2_summary.csv"
    with open(out_csv, "w") as f:
        f.write("dataset,variant,miou,f_at_5cm,chamfer_l2_m,n_pred\n")
        for v, miou, f5, cl2, n in r_rows:
            f.write(f"replica_room0,{v},{miou},{f5},{cl2},{n}\n")
        for v, miou, n in k_rows:
            f.write(f"kitti_seq08,{v},{miou},,,{n}\n")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
