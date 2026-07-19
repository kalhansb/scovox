"""E2.1b — bandwidth + lossy-merge overlap analysis.

For each Replica fusion scene (`results/fusion_replica_m2f/<scene>/`):
  - load solo_a, solo_b, fused NPZs
  - find overlap voxels by matching bit-packed (i,j,k) keys at voxel size
  - compute class agreement rate (top-1 argmax)
  - compute overlap-only mIoU vs GT for each solo and fused
  - report bandwidth: SCovox wire (32 B/voxel) vs dense Dirichlet (19*4 B/voxel)

Note: published NPZ does not preserve K_TOP=2 raw class IDs (only the
top-1 argmax via `semantic_class`), so the "top-2 set disagreement" is
not computed here — it would require extending the publish path.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_scovox_replica import build_gt_scovox_voxels
from eval_slimvdb_replica_miou import voxel_key


SCENES_DEFAULT = [
    "room0", "room1", "room2",
    "office0", "office1", "office2", "office3", "office4",
]
BYTES_SCOVOX_PER_VOXEL = 32         # SCovox on-wire footprint per voxel
BYTES_DENSE_DIR_PER_VOXEL = 19 * 4  # full Dirichlet (19 classes, float32)


def load_scovox(npz_path: Path, voxel_size: float):
    d = np.load(npz_path)
    xyz = d["points"].astype(np.float32)
    cls = d["semantic_class"].astype(np.int32)
    keys = voxel_key(xyz, voxel_size)
    return keys, cls, xyz


def overlap_miou(keys, cls, gt_dict, num_classes=19):
    """Score the given pred subset (keys+cls) against GT keymap."""
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    matched = 0
    for k, c in zip(keys, cls):
        gc = gt_dict.get(int(k))
        if gc is None:
            continue
        matched += 1
        if gc == c:
            inter[c] += 1
            union[c] += 1
        else:
            union[c] += 1
            union[gc] += 1
    valid = union > 0
    if not valid.any():
        return float("nan"), 0
    return float(np.mean(inter[valid] / union[valid])), matched


def main():
    home = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path,
                    default=home / "projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/replica_niceslam")
    ap.add_argument("--fusion_root", type=Path,
                    default=home / "projects/HMR_Exploration_Experiment/hmr_exploration_ws/src/robot_sw/distributed_mapping/scovox_eval/results/fusion_replica_m2f")
    ap.add_argument("--scenes", nargs="+", default=SCENES_DEFAULT)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--num_classes", type=int, default=19)
    ap.add_argument("--out_csv", type=Path,
                    default=home / "projects/HMR_Exploration_Experiment/hmr_exploration_ws/src/robot_sw/distributed_mapping/scovox_eval/results/fusion_replica_m2f/e2_1b_overlap.csv")
    args = ap.parse_args()

    rows = []
    print(f"{'scene':9s} {'|A|':>7s} {'|B|':>7s} {'|F|':>7s} "
          f"{'|A∩B|':>7s} {'agree':>7s} "
          f"{'mIoU_A^':>8s} {'mIoU_B^':>8s} {'mIoU_F^':>8s} "
          f"{'best':>6s} {'Δfused':>8s} "
          f"{'B_scov_KB':>10s} {'B_dense_KB':>11s} {'ratio':>6s}")
    for scene in args.scenes:
        scene_dir = args.fusion_root / scene
        npz_a = scene_dir / "solo_a.npz"
        npz_b = scene_dir / "solo_b.npz"
        npz_f = scene_dir / "fused.npz"
        if not (npz_a.exists() and npz_b.exists() and npz_f.exists()):
            print(f"  [skip] {scene}: missing npz")
            continue

        keys_a, cls_a, _ = load_scovox(npz_a, args.voxel_size)
        keys_b, cls_b, _ = load_scovox(npz_b, args.voxel_size)
        keys_f, cls_f, _ = load_scovox(npz_f, args.voxel_size)

        # A∩B overlap by bit-packed key intersection.
        idx_a_by_key = {int(k): i for i, k in enumerate(keys_a)}
        idx_b_by_key = {int(k): i for i, k in enumerate(keys_b)}
        idx_f_by_key = {int(k): i for i, k in enumerate(keys_f)}
        common_ab = set(idx_a_by_key.keys()) & set(idx_b_by_key.keys())
        if not common_ab:
            print(f"  [skip] {scene}: no overlap")
            continue
        ia = np.array([idx_a_by_key[k] for k in common_ab], dtype=np.int64)
        ib = np.array([idx_b_by_key[k] for k in common_ab], dtype=np.int64)
        cls_a_ov = cls_a[ia]
        cls_b_ov = cls_b[ib]
        agree = float(np.mean(cls_a_ov == cls_b_ov))

        # Three-way overlap (A ∩ B ∩ F): typically ≈ A∩B since fused covers both.
        common_abf = common_ab & set(idx_f_by_key.keys())
        if_idx = np.array([idx_f_by_key[k] for k in common_abf], dtype=np.int64)
        f_keys_ov = keys_f[if_idx] if len(if_idx) else np.array([], dtype=np.int64)
        f_cls_ov = cls_f[if_idx] if len(if_idx) else np.array([], dtype=np.int32)

        # Per-solo overlap-only keys/cls.
        a_keys_ov = keys_a[ia]
        b_keys_ov = keys_b[ib]

        # GT keymap (already bit-packed by build_gt_scovox_voxels).
        gt = build_gt_scovox_voxels(args.replica_root / scene,
                                     args.voxel_size, args.n_frames, args.stride)

        miou_a_ov, _ = overlap_miou(a_keys_ov, cls_a_ov, gt, args.num_classes)
        miou_b_ov, _ = overlap_miou(b_keys_ov, cls_b_ov, gt, args.num_classes)
        miou_f_ov = float("nan")
        if len(if_idx):
            miou_f_ov, _ = overlap_miou(f_keys_ov, f_cls_ov, gt, args.num_classes)

        best_solo = max(miou_a_ov, miou_b_ov)
        delta_fused = (miou_f_ov - best_solo) if miou_f_ov == miou_f_ov else float("nan")

        n_shared = len(common_ab)
        bytes_scov = n_shared * BYTES_SCOVOX_PER_VOXEL
        bytes_dense = n_shared * BYTES_DENSE_DIR_PER_VOXEL
        ratio = bytes_dense / bytes_scov

        print(f"{scene:9s} {len(keys_a):7d} {len(keys_b):7d} {len(keys_f):7d} "
              f"{n_shared:7d} {agree:7.3f} "
              f"{miou_a_ov:8.3f} {miou_b_ov:8.3f} {miou_f_ov:8.3f} "
              f"{best_solo:6.3f} {delta_fused:+8.3f} "
              f"{bytes_scov/1024:10.1f} {bytes_dense/1024:11.1f} {ratio:5.2f}x")

        rows.append({
            "scene": scene,
            "n_a": len(keys_a),
            "n_b": len(keys_b),
            "n_f": len(keys_f),
            "n_overlap": n_shared,
            "agree_top1": agree,
            "miou_a_overlap": miou_a_ov,
            "miou_b_overlap": miou_b_ov,
            "miou_f_overlap": miou_f_ov,
            "best_solo_miou": best_solo,
            "delta_fused_vs_best": delta_fused,
            "bytes_scovox": bytes_scov,
            "bytes_dense_dir": bytes_dense,
            "compression_ratio": ratio,
        })

    if rows:
        agree_mean = float(np.mean([r["agree_top1"] for r in rows]))
        deltas = [r["delta_fused_vs_best"] for r in rows
                  if r["delta_fused_vs_best"] == r["delta_fused_vs_best"]]
        delta_mean = float(np.mean(deltas)) if deltas else float("nan")
        ratio_mean = float(np.mean([r["compression_ratio"] for r in rows]))
        total_scov = sum(r["bytes_scovox"] for r in rows)
        total_dense = sum(r["bytes_dense_dir"] for r in rows)
        print(f"\n=== aggregate (n={len(rows)} scenes) ===")
        print(f"  top-1 class agreement (overlap):           {agree_mean:.3f}")
        print(f"  Δ(fused − best_solo) overlap mIoU:         {delta_mean:+.3f}")
        print(f"  bandwidth ratio (dense Dirichlet/SCovox):  {ratio_mean:.2f}x")
        print(f"  total overlap bytes — SCovox: {total_scov/1024:.1f} KB / dense: {total_dense/1024:.1f} KB")

        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w") as f:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
