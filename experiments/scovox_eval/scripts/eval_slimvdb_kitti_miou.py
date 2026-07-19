#!/usr/bin/env python3
"""Voxel-wise mIoU of SLIM-VDB semantic voxels against SemanticKITTI GT.

Paper-compatible methodology (SLIM-VDB RA-L 2026, Section IV-A.3):

  1. Load SLIM-VDB active voxels with argmax class from the
     `voxels.bin` files produced by `vdb_to_voxels` (one per scene).
  2. Build a GT voxel grid at the same resolution:
       - transform each velodyne scan to world frame using the same pose
         math the pipeline uses (T_velo_cam · P_cam2world · T_cam_velo)
       - bucket each point to a voxel and majority-vote the class.
       - classes are in learning-space [0..19] via `learning_map`.
  3. Voxel-wise confusion matrix:
       - voxels present in both → count (pred, gt)
       - voxels in GT but not in pred → FN for gt_class
       - voxels in pred but not in GT → "extra voxels" FP for pred_class
  4. Per-class IoU and mean IoU (excluding class 0 = unlabeled).

Usage:
    python eval_slimvdb_kitti_miou.py \
        --kitti_root data/semantickitti/dataset \
        --voxels_dir third_party_sw/slim_vdb/outputs/kitti \
        --sequences 06 07 08 09 10 --n_scans 100 \
        --voxel_size 0.10 --semantic_kitti_yaml src/sem_seg_pipeline/polarseg/semantic-kitti.yaml \
        --out_csv results/slimvdb_kitti_miou.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def load_voxels_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (N,3) float32 world coords and (N,) int32 class_ids."""
    raw = np.fromfile(path, dtype=np.uint8)
    rec = raw.view(np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "<i4")]))
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1)
    return xyz.astype(np.float32), rec["c"].astype(np.int32)


def read_kitti_calib(calib_file: Path) -> np.ndarray:
    """Returns the Tr (velodyne→cam0) 4×4 rigid transform."""
    Tr = np.eye(4)
    with open(calib_file) as f:
        for line in f:
            if line.startswith("Tr:"):
                vals = np.array([float(x) for x in line.split()[1:]], dtype=np.float64)
                Tr[:3, :4] = vals.reshape(3, 4)
                return Tr
    raise ValueError(f"no Tr line in {calib_file}")


def read_kitti_poses(poses_file: Path, Tr: np.ndarray):
    """Returns list of world-from-velodyne 4×4 transforms (matches the pipeline)."""
    Tr_inv = np.linalg.inv(Tr)
    out = []
    with open(poses_file) as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if len(vals) != 12:
                continue
            P = np.eye(4)
            P[:3, :4] = np.array(vals).reshape(3, 4)
            # T_world_velo = T_velo_cam · P_cam2world · T_cam_velo
            # pipeline builds (T_velo_cam * P * T_cam_velo)
            out.append(Tr_inv @ P @ Tr)
    return out


def build_learning_map(yaml_path: Path) -> np.ndarray:
    km = yaml.safe_load(open(yaml_path))["learning_map"]
    max_raw = max(km.keys())
    lut = np.zeros(max_raw + 1, dtype=np.int32)
    for raw, train in km.items():
        lut[raw] = train
    return lut


def voxelize_key(coords_m: np.ndarray, voxel_size: float) -> np.ndarray:
    """Floor-div world coords → integer voxel keys as packed int64 (x,y,z)."""
    ix = np.floor(coords_m[:, 0] / voxel_size).astype(np.int64)
    iy = np.floor(coords_m[:, 1] / voxel_size).astype(np.int64)
    iz = np.floor(coords_m[:, 2] / voxel_size).astype(np.int64)
    # Pack three int32-safe components into one int64 (range ±1.5e9 per axis).
    # mask & 0xFFFFFFFF avoids sign-bit bleed on negative coords.
    mask = (1 << 24) - 1  # 24-bit per axis is plenty for KITTI (≤40km at 10cm)
    keys = ((ix & mask) << 48) | ((iy & mask) << 24) | (iz & mask)
    return keys


def build_gt_voxels(kitti_root: Path, sequence: str, n_scans: int,
                    voxel_size: float, lmap: np.ndarray) -> dict[int, int]:
    """Returns {voxel_key: majority_class (learning-space)}."""
    seq_dir = kitti_root / "sequences" / sequence
    Tr = read_kitti_calib(seq_dir / "calib.txt")
    poses = read_kitti_poses(seq_dir / "poses.txt", Tr)

    velo_files = sorted((seq_dir / "velodyne").glob("*.bin"))[:n_scans]
    label_files = sorted((seq_dir / "labels").glob("*.label"))[:n_scans]
    assert len(velo_files) == len(label_files) == min(n_scans, len(velo_files))

    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for i, (vf, lf) in enumerate(zip(velo_files, label_files)):
        pts = np.fromfile(vf, dtype=np.float32).reshape(-1, 4)[:, :3]
        lbl_raw = np.fromfile(lf, dtype=np.uint32) & 0xFFFF
        lbl = lmap[lbl_raw]
        keep = lbl > 0  # drop unlabeled points
        pts, lbl = pts[keep], lbl[keep]

        T = poses[i].astype(np.float32)
        # Transform velo → world (pipeline applies this when apply_pose=True).
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pw = (T @ np.concatenate([pts, ones], axis=1).T).T[:, :3]

        keys = voxelize_key(pw, voxel_size)
        for k, c in zip(keys, lbl):
            counts[int(k)][int(c)] += 1

    # Majority class per voxel
    voxel2class = {k: max(d.items(), key=lambda kv: kv[1])[0] for k, d in counts.items()}
    return voxel2class


def compute_voxel_miou(pred_voxels: dict[int, int],
                        gt_voxels: dict[int, int],
                        num_classes: int,
                        class_names: list[str] | None = None
                        ) -> dict:
    """Voxel-wise confusion, per-class IoU, mIoU."""
    # Confusion matrix — counts in class 0 are treated as unknown/void; we
    # exclude class 0 from the mIoU average but still use it as a
    # "background" bucket to represent "no prediction".
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    all_keys = set(pred_voxels.keys()) | set(gt_voxels.keys())
    for k in all_keys:
        g = gt_voxels.get(k, 0)
        p = pred_voxels.get(k, 0)
        cm[g, p] += 1

    # Per-class IoU, exclude class 0 from the mean.
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    per_class = {
        i: (class_names[i] if class_names else f"c{i}", float(iou[i]) if np.isfinite(iou[i]) else None)
        for i in range(num_classes)
    }
    finite_nonzero = [iou[i] for i in range(1, num_classes) if np.isfinite(iou[i])]
    miou = float(np.mean(finite_nonzero)) if finite_nonzero else float("nan")
    return {
        "confusion": cm,
        "per_class_iou": per_class,
        "miou": miou,
        "n_pred_voxels": len(pred_voxels),
        "n_gt_voxels": len(gt_voxels),
        "n_intersection": sum(1 for k in pred_voxels if k in gt_voxels),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", type=Path, required=True)
    ap.add_argument("--voxels_dir", type=Path, required=True,
                    help="Root holding per-seq subdirs with voxels.bin")
    ap.add_argument("--sequences", nargs="+", default=["06", "07", "08", "09", "10"])
    ap.add_argument("--n_scans", type=int, default=100)
    ap.add_argument("--voxel_size", type=float, default=0.10)
    ap.add_argument("--semantic_kitti_yaml", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args()

    lmap = build_learning_map(args.semantic_kitti_yaml)
    yaml_cfg = yaml.safe_load(open(args.semantic_kitti_yaml))

    # Build learning-class names: learning_map_inv gives train_id → raw_id.
    lmi = yaml_cfg["learning_map_inv"]
    raw_names = yaml_cfg["labels"]
    num_classes = max(lmi.keys()) + 1
    class_names = [raw_names.get(lmi.get(i, 0), f"train_{i}") for i in range(num_classes)]

    print(f"[mIoU] {num_classes} classes; sequences={args.sequences}; voxel={args.voxel_size} m")

    per_seq_results = []
    for seq in args.sequences:
        voxels_bin = args.voxels_dir / seq / "voxels.bin"
        if not voxels_bin.exists():
            print(f"  seq {seq}: MISSING voxels.bin at {voxels_bin} — skipping")
            continue
        xyz, cls = load_voxels_bin(voxels_bin)
        keys = voxelize_key(xyz, args.voxel_size)
        pred_voxels = {int(k): int(c) for k, c in zip(keys, cls)}

        print(f"  seq {seq}: loading GT ({args.n_scans} scans) …", flush=True)
        gt_voxels = build_gt_voxels(args.kitti_root, seq, args.n_scans,
                                    args.voxel_size, lmap)

        res = compute_voxel_miou(pred_voxels, gt_voxels, num_classes, class_names)
        print(f"    pred_voxels={res['n_pred_voxels']}  gt_voxels={res['n_gt_voxels']}  "
              f"intersect={res['n_intersection']}  mIoU={res['miou']:.4f}")
        per_seq_results.append((seq, res))

    # Aggregate per-class IoU across sequences (mean per class, then mean over classes)
    if per_seq_results:
        miou_values = [r["miou"] for _, r in per_seq_results]
        print(f"\n=== SLIM-VDB KITTI mIoU (per-seq mean) = "
              f"{np.mean(miou_values):.4f} ± {np.std(miou_values):.4f} ===")
        print(f"{'class':<18s}  " + "  ".join(f"{s:>8s}" for s, _ in per_seq_results))
        for i, name in enumerate(class_names):
            row = [f"{name:<18s} "]
            for _, r in per_seq_results:
                v = r["per_class_iou"][i][1]
                row.append(f"{v:8.4f}" if v is not None else "    -   ")
            print("  ".join(row))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "class_id", "class_name", "iou",
                        "n_pred_voxels", "n_gt_voxels", "n_intersect"])
            for seq, r in per_seq_results:
                for cid, (nm, iv) in r["per_class_iou"].items():
                    w.writerow([seq, cid, nm, iv,
                                r["n_pred_voxels"], r["n_gt_voxels"], r["n_intersection"]])
        print(f"\nwrote per-class CSV → {args.out_csv}")


if __name__ == "__main__":
    main()
