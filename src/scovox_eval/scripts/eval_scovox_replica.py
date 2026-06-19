#!/usr/bin/env python3
"""SCovox Replica eval: mIoU (18-class SCovox space) + Chamfer vs GT mesh.

Reads the SCovox-produced `scovox.npz` (keys: points [N,3], semantic_class [N]),
computes:
  1. Voxel-wise mIoU in SCovox's 18-class space, against GT voxelized from
     depth + semantic_gt_fixed PNGs (same pipeline as the SLIM-VDB mIoU eval
     but with CATEGORY mapping applied so SCovox's internal [1..18] class IDs
     match GT).
  2. Bidirectional Chamfer L2 + F@5cm vs Replica mesh.ply.
"""
import argparse, sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).parent))
from eval_slimvdb_replica_miou import (
    voxel_key, read_poses, read_cam_params, compute_miou,
)
from eval_slimvdb_replica_chamfer import chamfer_and_f, sample_mesh


# Reproduces the SCovox 18-category mapping (see replica_replay_node.CATEGORY_COLORS
# + replica_eval.launch.py semantic_color_map_classes). Key = class name substring
# matched against Replica's info_semantic.json "classes" list; value = SCovox class id.
SCOVOX_CATS = {
    "wall": 1, "floor": 2, "ceiling": 3, "door": 4, "window": 5,
    "chair": 6, "table": 7, "sofa": 8, "bed": 9, "cushion": 10,
    "lamp": 11, "cabinet": 12, "blinds": 13, "book": 14, "picture": 15,
    "plant": 16, "rug": 17, "pillar": 18,
}


def class_id_remap(info_semantic: dict) -> np.ndarray:
    """Replica class_id (1..101) → SCovox category id (0 for unmapped).

    Matches longer keys first so "indoor-plant" → plant (16), not door (4).
    """
    sorted_cats = sorted(SCOVOX_CATS.items(), key=lambda kv: -len(kv[0]))
    max_id = max((c["id"] for c in info_semantic.get("classes", [])), default=0)
    lut = np.zeros(max_id + 2, dtype=np.int32)
    for cls in info_semantic.get("classes", []):
        name = cls.get("name", "").lower()
        for key, sid in sorted_cats:
            if key in name:
                lut[cls["id"]] = sid
                break
    return lut


def build_gt_scovox_voxels(scene_dir: Path, voxel_size: float, n_frames: int,
                           stride: int) -> dict[int, int]:
    root = scene_dir.parent
    fx, fy, cx, cy, W, H, depth_scale = read_cam_params(root)
    poses = read_poses(scene_dir / "poses.txt")
    sem_dir = scene_dir / "semantic_gt_fixed"
    depth_dir = scene_dir / "depth"
    info = json.load(open(scene_dir / "info_semantic.json"))
    cls_lut = class_id_remap(info)

    counts = defaultdict(lambda: defaultdict(int))
    n = min(n_frames, len(poses))
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u_grid.flatten().astype(np.float32)
    v_flat = v_grid.flatten().astype(np.float32)

    for i in range(0, n, stride):
        depth_img = cv2.imread(str(depth_dir / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        sem_img = cv2.imread(str(sem_dir / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth_img is None or sem_img is None:
            continue
        d = depth_img.flatten().astype(np.float32) / depth_scale
        raw_cls = sem_img.flatten().astype(np.int32)
        raw_cls = np.clip(raw_cls, 0, cls_lut.size - 1)
        s = cls_lut[raw_cls]  # map to SCovox 18-class space
        valid = (d > 0.01) & (d < 8.0) & (s > 0)
        d, s = d[valid], s[valid]
        u, v = u_flat[valid], v_flat[valid]
        x = (u - cx) / fx * d
        y = (v - cy) / fy * d
        z = d
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
        T = poses[i].astype(np.float32)
        pts_world = (T @ pts_cam.T).T[:, :3]
        keys = voxel_key(pts_world, voxel_size)
        for k, c in zip(keys, s):
            counts[int(k)][int(c)] += 1
    return {k: max(d.items(), key=lambda kv: kv[1])[0] for k, d in counts.items()}


def load_scovox_pred_voxels(npz_path: Path, voxel_size: float) -> tuple[dict, np.ndarray]:
    d = np.load(npz_path)
    xyz = d["points"].astype(np.float32)
    cls = d["semantic_class"].astype(np.int32)
    keys = voxel_key(xyz, voxel_size)
    return {int(k): int(c) for k, c in zip(keys, cls)}, xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", type=Path, required=True)
    ap.add_argument("--npz_root", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+",
                    default=["room0","room1","room2","office0","office1","office2","office3","office4"])
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--num_classes", type=int, default=19)  # 0..18
    ap.add_argument("--do_miou", action="store_true")
    ap.add_argument("--do_chamfer", action="store_true")
    ap.add_argument("--gt_points", type=int, default=200_000)
    ap.add_argument("--sample_k", type=int, default=500_000)
    args = ap.parse_args()

    if not (args.do_miou or args.do_chamfer):
        args.do_miou = args.do_chamfer = True

    miou_rows = []
    cham_rows = []
    for scene in args.scenes:
        npz = args.npz_root / scene / "scovox.npz"
        if not npz.exists():
            print(f"  [skip] no npz {npz}"); continue
        print(f"\n=== {scene} ===", flush=True)
        pred, pred_xyz = load_scovox_pred_voxels(npz, args.voxel_size)

        if args.do_miou:
            gt = build_gt_scovox_voxels(args.replica_root / scene,
                                         args.voxel_size, args.n_frames, args.stride)
            m = compute_miou(pred, gt, args.num_classes)
            print(f"  mIoU: pred={m['n_pred']} gt={m['n_gt']} ∩={m['n_intersect']} mIoU={m['miou']:.4f}")
            miou_rows.append((scene, m["miou"]))

        if args.do_chamfer:
            gt_mesh = args.replica_root / scene / "mesh.ply"
            if not gt_mesh.exists():
                print(f"  [chamfer skip] no mesh at {gt_mesh}")
            else:
                gt_pts = sample_mesh(gt_mesh, args.gt_points)
                cm = chamfer_and_f(pred_xyz.astype(np.float64), gt_pts, sample_k=args.sample_k)
                print(f"  Chamfer L2 = {cm['chamfer_l2_m']*100:.2f} cm   "
                      f"F@5cm = {cm['fscore_at_5cm']:.4f}   "
                      f"P={cm['precision_at_5cm']:.3f} R={cm['recall_at_5cm']:.3f}")
                cham_rows.append((scene, cm))

    if miou_rows:
        vals = [m for _, m in miou_rows if m == m]
        print(f"\n=== SCovox Replica mIoU (18-cls) = {np.mean(vals):.4f} ± {np.std(vals):.4f} ===")
    if cham_rows:
        l2 = np.array([r["chamfer_l2_m"] for _, r in cham_rows])
        f5 = np.array([r["fscore_at_5cm"] for _, r in cham_rows])
        print(f"=== SCovox Replica Chamfer = {l2.mean()*100:.2f} ± {l2.std()*100:.2f} cm   "
              f"F@5cm = {f5.mean():.4f} ± {f5.std():.4f} ===")


if __name__ == "__main__":
    main()
