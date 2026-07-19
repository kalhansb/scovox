#!/usr/bin/env python3
"""Voxel-wise mIoU for SCovox on SemanticKITTI.

Same methodology as eval_slimvdb_kitti_miou.py EXCEPT the GT voxel grid is
built in SCovox's world frame: T_world_velo = P @ Tr (matching
semantickitti_replay_node line 181). SLIM-VDB uses Tr_inv @ P @ Tr instead.
"""
import argparse, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).parent))
from eval_slimvdb_kitti_miou import (
    read_kitti_calib, build_learning_map, voxelize_key, compute_voxel_miou,
)


def read_kitti_poses_scovox(poses_file: Path, Tr: np.ndarray):
    """SCovox convention: T_world_velo = P @ Tr."""
    out = []
    with open(poses_file) as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if len(vals) != 12:
                continue
            P = np.eye(4)
            P[:3, :4] = np.array(vals).reshape(3, 4)
            out.append(P @ Tr)
    return out


def build_gt_voxels_scovox(kitti_root, sequence, n_scans, voxel_size, lmap):
    seq_dir = kitti_root / "sequences" / sequence
    Tr = read_kitti_calib(seq_dir / "calib.txt")
    poses = read_kitti_poses_scovox(seq_dir / "poses.txt", Tr)
    velo_files = sorted((seq_dir / "velodyne").glob("*.bin"))[:n_scans]
    label_files = sorted((seq_dir / "labels").glob("*.label"))[:n_scans]
    counts = defaultdict(lambda: defaultdict(int))
    for i, (vf, lf) in enumerate(zip(velo_files, label_files)):
        pts = np.fromfile(vf, dtype=np.float32).reshape(-1, 4)[:, :3]
        lbl_raw = np.fromfile(lf, dtype=np.uint32) & 0xFFFF
        lbl = lmap[lbl_raw]
        keep = lbl > 0
        pts, lbl = pts[keep], lbl[keep]
        T = poses[i].astype(np.float32)
        pw = (T @ np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1).T).T[:, :3]
        keys = voxelize_key(pw, voxel_size)
        for k, c in zip(keys, lbl):
            counts[int(k)][int(c)] += 1
    return {k: max(d.items(), key=lambda kv: kv[1])[0] for k, d in counts.items()}


# REPLAY → yaml learning-class LUT (per `[[kitti-miou-replay-bug]]` memo).
# PolarSeg's .topk files use the REPLAY scheme which inserts class 15 =
# "lane-marking" and shifts veg/trunk/terrain/pole/traffic-sign +1 vs
# yaml's learning_map (which has no lane-marking dim). Predicted voxels
# in REPLAY space (16..20) must be remapped to yaml space (15..19) before
# bucket-IoU against yaml-space GT; otherwise those classes score 0.
# Lane-marking (REPLAY 15) merges into road (yaml 9 = "road") — the
# PolarSeg convention treats it as a road sub-class.
_REPLAY_TO_YAML_LUT = np.arange(256, dtype=np.int32)
_REPLAY_TO_YAML_LUT[15] = 9   # lane-marking → road
_REPLAY_TO_YAML_LUT[16] = 15  # vegetation
_REPLAY_TO_YAML_LUT[17] = 16  # trunk
_REPLAY_TO_YAML_LUT[18] = 17  # terrain
_REPLAY_TO_YAML_LUT[19] = 18  # pole
_REPLAY_TO_YAML_LUT[20] = 19  # traffic-sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", type=Path, required=True)
    ap.add_argument("--npz_root", type=Path, required=True)
    ap.add_argument("--variant", default="scovox")
    ap.add_argument("--sequences", nargs="+", default=["06","07","08","09","10"])
    ap.add_argument("--n_scans", type=int, default=100)
    ap.add_argument("--voxel_size", type=float, default=0.10)
    ap.add_argument("--semantic_kitti_yaml", type=Path, required=True)
    ap.add_argument(
        "--replay_to_yaml_lut", action="store_true",
        help="Apply the REPLAY→yaml class-id LUT to predicted voxels before "
             "scoring. Required when the NPZ came from a soft-prob PolarSeg "
             "run (REPLAY layout: lane-marking at id 15, veg/trunk/terrain/"
             "pole/traffic-sign shifted +1). Without this flag, those 5 "
             "classes score 0 against the yaml-space GT, knocking ~0.10 off "
             "the apparent mIoU. See `[[kitti-miou-replay-bug]]`.")
    args = ap.parse_args()

    lmap = build_learning_map(args.semantic_kitti_yaml)
    yaml_cfg = yaml.safe_load(open(args.semantic_kitti_yaml))
    lmi = yaml_cfg["learning_map_inv"]; raw_names = yaml_cfg["labels"]
    num_classes = max(lmi.keys()) + 1
    class_names = [raw_names.get(lmi.get(i, 0), f"train_{i}") for i in range(num_classes)]

    print(f"[SCovox mIoU] variant={args.variant} seqs={args.sequences} voxel={args.voxel_size}m"
          f" replay_to_yaml_lut={args.replay_to_yaml_lut}")
    per_seq = []
    for seq in args.sequences:
        npz = args.npz_root / seq / f"{args.variant}.npz"
        if not npz.exists():
            print(f"  seq {seq}: MISSING {npz}"); continue
        d = np.load(npz)
        xyz = d["points"].astype(np.float32)
        cls = d["semantic_class"].astype(np.int32)
        if args.replay_to_yaml_lut:
            cls = _REPLAY_TO_YAML_LUT[np.clip(cls, 0, 255)]
        keys = voxelize_key(xyz, args.voxel_size)
        pred = {int(k): int(c) for k, c in zip(keys, cls)}
        print(f"  seq {seq}: building GT …", flush=True)
        gt = build_gt_voxels_scovox(args.kitti_root, seq, args.n_scans, args.voxel_size, lmap)
        res = compute_voxel_miou(pred, gt, num_classes, class_names)
        print(f"    pred={res['n_pred_voxels']} gt={res['n_gt_voxels']} "
              f"∩={res['n_intersection']}  mIoU={res['miou']:.4f}")
        per_seq.append((seq, res["miou"]))
    if per_seq:
        vals = [m for _, m in per_seq if m == m]
        print(f"\n=== SCovox {args.variant} KITTI mIoU = {np.mean(vals):.4f} ± {np.std(vals):.4f} ===")


if __name__ == "__main__":
    main()
