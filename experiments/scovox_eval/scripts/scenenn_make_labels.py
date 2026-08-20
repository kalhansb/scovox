#!/usr/bin/env python3
"""Render per-pixel NYU-40 GT labels for SceneNN frames from the annotated mesh.

SceneNN's 2D annotation package (label PNGs) is only on the dead Google Drive
folder; the HTTP mirror carries the annotated mesh instead. The mesh does have
per-vertex `nyu_class`, so labels are recovered by back-projection rather than
rasterisation: deproject each depth pixel, move it into the mesh frame with the
transform from scenenn_register_mesh.py, and take the nearest vertex's class.

Pixels further than --max-dist from any vertex stay class 0 (unknown) — that is
the honest answer for geometry the annotated mesh never covered, and SCovox's
Dirichlet path treats class 0 as "no semantic evidence".

Usage:
  python3 scenenn_make_labels.py --scene /home/jetsondevkit/datasets/scenenn/016 \
      --intrinsics /home/jetsondevkit/datasets/scenenn/asus.ini \
      --align /home/jetsondevkit/datasets/scenenn/016/mesh_align.json --jobs 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scovox_eval.scenenn_data import (  # noqa: E402
    deproject, load_annotated_ply, load_depth_m, load_intrinsics, load_trajectory, to_world,
)

_G = {}


def _init(mesh_path, intr_path, scene_dir, T, max_dist, out_dir, pose_offset, perframe_path):
    xyz, nyu, _ = load_annotated_ply(mesh_path)
    _G["tree"] = cKDTree(xyz)
    _G["nyu"] = nyu.astype(np.uint8)
    _G["K"] = load_intrinsics(intr_path)
    _G["poses"] = load_trajectory(Path(scene_dir) / "trajectory.log")
    _G["T"] = np.array(T)
    _G["max_dist"] = max_dist
    _G["out"] = Path(out_dir)
    _G["off"] = pose_offset
    _G["Ts"] = np.load(perframe_path)["T"] if perframe_path else None


def _one(item):
    idx, depth_file = item
    K = _G["K"]
    # Per-frame transform when available: a single global fit cannot absorb
    # trajectory drift, and a 7 cm misalignment silently relabels thin
    # structures (picture/window on a wall) as whatever is behind them.
    T = _G["Ts"][idx] if _G["Ts"] is not None else _G["T"]
    depth = load_depth_m(depth_file, flip_vertical=False)
    h, w = depth.shape
    labels = np.zeros((h, w), dtype=np.uint8)

    pts_cam, flat = deproject(depth, K, stride=1)
    if len(pts_cam):
        pts_w = to_world(pts_cam, _G["poses"][idx + _G["off"]])
        pts_m = pts_w @ T[:3, :3].T + T[:3, 3]
        dist, vid = _G["tree"].query(pts_m, k=1, workers=1)
        cls = _G["nyu"][vid]
        cls[dist > _G["max_dist"]] = 0
        labels.reshape(-1)[flat] = cls

    PILImage.fromarray(labels).save(_G["out"] / f"label{idx + 1:05d}.png")
    return idx, float((labels > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-dist", type=float, default=0.10,
                    help="max point-to-vertex distance before a pixel is left unknown")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--per-frame-align", default=None,
                    help="mesh_align_perframe.npz from scenenn_register_mesh.py --per-frame; "
                         "auto-detected next to --align when present")
    args = ap.parse_args()

    scene = Path(args.scene)
    mesh_path = Path(args.mesh) if args.mesh else next(scene.glob("*_nyu.ply"))
    out_dir = Path(args.out) if args.out else scene / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    align = json.loads(Path(args.align).read_text())
    T = align["T_mesh_from_traj"]
    pose_offset = int(align.get("pose_offset", 0))
    print(f"global align residual: {align['median_residual_m']:.4f} m "
          f"(inliers<5cm {align['inlier_frac_5cm']:.1%})")

    pf = args.per_frame_align
    if pf is None:
        cand = Path(args.align).with_name("mesh_align_perframe.npz")
        pf = str(cand) if cand.exists() else None
    if pf:
        z = np.load(pf)
        print(f"per-frame align: median {np.median(z['median_residual_m']):.4f} m, "
              f"mean inliers<5cm {z['inlier_frac_5cm'].mean():.1%} "
              f"({(z['inlier_frac_5cm'] < 0.8).sum()} weak frames)")
    else:
        print("per-frame align: NOT FOUND -- labels will use the global fit only")

    poses = load_trajectory(scene / "trajectory.log")
    df = sorted((scene / "frames" / "depth").glob("depth*.png"))
    n = min(len(df), len(poses) - pose_offset)
    if args.limit > 0:
        n = min(n, args.limit)
    items = [(i, df[i]) for i in range(n)]
    print(f"labelling {n} frames -> {out_dir} with {args.jobs} workers")

    from multiprocessing import Pool
    covs = []
    with Pool(args.jobs, initializer=_init,
              initargs=(mesh_path, args.intrinsics, str(scene), T,
                        args.max_dist, str(out_dir), pose_offset, pf)) as pool:
        for k, (idx, cov) in enumerate(pool.imap_unordered(_one, items, chunksize=4)):
            covs.append(cov)
            if k % 100 == 0:
                print(f"  {k}/{n} frames, mean labelled coverage {np.mean(covs):.1%}", flush=True)

    print(f"done: {n} label PNGs, mean labelled coverage {np.mean(covs):.1%}")


if __name__ == "__main__":
    main()
