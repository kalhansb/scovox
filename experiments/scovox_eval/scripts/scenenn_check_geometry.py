#!/usr/bin/env python3
"""Settle SceneNN's two replay ambiguities by measuring, not guessing.

1. Is the extracted depth PNG flipped vertically w.r.t. the mesh?
   (PlaybackSync.cpp flips colour but not depth -- see scenenn_data.)
2. Does depth frame k pair with trajectory pose k, k-1, or k+1?

Both are decided the same way: deproject depth with the candidate convention,
push it into world space with the candidate pose, and measure the median
distance to the annotated mesh. The correct combination sits at a few
centimetres (sensor noise + reconstruction error); every wrong one is
off by tens of centimetres or more.

Usage:
  python3 scenenn_check_geometry.py --scene /home/jetsondevkit/datasets/scenenn/016 \
      --intrinsics /home/jetsondevkit/datasets/scenenn/asus.ini
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scovox_eval.scenenn_data import (  # noqa: E402
    deproject, load_annotated_ply, load_depth_m, load_intrinsics, load_trajectory, to_world,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--mesh", default=None, help="defaults to <scene>/<id>_nyu.ply")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    scene = Path(args.scene)
    mesh_path = Path(args.mesh) if args.mesh else next(scene.glob("*_nyu.ply"))
    K = load_intrinsics(args.intrinsics)
    poses = load_trajectory(scene / "trajectory.log")
    depth_files = sorted((scene / "frames" / "depth").glob("depth*.png"))

    print(f"mesh    : {mesh_path.name}")
    print(f"poses   : {len(poses)}")
    print(f"depth   : {len(depth_files)} frames extracted so far")
    print(f"intrin  : fx={K['fx']:.3f} cx={K['cx']:.1f} cy={K['cy']:.1f} {K['width']}x{K['height']}")

    xyz, nyu, _ = load_annotated_ply(mesh_path)
    print(f"vertices: {len(xyz)}  nyu_class present: {nyu is not None}")
    tree = cKDTree(xyz)

    # Sample frames spread across the part of the sequence already extracted.
    n_avail = min(len(depth_files), len(poses))
    picks = np.linspace(0, n_avail - 1, args.frames).astype(int)

    print(f"\n{'flip':>6} {'offset':>7} {'median_dist_m':>14} {'p90_m':>9} {'inlier<5cm':>11}")
    print("-" * 52)
    results = {}
    for flip in (True, False):
        for offset in (-1, 0, 1):
            meds, p90s, inl = [], [], []
            for fi in picks:
                pose_idx = fi + offset
                if not (0 <= pose_idx < len(poses)):
                    continue
                depth = load_depth_m(depth_files[fi], flip_vertical=flip)
                pts_cam, _ = deproject(depth, K, stride=args.stride)
                if len(pts_cam) < 100:
                    continue
                pts_w = to_world(pts_cam, poses[pose_idx])
                d, _ = tree.query(pts_w, k=1, workers=-1)
                meds.append(np.median(d))
                p90s.append(np.percentile(d, 90))
                inl.append(float(np.mean(d < 0.05)))
            if not meds:
                continue
            key = (flip, offset)
            results[key] = (np.mean(meds), np.mean(p90s), np.mean(inl))
            print(f"{str(flip):>6} {offset:>7} {np.mean(meds):>14.4f} "
                  f"{np.mean(p90s):>9.4f} {np.mean(inl):>10.1%}")

    best = min(results, key=lambda k: results[k][0])
    print("-" * 52)
    print(f"BEST: flip_vertical={best[0]}  pose_offset={best[1]}  "
          f"(median {results[best][0]:.4f} m, inliers {results[best][2]:.1%})")
    print("\nInterpretation: depth frame k (1-indexed file) pairs with "
          f"poses[k-1{best[1]:+d}] = poses[k{best[1] - 1:+d}]")


if __name__ == "__main__":
    main()
