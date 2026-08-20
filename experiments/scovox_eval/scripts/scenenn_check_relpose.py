#!/usr/bin/env python3
"""Mesh-independent check of the SceneNN frame<->pose correspondence.

Takes two depth frames a few steps apart, maps the later cloud into the
earlier camera's frame using the *relative* trajectory pose, and measures
cloud-to-cloud residual. This isolates the question "do these poses describe
the motion between these frames?" from any mesh/world-frame convention, so
it stays valid even when world alignment is broken.

Correct correspondence + camera convention -> residual at sensor noise (~1-2 cm).
Wrong index offset or wrong optical-axis convention -> decimetres.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scovox_eval.scenenn_data import (  # noqa: E402
    deproject, load_depth_m, load_intrinsics, load_trajectory,
)

CONVENTIONS = {
    "std(x,y,z)": (1, 1, 1),
    "gl(x,-y,-z)": (1, -1, -1),
    "(x,-y,z)": (1, -1, 1),
    "(-x,y,z)": (-1, 1, 1),
    "(x,y,-z)": (1, 1, -1),
    "(-x,-y,z)": (-1, -1, 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--gap", type=int, default=5, help="frame separation")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--offsets", type=str, default="-3,-2,-1,0,1,2,3")
    ap.add_argument("--anchors", type=str, default="100,400,700")
    args = ap.parse_args()

    scene = Path(args.scene)
    K = load_intrinsics(args.intrinsics)
    poses = load_trajectory(scene / "trajectory.log")
    df = sorted((scene / "frames" / "depth").glob("depth*.png"))
    offsets = [int(v) for v in args.offsets.split(",")]
    anchors = [int(v) for v in args.anchors.split(",")]

    print(f"frames={len(df)} poses={len(poses)} gap={args.gap}\n")
    print(f"{'conv':>12} {'offset':>7} {'median_cm':>10} {'p90_cm':>8}")
    print("-" * 42)

    scores = {}
    for cname, sgn in CONVENTIONS.items():
        sgn = np.array(sgn, dtype=np.float64)
        for off in offsets:
            meds, p90s = [], []
            for a in anchors:
                b = a + args.gap
                pa, pb = a + off, b + off
                if not (0 <= pa < len(poses) and 0 <= pb < len(poses)):
                    continue
                if b >= len(df):
                    continue
                ca, _ = deproject(load_depth_m(df[a], flip_vertical=False), K, stride=args.stride)
                cb, _ = deproject(load_depth_m(df[b], flip_vertical=False), K, stride=args.stride)
                if len(ca) < 200 or len(cb) < 200:
                    continue
                ca, cb = ca * sgn, cb * sgn
                # Map cloud b into camera a: T_rel = pose_a^-1 @ pose_b
                T_rel = np.linalg.inv(poses[pa]) @ poses[pb]
                cb_in_a = cb @ T_rel[:3, :3].T + T_rel[:3, 3]
                d, _ = cKDTree(ca).query(cb_in_a, k=1, workers=-1)
                meds.append(np.median(d))
                p90s.append(np.percentile(d, 90))
            if meds:
                scores[(cname, off)] = np.mean(meds)
                print(f"{cname:>12} {off:>7} {np.mean(meds) * 100:>10.2f} {np.mean(p90s) * 100:>8.2f}")

    best = min(scores, key=scores.get)
    print("-" * 42)
    print(f"BEST: convention={best[0]}  pose_offset={best[1]}  median={scores[best] * 100:.2f} cm")


if __name__ == "__main__":
    main()
