#!/usr/bin/env python3
"""Convert WorldQuerySystem CSV output to NPZ for the scovox_eval toolkit.

Usage:
    python gz_gt_to_npz.py world_query_results.csv gt_voxels.npz
    python gz_gt_to_npz.py world_query_results.csv gt_voxels.npz --occupied-only

Output NPZ keys:
    points       (N, 3) float32 — voxel centres
    gt_binary    (N,)   float32 — 1.0 = occupied, 0.0 = free
    semantic_class (N,) int32   — semantic label ID (0 = unknown/free)
"""

import argparse
import csv

import numpy as np


def convert(csv_path: str, npz_path: str, occupied_only: bool = False):
    xs, ys, zs = [], [], []
    occ, labels = [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_occ = int(row["occupied"])
            if occupied_only and not is_occ:
                continue
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            occ.append(float(is_occ))
            labels.append(int(row["semantic_label"]))

    points = np.column_stack([xs, ys, zs]).astype(np.float32)
    gt_binary = np.array(occ, dtype=np.float32)
    semantic_class = np.array(labels, dtype=np.int32)

    np.savez_compressed(
        npz_path,
        points=points,
        gt_binary=gt_binary,
        semantic_class=semantic_class,
    )

    n_occ = int(gt_binary.sum())
    print(f"Saved {len(points)} voxels ({n_occ} occupied, "
          f"{len(points) - n_occ} free) to {npz_path}")
    unique_labels = sorted(set(labels))
    print(f"Semantic labels present: {unique_labels}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert WorldQuerySystem CSV to NPZ"
    )
    parser.add_argument("csv", help="Input CSV from WorldQuerySystem")
    parser.add_argument("npz", help="Output NPZ file")
    parser.add_argument("--occupied-only", action="store_true",
                        help="Only include occupied voxels")
    args = parser.parse_args()
    convert(args.csv, args.npz, args.occupied_only)


if __name__ == "__main__":
    main()
