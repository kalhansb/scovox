#!/usr/bin/env python3
"""Extract voxel centres from a SCovox .npz file to a CSV for WorldQuerySystem.

Usage:
    python npz_to_query_csv.py scovox.npz query_points.csv
    python npz_to_query_csv.py scovox.npz query_points.csv --key points

The output CSV has the format expected by WorldQuerySystem:
    x,y,z
    1.05,2.35,0.15
    ...
"""

import argparse

import numpy as np


def convert(npz_path: str, csv_path: str, key: str = "points"):
    data = np.load(npz_path)
    points = data[key]
    n = len(points)

    with open(csv_path, "w") as f:
        f.write("x,y,z\n")
        for p in points:
            f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n")

    print(f"Wrote {n} query points to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract voxel centres from NPZ to CSV for WorldQuerySystem"
    )
    parser.add_argument("npz", help="Input .npz file with 'points' array")
    parser.add_argument("csv", help="Output CSV file")
    parser.add_argument("--key", default="points",
                        help="NPZ key for point array (default: points)")
    args = parser.parse_args()
    convert(args.npz, args.csv, args.key)


if __name__ == "__main__":
    main()
