#!/usr/bin/env python3
"""Generate binary occupancy ground truth for Replica predictions.

Voxelises the GT mesh at the same resolution as the map. For each predicted
voxel, snaps to the nearest GT grid cell. If that cell is occupied (mesh
triangle intersects it) → gt_binary=1, else → gt_binary=0.

This approach uses the same discretisation for both prediction and GT,
eliminating boundary threshold ambiguity.

Produces per-variant GT NPZ files compatible with `compute-ece`:
    {scene}/{variant}_gt_occ.npz  with keys: occupancy_prob, gt_binary

Usage:
    # Single scene + variant:
    python replica_gt_binary.py --scene room0 --variant scovox --resolution 0.05

    # All scenes, all variants, all resolutions:
    python replica_gt_binary.py --all

    # Then run ECE:
    compute-ece results/replica_5cm/room0/scovox_gt_occ.npz results/replica_5cm/room0/scovox_gt_occ.npz
    (occupancy_prob and gt_binary are in the same file)
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


SCENES = ["room0", "room1", "room2",
          "office0", "office1", "office2", "office3", "office4"]
VARIANTS = ["scovox", "covox", "logodds"]
RESOLUTIONS = {"5cm": 0.05, "10cm": 0.10, "20cm": 0.20}

DATASET_ROOT = Path(__file__).resolve().parents[5] / "data" / "replica_niceslam"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def voxelise_mesh(mesh, resolution: float):
    """Voxelise a mesh and return a set of occupied voxel indices.

    Returns the voxel grid origin and a set of (ix, iy, iz) tuples for
    occupied cells.
    """
    vox = mesh.voxelized(pitch=resolution)
    # vox.matrix is a 3D boolean array; vox.transform gives world-frame origin
    origin = vox.transform[:3, 3]
    # Get indices of occupied cells
    occupied_indices = set(map(tuple, np.argwhere(vox.matrix)))
    return origin, occupied_indices, vox.matrix.shape


def generate_gt_binary(
    pred_npz_path: Path,
    gt_mesh_path: Path,
    output_path: Path,
    resolution: float,
):
    """Generate occupancy GT for a single prediction file."""
    pred = np.load(pred_npz_path)
    pred_points = pred["points"]
    occupancy_prob = pred["occupancy_prob"]

    # Voxelise GT mesh at the same resolution
    mesh = trimesh.load(str(gt_mesh_path))
    origin, occupied_set, grid_shape = voxelise_mesh(mesh, resolution)

    # For each predicted voxel, snap to nearest GT grid cell
    # Convert world coords to grid indices
    grid_indices = np.round((pred_points - origin) / resolution).astype(int)

    # Look up each predicted voxel in the occupied set
    gt_binary = np.zeros(len(pred_points), dtype=np.float32)
    for i, idx in enumerate(grid_indices):
        if tuple(idx) in occupied_set:
            gt_binary[i] = 1.0

    n_occ = int(gt_binary.sum())
    n_free = len(gt_binary) - n_occ
    pct = 100 * n_occ / len(gt_binary)

    np.savez_compressed(
        output_path,
        occupancy_prob=occupancy_prob,
        gt_binary=gt_binary,
    )

    return n_occ, n_free, pct


def process_scene(scene: str, variant: str, res_name: str, resolution: float):
    """Process a single scene + variant + resolution."""
    results_dir = RESULTS_ROOT / f"replica_{res_name}" / scene
    pred_path = results_dir / f"{variant}.npz"
    gt_mesh = DATASET_ROOT / scene / "mesh.ply"
    output_path = results_dir / f"{variant}_gt_occ.npz"

    if not pred_path.exists():
        print(f"  SKIP {scene}/{variant}@{res_name}: no prediction")
        return
    if not gt_mesh.exists():
        print(f"  SKIP {scene}/{variant}@{res_name}: no GT mesh at {gt_mesh}")
        return

    n_occ, n_free, pct = generate_gt_binary(
        pred_path, gt_mesh, output_path, resolution
    )
    print(f"  {scene}/{variant}@{res_name}: "
          f"{n_occ} occ + {n_free} free ({pct:.1f}% occ) → {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate occupancy GT for Replica predictions (voxelised mesh)"
    )
    parser.add_argument("--scene", choices=SCENES, default=None)
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--resolution", type=float, default=None,
                        help="Voxel resolution in meters (e.g. 0.05)")
    parser.add_argument("--all", action="store_true",
                        help="Process all scenes × variants × resolutions")
    args = parser.parse_args()

    if args.all:
        for res_name, res_val in RESOLUTIONS.items():
            print(f"\n=== {res_name} ===")
            for scene in SCENES:
                for variant in VARIANTS:
                    process_scene(scene, variant, res_name, res_val)
    elif args.scene and args.variant and args.resolution:
        res_name = f"{int(args.resolution*100)}cm"
        process_scene(args.scene, args.variant, res_name, args.resolution)
    else:
        parser.error("Specify --scene/--variant/--resolution or --all")


if __name__ == "__main__":
    main()
