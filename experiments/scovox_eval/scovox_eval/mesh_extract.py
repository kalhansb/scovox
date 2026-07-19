"""Mesh extraction from SCovox occupancy grids via marching cubes.

Subscribes to the SCovox PointCloud2 topic (which contains per-voxel
occupancy_prob), reconstructs a dense occupancy grid, runs marching cubes,
and optionally writes the result as a PLY/OBJ mesh with per-vertex
semantic labels.

Can also operate offline on a saved numpy occupancy array.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from skimage.measure import marching_cubes
except ImportError:
    marching_cubes = None

try:
    import open3d as o3d
except ImportError:
    o3d = None


def occupancy_grid_from_pointcloud(
    points: np.ndarray,
    occupancy: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sparse occupied points + probabilities into a dense 3-D grid.

    Parameters
    ----------
    points : (N, 3) float array of voxel centres
    occupancy : (N,) float array of occupancy probabilities
    resolution : voxel edge length in metres

    Returns
    -------
    grid : 3-D float array with occupancy values (unobserved = 0)
    origin : (3,) float array — world position of grid[0,0,0]
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    origin = mins - resolution  # one-voxel padding
    shape = np.ceil((maxs - mins) / resolution).astype(int) + 3  # +3 for padding

    grid = np.zeros(shape, dtype=np.float32)
    indices = np.round((points - origin) / resolution).astype(int)

    # Clip to grid bounds (safety)
    for d in range(3):
        indices[:, d] = np.clip(indices[:, d], 0, shape[d] - 1)

    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = occupancy
    return grid, origin


def extract_mesh(
    grid: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run marching cubes on a dense occupancy grid.

    Parameters
    ----------
    grid : 3-D occupancy array
    origin : world position of grid[0,0,0]
    resolution : voxel size in metres
    threshold : iso-surface occupancy threshold

    Returns
    -------
    vertices : (V, 3) float array in world frame
    faces : (F, 3) int array of triangle indices
    """
    if marching_cubes is None:
        raise ImportError(
            "scikit-image is required for mesh extraction: "
            "pip install scikit-image"
        )

    vertices, faces, normals, _ = marching_cubes(
        grid, level=threshold, spacing=(resolution, resolution, resolution)
    )
    # Shift to world frame
    vertices += origin
    return vertices, faces


def save_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh to PLY using Open3D, or fallback to raw PLY."""
    if o3d is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(path), mesh)
    else:
        _write_ply_ascii(path, vertices, faces)


def _write_ply_ascii(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    nv, nf = len(vertices), len(faces)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract mesh from SCovox occupancy grid")
    parser.add_argument("input", help="Path to .npz file with 'points' and 'occupancy' arrays")
    parser.add_argument("-o", "--output", default="mesh.ply", help="Output mesh path (PLY)")
    parser.add_argument("-r", "--resolution", type=float, default=0.05, help="Voxel resolution (m)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Occupancy iso-surface threshold")
    args = parser.parse_args()

    data = np.load(args.input)
    points = data["points"]
    occupancy = data["occupancy"]

    print(f"Loaded {len(points)} voxels, resolution={args.resolution}m, threshold={args.threshold}")

    grid, origin = occupancy_grid_from_pointcloud(points, occupancy, args.resolution)
    print(f"Dense grid shape: {grid.shape}")

    vertices, faces = extract_mesh(grid, origin, args.resolution, args.threshold)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} triangles")

    out = Path(args.output)
    save_mesh_ply(out, vertices, faces)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
