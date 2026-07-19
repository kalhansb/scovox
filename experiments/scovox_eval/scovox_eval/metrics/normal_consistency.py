"""Compute normal consistency between predicted point cloud and GT mesh."""

from __future__ import annotations

import argparse

import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None


def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate normals via PCA on k-nearest neighbors."""
    tree = KDTree(points)
    _, idx = tree.query(points, k=k)
    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[idx[i]]
        cov = np.cov(neighbors.T)
        _, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]  # smallest eigenvector = normal
    return normals


def compute_normal_consistency(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    gt_normals: np.ndarray | None = None,
    k: int = 20,
) -> dict[str, float]:
    """Normal consistency: mean |dot(n_pred, n_gt)| at nearest points.

    Parameters
    ----------
    pred_points : (N, 3) predicted points
    gt_points : (M, 3) GT points (or mesh vertices)
    gt_normals : (M, 3) GT normals. If None, estimated from gt_points.
    k : neighbors for normal estimation

    Returns
    -------
    dict with normal_consistency (float in [0, 1])
    """
    if KDTree is None:
        raise ImportError("scipy required: pip install scipy")

    # Estimate normals for predicted points
    pred_normals = estimate_normals(pred_points, k)

    # Get GT normals
    if gt_normals is None:
        gt_normals = estimate_normals(gt_points, k)

    # For each predicted point, find nearest GT point and compare normals
    gt_tree = KDTree(gt_points)
    dists, indices = gt_tree.query(pred_points)

    # Only compare points within 10cm of GT
    mask = dists < 0.10
    if mask.sum() == 0:
        return {"normal_consistency": 0.0, "matched_points": 0}

    dots = np.abs(np.sum(
        pred_normals[mask] * gt_normals[indices[mask]], axis=1
    ))
    # Clamp to [0,1] for safety
    dots = np.clip(dots, 0.0, 1.0)

    return {
        "normal_consistency": float(dots.mean()),
        "matched_points": int(mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute normal consistency")
    parser.add_argument("predicted", help=".npz with 'points'")
    parser.add_argument("ground_truth", help=".ply GT mesh or .npz with 'points'")
    parser.add_argument("-k", type=int, default=20, help="Neighbors for PCA normals")
    args = parser.parse_args()

    pred = np.load(args.predicted)["points"]

    if args.ground_truth.endswith(".ply"):
        try:
            import trimesh
            mesh = trimesh.load(args.ground_truth)
            gt_points = np.asarray(mesh.vertices, dtype=np.float32)
            if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
                gt_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            else:
                gt_normals = None
        except ImportError:
            raise ImportError("trimesh required for PLY: pip install trimesh")
    else:
        gt_data = np.load(args.ground_truth)
        gt_points = gt_data["points"]
        gt_normals = gt_data.get("normals", None)

    result = compute_normal_consistency(pred, gt_points, gt_normals, args.k)
    print(f"Normal consistency: {result['normal_consistency']:.4f}")
    print(f"Matched points: {result['matched_points']}")


if __name__ == "__main__":
    main()
