"""Compute F-score between predicted and ground-truth meshes.

F-score is the harmonic mean of precision and recall at a distance threshold.
Precision: fraction of predicted points within threshold of GT.
Recall: fraction of GT points within threshold of predicted.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None


def sample_mesh_points(vertices: np.ndarray, faces: np.ndarray, n_samples: int = 100_000) -> np.ndarray:
    """Uniformly sample points on a triangle mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / areas.sum()
    chosen = np.random.choice(len(faces), size=n_samples, p=probs)
    r1 = np.sqrt(np.random.rand(n_samples, 1))
    r2 = np.random.rand(n_samples, 1)
    pts = (1 - r1) * v0[chosen] + r1 * (1 - r2) * v1[chosen] + r1 * r2 * v2[chosen]
    return pts


def compute_fscore(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.05,
) -> dict[str, float]:
    """Compute precision, recall, and F-score at given threshold.

    Parameters
    ----------
    pred_points : (N, 3) predicted surface points
    gt_points : (M, 3) ground-truth surface points
    threshold : distance threshold in metres

    Returns
    -------
    dict with keys: precision, recall, fscore
    """
    if KDTree is None:
        raise ImportError("scipy is required: pip install scipy")

    gt_tree = KDTree(gt_points)
    pred_tree = KDTree(pred_points)

    d_pred_to_gt, _ = gt_tree.query(pred_points)
    d_gt_to_pred, _ = pred_tree.query(gt_points)

    precision = float(np.mean(d_pred_to_gt < threshold))
    recall = float(np.mean(d_gt_to_pred < threshold))

    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return {"precision": precision, "recall": recall, "fscore": fscore}


def main():
    parser = argparse.ArgumentParser(description="Compute F-score between predicted and GT meshes")
    parser.add_argument("predicted", help="Predicted mesh (.ply) or point cloud (.npz)")
    parser.add_argument("ground_truth", help="Ground-truth mesh (.ply) or point cloud (.npz)")
    parser.add_argument("-t", "--threshold", type=float, default=0.05, help="Distance threshold (m)")
    parser.add_argument("-n", "--n-samples", type=int, default=100_000, help="Points to sample from meshes")
    args = parser.parse_args()

    pred = _load_points(args.predicted, args.n_samples)
    gt = _load_points(args.ground_truth, args.n_samples)

    result = compute_fscore(pred, gt, args.threshold)
    print(f"Threshold: {args.threshold:.3f}m")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F-score:   {result['fscore']:.4f}")


def _load_points(path: str, n_samples: int) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        return data["points"]
    elif path.endswith(".ply"):
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d required to load PLY files: pip install open3d")
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(n_samples)
            return np.asarray(pcd.points)
        else:
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported format: {path}")


if __name__ == "__main__":
    main()
