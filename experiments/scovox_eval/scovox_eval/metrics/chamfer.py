"""Compute Chamfer-L1 distance between predicted and GT point clouds."""

from __future__ import annotations

import argparse

import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None


def compute_chamfer_l1(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
) -> dict[str, float]:
    """Bidirectional mean nearest-point distance (Chamfer-L1).

    Returns
    -------
    dict with chamfer_l1 (mean of both directions), pred_to_gt, gt_to_pred in metres.
    """
    if KDTree is None:
        raise ImportError("scipy is required: pip install scipy")

    gt_tree = KDTree(gt_points)
    pred_tree = KDTree(pred_points)

    d_pred_to_gt, _ = gt_tree.query(pred_points)
    d_gt_to_pred, _ = pred_tree.query(gt_points)

    pred_to_gt = float(np.mean(d_pred_to_gt))
    gt_to_pred = float(np.mean(d_gt_to_pred))

    return {
        "chamfer_l1": (pred_to_gt + gt_to_pred) / 2.0,
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": gt_to_pred,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Chamfer-L1 distance")
    parser.add_argument("predicted", help="Predicted points (.npz or .ply)")
    parser.add_argument("ground_truth", help="Ground-truth points (.npz or .ply)")
    args = parser.parse_args()

    from .fscore import _load_points

    pred = _load_points(args.predicted, 100_000)
    gt = _load_points(args.ground_truth, 100_000)

    result = compute_chamfer_l1(pred, gt)
    print(f"Chamfer-L1: {result['chamfer_l1'] * 100:.2f} cm")
    print(f"  pred->GT: {result['pred_to_gt'] * 100:.2f} cm")
    print(f"  GT->pred: {result['gt_to_pred'] * 100:.2f} cm")


if __name__ == "__main__":
    main()
