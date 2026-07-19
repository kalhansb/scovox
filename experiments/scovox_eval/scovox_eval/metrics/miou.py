"""Compute mean Intersection-over-Union for semantic voxel maps."""

from __future__ import annotations

import argparse

import numpy as np


def compute_miou(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int | None = None,
    ignore_label: int = 0,
) -> dict[str, float | np.ndarray]:
    """Per-class IoU and mean IoU.

    Parameters
    ----------
    pred_labels : (N,) predicted class IDs
    gt_labels : (N,) ground-truth class IDs
    num_classes : total number of classes (auto-detected if None)
    ignore_label : label to exclude from evaluation

    Returns
    -------
    dict with miou, per_class_iou (array), per_class_count (array)
    """
    mask = gt_labels != ignore_label
    pred_labels = pred_labels[mask]
    gt_labels = gt_labels[mask]

    if num_classes is None:
        num_classes = int(max(pred_labels.max(), gt_labels.max())) + 1

    per_class_iou = np.full(num_classes, np.nan)
    per_class_count = np.zeros(num_classes, dtype=int)

    for c in range(num_classes):
        pred_c = pred_labels == c
        gt_c = gt_labels == c
        intersection = np.sum(pred_c & gt_c)
        union = np.sum(pred_c | gt_c)
        per_class_count[c] = int(np.sum(gt_c))

        if union > 0:
            per_class_iou[c] = intersection / union

    valid = ~np.isnan(per_class_iou)
    miou = float(np.mean(per_class_iou[valid])) if valid.any() else 0.0

    return {
        "miou": miou,
        "per_class_iou": per_class_iou,
        "per_class_count": per_class_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute mIoU for semantic maps")
    parser.add_argument("predicted", help=".npz with 'points' and 'semantic_class'")
    parser.add_argument("ground_truth", help=".npz with 'points' and 'semantic_class'")
    parser.add_argument("-n", "--num-classes", type=int, default=20)
    parser.add_argument("-d", "--max-dist", type=float, default=0.05,
                        help="Max distance (m) for nearest-neighbor matching")
    args = parser.parse_args()

    pred_data = np.load(args.predicted)
    gt_data = np.load(args.ground_truth)

    pred_labels = pred_data["semantic_class"].astype(int)
    gt_labels = gt_data["semantic_class"].astype(int)

    # If point counts differ, use nearest-neighbor matching
    if len(pred_labels) != len(gt_labels):
        from scipy.spatial import KDTree
        pred_pts = pred_data["points"]
        gt_pts = gt_data["points"]
        gt_tree = KDTree(gt_pts)
        dists, indices = gt_tree.query(pred_pts)
        # Only keep matches within max_dist
        mask = dists < args.max_dist
        pred_labels = pred_labels[mask]
        gt_labels = gt_labels[indices[mask]]
        print(f"Matched {mask.sum()}/{len(mask)} predicted voxels "
              f"(within {args.max_dist}m of GT)")

    result = compute_miou(pred_labels, gt_labels, args.num_classes)
    print(f"mIoU: {result['miou']:.4f}")
    for c, (iou, cnt) in enumerate(zip(result["per_class_iou"], result["per_class_count"])):
        if cnt > 0:
            print(f"  class {c:3d}: IoU={iou:.4f}  ({cnt} voxels)")


if __name__ == "__main__":
    main()
