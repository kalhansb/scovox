"""Compute multiclass Brier score for semantic voxel predictions.

Brier score = mean over voxels of sum_k (p_k - y_k)^2, where y is
one-hot ground truth. Supports stratification by Dirichlet alpha_0.
"""

from __future__ import annotations

import argparse

import numpy as np

from ._stratify import compute_alpha0, make_strata_masks


def compute_semantic_brier(
    semantic_probs: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Multiclass Brier score.

    Parameters
    ----------
    semantic_probs : (N, K) predicted class probabilities
    gt_labels : (N,) ground-truth class IDs
    num_classes : total classes (inferred from semantic_probs if None)

    Returns
    -------
    dict with brier_score
    """
    n, k = semantic_probs.shape
    if num_classes is None:
        num_classes = k

    if gt_labels.max() >= num_classes:
        raise ValueError(
            f"gt_labels contains class {gt_labels.max()} but num_classes={num_classes}"
        )

    y_onehot = np.zeros((n, num_classes), dtype=np.float32)
    y_onehot[np.arange(n), gt_labels] = 1.0

    # Pad semantic_probs if num_classes > k
    if num_classes > k:
        pad = np.zeros((n, num_classes - k), dtype=semantic_probs.dtype)
        probs = np.concatenate([semantic_probs, pad], axis=1)
    else:
        probs = semantic_probs

    per_voxel = np.sum((probs - y_onehot) ** 2, axis=1)
    brier_score = float(np.mean(per_voxel))

    return {"brier_score": brier_score}


def compute_stratified_semantic_brier(
    semantic_probs: np.ndarray,
    gt_labels: np.ndarray,
    alpha0: np.ndarray,
    num_classes: int | None = None,
) -> dict[str, dict]:
    """Semantic Brier score stratified by alpha_0: <5, 5-20, >20."""
    strata = make_strata_masks(alpha0)
    results = {}
    for name, mask in strata.items():
        if mask.sum() > 0:
            results[name] = compute_semantic_brier(
                semantic_probs[mask], gt_labels[mask], num_classes
            )
        else:
            results[name] = {"brier_score": float("nan")}
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute multiclass Brier score for semantic predictions"
    )
    parser.add_argument("predicted", help=".npz with 'semantic_probs' (N,K) array")
    parser.add_argument("ground_truth", help=".npz with 'semantic_class' (N,) array")
    parser.add_argument("-n", "--num-classes", type=int, default=None)
    args = parser.parse_args()

    pred = np.load(args.predicted)
    gt_data = np.load(args.ground_truth)

    semantic_probs = pred["semantic_probs"]
    gt_labels = gt_data["semantic_class"].astype(int)

    result = compute_semantic_brier(semantic_probs, gt_labels, args.num_classes)
    print(f"Brier: {result['brier_score']:.4f}")

    # Stratified if evidence available
    try:
        evidence = pred["semantic_evidence"]
        a_unk = pred["a_unk"] if "a_unk" in pred else None
        alpha0 = compute_alpha0(evidence, a_unk)
        strat = compute_stratified_semantic_brier(
            semantic_probs, gt_labels, alpha0, args.num_classes
        )
        for name, res in strat.items():
            print(f"  {name}: Brier={res['brier_score']:.4f}")
    except KeyError:
        pass


if __name__ == "__main__":
    main()
