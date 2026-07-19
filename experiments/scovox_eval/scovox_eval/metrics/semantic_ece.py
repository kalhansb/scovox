"""Compute Expected Calibration Error (ECE) for semantic predictions.

Bins voxels by max predicted class probability and compares to empirical
accuracy (was the argmax prediction correct?) per bin. Supports
stratification by Dirichlet concentration alpha_0.

Works with two data formats:
  - Full probability vectors: 'semantic_probs' (N, K)
  - Confidence + class: 'semantic_confidence' (N,) + 'semantic_class' (N,)
"""

from __future__ import annotations

import argparse

import numpy as np

from ._stratify import make_strata_masks


def compute_semantic_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float | np.ndarray]:
    """Semantic ECE, MCE, and per-bin reliability data.

    Parameters
    ----------
    confidences : (N,) predicted confidence for the chosen class
    correct : (N,) binary array — 1 if prediction matches GT, 0 otherwise
    n_bins : number of equal-width bins over [0, 1]

    Returns
    -------
    dict with semantic_ece, semantic_mce, bin_edges, bin_accuracy,
    bin_confidence, bin_count
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        cnt = mask.sum()
        bin_count[i] = cnt
        if cnt > 0:
            bin_accuracy[i] = correct[mask].mean()
            bin_confidence[i] = confidences[mask].mean()

    weights = bin_count / bin_count.sum() if bin_count.sum() > 0 else np.zeros(n_bins)
    gaps = np.abs(bin_accuracy - bin_confidence)
    ece = float(np.sum(weights * gaps))
    mce = float(gaps[bin_count > 0].max()) if (bin_count > 0).any() else 0.0

    return {
        "semantic_ece": ece,
        "semantic_mce": mce,
        "bin_edges": bin_edges,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_count": bin_count,
    }


def compute_stratified_semantic_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    alpha0: np.ndarray,
    n_bins: int = 15,
) -> dict[str, dict]:
    """Semantic ECE stratified by alpha_0: <5, 5-20, >20."""
    strata = make_strata_masks(alpha0)
    results = {}
    for name, mask in strata.items():
        if mask.sum() > 0:
            results[name] = compute_semantic_ece(
                confidences[mask], correct[mask], n_bins
            )
            results[name]["n_voxels"] = int(mask.sum())
        else:
            results[name] = {
                "semantic_ece": float("nan"),
                "semantic_mce": float("nan"),
                "n_voxels": 0,
            }
    return results
