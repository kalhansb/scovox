"""Compute Expected Calibration Error (ECE) for occupancy predictions.

Compares predicted occupancy probabilities against ground-truth binary labels.
Produces reliability diagrams and stratified ECE by evidence level.
"""

from __future__ import annotations

import argparse

import numpy as np


def compute_ece(
    pred_prob: np.ndarray,
    gt_binary: np.ndarray,
    n_bins: int = 20,
) -> dict[str, float | np.ndarray]:
    """Compute ECE, MCE, and Brier score.

    Parameters
    ----------
    pred_prob : (N,) predicted occupancy probabilities in [0, 1]
    gt_binary : (N,) ground-truth binary labels (1 = occupied, 0 = free)
    n_bins : number of bins for reliability diagram

    Returns
    -------
    dict with ece, mce, brier, bin_edges, bin_accuracy, bin_confidence, bin_count
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (pred_prob >= lo) & (pred_prob <= hi)
        else:
            mask = (pred_prob >= lo) & (pred_prob < hi)

        cnt = mask.sum()
        bin_count[i] = cnt
        if cnt > 0:
            bin_accuracy[i] = gt_binary[mask].mean()
            bin_confidence[i] = pred_prob[mask].mean()

    # Weighted ECE
    weights = bin_count / bin_count.sum() if bin_count.sum() > 0 else np.zeros(n_bins)
    gaps = np.abs(bin_accuracy - bin_confidence)
    ece = float(np.sum(weights * gaps))
    mce = float(gaps[bin_count > 0].max()) if (bin_count > 0).any() else 0.0

    # Brier score
    brier = float(np.mean((pred_prob - gt_binary) ** 2))

    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "bin_edges": bin_edges,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_count": bin_count,
    }


def compute_stratified_ece(
    pred_prob: np.ndarray,
    gt_binary: np.ndarray,
    evidence: np.ndarray,
    n_bins: int = 20,
) -> dict[str, dict]:
    """ECE stratified by evidence level: <5, 5-20, >20 observations."""
    strata = {
        "low (<5)": evidence < 5,
        "medium (5-20)": (evidence >= 5) & (evidence <= 20),
        "high (>20)": evidence > 20,
    }
    results = {}
    for name, mask in strata.items():
        if mask.sum() > 0:
            results[name] = compute_ece(pred_prob[mask], gt_binary[mask], n_bins)
        else:
            results[name] = {"ece": float("nan"), "mce": float("nan"), "brier": float("nan")}
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute ECE for occupancy predictions")
    parser.add_argument("predicted", help=".npz with 'occupancy_prob' array")
    parser.add_argument("ground_truth", help=".npz with 'gt_binary' (1=occupied, 0=free)")
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    pred = np.load(args.predicted)["occupancy_prob"]
    gt = np.load(args.ground_truth)["gt_binary"].astype(float)

    result = compute_ece(pred, gt, args.bins)
    print(f"ECE:   {result['ece']:.4f}")
    print(f"MCE:   {result['mce']:.4f}")
    print(f"Brier: {result['brier']:.4f}")

    # Stratified if evidence available
    try:
        evidence = np.load(args.predicted)["evidence"]
        strat = compute_stratified_ece(pred, gt, evidence, args.bins)
        for name, res in strat.items():
            print(f"  {name}: ECE={res['ece']:.4f}")
    except KeyError:
        pass


if __name__ == "__main__":
    main()
