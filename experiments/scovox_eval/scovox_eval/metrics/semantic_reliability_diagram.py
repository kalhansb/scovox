"""Reliability diagrams for semantic predictions, stratified by alpha_0.

Produces binned calibration data suitable for plotting, plus optional
matplotlib figure generation.
"""

from __future__ import annotations

import argparse

import numpy as np

from ._stratify import compute_alpha0, make_strata_masks
from .semantic_ece import compute_semantic_ece


def compute_reliability_data(
    semantic_probs: np.ndarray,
    gt_labels: np.ndarray,
    alpha0: np.ndarray,
    n_bins: int = 20,
) -> dict[str, dict[str, float | np.ndarray]]:
    """Reliability diagram data for all voxels and per alpha_0 stratum.

    Parameters
    ----------
    semantic_probs : (N, K) predicted class probabilities
    gt_labels : (N,) ground-truth class IDs
    alpha0 : (N,) Dirichlet concentration per voxel
    n_bins : number of bins

    Returns
    -------
    dict keyed by "all", "low (<5)", "medium (5-20)", "high (>20)",
    each containing semantic_ece, semantic_mce, bin_edges, bin_accuracy,
    bin_confidence, bin_count
    """
    results = {"all": compute_semantic_ece(semantic_probs, gt_labels, n_bins)}

    strata = make_strata_masks(alpha0)
    for name, mask in strata.items():
        if mask.sum() > 0:
            results[name] = compute_semantic_ece(
                semantic_probs[mask], gt_labels[mask], n_bins
            )
        else:
            results[name] = {
                "semantic_ece": float("nan"),
                "semantic_mce": float("nan"),
                "bin_edges": np.linspace(0, 1, n_bins + 1),
                "bin_accuracy": np.zeros(n_bins),
                "bin_confidence": np.zeros(n_bins),
                "bin_count": np.zeros(n_bins, dtype=int),
            }
    return results


def plot_reliability_diagram(
    data: dict[str, dict[str, float | np.ndarray]],
    output_path: str | None = None,
) -> None:
    """Plot reliability curves in a 2x2 grid. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install scovox_eval[plot]"
        )

    panel_order = ["all", "low (<5)", "medium (5-20)", "high (>20)"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, panel_name in zip(axes.flat, panel_order):
        if panel_name not in data:
            ax.set_visible(False)
            continue

        d = data[panel_name]
        bin_edges = d["bin_edges"]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_acc = d["bin_accuracy"]
        bin_count = d["bin_count"]
        ece = d["semantic_ece"]

        # Bar chart of accuracy per bin
        width = bin_edges[1] - bin_edges[0]
        ax.bar(
            bin_centers, bin_acc, width=width * 0.9,
            alpha=0.6, color="steelblue", edgecolor="black", linewidth=0.5,
        )
        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        n_total = int(bin_count.sum())
        ax.set_title(f"{panel_name}  (ECE={ece:.3f}, n={n_total})")
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic reliability diagrams stratified by alpha_0"
    )
    parser.add_argument("predicted", help=".npz with 'semantic_probs' and 'semantic_evidence'")
    parser.add_argument("ground_truth", help=".npz with 'semantic_class'")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--output-plot", default=None, help="Save figure to path (e.g. fig.png)")
    args = parser.parse_args()

    pred = np.load(args.predicted)
    gt_data = np.load(args.ground_truth)

    semantic_probs = pred["semantic_probs"]
    gt_labels = gt_data["semantic_class"].astype(int)
    evidence = pred["semantic_evidence"]
    a_unk = pred["a_unk"] if "a_unk" in pred else None
    alpha0 = compute_alpha0(evidence, a_unk)

    data = compute_reliability_data(semantic_probs, gt_labels, alpha0, args.bins)

    for name, d in data.items():
        ece = d["semantic_ece"]
        n = int(d["bin_count"].sum())
        print(f"{name:>15}: ECE={ece:.4f}  (n={n})")

    if args.output_plot:
        plot_reliability_diagram(data, args.output_plot)


if __name__ == "__main__":
    main()
