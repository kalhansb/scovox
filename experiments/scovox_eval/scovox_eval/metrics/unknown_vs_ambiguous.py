"""Unknown vs ambiguous voxel separation using posterior variance.

This is the figure that sells the paper: SCovox (Beta posteriors) can
distinguish truly unknown voxels (unobserved, low evidence) from genuinely
ambiguous voxels (surface boundaries, conflicting evidence) using posterior
variance. LogOdds cannot — both types have p ≈ 0.5 with no variance signal.

Pipeline:
    1. Filter to voxels with occupancy_prob in [0.4, 0.6] (uncertain band)
    2. Compute Beta variance: a_occ * a_free / ((a_occ+a_free)^2 * (a_occ+a_free+1))
    3. Label each voxel:
       - "unknown"  = far from any GT surface (dist > resolution)
       - "boundary" = near a GT surface edge (dist < resolution/2)
    4. ROC-AUC of variance for separating unknown vs boundary
       SCovox: unknown → high variance (low evidence), boundary → lower variance → AUC > 0.90
       LogOdds: no variance signal → AUC ≈ 0.50
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial import KDTree


def beta_variance(a_occ: np.ndarray, a_free: np.ndarray) -> np.ndarray:
    """Posterior variance of the Beta distribution."""
    s = a_occ + a_free
    return (a_occ * a_free) / (s * s * (s + 1))


def compute_unknown_vs_ambiguous(
    points: np.ndarray,
    occupancy_prob: np.ndarray,
    a_occ: np.ndarray,
    a_free: np.ndarray,
    gt_surface_points: np.ndarray,
    resolution: float,
    prob_lo: float = 0.4,
    prob_hi: float = 0.6,
) -> dict[str, float | np.ndarray]:
    """Compute ROC-AUC for separating unknown from boundary voxels.

    Parameters
    ----------
    points : (N, 3) voxel positions
    occupancy_prob : (N,) predicted occupancy probability
    a_occ, a_free : (N,) Beta distribution parameters
    gt_surface_points : (M, 3) dense GT mesh surface samples
    resolution : voxel resolution in metres
    prob_lo, prob_hi : occupancy probability band for uncertain voxels

    Returns
    -------
    dict with auc, n_unknown, n_boundary, n_total_uncertain,
    variance_unknown (array), variance_boundary (array),
    labels (array), scores (array)
    """
    # Step 1: filter to uncertain band
    uncertain_mask = (occupancy_prob >= prob_lo) & (occupancy_prob <= prob_hi)
    n_uncertain = uncertain_mask.sum()

    if n_uncertain == 0:
        return {
            "auc": float("nan"),
            "n_unknown": 0,
            "n_boundary": 0,
            "n_total_uncertain": 0,
        }

    unc_points = points[uncertain_mask]
    unc_a_occ = a_occ[uncertain_mask]
    unc_a_free = a_free[uncertain_mask]

    # Step 2: compute Beta variance
    var = beta_variance(unc_a_occ, unc_a_free)

    # Step 3: label unknown vs boundary
    tree = KDTree(gt_surface_points)
    dists, _ = tree.query(unc_points)

    # Unknown: far from surface (not observed, just prior)
    unknown_mask = dists > resolution
    # Boundary: near surface (conflicting occupied/free observations)
    boundary_mask = dists < resolution / 2.0

    n_unknown = unknown_mask.sum()
    n_boundary = boundary_mask.sum()

    if n_unknown == 0 or n_boundary == 0:
        return {
            "auc": float("nan"),
            "n_unknown": int(n_unknown),
            "n_boundary": int(n_boundary),
            "n_total_uncertain": int(n_uncertain),
        }

    # Step 4: ROC-AUC
    # Label: 1 = unknown (high variance expected), 0 = boundary (lower variance)
    # Score: variance (higher → more likely unknown)
    combined_mask = unknown_mask | boundary_mask
    labels = unknown_mask[combined_mask].astype(int)
    scores = var[combined_mask]

    # Rank-based AUC (Mann–Whitney U), no sklearn dependency.
    pos = scores[labels == 1]; neg = scores[labels == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    n_p, n_n = len(pos), len(neg)
    auc = float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))

    return {
        "auc": auc,
        "n_unknown": int(n_unknown),
        "n_boundary": int(n_boundary),
        "n_total_uncertain": int(n_uncertain),
        "variance_unknown_mean": float(var[unknown_mask].mean()),
        "variance_boundary_mean": float(var[boundary_mask].mean()),
        "evidence_unknown_mean": float((unc_a_occ[unknown_mask] + unc_a_free[unknown_mask]).mean()),
        "evidence_boundary_mean": float((unc_a_occ[boundary_mask] + unc_a_free[boundary_mask]).mean()),
        "labels": labels,
        "scores": scores,
    }


def compute_logodds_baseline(
    points: np.ndarray,
    occupancy_prob: np.ndarray,
    gt_surface_points: np.ndarray,
    resolution: float,
    prob_lo: float = 0.4,
    prob_hi: float = 0.6,
) -> dict[str, float]:
    """Compute AUC for LogOdds using occupancy_prob as the only signal.

    LogOdds has no variance — all voxels at p ≈ 0.5 are indistinguishable.
    The AUC should be ~0.50 (random).
    """
    uncertain_mask = (occupancy_prob >= prob_lo) & (occupancy_prob <= prob_hi)
    n_uncertain = uncertain_mask.sum()

    if n_uncertain == 0:
        return {"auc": float("nan"), "n_unknown": 0, "n_boundary": 0,
                "n_total_uncertain": 0}

    unc_points = points[uncertain_mask]
    unc_prob = occupancy_prob[uncertain_mask]

    tree = KDTree(gt_surface_points)
    dists, _ = tree.query(unc_points)

    unknown_mask = dists > resolution
    boundary_mask = dists < resolution / 2.0

    n_unknown = unknown_mask.sum()
    n_boundary = boundary_mask.sum()

    if n_unknown == 0 or n_boundary == 0:
        return {"auc": float("nan"), "n_unknown": int(n_unknown),
                "n_boundary": int(n_boundary), "n_total_uncertain": int(n_uncertain)}

    combined_mask = unknown_mask | boundary_mask
    labels = unknown_mask[combined_mask].astype(int)
    # LogOdds only has probability — use it as the score
    scores = unc_prob[combined_mask]

    # Rank-based AUC (Mann–Whitney U), no sklearn dependency.
    pos = scores[labels == 1]; neg = scores[labels == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    n_p, n_n = len(pos), len(neg)
    auc = float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))

    return {
        "auc": auc,
        "n_unknown": int(n_unknown),
        "n_boundary": int(n_boundary),
        "n_total_uncertain": int(n_uncertain),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unknown vs ambiguous voxel separation (ROC-AUC)"
    )
    parser.add_argument("predicted", help=".npz with points, occupancy_prob, a_occ, a_free")
    parser.add_argument("gt_mesh", help="GT mesh (.ply) for surface distance")
    parser.add_argument("--resolution", type=float, required=True,
                        help="Voxel resolution in metres")
    parser.add_argument("--prob-lo", type=float, default=0.4)
    parser.add_argument("--prob-hi", type=float, default=0.6)
    parser.add_argument("--gt-samples", type=int, default=500_000,
                        help="Number of surface samples from GT mesh")
    args = parser.parse_args()

    import trimesh
    pred = np.load(args.predicted)
    mesh = trimesh.load(args.gt_mesh)
    gt_surface = mesh.sample(args.gt_samples).astype(np.float32)

    if "a_occ" in pred and "a_free" in pred:
        result = compute_unknown_vs_ambiguous(
            pred["points"], pred["occupancy_prob"],
            pred["a_occ"], pred["a_free"],
            gt_surface, args.resolution,
            args.prob_lo, args.prob_hi,
        )
        print(f"Beta variance AUC: {result['auc']:.4f}")
        print(f"  Unknown:  n={result['n_unknown']}, "
              f"var_mean={result.get('variance_unknown_mean', 0):.6f}, "
              f"evidence_mean={result.get('evidence_unknown_mean', 0):.1f}")
        print(f"  Boundary: n={result['n_boundary']}, "
              f"var_mean={result.get('variance_boundary_mean', 0):.6f}, "
              f"evidence_mean={result.get('evidence_boundary_mean', 0):.1f}")
        print(f"  Total uncertain (p∈[{args.prob_lo},{args.prob_hi}]): "
              f"{result['n_total_uncertain']}")
    else:
        # LogOdds mode — no a_occ/a_free
        result = compute_logodds_baseline(
            pred["points"], pred["occupancy_prob"],
            gt_surface, args.resolution,
            args.prob_lo, args.prob_hi,
        )
        print(f"LogOdds baseline AUC: {result['auc']:.4f}")
        print(f"  Unknown: n={result['n_unknown']}")
        print(f"  Boundary: n={result['n_boundary']}")
        print(f"  Total uncertain: {result['n_total_uncertain']}")


if __name__ == "__main__":
    main()
