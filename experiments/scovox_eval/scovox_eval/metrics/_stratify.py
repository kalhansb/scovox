"""Shared stratification helpers for semantic calibration metrics.

All semantic calibration metrics stratify by Dirichlet concentration
alpha_0 using the same thresholds: <5 (low), 5-20 (medium), >20 (high).
"""

from __future__ import annotations

import numpy as np


def compute_alpha0(
    semantic_evidence: np.ndarray,
    a_unk: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Dirichlet concentration alpha_0 per voxel.

    Parameters
    ----------
    semantic_evidence : (N, K) per-class evidence counts
    a_unk : (N,) unknown/prior evidence (optional)

    Returns
    -------
    (N,) alpha_0 = sum_k(evidence_k) + a_unk
    """
    alpha0 = semantic_evidence.sum(axis=1)
    if a_unk is not None:
        alpha0 = alpha0 + a_unk
    return alpha0


def make_strata_masks(alpha0: np.ndarray) -> dict[str, np.ndarray]:
    """Return boolean masks for low/medium/high evidence strata."""
    return {
        "low (<5)": alpha0 < 5,
        "medium (5-20)": (alpha0 >= 5) & (alpha0 <= 20),
        "high (>20)": alpha0 > 20,
    }
