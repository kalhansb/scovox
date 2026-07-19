"""Compute expected information gain metrics for exploration evaluation.

Given a partially-mapped environment and candidate viewpoints, compute
total EIG for each viewpoint from the map representation. Used to evaluate
whether EIG-based exploration outperforms entropy/frontier baselines.
"""

from __future__ import annotations

import argparse

import numpy as np


def beta_eig(a_occ: np.ndarray, a_free: np.ndarray) -> np.ndarray:
    """Vectorised Beta EIG using digamma.

    EIG = psi(a+b+1) - p*psi(a+1) - (1-p)*psi(b+1)
          - psi(a+b) + p*psi(a) + (1-p)*psi(b)
    where p = a/(a+b).
    """
    from scipy.special import digamma

    s = a_occ + a_free
    p = a_occ / np.maximum(s, 1e-12)

    eig = (
        digamma(s + 1)
        - p * digamma(a_occ + 1)
        - (1 - p) * digamma(a_free + 1)
        - digamma(s)
        + p * digamma(a_occ)
        + (1 - p) * digamma(a_free)
    )
    return np.maximum(eig, 0.0)


def beta_variance(a_occ: np.ndarray, a_free: np.ndarray) -> np.ndarray:
    """Vectorised Beta posterior variance."""
    s = a_occ + a_free
    return (a_occ * a_free) / (s * s * (s + 1))


def logodds_entropy(prob: np.ndarray) -> np.ndarray:
    """Shannon entropy of Bernoulli — the best log-odds can offer."""
    p = np.clip(prob, 1e-12, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def main():
    parser = argparse.ArgumentParser(description="Compute EIG statistics for a map snapshot")
    parser.add_argument("input", help=".npz with 'a_occ' and 'a_free' arrays (SCovox map)")
    args = parser.parse_args()

    data = np.load(args.input)
    a_occ = data["a_occ"]
    a_free = data["a_free"]
    prob = a_occ / np.maximum(a_occ + a_free, 1e-12)

    eig = beta_eig(a_occ, a_free)
    var = beta_variance(a_occ, a_free)
    ent = logodds_entropy(prob)

    print(f"Voxels: {len(a_occ)}")
    print(f"EIG     — mean: {eig.mean():.6f}  max: {eig.max():.6f}  >0.1: {(eig > 0.1).sum()}")
    print(f"Variance — mean: {var.mean():.6f}  max: {var.max():.6f}")
    print(f"Entropy  — mean: {ent.mean():.6f}  max: {ent.max():.6f}")
    print(f"Correlation(EIG, entropy): {np.corrcoef(eig, ent)[0, 1]:.4f}")
    print(f"Correlation(EIG, variance): {np.corrcoef(eig, var)[0, 1]:.4f}")


if __name__ == "__main__":
    main()
