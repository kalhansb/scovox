#!/usr/bin/env python3
"""Offline, numpy port of scovox's per-voxel uncertainty closed forms.

Every function here mirrors, line for line, the C++ in
  scovox/src/scovox_core/src/uncertainty.cpp
  scovox/src/scovox_core/include/scovox/uncertainty.hpp
so the calibration / decomposition experiments (E1-E4) can be computed OFFLINE
from a captured `scovox_bin` map snapshot — no C++ instrumentation required.
The signals the plan (UNCERTAINTY_PLAN.md §0) asks to "expose"/"add"
(betaExpectedEntropy, betaPredictiveEntropy, semanticExpectedEntropy,
semanticMI) are implemented here rather than in the node.

Inputs are numpy arrays over N voxels. Occupancy uses a_occ,a_free. Semantics
uses cnt (N,K_TOP), a_unk (N,), and header C=num_classes, alpha0. K_TOP=2 in
the shipped build (voxel.hpp:11); the code is written for general K_TOP.

Fidelity contract: these must equal what scovox computes at query time, so the
Hutter query-time floor (effectiveResidual) is applied for the semantic
categorical exactly as semanticEntropy()/argmaxClassConfidence do. Occupancy
E_H / H_y / EIG are the terms inside expectedInformationGain().
"""
from __future__ import annotations
import numpy as np
from scipy.special import digamma

K_TOP = 2

# --------------------------------------------------------------------------
# Occupancy (Beta) — mirrors uncertainty.cpp:63-85 (expectedInformationGain)
# --------------------------------------------------------------------------

def beta_predictive_entropy(a_occ, a_free):
    """H_y = Bernoulli entropy of the mean p = a/(a+b). The 'total' occupancy
    uncertainty. (The H_y term in expectedInformationGain.)"""
    a = np.asarray(a_occ, float); b = np.asarray(a_free, float)
    s = a + b
    p = np.divide(a, s, out=np.full_like(a, 0.5), where=s > 0)
    H = np.zeros_like(p)
    m = (p > 1e-7) & (p < 1 - 1e-7)
    H[m] = -p[m]*np.log(p[m]) - (1-p[m])*np.log(1-p[m])
    return H


def beta_expected_entropy(a_occ, a_free):
    """E_H = E_{θ~Beta(a,b)}[ H_Bernoulli(θ) ]
          = ψ(s+1) − p·ψ(a+1) − (1−p)·ψ(b+1).
    The 'aleatoric' occupancy term (§0 item 1). Verbatim uncertainty.cpp:74-76."""
    a = np.asarray(a_occ, float); b = np.asarray(a_free, float)
    s = a + b
    p = np.divide(a, s, out=np.full_like(a, 0.5), where=s > 0)
    E_H = digamma(s + 1.0) - p*digamma(a + 1.0) - (1-p)*digamma(b + 1.0)
    return np.where(s > 0, E_H, 0.0)


def expected_information_gain(a_occ, a_free):
    """EIG = max(0, H_y − E_H). Exact Beta-Bernoulli BALD mutual information —
    the 'epistemic' occupancy term. Verbatim uncertainty.cpp:63-85."""
    return np.maximum(0.0, beta_predictive_entropy(a_occ, a_free)
                           - beta_expected_entropy(a_occ, a_free))


def beta_variance(a_occ, a_free):
    a = np.asarray(a_occ, float); b = np.asarray(a_free, float)
    s = a + b
    return np.divide(a*b, s*s*(s+1.0), out=np.zeros_like(a), where=s > 0)


def _bernoulli_kl(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    kl = np.zeros_like(p)
    m1 = p > 1e-7;       kl = np.where(m1, kl + p*np.log(np.divide(p, q, out=np.ones_like(p), where=q>0)), kl)
    m2 = p < 1 - 1e-7;   kl = np.where(m2, kl + (1-p)*np.log(np.divide(1-p, 1-q, out=np.ones_like(p), where=(1-q)>0)), kl)
    return np.maximum(kl, 0.0)


def ssmi_occ_kl(a_occ, a_free):
    a = np.asarray(a_occ, float); b = np.asarray(a_free, float); s = a+b
    p = np.divide(a, s, out=np.zeros_like(a), where=s>0)
    p_post = np.divide(a+1.0, s+1.0, out=np.zeros_like(a), where=s>0)
    return np.where(s > 0, _bernoulli_kl(p, p_post), 0.0)


def ssmi_free_kl(a_occ, a_free):
    a = np.asarray(a_occ, float); b = np.asarray(a_free, float); s = a+b
    p = np.divide(a, s, out=np.zeros_like(a), where=s>0)
    p_post = np.divide(a, s+1.0, out=np.zeros_like(a), where=s>0)
    return np.where(s > 0, _bernoulli_kl(p, p_post), 0.0)


# --------------------------------------------------------------------------
# Semantics (Dirichlet) — mirrors uncertainty.hpp:46-96 + uncertainty.cpp:87-124
# --------------------------------------------------------------------------

def _hutter_escape_mass(m, N):
    """β* = m / (2 ln((N+1)/m)), clamped to [0, N]. uncertainty.hpp:67-79."""
    m = np.asarray(m, float); N = np.asarray(N, float)
    ratio = np.divide(N + 1.0, m, out=np.full_like(N, np.inf), where=m > 0)
    raw = np.divide(m, 2.0*np.log(ratio), out=np.zeros_like(N),
                    where=ratio > 1.0)
    esc = np.minimum(raw, N)
    esc = np.where(ratio <= 1.0, N, esc)          # degenerate m>=N+1 -> N
    esc = np.where((m <= 0) | (N <= 0), 0.0, esc)
    return esc


def effective_residual(cnt, a_unk):
    """max(a_unk, hutterEscapeMass(m, N)). uncertainty.hpp:91-96.
    cnt: (N,K_TOP) raw sem counts; a_unk: (N,)."""
    cnt = np.asarray(cnt, float); a_unk = np.asarray(a_unk, float)
    m = (cnt > 0).sum(axis=1) + (a_unk > 0)
    m = np.maximum(m, 1)                          # floor at 1 (log(0) guard)
    N = a_unk + cnt.sum(axis=1)
    return np.maximum(a_unk, _hutter_escape_mass(m, N))


def _sem_alpha_matrix(cnt, a_unk):
    """Build the query-time categorical alphas used by semanticEntropy:
    one slot per resident class with cnt>0 (alpha = cnt+1), plus a single
    OTHER slot (alpha = effectiveResidual+1). Returns (A, valid) of shape
    (N, K_TOP+1); the last column is always the OTHER/residual slot."""
    cnt = np.asarray(cnt, float)
    Nn, K = cnt.shape
    A = np.zeros((Nn, K + 1)); valid = np.zeros((Nn, K + 1), bool)
    for i in range(K):
        vi = cnt[:, i] > 0
        A[vi, i] = cnt[vi, i] + 1.0; valid[vi, i] = True
    A[:, K] = effective_residual(cnt, a_unk) + 1.0
    valid[:, K] = True
    return A, valid


def semantic_entropy(cnt, a_unk):
    """H(E[p]) — Shannon entropy of the MEAN categorical (the predictive /
    'total' semantic uncertainty). Verbatim semanticEntropy, uncertainty.cpp.
    Returns 0 where fewer than 2 categorical slots are active (K<2)."""
    A, V = _sem_alpha_matrix(cnt, a_unk)
    K = V.sum(1)
    S = np.where(V, A, 0.0).sum(1)
    P = np.divide(A, S[:, None], out=np.zeros_like(A), where=(S[:, None] > 0) & V)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where((P > 1e-7) & V, np.log(P), 0.0)
    H = -(np.where(V, P, 0.0) * logP).sum(1)
    return np.where(K >= 2, H, 0.0)


def semantic_expected_entropy(cnt, a_unk):
    """E[H(p)] = ψ(S+1) − Σ_i (α_i/S)·ψ(α_i+1) over the SAME categorical as
    semantic_entropy (§0 item 2). The 'aleatoric' semantic term."""
    A, V = _sem_alpha_matrix(cnt, a_unk)
    K = V.sum(1)
    S = np.where(V, A, 0.0).sum(1)
    term = np.where(V, (A / S[:, None]) * digamma(A + 1.0), 0.0).sum(1)
    E_H = digamma(S + 1.0) - term
    return np.where(K >= 2, np.maximum(E_H, 0.0), 0.0)


def semantic_mi(cnt, a_unk):
    """Epistemic semantic MI = max(0, H(E[p]) − E[H(p)]) — exact Dirichlet
    mutual information over the {resident top-K, OTHER} partition (§0 item 2)."""
    return np.maximum(0.0, semantic_entropy(cnt, a_unk)
                           - semantic_expected_entropy(cnt, a_unk))


def p_other(cnt, a_unk):
    """Mean probability mass on the OTHER (residual) bucket of the query-time
    categorical. Used for the C3 coarsening-bound scatter (E4b)."""
    A, V = _sem_alpha_matrix(cnt, a_unk)
    S = np.where(V, A, 0.0).sum(1)
    return np.divide(A[:, -1], S, out=np.zeros_like(S), where=S > 0)


def vacuity(cnt, a_unk, num_classes, alpha0):
    """Subjective-logic uncertainty mass u = W / (S_raw + W), W = C·α0 the
    total prior pseudo-mass. NOTE: UNCERTAINTY_PLAN.md writes vacuity as
    '(K+1)/(S+K+1)', which is ambiguous about K; this is the principled SL
    form and should be reconciled with the author before it goes in a table."""
    cnt = np.asarray(cnt, float); a_unk = np.asarray(a_unk, float)
    W = float(num_classes) * float(alpha0)
    S_raw = cnt.sum(axis=1) + a_unk
    return W / (S_raw + W)
