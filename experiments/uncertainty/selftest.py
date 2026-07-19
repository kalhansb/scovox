#!/usr/bin/env python3
"""Validate the offline functionals AND the plan's core math claims (C2, C3,
§0 unit tests) in pure Python — no ROS, no C++. This is the 'is the math
valid' half of the assessment, made executable.

Run: python3 experiments/uncertainty/selftest.py
"""
import numpy as np
from scipy.special import digamma
import functionals as F

rng = np.random.default_rng(0)
ok = True
def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("== §0 unit tests / C2 occupancy decomposition ==")
# random Beta voxels
a = rng.uniform(1, 200, 5000); b = rng.uniform(1, 200, 5000)
EIG = F.expected_information_gain(a, b)
Hy  = F.beta_predictive_entropy(a, b)
EH  = F.beta_expected_entropy(a, b)
check("occ EIG >= 0 (BALD non-negativity)", np.all(EIG >= -1e-9))
check("occ E_H <= H_y (expected <= total, Jensen)", np.all(EH <= Hy + 1e-9))
# MI -> 0 as evidence -> inf at fixed mean p=0.5
s_grid = np.array([2., 20., 200., 2000., 20000.])
eig_s = F.expected_information_gain(s_grid/2, s_grid/2)
check("occ EIG decreases monotonically with evidence (fixed mean)",
      np.all(np.diff(eig_s) < 0))
check("occ EIG -> 0 as s -> inf", eig_s[-1] < 1e-3)

# --- The C2 caveat, made explicit: E_H is NOT evidence-invariant ---
eh_s = F.beta_expected_entropy(s_grid/2, s_grid/2)
print(f"     E_H at fixed mean p=0.5 vs s={s_grid.tolist()}:")
print(f"       {np.round(eh_s,4).tolist()}  (H_y=ln2={np.log(2):.4f})")
check("occ E_H RISES toward H_y as evidence grows (aleatoric not invariant)",
      np.all(np.diff(eh_s) > 0) and abs(eh_s[-1]-np.log(2)) < 1e-2)

print("== §0 unit tests / semantic Dirichlet MI decomposition ==")
# random Dirichlet voxels: 2 resident slots + a_unk, C classes
Nn = 5000
cnt = rng.uniform(0, 50, (Nn, F.K_TOP)) * (rng.random((Nn, F.K_TOP)) > 0.3)
a_unk = rng.uniform(0, 10, Nn)
Htot = F.semantic_entropy(cnt, a_unk)
Hexp = F.semantic_expected_entropy(cnt, a_unk)
MI   = F.semantic_mi(cnt, a_unk)
active = (cnt > 0).sum(1) + 1 >= 2
check("sem MI >= 0", np.all(MI >= -1e-9))
check("sem E[H] <= H(mean) where active (expected <= total)",
      np.all(Hexp[active] <= Htot[active] + 1e-9))
# MI -> 0 as S -> inf at fixed mean (scale a single voxel's counts up)
base = np.array([[10.0, 5.0]]); au = np.array([1.0])
scales = np.array([1, 10, 100, 1000, 10000.])
mis = np.array([F.semantic_mi(base*s, au*s)[0] for s in scales])
check("sem MI decreases monotonically with evidence (fixed mean)",
      np.all(np.diff(mis) < 1e-9) and mis[-1] < 1e-2)

print("== C3 coarsening bound: gap <= p_OTHER * ln(C - K_top) (floor-off) ==")
# Build synthetic FULL-C Dirichlet means, coarsen (C-K) tail classes into one
# OTHER bucket, and check the entropy grouping identity + bound. This validates
# the E4b claim analytically, independent of scovox.
def coarsen_check(C, K, trials=20000):
    viol = 0; gaps = []; bounds = []
    for _ in range(trials):
        alpha = rng.uniform(0.01, 5.0, C)
        p = alpha/alpha.sum()
        # keep top-K classes as singletons, group the rest into OTHER
        idx = np.argsort(p)[::-1]
        keep, other = idx[:K], idx[K:]
        p_keep = p[keep]; p_oth = p[other].sum()
        H_full = -(p[p>0]*np.log(p[p>0])).sum()
        parts = list(p_keep) + [p_oth]
        parts = np.array([x for x in parts if x > 0])
        H_coarse = -(parts*np.log(parts)).sum()
        gap = H_full - H_coarse
        bound = p_oth*np.log(C-K) if C-K > 1 else 0.0
        # grouping identity: gap == p_oth * H(conditional within OTHER)
        pc = p[other]/p_oth if p_oth > 0 else np.zeros_like(p[other])
        H_within = -(pc[pc>0]*np.log(pc[pc>0])).sum()
        ident = p_oth*H_within
        if gap > bound + 1e-9: viol += 1
        gaps.append(gap); bounds.append(bound)
        if abs(gap-ident) > 1e-9: viol += 1000  # identity must hold exactly
    return viol, np.array(gaps), np.array(bounds)

for (C, K) in [(19, 2), (14, 2)]:
    viol, gaps, bounds = coarsen_check(C, K)
    check(f"C={C},K={K}: 0 bound violations & grouping identity exact "
          f"(max gap {gaps.max():.3f}, max bound {bounds.max():.3f})", viol == 0)

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
raise SystemExit(0 if ok else 1)
