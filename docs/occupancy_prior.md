# The occupancy prior: choosing a symmetric `Beta(1,1)`

**Decision (TL;DR).** The split-substrate occupancy prior is the symmetric
**uniform / Bayes–Laplace prior `Beta(1, 1)`** — `a_occ = a_free = 1.0`, so an
**unobserved voxel has prior `p_occ = 0.5`**. This replaces the previous
*calibrated* prior `Beta(C·α₀, α₀)` (prior `p_occ = C/(C+1) ≈ 0.933` for NYU13).
The single source of truth is `kBetaOccPrior` / `kBetaFreePrior` in
[`beta_voxel.hpp`](../src/scovox_core/include/scovox/beta_voxel.hpp); the
Jeffreys prior `Beta(0.5, 0.5)` is the documented runner-up (§11).

This document records *why* — the change is behavioural and reversible, so the
reasoning should outlive the diff.

---

## 1. Background: the Beta–Bernoulli occupancy counter

Occupancy in the split substrate is a conjugate Beta–Bernoulli counter. A
[`BetaVoxel`](../src/scovox_core/include/scovox/beta_voxel.hpp) stores two
pseudo-counts `(a_occ, a_free)`; the posterior occupancy is

```
p_occ = a_occ / (a_occ + a_free),   posterior ~ Beta(a_occ, a_free).
```

Each ray **hit** adds `w_occ·quality` to `a_occ` at the surface voxel
(`w_occ = 1.0`); each **carve** adds `w_free·quality` to `a_free` along the ray
(`w_free = 0.5`). Updates are `O(1)` per observation. A first-touch **prior**
must be stored (Bonxai zero-inits leaf blocks), and it must be *proper and
strictly positive* so `p_occ`, the Beta variance `p(1−p)/(s+1)`, and the entropy
are all defined for an unobserved voxel.

## 2. The problem with the old prior

The calibrated prior `Beta(C·α₀, α₀) = Beta(0.14, 0.01)` was chosen so the split
Beta marginal would *match the unified `SemDirVoxel` occupancy marginal*,
`p_occ = s_occ/s_total = C·α₀ / ((C+1)·α₀) = C/(C+1) ≈ 0.933`. But a prior that
declares an **unobserved** voxel ~93 % occupied is wrong for a fresh-map
regulariser: it biases frontier detection and EIG, and it primes the carve
wall-guard. We want a symmetric prior, mean `0.5`.

## 3. The family, and the one real knob

Fixing the mean at `0.5` means a **symmetric** prior `Beta(c, c)`. The only
remaining choice is the **concentration** `s = 2c` — the prior's worth in
*pseudo-observations* before data takes over. Three candidates span the range:

| prior | `c` | concentration `s=2c` | character |
|---|---|---|---|
| Haldane-limit `Beta(α₀,α₀)`=`Beta(0.01,0.01)` | 0.01 | 0.02 | near-improper, data-dominant (posterior ≈ MLE) |
| **Jeffreys** `Beta(0.5,0.5)` | 0.5 | 1 | objective / reference prior |
| **Uniform** `Beta(1,1)` | 1 | 2 | max-entropy flat density; classic occupancy-grid prior |

## 4. The objective-Bayes case for Jeffreys (stated fairly)

For a Bernoulli rate the **Jeffreys prior is the formal objective prior**.
Fisher information is `I(p) = 1/(p(1−p))`; the Jeffreys rule
`π(p) ∝ √I(p) = p^(−1/2)(1−p)^(−1/2) = Beta(½, ½)`. It is the unique symmetric
Beta that is **invariant under reparameterization** (log-odds, arcsine-√p) — the
coordinate-free notion of "non-informative" — and for a scalar parameter it
coincides with the Bernardo *reference* prior. Uniform `Beta(1,1)` is flat in
`p` but *not* under any monotone transform, so its "non-informativeness" is an
artifact of the parameterization. On pure prior theory, **Jeffreys wins**.

## 5. …but "objective" is the wrong objective here

The occupancy prior is **not an analyst's ignorance device** for pooled
inference. It is a per-voxel **regulariser** that (a) must survive genuine
single-ray sensor noise on the *first* observation, (b) feeds **hard
thresholds** (the carve wall-guard, the Dirichlet commit gate), (c) seeds
frontier/EIG, and (d) is **subtracted** by the consensus merge. Reparameterization
invariance — Jeffreys' winning property — is read by **no scovox consumer**;
none of them transform the occupancy coordinate. With invariance worth nothing
here, the decision turns on four system interactions (§7–§10), and there Uniform
dominates.

## 6. Properness rules out Haldane

The stored prior must be proper and strictly positive. The Haldane prior
`Beta(0,0)` is the improper `c→0` limit (posterior = MLE); only its `ε>0`
surrogate `Beta(0.01,0.01)` is admissible, and it is *near*-improper — one noisy
ray drives it to the MLE (§7). Jeffreys and Uniform are both proper.

## 7. Single-ray noise vs the hard thresholds (the decisive factor)

The carve wall-guard fires at `p_occ > carve_skip` (0.95 on the split path) and
the Dirichlet commit gate at `p_occ ≥ 0.5`. After **one** hit (`w_occ = 1`,
`quality = q`), `p_occ` lands at:

| prior | `p_occ` after 1 hit (q=1) | (q=0.5) | after 5 hits (q=1) |
|---|---|---|---|
| Haldane `Beta(0.01,0.01)` | **0.990** | 0.981 | 0.999 |
| Jeffreys `Beta(0.5,0.5)` | 0.750 | 0.667 | 0.846 |
| **Uniform `Beta(1,1)`** | **0.667** | 0.600 | 0.857 |

A single spurious/grazing/max-range ray under **Haldane trips the 0.95
wall-guard**, blocking carving along the rest of that ray — the worst single-ray
failure mode. **Uniform sits furthest below the wall-guard** while still clearing
the `0.5` commit gate on one *genuine* hit (`0.667 ≥ 0.5`), so real evidence
still commits semantics. This is the classic bias-toward-the-prior shrinkage the
noisy-classifier regime demands, and it is the deciding consideration.

## 8. Evidence saturation favours a larger `c`

`applyBetaSaturation` rescales `a_occ, a_free` by `k = cap/s` when the total
concentration exceeds the cap, exactly preserving `p_occ` — so the prior
persists strictly **in proportion** to its share of the budget. A larger `c`
keeps a larger fixed nonzero share after the cap; Haldane's `0.02` share is
functionally erased. This argues monotonically for `c ≥ 1` and against Haldane.

## 9. Consensus merge stays clean

`mergeBeta` computes `a_fused = a_A + a_B − prior`, floored at the prior, so a
prior shared by both sources is not double-counted. With the symmetric
`prior = 1` per dimension the subtracted mass is the **natural unit**, is
**class-count-independent** (no `C` to reconstruct on the occupancy side — the
prior is now decoupled from `(num_classes, α₀)`), and the floor `max(1, ·)`
keeps fused alphas `≥ 1`, preserving properness. Because the prior is a
compile-time constant on both ends, it is **not carried on the wire**. To stop a
mixed-prior fleet from silently corrupting fused mass, the v4 blob codec
revision (`BinarySerializerV4::FORMAT_VERSION`) was bumped **4 → 5** with this
change: the wire *layout* is byte-identical, but a revision-4 (calibrated-prior)
node and a revision-5 (`Beta(1,1)`) node now reject each other's frames at
`deserialize` (the frame is dropped with a warning) instead of merging under
mismatched priors. The ROS envelope `version` stays `4` (the v4 format-family
router).

## 10. Entropy / EIG / SSMI

Uniform's density is **bounded** on `[0,1]`, so early-observation entropy and
Beta variance are smooth. Jeffreys' density spikes to `+∞` at `p = 0, 1`
(integrable but awkward for a threshold/visualisation pipeline). All candidates
are consistent (`p_occ(n) → 1`), so the only cost of the heavier prior is
slightly slower commitment — which is exactly the desired noise robustness.

## 11. Decision, and why not Jeffreys

**Ship `Beta(1,1)`.** It is the most first-observation-robust *proper* prior,
sits safest below the wall-guard while still clearing the commit gate, keeps the
largest share through saturation, gives an integer-clean merge subtraction, and
is the canonical log-odds occupancy-grid prior (`l₀ = log(0.5/0.5) = 0`,
Moravec–Elfes / Thrun / OctoMap) and the max-entropy flat density.

**Jeffreys `Beta(0.5, 0.5)` is the honest runner-up** — the textbook objective
prior. We would switch to it only if information-geometric / EIG
reparameterization invariance is ever promoted to a *hard* requirement (e.g. an
SSMI formulation that integrates over a transformed occupancy coordinate). Until
then, invariance buys nothing and costs first-ray robustness, so Uniform wins.

Haldane `Beta(0.01,0.01)` is rejected on two hard grounds: near-improperness,
and tripping the wall-guard on a single noisy ray. Its only argument — magnitude
consistency with the semantic `α₀ = 0.01` — is a soft, non-binding design value.

## 12. Implementation

The prior is a fixed symmetric constant, so it is **decoupled** from the
semantic `(num_classes, α₀)` plumbing. Single source of truth:

```cpp
// beta_voxel.hpp
constexpr float kBetaOccPrior  = 1.0f;
constexpr float kBetaFreePrior = 1.0f;
```

These are referenced — identically on sender and receiver — at every site that
(re)constructs the occupancy prior, so none can drift:

| site | file | role |
|---|---|---|
| allocation | `sem_split_map.cpp` `beta_occ_prior_/beta_free_prior_` | first-touch `BetaVoxel` |
| merge | `consensus_merge_v4.hpp` `mergeBeta` | `a_fused = a_A+a_B − prior` |
| at-prior detection | `dscovox_node.cpp` `isPriorBeta` | refold / publish gate |
| refold reset | `dscovox_node.cpp` `refoldCellBeta` | reset before fold |
| sender emit gate | `scovox_node.cpp` (v4 `emit_beta`) | keep prior-only voxels off the wire |
| viz gate | `scovox_node.cpp` `has_beta_evidence` | don't publish prior-only cells |
| SSMI baseline | `dscovox_node.cpp` (v4 `unobserved`) | unallocated-cell scoring prior |

`mergeBeta` / `isPriorBeta` keep their `(num_classes, α₀)` parameters for
call-site symmetry with the *semantic* (`mergeDir` / `isPriorDir`) path but
ignore them for occupancy. The prior-agnostic factory
`defaultBetaVoxel(occ, free)` still reproduces the **calibrated** prior
`Beta(C·α₀, α₀)` on explicit request, so that marginal-matching ablation remains
available. Guarded by the `p_occ == 0.5` fresh-voxel + two-source-merge tests in
`test_sem_split_map.cpp` / `test_consensus_merge_v4.cpp`.

**Scope:** this change is the **split (v4) `BetaVoxel` path only**. The unified
`SemDirVoxel` substrate keeps its calibrated `C/(C+1)` occupancy marginal (its
`s_occ` is structural and feeds the semantic Dirichlet; flipping it to `0.5`
needs the FREE-dual technique — `alpha_free = C·α₀` — and is out of scope here).

## 13. References

- Jeffreys, H. (1946). *An invariant form for the prior probability in
  estimation problems.* Proc. R. Soc. A.
- Bernardo, J. M. (1979). *Reference posterior distributions for Bayesian
  inference.* JRSS-B. (Reference prior ≡ Jeffreys for a scalar parameter.)
- Tutz / standard Bayesian texts: the uniform `Beta(1,1)` (Bayes–Laplace) as the
  max-entropy prior on `[0,1]`.
- Moravec & Elfes (1985); Thrun, Burgard & Fox, *Probabilistic Robotics* (2005),
  ch. 9 — log-odds occupancy grids with prior `p = 0.5` (`l₀ = 0`).
- Hornung et al. (2013), *OctoMap* — log-odds occupancy with `0.5` prior.
