# SCovox: Methods Document

**Author:** Kalhan
**Companion to:** [experiments_and_results.md](experiments_and_results.md) for experiments, results, and baseline comparisons.

This document describes the representation, update rules, uncertainty
queries, and evaluation protocols of SCovox. It is a pure methods
reference — experiment specifics and results are kept out of this file.

---

## 1. Overview

**SCovox** (Semantic Covariance Voxel) is a CPU-resident Bayesian
volumetric map that jointly models, at every voxel, (a) a Beta
posterior over occupancy and (b) a Dirichlet posterior over a small
set of semantic classes. It is implemented on top of the Bonxai sparse
voxel container.

---

## 2. Voxel Representation

Each voxel stores a 32-byte trivially-copyable struct
([scovox_core/include/scovox/voxel.hpp](../../scovox_core/include/scovox/voxel.hpp)):

```text
α_occ, α_free, α_unk        : Beta concentrations + Dirichlet "unknown" residual (3 × 4 B)
sem_cnt[K_TOP]              : Dirichlet counts for the top-K observed classes (K × 4 B)
sem_cls[K_TOP]              : Class IDs for those K slots (K × 2 B, uint16)
tsdf_distance, tsdf_weight  : TSDF surface state (2 × 4 B; valid only when tsdf_weight > 0)
```

**K_TOP = 2.** Each voxel keeps the two most-observed classes;
additional classes spill into `α_unk`. Semantic per-voxel cost is 12 B
regardless of class cardinality — a key difference from BKI-family
systems that allocate a full *C*-dim Dirichlet per voxel. Memory
savings vs a dense per-voxel posterior of length *C* scale linearly
with *C*.

**Prior.** Voxels initialise to Beta(1,1) on occupancy (uniform) and
store `α_unk = 0`, `sem_cnt = 0` as raw evidence. A Dirichlet(+1) prior
is added at *query time* in `semanticEntropy` / `semanticVariance`,
not persisted. This keeps the per-voxel byte budget minimal while
preserving conjugacy.

**Sparse-add policy.** When an observation arrives for a class not
already in one of the two slots and both slots are occupied, we evict
the smaller-count slot only if the incoming increment strictly exceeds
it; otherwise the new mass is routed into `α_unk`. Mass is conserved
on eviction (evicted slot's count is added to `α_unk`). The strict
inequality prevents a noisy classifier emitting tied increments from
thrashing slots indefinitely.

---

## 3. Update Rules

### 3.1 Occupancy (Beta conjugate)

For each ray cast from the sensor origin to the measurement endpoint
(3D DDA traversal via [ray_iterator.hpp](../../scovox_core/include/scovox/ray_iterator.hpp)):

- Voxels traversed before the endpoint receive `α_free += w_free · ψ(r, θ)`
- The endpoint voxel receives `α_occ += w_occ · ψ(r, θ)`

The weight ψ(r, θ) is a range/grazing-angle modulation: evidence
decays exponentially with range (`range_decay_length = 5 m`) and is
suppressed when the ray hits at an oblique angle (cosine below
`grazing_angle_threshold`). A carve-skip guard
(`carve_skip_occ_threshold = 0.7`) prevents a single stray free
traversal from erasing a strongly-occupied voxel.

The posterior occupancy probability is
`p_occ = α_occ / (α_occ + α_free)`. Evidence is saturated at
`evidence_saturation = 1000` to prevent unbounded confidence in
dynamic environments and to keep later ray updates informative.

### 3.2 Semantic update (Dirichlet conjugate)

Semantic fusion is **Bayesian-soft, hard-gated**
([scovoxmap.cpp:141-166](../../scovox_mapping/src/scovoxmap.cpp#L141-L166),
[semantics.hpp::dirichlet_update_semantics](../../scovox_core/include/scovox/semantics.hpp#L24-L64)).
A voxel's Dirichlet counts are touched only when its current
`p_occ ≥ dirichlet_min_p_occ` (default 0.5). Past this gate, the
per-observation weight is `w = κ₀ · p_occ · q`, so confidently-
occupied voxels accumulate fully and weakly-occupied voxels just above
the gate accumulate less — the `p_occ` factor marginalises the class
observation over the current Beta posterior on occupancy.

Given a per-class probability vector **p** (from the upstream
segmentation network) and a per-observation quality `q ∈ [0,1]`, when
`p_occ ≥ dirichlet_min_p_occ`:

- Effective weight: `w = κ₀ · p_occ · q` (κ₀ = 2.0).
- For each class i: add `w · p_i / norm` to the Dirichlet slot for
  class i (via `sparse_add`).
- The unobserved-mass residual `w · (1 − Σ p_i / norm)` flows into
  `α_unk`.

The normalisation `norm = max(1, Σ p_i)` preserves the invariant that
each admitted observation contributes exactly `w` pseudocounts total,
even if the caller passes un-normalised scores. For one-hot or softmax
inputs (Σ p ≤ 1), `norm = 1` and the `α_unk` increment is the
classifier's explicit unknown-mass.

After the update, `apply_evidence_saturation` proportionally clamps
`(α_occ, α_free, sem_cnt)` so no voxel exceeds `evidence_saturation`
total pseudocounts (preserving `p_occ` and class proportions while
keeping later observations informative).

**Load-bearing parameters** ([map_interface.hpp](../../scovox_core/include/scovox/map_interface.hpp)):
`evidence_saturation = 1000`, `dirichlet_min_p_occ = 0.5`,
`carve_skip_occ_threshold = 0.7`, `range_decay_length = 5.0 m`,
`kappa0 = 2.0`. Earlier revisions also exposed `tau`, `k_gate`,
`s_min`, and `semantic_min_confidence`; the residual-attribution
sweep showed these were redundant and they have been removed (audit at
[issues/gate_simplification.md](issues/gate_simplification.md)).

### 3.3 Update locality along the truncation band

SCovox writes the **Dirichlet update on the hit voxel only**: the
unique voxel containing the measurement endpoint
([scovoxmap.cpp:306-310](../../scovox_mapping/src/scovoxmap.cpp#L306-L310)).
In-band non-hit voxels along `[depth − sdf_trunc, depth + sdf_trunc]`
receive a TSDF-fusion update and (if appropriate) a Beta-free carve,
but no Dirichlet write. One measurement contributes one Dirichlet
observation, on the voxel the measurement geometrically concerns.

This keeps Dirichlet evidence accumulation independent of the
geometric truncation parameter and confines class-label mass to
voxels at the measured surface. (A band-wide alternative — writing
the categorical vote on every band voxel — accumulates evidence
faster but couples the semantic posterior to `sdf_trunc`.)

### 3.4 Soft-probability ingestion (post-hoc class merge)

The Dirichlet update naturally accepts a full per-class distribution
**p**, not just an argmax label. When the upstream segmentation
network emits a logits/softmax vector over its native class space
*S_net* (e.g. ADE-150 for Mask2Former) and the SCovox class space
*S_voxel* is a coarsening of *S_net* (e.g. 19 indoor classes), the
correct projection is **post-hoc class merge**:

1. Apply softmax in *S_net*.
2. Marginalise over the *S_net* → *S_voxel* mapping by summing
   probabilities of source classes that map to the same target class.
3. Pass the resulting distribution in *S_voxel* to
   `dirichlet_update_semantics` with the standard `α_unk` residual.

Σ p = 1 is preserved by construction (it's marginalisation, not
re-weighting), and source classes with no target collapse to
`α_unk`. The merge LUT is built from a name-substring match against
the SCovox class names (longest match wins), with manual aliases for
classes whose canonical names don't substring-match
([build_ade_collapse_map.py](../scovox_eval/scripts/build_ade_collapse_map.py)).

The dense distribution is serialised as a `(H, W, C)` (image) or
`(N, C)` (pointcloud) `uint8` array quantised ×255, with a small
binary header. Slot 0 is reserved for "unknown" so a one-hot label can
be passed through the same channel by setting that slot to 1 and the
rest to 0.

### 3.5 Band-only integration mode (optional)

A `Params::band_only_integration` flag (default `false`) shrinks the
DDA range from `[posToCoord(origin), posToCoord(hit + sdf_trunc·u)]`
to `[posToCoord(hit − sdf_trunc·u), posToCoord(hit + sdf_trunc·u)]`,
matching a band-only TSDF protocol. Beta-free still fires for in-band
non-hit voxels in front of the surface (~3 voxels per ray), so the
surface neighbourhood retains an occupancy posterior. Voxels outside
the band are not visited.

This trades free-space carving along the full ray for a smaller
per-ray work cost. Use the full-ray default when downstream planners
rely on free-space carving along the entire ray; use band-only for
offline reconstruction-only workloads where free-space mass beyond
the surface band carries no information consumed downstream.

### 3.6 Sparse Dirichlet — theoretical framing

The K_TOP-slot tracker described in §2 (Sparse-add policy) and §3.2
is equivalent to the **Space-Saving / Misra-Gries** streaming
heavy-hitter algorithm (Misra & Gries 1982; Metwally, Agrawal & El
Abbadi, ICDT 2005). The eviction comparison `inc > sem_cnt[min_i]`
is the posterior-predictive swap test under a symmetric Dirichlet
prior: with α₀ shared across classes and small per-observation
increments, swapping incoming class *c* into the tracking set is
optimal exactly when its evidence exceeds the weakest tracked slot's,
recovering the Space-Saving criterion. The strict `>` (not `>=`) is a
deliberate stability choice — tied newcomers are routed to `α_unk` to
prevent thrashing under noisy classifiers.

**Error bound on K.** Per Cormode (2016, *Encyclopedia of
Algorithms*, Springer): any class with true frequency `> n/K` at a
voxel is guaranteed to be tracked, and the additive count error on
any tracked class is at most `n/K`, where *n* is the total
observations at that voxel. K_TOP = 2 tracks any class with
> 50 % observation share and bounds count error at *n/2*.

**Residual interpretation (Hutter 2013).** Raw `α_unk` is a dump
bucket — it accumulates evicted and dropped evidence with no
probabilistic meaning. A principled interpretation is provided at
**query time** via the adaptive escape mass of Hutter (AISTATS 2013,
*Sparse Adaptive Dirichlet-Multinomial-like Processes*):

```text
β*  =  m / [2 ln((N+1)/m)]
```

where *m* is a lower bound on distinct classes ever observed at that
voxel and *N* is total semantic observations
(`α_unk + Σ sem_cnt`). The effective residual is
`max(α_unk, β*)`, reflecting the Dirichlet posterior mass allocated
to untracked classes under a sparse Dirichlet-Multinomial model.

**Crucially, the floor is applied only at query time** (entropy,
variance, class prediction, visualisation — see
[`uncertainty.hpp::effectiveResidual`](../../scovox_core/include/scovox/uncertainty.hpp)).
The update path (`sparse_add`, `dirichlet_update_semantics`,
`consensusMerge`, the wire format) is unchanged: raw evidence is
conserved exactly across fusion and serialisation. Only consumers of
the posterior see the lifted residual.

---

## 4. Uncertainty Queries

Closed-form quantities computed on demand from the stored counts
([uncertainty.cpp](../../scovox_core/src/uncertainty.cpp)):

| Quantity                | Formula                                              | Use                                         |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------- |
| Occupancy variance      | `αβ / (s²(s+1))` with `s = α+β`                      | Confidence bars, planning                   |
| Occupancy entropy       | Differential entropy of Beta(α,β)                    | Frontier scoring                            |
| Expected info. gain     | `H(y) − E[H(p\|y)]` via digamma expansion            | SSMI-style exploration targets              |
| Semantic entropy        | Differential entropy of Dirichlet with +1 prior      | Semantic frontier selection                 |
| Semantic variance       | Per-class Dirichlet marginal variance                | Per-class confidence                        |
| SSMI occ/free KL        | KL(Bern(p) ‖ Bern(p_post)) after hypothetical update | Mutual-information planning                 |
| Inter-voxel Beta-KL     | KL(Beta(α₁,β₁) ‖ Beta(α₂,β₂))                        | Consensus fusion across robots              |

A scalar log-odds occupancy representation cannot provide these
signals — it stores a single value per voxel and conflates "never
observed" with "observed many times near *p* = 0.5". The closed-form
Beta variance and Dirichlet entropy / variance terms above are
specifically what motivates carrying the full posterior.

---

## 5. Evaluation Protocol

### 5.1 Offline replay harness

Each dataset is driven through a headless ROS 2 replay node that
publishes sensor messages at a deterministic rate. Each variant is a
drop-in `MapInterface` backend, so timings reflect pure map update
cost (no rendering, no visualisation). Per-frame metrics captured:

- Per-stage wall time (`tf`, `raycast`, `semantic`, `publish`)
- Active voxel count, RSS, map bytes
- On shutdown: a full NPZ dump of (x, y, z, p_occ, class_id,
  confidence, evidence_count) for all active voxels

### 5.2 Metrics ([scovox_eval/metrics/](../scovox_eval/metrics/))

**Geometry** (vs GT mesh sampled at 100 k points):

- **F-score @ τ** — harmonic mean of precision/recall at distance
  threshold τ ([fscore.py](../scovox_eval/metrics/fscore.py))
- **Chamfer distance (cm)** — bidirectional nearest-neighbour,
  KDTree-accelerated ([chamfer.py](../scovox_eval/metrics/chamfer.py))
- **Normal consistency** — mean |cos θ| between matched
  predicted/GT surface normals ([normal_consistency.py](../scovox_eval/metrics/normal_consistency.py))

**Semantics:**

- **mIoU** — unweighted mean of per-class IoU, configurable
  ignore label ([miou.py](../scovox_eval/metrics/miou.py))
- **Semantic ECE / MCE** — 20-bin top-label calibration error
  ([semantic_ece.py](../scovox_eval/metrics/semantic_ece.py))

**Calibration** (occupancy, after rendering to a binary GT grid):

- **ECE** — weighted |acc − conf| over 20 equal-width bins of `p_occ`
  ([ece.py](../scovox_eval/metrics/ece.py))
- **MCE** — max over bins
- **Brier score** — mean squared error of `p_occ` vs binary GT
- **Evidence-stratified ECE** — computed separately for low (<5),
  medium (5–20), and high (>20) observation counts, to detect
  miscalibration that appears only in informative regions

**Uncertainty separability:**

- **Unknown-vs-ambiguous AUC** — ROC AUC of a "genuinely unobserved"
  score (small `α_unk` / large variance) at separating truly unseen
  voxels from surface-boundary voxels that are merely in the
  *p ≈ 0.5* band

**Runtime:** median and p95 of per-frame `total` time, FPS, peak RSS,
bytes-per-voxel.

### 5.3 Statistical tests

For paired variant comparisons across small samples (e.g. 8 Replica
scenes) we use the **Wilcoxon signed-rank test** (non-parametric, no
normality assumption). Significance threshold α = 0.05. Tests are
two-sided unless otherwise noted. Effect sizes are reported as
absolute differences — not percentage improvements — when the raw
metric range is narrow.

### 5.4 Baselines

For SLIM-VDB CLOSED head-to-head we re-run the upstream system in
its docker image on the same hardware as SCovox, consuming the same
sensor + label inputs, scoring with the same evaluator. For wider
field positioning we cite published numbers from prior systems
(ConvBKI, SEE-CSOM, Voxfield Panmap, …) on their reported protocols
without re-running. Both axes are tagged in
[experiments_and_results.md](experiments_and_results.md).

---

## 6. Reproducibility

Shell drivers in [scovox_eval/scripts/](../scovox_eval/scripts/)
regenerate artefacts end-to-end. Per-run artefacts (NPZ maps,
metrics JSON, per-stage timing CSV) land under
`results/<dataset>_<config>/<variant>/<scene>/`. Per-cell aggregator
scripts produce CSV summaries for cross-cell analysis.

---

## 7. Cross-references

- Experiments + results: [experiments_and_results.md](experiments_and_results.md)
- Per-experiment archive: [archive/](archive/)
- Voxel struct / priors: [scovox_core/include/scovox/voxel.hpp](../../scovox_core/include/scovox/voxel.hpp)
- Occupancy gate + semantic update: [scovox_core/include/scovox/semantics.hpp](../../scovox_core/include/scovox/semantics.hpp)
- Uncertainty closed forms: [scovox_core/src/uncertainty.cpp](../../scovox_core/src/uncertainty.cpp)
- Parameters: [scovox_core/include/scovox/map_interface.hpp](../../scovox_core/include/scovox/map_interface.hpp)
- Metric implementations: [scovox_eval/metrics/](../scovox_eval/metrics/)

---

## 8. References

- Misra, J. & Gries, D. (1982). *Finding repeated elements*. Science of Computer Programming 2(2), 143–152.
- Metwally, A., Agrawal, D. & El Abbadi, A. (2005). *Efficient computation of frequent and top-k elements in data streams*. ICDT, 398–412.
- Hutter, M. (2013). *Sparse Adaptive Dirichlet-Multinomial-like Processes*. AISTATS.
- Cormode, G. (2016). *Space-Saving*. In *Encyclopedia of Algorithms*, Springer. doi:10.1007/978-1-4939-2864-4_572.
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML.
- Asgharivaskasi, A. & Atanasov, N. (2023). *Semantic OctoMap*. IEEE T-RO. (SSMI-style ray-marginalised MI in §4.)
