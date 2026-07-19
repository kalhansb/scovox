# SCovox Experiment Plan — Uncertainty Assessment on SemanticKITTI and SceneNet

**Scope.** The previous experimental round established accuracy–memory trade-offs (K_top sweep), update-rule ablations, a system-level baseline comparison, gating behaviour, map combination, and embedded feasibility. What it did not test is the central representational claim: that the Beta/Dirichlet substrate yields *calibrated* uncertainty and a meaningful *aleatoric/epistemic decomposition* under the top-K + OTHER truncation. This plan closes that gap, and simultaneously fixes the main methodological limitation of the prior round (single executions per configuration).

**Claims under test**

- **C1 (Calibration).** Per-voxel p_occ and the predictive semantic categorical are calibrated against voxel references, and at least as well calibrated as the dense-Dirichlet baseline and a no-accumulation segmenter baseline.
- **C2 (Occupancy decomposition).** EIG behaves as epistemic uncertainty (decays with evidence; high in under-observed regions) and E_H = ψ(s+1) − p·ψ(a+1) − (1−p)·ψ(b+1) behaves as aleatoric uncertainty (persists in genuinely ambiguous voxels; invariant to additional evidence).
- **C3 (Semantic decomposition under truncation).** The decomposition over {resident top-K, OTHER} is faithful: (a) the stored sparse state tracks the exactly-marginalised full-C posterior (fidelity), and (b) the full-space vs coarsened-space gap obeys the analytic bound p_OTHER · ln(C_sem − K_top) per voxel.
- **C4 (Utility).** Uncertainty signals rank errors (sparsification) and identify informative observations (frame selection; predicted vs realised gain).
- **C5 (Fusion).** Evidence-summing combination preserves calibration where observations are independent, and the overconfidence induced by duplicated observations in the overlap construction is measurable and mitigable.

---

## 0. Prerequisites (instrumentation, ~1 week)

**Code additions** (all closed-form; follow existing query-time +1 and `effectiveResidual` conventions):

1. Expose `betaExpectedEntropy(v)` (the E_H term currently internal to `expectedInformationGain`) as a public aleatoric signal; `betaPredictiveEntropy(v)` = Bernoulli H(p̄) as the total.
2. Add `semanticExpectedEntropy(v)` = ψ(S+1) − Σᵢ (αᵢ/S)·ψ(αᵢ+1) over the same categorical used by `semanticEntropy` (resident slots + Hutter-floored residual, +1 prior), and `semanticMI(v)` = max(0, semanticEntropy − semanticExpectedEntropy).
3. Add per-voxel exports: a_occ, a_free, s, EIG, E_H, H_y, sem alphas + a_unk (raw and floored), S, p_OTHER, semanticEntropy, semanticExpectedEntropy, semanticMI, vacuity (K+1)/(S+K+1), observation counters.
4. A query-mode flag to disable the Hutter floor (ablation A3), leaving update paths untouched.
5. A snapshot dump (v5 binary → npz) at final frame and at intermediate checkpoints {25, 50, 100} scans / {75, 150, 300} frames.
6. Unit tests: MI ≥ 0; MI → 0 as S → ∞ at fixed mean; expected entropy ≤ total entropy; marginalisation identity Dir(α) grouped = exact coarsened Dirichlet.

**Reference construction.**

- *Occupied reference*: existing protocol (aggregated labelled points, majority class per voxel).
- *Free reference*: voxels traversed by ≥ m ground-truth-pose rays with zero endpoint hits across the full replay, inside the range gate; m = 3 default, sensitivity check m ∈ {1, 3, 5}. Never-observed voxels excluded.
- *SemanticKITTI dynamics*: report occupancy calibration twice — all classes, and excluding vehicle/person voxels — since moving objects corrupt a static occupancy reference. This is a proxy reference; treat absolute ECE with caution and lean on paired comparisons.

**Uncertainty baselines**, matched inputs/poses/resolution as in the prior round:

| Baseline | Uncertainty extracted | Purpose |
|---|---|---|
| SLIM-VDB (dense Dirichlet) | same functionals from its per-voxel alphas | representation-level comparison |
| Segmenter-only (last-observation projection of PolarNet soft probs / oracle labels at endpoints) | per-frame softmax entropy/max-prob | value of accumulation |
| NAIVE / MV ablations | NAIVE: constant entropy; MV: vote margin | shows counting is what produces usable uncertainty |

---

## 1. Experiment matrix

Common protocol: sequences 06–10 (100 scans, 10 cm) and 13 SceneNet validation trajectories (300 frames, 5 cm); principal parameters as before (w_occ = 6, w_free = 1, κ₀ = 2, α₀ = 0.01, S_max = 1000, K_top = 2 unless varied). **Every principal-configuration cell is executed n = 3 times**; run-to-run SD is reported separately from across-sequence SD. Statistics: paired by sequence/trajectory, mean difference ± SE, 95% CI (paired t), sign tests where informative — as in the prior round.

### E1 — Occupancy calibration and decomposition (P0)

| | |
|---|---|
| Conditions | principal config; checkpoints at 25/50/100% of scans |
| Metrics | reliability diagram (15 equal-mass bins), ECE, MCE, Brier, NLL on occupied/free reference; sparsification of occupancy error ranked by Var[p_occ] and by EIG, with AUSE vs oracle ranking |
| Hypotheses (stated in advance) | H1.1: median EIG decreases monotonically across checkpoints on voxels observed at 25%. H1.2: E_H is stable across checkpoints (effect size reported; no significance claim). H1.3: E_H distribution is right-shifted for boundary voxels (reference voxels adjacent to free reference) and, on KITTI, for vegetation/fence classes, relative to planar road/building voxels. |
| Deliverables | Fig: reliability (per dataset); Fig: EIG and E_H vs evidence s (binned); Table: ECE/Brier/NLL vs baselines |

### E2 — Semantic calibration, error detection, sparsification (P0)

| | |
|---|---|
| Conditions | principal config; baselines from §0; Hutter on/off (A3) |
| Metrics | top-label ECE + classwise ECE, NLL, Brier of the predictive categorical on reference voxels; AUROC/AUPR of each signal {total, aleatoric, epistemic MI, vacuity, MV margin, segmenter max-prob} for (a) misclassified voxels and (b) voxels whose true class is outside the resident top-K; sparsification curves (mIoU vs rejection fraction) + AUSE per signal |
| Hypotheses | H2.1: MI and vacuity beat total entropy at detecting (b); total/aleatoric competitive at (a). H2.2: SCovox accumulation improves ECE and AUSE over segmenter-only. H2.3: NAIVE provides no usable ranking (AUSE ≈ random). |
| Deliverables | Fig: reliability; Fig: sparsification; Table: AUROC/AUPR/AUSE matrix (signal × task × dataset) |

### E3 — Decomposition validation by controlled manipulation (P0; the core scientific test)

Two independent manipulations isolate the two axes:

**E3a — Evidence scaling (targets epistemic).** Subsample replay uniformly to {25%, 50%, 100%} of frames. On the matched voxel set observed in all conditions: epistemic (MI, EIG) should fall with more data; aleatoric (expected entropy, E_H) should be approximately invariant. Report paired per-voxel effect sizes.

**E3b — Observation-noise injection (targets aleatoric).** SceneNet only (oracle labels make this clean): corrupt each per-observation label with symmetric noise at rate ε ∈ {0, 0.1, 0.2, 0.4} (occupancy inputs untouched). Predictions: aleatoric increases monotonically in ε and converges to the entropy of the corrupted label distribution at high count; epistemic at matched count is approximately invariant; mIoU degrades gracefully and high-noise voxels are flagged by aleatoric, not epistemic. KITTI analogue (secondary): temperature-scale PolarNet probabilities T ∈ {0.5, 1, 2} — total per-observation mass is unchanged by design (normalisation in `dirichlet_update_semantics`), so only allocation sharpness varies.

Deliverables: Fig: aleatoric/epistemic vs ε (the signature plot — crossing behaviour is the claim); Fig: MI vs observation count; Table: paired effects.

### E4 — Truncation: sparse-state fidelity and the coarsening bound (P0; ties to C3)

Uses the existing full-support runs (K_top = 19 / 13) as the reference posterior.

**E4a — Fidelity.** For each K_top = 2 voxel, exactly marginalise the matched full-K voxel onto that voxel's resident partition {c₁, c₂, OTHER, unk}. Compare stored vs exact-marginal: total variation of mean categoricals, and Δ in each of {total, aleatoric, epistemic}. This measures accumulated eviction-path and residual-mixing drift — the honest cost of the online sparse approximation. Report gap CDFs and their dependence on p_OTHER and eviction count.

**E4b — Bound.** On full-K voxels, compute the full-space decomposition and its exact coarsened counterpart; verify per-voxel gap ≤ p_OTHER · ln(C_sem − K_top) (violation rate should be 0 to float tolerance), and report the gap distribution and the fraction of voxels with p_OTHER above {0.05, 0.1, 0.25}. Also verify coarsened MI ≤ full MI (safe-direction property for exploration).

Deliverables: Fig: gap vs p_OTHER scatter with bound line; Table: fidelity summary per dataset; one paragraph of paper-ready wording for the C3 claim.

### E5 — Downstream utility (P1)

**E5a — Informative frame selection.** Budget B ∈ {10, 25, 50} of 100 scans (KITTI) / {30, 75, 150} of 300 (SceneNet). Policies: uniform stride, random (5 seeds), greedy by predicted map-level EIG, greedy by semantic MI, frontier count. Metric: occupancy F1 and mIoU of the resulting map vs the full-replay map. Hypothesis: EIG/MI selection dominates random at small B; converges by large B.

**E5b — Predicted vs realised gain.** For a held-out scan stream, correlate per-ray predicted SSMI score (`ssmiOccKL`/`ssmiFreeKL` marginalised along the ray) with realised `betaKL` between pre- and post-integration voxel states. Report Spearman ρ per sequence.

### E6 — Fusion and uncertainty (P1)

Trajectory-partitioned SceneNet setup as before, evaluated in the 100-frame overlap region only. Metrics: ECE/NLL for Robot A, Robot B, and combined; effective-evidence inflation S_combined / max(S_A, S_B). Prediction (stated up front): evidence summing over *duplicated* observations double-counts and worsens calibration in the overlap — quantify it, then show a peer-evidence scaling ablation γ ∈ {1.0, 0.5} mitigates. This converts the prior round's acknowledged independence violation into a measured, mitigated finding rather than a caveat.

### E7 — Recalibration transfer (P1)

Fit a single effective-sample-size temperature (S → S/τ for semantics; s → s/τ for occupancy) on one held-out unit (KITTI seq 06; 3 SceneNet trajectories); evaluate ECE/NLL transfer on the rest. Distinguishes global-scale miscalibration (one-parameter fix, expected because pseudo-counts are κ₀·q·p_occ-weighted rather than unit counts) from structural miscalibration.

### A — Supporting ablations (P2)

A1: κ₀ ∈ {0.5, 2, 8} — calibration and MI magnitude vs κ₀ (effective sample size). A2: S_max ∈ {250, 1000, ∞} — saturation clamps epistemic decay. A3: Hutter floor on/off at query time — effect on ECE and unknown-detection AUROC. A4: soft vs argmax inputs, rescored for *uncertainty* metrics (prior round showed mIoU insensitivity; calibration may differ — that is the interesting question).

---

## 2. Metric definitions (for reproducibility)

ECE with 15 equal-mass bins on top-label confidence (semantics) or p_occ (occupancy); MCE = max bin gap. Brier: mean squared error of the full predictive vector. NLL: −log predictive probability of the reference label. AUSE: normalised area between the model's sparsification curve (error on retained set vs fraction rejected, ranked by the signal) and the oracle curve (ranked by true error). AUROC/AUPR: per-voxel binary tasks as defined in E2. All semantic metrics on the union-support protocol restricted to voxels with a reference label; Unknown handled as in the prior round.

## 3. Validity threats (acknowledge in the paper)

1. Free-space reference is a visibility proxy; occupancy ECE is therefore comparative, not absolute.
2. Dynamic objects contaminate the KITTI occupancy reference; dual reporting per §0.
3. Fractional, quality-weighted pseudo-counts mean the posterior is a tempered-likelihood object, not exact Bayes; E7 measures the practical consequence.
4. "Aleatoric" here absorbs pose error and ray-casting artefacts along with genuine observation noise; E3b isolates the controllable part only.
5. Eviction path-dependence (E4a) means the coarsened partition is per-voxel and history-dependent; claims are phrased over the resident partition at query time.
6. Single-execution variance observed previously on SceneNet motivates the n = 3 rule; where n = 1 remains (ablation tier A), results stay descriptive.

## 4. Priorities and rough budget

| Tier | Experiments | Runs (KITTI + SceneNet) | Notes |
|---|---|---|---|
| P0 | E1, E2, E3, E4 | ≈ (5+13) × 3 principal + 4 noise levels × 13 + subsample checkpoints reuse dumps | E4 reuses existing full-K runs; checkpoints avoid re-runs |
| P1 | E5, E6, E7 | E5a is the heaviest (policies × budgets × seeds); cap at 3 KITTI + 5 SceneNet units | |
| P2 | A1–A4 | anchors only (seq 06/08/10; 0_789/0_723/0_485), n = 1, descriptive | matches prior anchor convention |

All map integration remains CPU-only Release builds; the added cost is dominated by replay count, not per-run compute. P0 is the minimum set that supports the aleatoric/epistemic claim in print.

## 5. Reporting artifacts

Figures: reliability diagrams (occ + sem); sparsification curves; EIG/E_H vs evidence; aleatoric–epistemic vs ε (signature plot); MI vs observation count; truncation gap vs p_OTHER with analytic bound; frame-selection accuracy vs budget. Tables: calibration summary vs baselines; AUROC/AUPR/AUSE matrix; manipulation effect sizes; fidelity/bound summary; overlap-region fusion calibration; recalibration transfer.

**Paper-ready claim (if P0 supports it):** SCovox represents and computes an exact aleatoric/epistemic decomposition over the tracked-classes + OTHER partition (occupancy: closed-form BALD decomposition; semantics: closed-form Dirichlet MI), empirically validated by independent evidence-scaling and noise-injection manipulations, with the full-taxonomy gap bounded per voxel by p_OTHER · ln(C_sem − K_top) and measured to be small at K_top = 2.
