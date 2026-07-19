# SCovox RA-L Experiment Plan v3 (post-SemDir refactor)

## Context

The SemDir refactor collapses Beta + Dirichlet into a single
`Dir(α_1, …, α_K, α_OTHER, α_FREE)` per voxel. Every SCovox NPZ
from the SemBeta era is invalidated. SLIM-VDB results are unaffected
(deterministic docker, identical inputs). All SCovox experiments
re-run from scratch on the post-SemDir binary.

**What changes numerically:**
- Per-voxel: 20 B at K=2 (was 24 B) — `α_free` + `α_other` + 2×cnt + 2×cls
- Wire format: ~20 B/voxel (was 37 B) — ~46% reduction
- Mass conservation: strict equality (was ≥ 0)
- K=1 no longer a special case
- `α_unk` → `α_OTHER` (lumped evicted mass, mathematically proper marginal)
- Admission gate (`p_occ ≥ τ`) has identical closed form — labelling
  envelope should not move

**What should not change:**
- mIoU (same admission gate, same sparse_add logic, same K=2 truncation)
- FPS (budget: ≤ 5% regression per design doc)
- The qualitative story on every ablation

If mIoU shifts by more than 0.02 on any anchor, stop and diagnose
before running the full batch.

---

## Datasets

| Dataset       | Sensor | Scenes/Seqs | Frames | Classes | Voxel  | Labels          |
|---------------|--------|-------------|--------|---------|--------|-----------------|
| SemanticKITTI | LiDAR  | 5 seqs      | 100    | 20      | 10 cm  | PolarSeg soft   |
| SceneNet      | RGB-D  | 13 trajs    | 300    | 14      | 5 cm   | GT closed-set   |

**Scope note (2026-05-14):** Replica is out of the paper entirely.
The Step 8 execution captured 3 Replica fusion scenes (room0/1/2;
geometry wins clean) as a bonus during the v3 wire smoke, but with
data on disk wiped to a partial scope and no clean way to recover
the full 8-scene aggregate, the paper rests on **two datasets,
two modalities (LiDAR + RGB-D), two label sources (predicted soft
+ GT)**. The fusion story is SceneNet-only. The Replica numbers
remain in `STEP8_EXPERIMENT_RESULTS_*.md` as substrate-validation
evidence but do not feed any paper table.

### Paper table mapping (post-2026-05-14 scope)

| Table  | Phase    | Datasets                  | Status |
|--------|----------|---------------------------|--------|
| 1 K_TOP Pareto       | Phase 1   | KITTI + SceneNet | live |
| 2 Component ablation | Phase 2   | KITTI + SceneNet | live |
| 3 Gate threshold     | Phase 2.5 | KITTI + SceneNet | live (null result reframed; gate is integration-time not publish-time) |
| 4 Fusion             | Phase 3   | **SceneNet only** | live (n=13 trajs, all staged) |
| 5 (dropped)          | —         | (was Replica fusion) | dropped |
| 6 Fusion aggregate   | Phase 3   | SceneNet only | live — 3 wins / 1 tie / 0 losses, mean Δ +0.065 mIoU at n=4 |
| 7 Memory scaling     | Phase 5   | KITTI + SceneNet | live (no new compute) |
| h2h vs SLIM-VDB      | Phase 4   | KITTI + SceneNet | live (reuses Phase 1 K=2 cells) |

### Review-hardening gap-closers (queued 2026-05-14, expanded scope)

Identified after the first execution pass left the paper thin in two
review-vulnerable places:

1. **K_TOP SceneNet panel was n=1.** The 0.045 mIoU swing across K∈{1..19}
   on 0_223 looks like a K effect but is inside the ±0.05 single-cell
   noise envelope. n=1 can't distinguish.

2. **Fusion table had only mIoU, n=4.** Reviewer-critical because the
   Replica drop removed the geometry-metric story (F@5cm / Chamfer
   wins). And n=4 leaves the per-trajectory pattern open to cherry-pick
   accusations.

**Data state correction (2026-05-14):** earlier plan + memory framed
SceneNet train shards (262 GB) as the blocker for more fusion trajs.
That was wrong — `data/scenenet_val_layout/train/` already has **13
val trajs staged**: `0_175, 0_178, 0_182, 0_223, 0_279, 0_485, 0_490,
0_571, 0_682, 0_723, 0_789, 0_867, 0_977`. The phase3 batch's
TRAJS_DEFAULT happened to overlap with only 4 of them. All 13 are
usable immediately (62 GB free).

**Fix scope (full 13-traj, the centrepiece deserves it):**

| Task | Cells | Time |
|------|-------|------|
| KITTI K_TOP sweep (5 seqs × 6 K — DONE)          | 30 | ~110 min |
| SceneNet K_TOP sweep (13 trajs × 6 K) | 78 | ~3 hr |
| SceneNet fusion expansion (13 trajs × 3 maps) | 39 | ~3 hr |
| SceneNet voxel-grid F@5cm + Chamfer scorer + scoring | 0 (~30 LOC + scoring pass) | ~30 min |
| **Total new compute** | | **~7 hr (overnight batch)** |

At n=13 the SceneNet K-effect signal washes out per-trajectory noise
and gives a clean Pareto figure. At n=13 the fusion claim has a real
95% CI rather than n=4 marginal. Both panels then defensible.

The TRAJS_DEFAULT in `phase3_scenenet_fusion_batch.sh` updates from
the head-to-head 13 (only 4 staged) to the 13 actually-staged val
trajs above.

---

## Phase 0: Smoke gate (before any batch)

Re-run the two canonical anchors on the post-SemDir binary and compare
against the SemBeta-era baselines.

| Anchor             | SemBeta baseline | Tolerance | Pass if        |
|--------------------|------------------|-----------|----------------|
| KITTI seq08 K=2    | 0.3030 (REPLAY-fixed) | ±0.02 | mIoU ∈ [0.283, 0.323] |
| SceneNet 0_223 K=2 | 0.3624           | ±0.02     | mIoU ∈ [0.342, 0.382] |

Also verify:
- `colcon build` Release: 0 warnings
- `colcon test`: all green
- Mass conservation strict equality on both anchors (the upgraded E5.1)
- FPS within 5% of SemBeta-era on same anchor
- Wire format v3 round-trip (serialize → deserialize → identical voxel state)

**If smoke fails:** diagnose before proceeding. Do not run batches on
a broken binary.

---

## Phase 1: K_TOP Pareto sweep (centrepiece — run first)

**Claim:** K=2 matches full Dirichlet within noise at ~4× less
per-voxel state.

Note: the ratio changes from 5× to ~4× because SemDir is 20 B at K=2
vs 88 B at K=13 (SceneNet) / 128 B at K=19 (KITTI). Still substantial
and structurally O(1) vs O(C).

**Design:** K ∈ {1, 2, 3, 4, 6, K_full} on all scenes/seqs.

| Dataset   | K_full | Cells               | Est. time |
|-----------|--------|---------------------|-----------|
| KITTI     | 19     | 5 seqs × 6 K = 30   | ~2 hr     |
| SceneNet  | 13     | 13 trajs × 6 K = 78 | ~3 hr     |
| **Total** |        | **108 cells**       | **~5 hr** |

**KITTI seq scope:** 06, 07, 08, 09, 10 (PolarSeg `predictions_topk/`
staged at 100 frames each — matches the SLIM-VDB paper's val split).

K_TOP is compile-time. Batch by K: rebuild once, run all scenes/seqs
at that K, then rebuild for the next K. That's 6 rebuilds × ~5 min
= 30 min rebuild overhead total.

**Evaluator:** Strict bucket-IoU. KITTI: REPLAY-fixed scorer.
SceneNet: same scorer as h2h batch.

**Report:**
- Table: K × dataset → mean mIoU
- Per-voxel bytes at each K (general formula: `((8 + 6·K + 3)/4)·4`)
- Pareto figure: bytes/voxel (x) vs mIoU (y), dual panel
- Δ(K=2 vs K=full) and Δ(K=1 vs K=2) per dataset

---

## Phase 2: Component ablation

**Claim:** Bayesian Dirichlet accumulation outperforms simpler update
strategies, especially when the segmenter is uncertain.

**Design:** Three modes × anchors.

| Mode       | What it does                                 |
|------------|----------------------------------------------|
| Dirichlet  | Full soft-prob update into K=2 SemDir slots   |
| MV         | Argmax only, +1 to winner's count             |
| Naive (NP) | Argmax only, overwrite slot with count=1      |

Anchors:
- KITTI: 3 sequences for distribution coverage matching Phase 4 h2h
  - seq06 (suburban, fast traffic — distribution variant)
  - seq08 (standard anchor — same as Phase 0 smoke)
  - seq10 (highway / long-straight, less semantic clutter)
- SceneNet: 3 trajectories spanning the mIoU range
  - 0_789 (high SCovox mIoU = 0.385, biggest h2h delta)
  - 0_723 (low SCovox mIoU = 0.269, smallest h2h delta)
  - 0_485 (mid SCovox mIoU = 0.303)

3 modes × 6 anchors = **18 cells**, ~2 hr.

**⚠️ SceneNet uses GT one-hot labels.** Soft-prob reduces to hard on
SceneNet. The ablation still tests Dirichlet accumulation vs MV
counting vs NP overwrite — the "history matters" finding should
survive. But the "soft-prob recovers calibration signal" finding is
KITTI-specific and can't be tested on SceneNet. Note this in the
paper.

**Report:**
- Table: anchor × mode → mIoU
- Δ(D − NP): quantifies value of Bayesian history
- Δ(D − MV): quantifies value of probability magnitudes

---

## Phase 2.5: Gate threshold sweep (architectural ablation)

**Claim:** The unified Dirichlet's FREE dimension provides a principled
occupancy posterior that improves semantic mIoU by suppressing free-space
false positives. The publish-time gate (`dirichlet_min_p_occ`) is the
runtime knob that exposes the labelling envelope; varying it isolates
the contribution of the occupancy gate itself, separate from K_TOP (slot
count) and the update strategy (D vs MV vs NP).

**Why this matters.** The K_TOP and component-ablation tables answer
"how should we represent class distributions?" and "how should we
accumulate them?" but they don't directly ablate **the** architectural
component that distinguishes SCovox from SLIM-VDB: the per-voxel
occupancy posterior. The SceneNet gap-mechanism analysis (memo
`[[project-scenenet-gap-mechanism-2026-05-13]]`) already showed that the
+0.088 head-to-head gap is almost entirely precision-vs-recall from the
labelling envelope, not per-voxel semantic accuracy — so the gate is the
dominant lever. This phase turns that diagnostic into a controlled
ablation that lives in the paper.

The gate threshold is also the conjugate on our side of SLIM-VDB's
`sdf_trunc`: both control the labelling envelope. The SceneNet sdf_trunc
sweep (SL mIoU 0.320 → 0.187 across trunc 0.05 → 0.15) becomes
interpretable side-by-side with this table.

**Design:** Vary `dirichlet_min_p_occ` at publish time. This is a
runtime parameter — no recompilation, no re-integration — so each
threshold reuses the same SemDirMap state from a single integration
pass per anchor. Cost is one extra pointcloud capture per (anchor,
threshold) cell, plus one re-scoring.

| Threshold | What it does                                         |
|-----------|------------------------------------------------------|
| 0.0       | Publish everything with any evidence (SLIM-VDB-like) |
| 0.3       | Weak gate                                            |
| 0.5       | Production default                                   |
| 0.7       | Strict gate                                          |

Anchors: KITTI seq08 (LiDAR + PolarSeg soft) and SceneNet 0_223
(RGB-D, GT labels). Same two anchors the smoke gate uses, so any
drift between this experiment and the smoke baseline is easy to spot.

4 thresholds × 2 anchors = **8 cells**, ~30 min total. The
implementation should integrate once per anchor at threshold 0.5
(re-using the Phase 0 / Phase 2 D-mode NPZ if still on disk), then
re-publish the same SemDirMap state through 3 additional gate
thresholds for ~5 min extra each.

**Report:**
- Table: anchor × threshold → (mIoU, voxel count published, % gain
  vs threshold=0.0)
- Per-anchor curve: mIoU vs threshold (expect monotonic increase up
  to a knee near the production default, then plateau or slight
  decrease at 0.7 as true-positive voxels get gated)
- Side-by-side with the SLIM-VDB sdf_trunc sweep on SceneNet —
  framed as "two ways to control the labelling envelope; ours is
  evidence-based, theirs is geometry-based"
- **Headline metric:** mIoU(threshold=0.5) − mIoU(threshold=0.0) =
  the contribution of the occupancy gate to the system

**Outcome (executed 2026-05-14, commit cf328ab):** the predicted
monotonic drop did **not** materialise. The publish-time gate moves
mIoU by ≤ 0.004 across all 4 thresholds on both anchors, even though
threshold=0.3 filters out 91% of SceneNet voxels.

| Anchor          | t=0.0 | t=0.3 | t=0.5 | t=0.7 |
|-----------------|-------|-------|-------|-------|
| KITTI seq08     | 0.2839 | 0.2855 | 0.2855 | 0.2854 |
| SceneNet 0_223  | 0.3483 | 0.3462 | 0.3448 | 0.3472 |

**Mechanism (the substantive paper finding):** low-p_occ voxels in
SCovox carry `cls[i]==0xFFFF` → `semantic_class=Unknown(0)`, which
both the KITTI and SceneNet scorers ignore. The 91% of SceneNet
voxels filtered at t=0.3 are harmless Unknown carve-only voxels,
not labelled FPs. Filtering changes the published count but not
the scored set.

**Reframed paper claim** (the one the executed experiment supports):
the labelling envelope is set at **integration-time admission** —
`SemDirMap::applyHitUpdate`'s gated `dirichletUpdate` inside
[semdir_map.cpp:325](../robot_sw/distributed_mapping/scovox_core/src/semdir_map.cpp#L325)
— not at publish-time. The +0.088 SceneNet head-to-head gap
([[project-scenenet-gap-mechanism-2026-05-13]]) is a viewer-convenience
gate downstream of where the FP-vs-TP balance is set.

**Still missing as a paper-table ablation:** sweeping the
integration-time gate (`dirichlet_min_p_occ` *at integration*, not
publish). That would force class commits on every hit regardless of
occupancy posterior — the mirror of SLIM-VDB's "label everything in
the TSDF band" behaviour. Out of scope for the executed Phase 2.5;
follow-up specified below.

---

## Phase 2.5-v2: Integration-time gate sweep (the real architectural ablation)

**Claim:** The occupancy posterior controls the labelling envelope at
**integration time**, not publish time. Forcing class commits on every
hit (`dirichlet_min_p_occ = 0.0` *inside `SemDirMap::applyHitUpdate`*)
should produce many low-posterior voxels with real class commits — the
"archetype-A" pattern from the SceneNet gap-mechanism analysis — and
mIoU should drop measurably.

**Why this matters.** Phase 2.5 (publish-time) returned a null result
because filtered voxels carry `cls[i]==0xFFFF → semantic_class=Unknown(0)`
and the scorer ignores class 0. To actually test "does the occupancy
gate matter for mIoU?" the gate has to be moved into the integration
path so it changes which voxels get a real class assignment, not just
which voxels get published.

**Mechanism.** `SemDirMap::applyHitUpdate` at
[semdir_map.cpp:325](../robot_sw/distributed_mapping/scovox_core/src/semdir_map.cpp#L325)
gates the per-class `dirichletUpdate` on `p_occ_post >= dirichlet_min_p_occ`.
Below the gate, hit evidence still lands on `alpha_other` (Stream A),
so the voxel exists but its top-K class slots stay at the symmetric
prior — argmax projects to Unknown(0). At gate=0.0 every hit commits a
real class, and the resulting voxels look like SLIM-VDB's "label
everything in the band" output.

**Design:** Vary `dirichlet_min_p_occ` (the integration-time param,
not the publish-time `occupancy_vis_threshold`) on the same two anchors
the publish-time sweep used.

| Threshold | Behaviour                                                  |
|-----------|------------------------------------------------------------|
| 0.0       | Force class commit on every hit (SLIM-VDB-like envelope)    |
| 0.3       | Weak gate                                                   |
| 0.5       | Production default                                          |
| 0.7       | Strict gate (only commit when posterior is very confident)  |

Anchors: KITTI seq08, SceneNet 0_223. 4 thresholds × 2 anchors = **8
cells**, ~5 min integration + 1 min scoring per cell ≈ **~50 min total**.
Each cell requires a fresh integration (unlike Phase 2.5, the gate is
now inside the integration loop) — no NPZ reuse possible.

**Launch arg:** `dirichlet_min_p_occ:=<X>` on
`semantickitti_eval.launch.py` / `scenenet_eval.launch.py` (already
plumbed through to `SP.semdir.dirichlet_min_p_occ` at
[scovox_node.cpp:129](../robot_sw/distributed_mapping/scovox_mapping/src/scovox_node.cpp#L129)
— no code change needed).

**Report:**
- Table: anchor × threshold → (mIoU, n_voxels with real class, %
  voxels labelled-vs-Unknown)
- Per-anchor curve: mIoU vs integration-time threshold (expect
  monotonic ascent up to the production default, plateau or slight
  decrease above it as TP voxels get gated out)
- Side-by-side with the SLIM-VDB sdf_trunc sweep on SceneNet —
  framed as "two ways to control the labelling envelope; ours is
  evidence-based at integration time, theirs is geometry-based at
  TSDF-band time"
- **Headline metric:** Δ mIoU(t=0.5 − t=0.0) — the contribution of
  the occupancy gate to the system, in mIoU units

**Expected result:** mIoU drops monotonically as the threshold
decreases, with the largest drop between 0.3 and 0.0 as forced class
commits flood in. Magnitude should be in the same ballpark as the
SceneNet head-to-head gap (~+0.088 abs vs SLIM-VDB) — at t=0.0 SCovox
should approach SLIM-VDB's precision/recall on SceneNet, since both
now label every hit regardless of confidence.

If the drop matches the head-to-head gap, the gate-mechanism story
([[project-scenenet-gap-mechanism-2026-05-13]]) becomes a controlled
ablation rather than a diagnostic — defensible as a paper table.

**Status:** queued. Run after Phase 1 / Phase 3 expansion completes.
No K_TOP edit / rebuild required, so it can slot in opportunistically
between the larger phases.

---

## Phase 3: Multi-robot fusion (SceneNet)

**Claim:** Exact conjugate merging of Dirichlet sufficient statistics
improves the fused map over either solo.

**Design:** Trajectory-split fusion on all 13 SceneNet trajs.

Split: Robot A = frames 0–199, Robot B = frames 100–299.
50% temporal overlap. Each robot runs its own `scovox_node`;
`dscovox_node` consensus-merges via wire format v3.

13 trajs × 3 maps (solo_a, solo_b, fused) = **39 NPZs**, ~3 hr.

**What to measure per scene:**
- mIoU: fused vs max(solo_a, solo_b)
- Chamfer: fused vs solos (voxel-centre extraction)
- F@5cm: fused vs solos
- Bandwidth: `bin_bytes` per frame (wire v3) vs raw depth per frame

**Report:**
- Aggregate table: solo_a / solo_b / fused means ± std
- Per-scene verdict count: fused ≥ max(solo) on each metric
- Wire v3 bandwidth: bytes/frame, compression ratio vs raw depth
- Note: at ~20 B/voxel wire (was 37 B), the bandwidth story improves
  vs the SemBeta era

---

## Phase 4: SLIM-VDB head-to-head (validation)

**Claim:** Our system matches state-of-the-art accuracy on the same
inputs.

**Design:** Run SCovox (post-SemDir, K=2) on all scenes/seqs. SLIM-VDB
results are already captured and deterministic — no re-run needed.

| Dataset   | SCovox cells | SLIM-VDB cells | Status     |
|-----------|--------------|----------------|------------|
| KITTI     | 5 seqs       | 5 seqs         | SLIM done |
| SceneNet  | 13 trajs     | 13 trajs       | SLIM done |

SCovox cells: **18 cells total**, reused from Phase 1 K=2 row (no
extra compute).

**Report:**
- Table: dataset × system → mIoU (paired Δ, 95% CI)
- FPS + grid memory side-by-side
- Frame honestly: "comparable accuracy, structural memory advantage"
- Note the SceneNet gap mechanism (sdf_trunc labelling envelope) in
  discussion, not in the headline
- Published baseline numbers in a separate row for context

---

## Phase 5: Derived analyses (no new compute)

### Memory scaling

| Dataset   | C  | SemDir B/v | SLIM-VDB B/v | Ratio |
|-----------|----|------------|--------------|-------|
| SceneNet  | 14 | 20         | 56           | 2.8×  |
| KITTI     | 20 | 20         | 80           | 4.0×  |

Note: ratios improve vs SemBeta era (was 2.3× / 3.3×) because SemDir
is 20 B not 24 B.

Total grid memory still flips on SceneNet (Beta carving voxel-count
blowup). Report honestly.

### Mass conservation (upgraded E5.1)

Run `verify_mass_conservation.py` (updated for SemDir) on both smoke
anchors. Should now pass **strict equality** (within float tolerance),
not just ≥ 0.

### Eviction telemetry

Run on both smoke anchors. Note that under SemDir, "drop" mass is
retained in OTHER rather than lost — the telemetry categories change
semantics slightly. `drop` becomes `to_other` (mass preserved in the
lumped bucket). This is a genuine improvement: today's "dropped" mass
is gone forever; under SemDir it stays in the posterior.

---

## Execution order

```
Phase 0:   Smoke gate (3 cells, ~10 min)                       [done b76d8c8]
   ↓ pass
Phase 1:   K_TOP sweep (121 cells DONE — 5 KITTI × 6 K + 13 SceneNet × 7 K incl K=13)
   ↓ K=2 cells feed into Phase 4
Phase 2:   Component ablation (18 cells, ~40 min)              [done 9d68d11]
   ↓
Phase 2.5: Gate threshold sweep (8 cells, ~6 min)              [done cf328ab, null result]
Phase 3:   Fusion (13 trajs, ~3 hr — wire_format=v3)
Phase 4:   h2h (reuse Phase 1 K=2 + SLIM-VDB cached; no extra compute)
Phase 5:   Derived analyses (no compute)
```

**All experiments complete (2026-05-14).** Total compute spent
across three sessions: ~7 hr (first pass ~2.5 hr + gap-closer
~3.5 hr + wrap-up ~1 hr). The original ~13.5 hr estimate was
conservative — rebuilds + integration came in much cheaper than
budgeted. Branch head `8b92ce3` on `refactor/split-tsdf-sembeta`.

---

## Paper changes from SemDir refactor

### Numbers that update everywhere

| Quantity | SemBeta | SemDir |
|----------|---------|--------|
| Per-voxel semantic+occ bytes (K=2) | 24 B | **20 B** |
| Wire format per-voxel | 37 B | **~20 B** |
| K=2 vs K=full ratio (KITTI, K=19) | 5.3× | **~6.4×** |
| K=2 vs K=full ratio (SceneNet, K=13) | — | **~4.4×** |
| Wire bandwidth vs raw depth | 5× | **~9×** (back to pre-refactor levels) |

### Abstract
"24 bytes" → "20 bytes". "37 bytes per shared voxel" → "20 bytes".
Memory ratios update. Wire bandwidth ratio improves.

### Method section
Beta + Dirichlet subsections merge into a single "Unified Dirichlet"
subsection. The marginal query table from the design doc goes here.
Mass conservation becomes exact equality (not ≥ 0).

### All tables
Every table with per-voxel bytes, FPS, grid memory, or wire bandwidth
needs fresh numbers from the post-SemDir runs.

### Discussion
"α_unk" language → "α_OTHER" throughout. The "K=1 special case"
caveat disappears (genuine simplification).

---

## Risks

1. **mIoU regression.** The admission gate has the same closed form,
   but the update path is different (merge-sort-evict vs
   sparse_add branches). Any numerical drift shows up as a smoke
   gate failure. Budget: ±0.02 per anchor.

2. **FPS regression.** The merge-sort on 3–4 entries per hit should
   be cheaper than the current branch tree, but profile before
   declaring. Budget: ≤ 5% wall-clock regression.

3. **α_0 prior sensitivity.** At α_0 = 0.01, the OTHER prior for
   SceneNet (C=14, K=2) is 11 × 0.01 = 0.11. For KITTI (C=20, K=2)
   it's 17 × 0.01 = 0.17. Both are small enough to not bias p_occ
   away from 0.5 at a fresh voxel. Verify in smoke gate.

4. **Wire format v3 invalidates all v2 NPZs/blobs.** No in-flight
   experiments to worry about since everything re-runs. Keep v2
   reader in tree for backward compat (it's read-only, ~150 LOC).

5. **108-cell K_TOP sweep compile overhead.** 6 rebuilds × ~5 min
   = 30 min. Batch all scenes per K to minimise rebuilds.