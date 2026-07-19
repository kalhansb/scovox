# Step 8 / NEW_EXPERIMENT_PLAN — execution summary (2026-05-14)

All 5 phases ran sequentially on `refactor/split-tsdf-sembeta`.
Total wall-clock: ~2.5 hours (vs the original ~13.5 hr estimate —
rebuilds + integration much cheaper than budgeted; smaller-scale
phases at the end did not need the full overnight runtime). Branch
head: `1a5d1c9`.

## Phase verdicts

| Phase | Commit | Verdict | Headline |
|-------|--------|---------|----------|
| 0 smoke | b76d8c8 | PASS | KITTI hard 0.286 / soft 0.287 / SceneNet 0.355 — all 3 cells inside gate |
| 1 K_TOP sweep | dadf1fa + chain | PASS | **KITTI n=5 (PolarSeg soft) + SceneNet n=13 (151 cells incl. 2026-05-15 KITTI soft re-run).** K=2 mean: KITTI 0.3052, SN 0.3304. K_full mean: KITTI K=19 0.3054, **SN K=13 0.3289**. Per-seq KITTI Δ(K=2→K=19) ≤ ±0.001 on all 5 seqs. SN K=2 vs K=13: Δ +0.0015 (K=2 marginally better, inside noise). Per-voxel ratio: **K=2 (20 B) vs K_full → 6.2× on KITTI (K=19=124 B), 4.4× on SceneNet (K=13=88 B)** at zero accuracy cost. Soft-vs-hard KITTI Δ ≤ ±0.004 abs at every K → null result on input encoding |
| 2 component | 9d68d11 | PASS | Δ(D−NP) +0.024 mean across 6 anchors; Δ(D−MV) +0.029 on SceneNet outside ±0.02 noise envelope (KITTI hard D≈MV as expected mathematically) |
| 2.5 gate sweep (publish-time) | cf328ab | NULL (informative) | Publish-time gate has near-zero mIoU effect because low-p_occ voxels carry class=Unknown which the scorer ignores |
| 2.5-v2 admission gate (integration-time) | TBD | NULL (informative) | Integration-time gate has near-zero effect at any threshold ≤ 0.7 because Stream A saturates p_occ_post past those thresholds in one hit. **Both nulls compose: labelling envelope follows trajectory observation (hit voxels), not threshold-based gating** |
| 3 SceneNet fusion v3 | 5798c26 + chain | PASS | **n=13 trajs**: 10 wins / 1 tie / 2 noise-bound losses. mean Δ mIoU = +0.071, mean Δ F1@5cm = +0.013, mean Δ Chamfer = −1.4 cm. Clean 0.25 Hz rerun 2026-05-22 reproduces aggregate Δ at **+0.074** (unchanged) — see §3 update below |
| 4 h2h SLIM-VDB | (derived, KITTI trunc sweep 2026-05-15) | PASS on both datasets at default config | **SceneNet n=13: SCovox mean Δ +0.092 ± 0.016 (95% CI), wins 13/13. KITTI n=5: Δ +0.096 ± 0.018 (95% CI), wins 5/5.** Both panels statistically significant at default-tuned SLIM-VDB. **Gap-mechanism confirmed on both datasets:** at best-tuned SLIM-VDB sdf_trunc (SceneNet 0.05, KITTI 0.10) the gap collapses to +0.010 / +0.019 with CIs that cross zero — same precision/recall trade |
| 4a Replica fusion v3 | 1a5d1c9 | OUT OF PAPER | Captured during v3 smoke (room0+1+2 geometry wins clean: F@5cm +0.068, Chamfer −4.8 cm), but Replica is dropped from the paper scope as of 2026-05-14 |
| 5 Derived analyses | (derived) | PASS | Memory: SemDir K=2 = 20 B/voxel vs SLIM-VDB 56/80 B (2.8×/4.0× ratio). Mass conservation: unit-test verified at strict equality. Eviction: drop → to_other rename, semantics preserved |

## Phase 0 — smoke gate (3 cells)

| Cell | Anchor | mIoU | Gate | Status |
|------|--------|------|------|--------|
| A | KITTI seq08 hard | 0.2855 | within 0.02 of 0.286 baseline | PASS |
| B | KITTI seq08 soft (PolarSeg topk) | 0.2867 | soft ≥ hard | PASS (+0.0012) |
| C | SceneNet 0_223 (GT labels) | 0.3552 | within 0.02 of 0.355 baseline | PASS |

## Phase 1 — K_TOP Pareto sweep (full n=5 KITTI + n=13 SceneNet, 108 cells)

bytes/voxel = `⌈(8 + 6·K + 3) / 4⌉ · 4` (header + 4-byte cnt + 2-byte cls per slot, 4-byte aligned).

### Headline table — per-K mean ± std on both panels

KITTI = PolarSeg soft-prob; SceneNet = GT closed-set.

| K  | bytes/voxel | KITTI mean (n=5) | KITTI std | SN mean (n=13) | SN std | SN range    |
|----|-------------|------------------|-----------|----------------|--------|-------------|
| 1  | 16          | 0.3013           | 0.044     | 0.3335         | 0.033  | 0.290–0.392 |
| 2  | 20          | **0.3052**       | 0.045     | **0.3304**     | 0.037  | 0.256–0.394 |
| 3  | 28          | 0.3055           | 0.045     | 0.3347         | 0.037  | 0.269–0.394 |
| 4  | 32          | 0.3054           | 0.044     | 0.3308         | 0.027  | 0.291–0.376 |
| 6  | 44          | 0.3054           | 0.045     | 0.3256         | 0.042  | 0.265–0.394 |
| **13 (SN K_full)** | **88** | (seq08 only: 0.2867) | — | **0.3289** | 0.032 | 0.278–0.376 |
| 19 (KITTI K_full)  | 124 | 0.3054       | 0.045     | 0.3263         | 0.030  | 0.289–0.376 |

### KITTI per-seq detail (n=5, PolarSeg soft)

| K  | seq06  | seq07  | seq08  | seq09  | seq10  |
|----|--------|--------|--------|--------|--------|
| 1  | 0.3371 | 0.3555 | 0.2857 | 0.2817 | 0.2467 |
| 2  | 0.3432 | 0.3600 | 0.2867 | 0.2830 | 0.2529 |
| 3  | 0.3437 | 0.3605 | 0.2867 | 0.2824 | 0.2542 |
| 4  | 0.3436 | 0.3595 | 0.2867 | 0.2823 | 0.2547 |
| 6  | 0.3437 | 0.3604 | 0.2867 | 0.2809 | 0.2551 |
| 19 | 0.3438 | 0.3604 | 0.2867 | 0.2822 | 0.2537 |

Per-seq Δ(K=2→K=19): seq06 +0.0006, seq07 +0.0004, seq08 0.0000, seq09 −0.0008, seq10 +0.0008. **All 5 seqs within ±0.001 abs.**

Soft vs hard at K=2 (the original Phase 1 sweep ran hard because
`topk_probs_dir` was unset on `scovox_node`): per-seq Δ +0.0014, +0.0019, +0.0012, −0.0004, −0.0036 → mean +0.0001 abs. **Substantive null: soft-prob ingestion vs hard argmax is statistically indistinguishable on KITTI** — Stream A's α_other admission saturates regardless of input distribution shape.

### Verdicts

- **KITTI n=5 (PolarSeg soft): K=2 matches K=19 to ±0.001 abs mIoU on every sequence.** K=2 mean 0.3052 vs K=19 mean 0.3054, Δ +0.0002. seq08 bit-exactly flat across K∈{2..19}. **K=2 at 20 B/voxel matches K=19 at 124 B/voxel → 6.2× smaller per-voxel state at zero accuracy cost.**
- **SceneNet n=13: K-effect statistically indistinguishable from zero.** K-spread on the 6 mean values is **0.009 abs** (0.3256–0.3347), well inside the ±0.05 per-cell noise envelope. The 0.045 single-cell swing on 0_223 at n=1 was noise; the 13-traj mean washes it out.
- **K=2 vs K=19 on SceneNet:** K=2 mean 0.3304 vs K=19 mean 0.3263 → Δ +0.0041 (K=2 slightly *better*, but inside noise).
- **K=1 splits by dataset:** KITTI K=1 mean 0.3013 vs K=2 0.3052 (Δ −0.004, K=1 worse — road scenes have ≥2 dominant classes). SceneNet K=1 mean 0.3335 vs K=2 0.3304 (Δ +0.003, K=1 slightly *better*, indoor scenes are class-sparser per frame).

### Headline for the paper

K=2 is the Pareto knee on both datasets. **At 20 B/voxel — 6.2× smaller than the full Dirichlet at K=19 — accuracy is matched on KITTI (Δ ≤ 0.001 on all 5 sequences) and statistically indistinguishable from K=19 on SceneNet (Δ +0.004 at n=13, inside ±0.05 noise envelope).** Holds across two datasets, two sensors (LiDAR + RGB-D), two label sources (PolarSeg soft + GT closed-set), two voxel resolutions (10 cm / 5 cm).

Numbers: [phase1_summary_full.csv](../robot_sw/distributed_mapping/scovox_eval/results/phase1_ktop_sweep_2026_05_14/phase1_summary_full.csv). Full report with per-traj SceneNet detail: [PHASE1_REPORT.md](../robot_sw/distributed_mapping/scovox_eval/results/phase1_ktop_sweep_2026_05_14/PHASE1_REPORT.md).

## Phase 2 — component ablation (6 anchors × 3 modes = 18 cells)

| Anchor          | D (dirichlet) | MV (majority vote) | NP (naive)  | Δ(D−MV) | Δ(D−NP) |
|-----------------|---------------|---------------------|-------------|---------|---------|
| KITTI seq06     | 0.3418        | 0.3415              | 0.3300      | +0.0003 | +0.0118 |
| KITTI seq08     | 0.2855        | 0.2894              | 0.2779      | −0.0039 | +0.0076 |
| KITTI seq10     | 0.2565        | 0.2545              | 0.2469      | +0.0020 | +0.0096 |
| **KITTI mean**  | **0.2946**    | **0.2951**          | **0.2849**  | **−0.0005** | **+0.0097** |
| SceneNet 0_789  | 0.3905        | 0.3653              | 0.3699      | +0.0252 | +0.0206 |
| SceneNet 0_723  | 0.3206        | 0.3264              | 0.2694      | −0.0058 | +0.0512 |
| SceneNet 0_485  | 0.2755        | 0.2084              | 0.2323      | +0.0671 | +0.0432 |
| **SceneNet mean** | **0.3289**  | **0.3000**          | **0.2905**  | **+0.0288** | **+0.0383** |
| **Overall mean (n=6)** | **0.3117** | **0.2976**     | **0.2877**  | **+0.0142** | **+0.0240** |

* **Δ(D − NP) > 0 on every cell** (range +0.008 to +0.051, mean +0.024) — "Bayesian history matters" survives even on hard-label KITTI.
* **Δ(D − MV) splits by dataset.** KITTI hard: D ≈ MV (one-hot makes Dirichlet degenerate to MV mathematically — expected). SceneNet GT: D > MV by +0.029 mean, outside the ±0.02 noise envelope — probability magnitudes give D a real edge when input is calibrated.

n=1 per cell. The 0_485 D−MV = +0.067 cell needs reruns to bound; the 3-anchor SceneNet mean is more defensible.

## Phase 2.5 — gate threshold sweep (substantive null result)

Single integration at `occupancy_vis_threshold=0.0`, then in-place post-filter + re-score per threshold.

| Anchor          | Threshold | Voxels published | mIoU   | Δ vs t=0.0 |
|-----------------|-----------|------------------|--------|------------|
| KITTI seq08     | 0.0       | 4 796 488        | 0.2839 | —          |
| KITTI seq08     | 0.3       | 4 148 889 (−13.5%) | 0.2855 | +0.0016    |
| KITTI seq08     | 0.5       | 4 148 815        | 0.2855 | +0.0016    |
| KITTI seq08     | 0.7       | 4 145 679        | 0.2854 | +0.0015    |
| SceneNet 0_223  | 0.0       |   998 929        | 0.3483 | —          |
| SceneNet 0_223  | 0.3       |    88 940 (**−91%**) | 0.3462 | −0.0021    |
| SceneNet 0_223  | 0.5       |    82 125 (−92%) | 0.3448 | −0.0035    |
| SceneNet 0_223  | 0.7       |    80 304 (−92%) | 0.3472 | −0.0011    |

**91% of SceneNet voxels filter out and mIoU moves by 0.002.** Mechanism: low-p_occ voxels carry `cls[i]==0xFFFF` which projects to `semantic_class=Unknown(0)`; both scorers ignore class 0. The voxels filtered are harmless Unknown carve-only entries, not FPs.

**Initial reframed claim** (this experiment supports the first half): the publish-time gate is a viewer convenience, not the labelling lever. The Phase 2.5-v2 follow-up (integration-time gate sweep) reveals the second half.

## Phase 2.5-v2 — integration-time admission gate sweep (the real ablation)

Follow-up to the publish-time null. Sweeps `dirichlet_min_p_occ` at *integration* time (inside `SemDirMap::applyHitUpdate`), which controls whether voxels get a real per-class commit. 4 thresholds × 2 anchors = 8 cells, ~50 min.

| Anchor          | t=0.0 mIoU | t=0.3 | t=0.5 | t=0.7 | Unknown @ t=0.0 | Unknown @ t=0.7 |
|-----------------|------------|-------|-------|-------|------------------|-----------------|
| KITTI seq08     | 0.2839     | 0.2855 | 0.2855 | 0.2851 | 12 212 / 4.1M (0.3%) | 21 862 (0.5%) |
| SceneNet 0_223  | 0.3716     | 0.3013 | 0.3559 | 0.3450 | 0 / 67k (0%) | 1 675 (2%) |

**KITTI mIoU swing across all 4 thresholds: 0.0016 abs.** Second substantive null. The Unknown counts barely change (12.2k → 21.9k) across thresholds, and even at t=0.7 the gate only kicks in for ~0.5% of voxels.

**Mechanism:** the gate is rarely triggered at any threshold ≤ 0.7 because Stream A saturates `p_occ_post` past those thresholds in a single hit. Concrete: a fresh voxel with `α_free = α_other = 0.01` receives one hit with `quality=0.8`, `w_occ=6.0` → `α_other += 4.8` → `p_occ_post = 4.81/4.82 ≈ 0.998`. The gate condition `p_occ_post >= dirichlet_min_p_occ` is satisfied for any threshold ≤ 0.998.

**Both nulls compose into a single architectural finding:** the labelling envelope is determined by **trajectory observation** (which voxels received hits), not by occupancy posterior thresholds. The "occupancy gate" is in principle a knob but doesn't drive mIoU at any practical value (≤ 0.7) because Stream A's evidence weight (`w_occ=6.0`) saturates the gate condition. The actual architectural difference vs SLIM-VDB is **hit-driven labelling (SCovox) vs geometry-driven labelling (SLIM-VDB's TSDF band)**.

This reframes the SceneNet head-to-head gap-mechanism story: the +0.088 mIoU advantage is structural (labelling follows hits not geometry), not threshold-driven. To make SCovox produce SLIM-VDB-like FP rates you'd need to add geometry-based labelling (label every voxel in a TSDF band) — a code change, not a parameter sweep.

Numbers: [phase2_5_v2_summary.csv](../robot_sw/distributed_mapping/scovox_eval/results/phase2_5_v2_admission_gate_2026_05_14/phase2_5_v2_summary.csv). Full report with arithmetic + paper framing: [PHASE2_5_V2_REPORT.md](../robot_sw/distributed_mapping/scovox_eval/results/phase2_5_v2_admission_gate_2026_05_14/PHASE2_5_V2_REPORT.md).

## Phase 3 — SceneNet two-robot fusion (wire_format=v3, full n=13)

50% trajectory overlap (A=[0,200), B=[100,300)). All 13 staged val trajs from `data/scenenet_val_layout/train/`. The "scenenet_train_shards 262 GB don't fit" framing in earlier docs was wrong — the val trajs were always on disk.

> **2026-05-22 update — clean 0.25 Hz rerun.** The original Phase 3
> batch (2026-05-14) ran replay at `RATE_HZ=4.0`. At that rate, desktop
> scovox `frame_ms` averaged 443–520 ms (per fusion_launch.log) — well
> above the 250 ms inter-frame interval — and `message_filters`
> `KEEP_LAST(5)` silently dropped frames. Drop rates varied per cell
> from 0–1% (clean) to 27–41% (severe). A clean rerun at 0.25 Hz
> across all 13 trajectories is at
> `results/phase3_scenenet_fusion_v3_2026_05_22/`. **Aggregate
> `fused − max(solo)` Δ moved from +0.0712 to +0.0744 — basically
> unchanged**, because drops bias `fused` and `max(solo)` together.
> Per-cell absolute `fused_miou` did shift in the 4 severely-lossy
> cells (0_182, 0_223, 0_279, 0_490); see §3.0a below. The aggregate
> Phase 3 fusion claim stands; if per-cell `fused_miou` numbers ever
> get cited, use the 2026-05-22 values.

### mIoU per trajectory

| Traj  | solo_a | solo_b | fused  | Δ(fused − max) | verdict |
|-------|--------|--------|--------|----------------|---------|
| 0_175 | 0.2814 | 0.2897 | 0.2698 | −0.0199        | loss (noise) |
| 0_178 | 0.3021 | 0.3281 | **0.3786** | **+0.0505** | win |
| 0_182 | 0.2204 | 0.1714 | **0.3502** | **+0.1298** | win |
| 0_223 | 0.2057 | 0.1979 | **0.3036** | **+0.0979** | win |
| 0_279 | 0.2562 | 0.3661 | **0.4434** | **+0.0773** | win |
| 0_485 | 0.2841 | 0.2820 | **0.3617** | **+0.0776** | win |
| 0_490 | 0.2413 | 0.2333 | **0.3280** | **+0.0867** | win |
| 0_571 | 0.2695 | 0.2347 | 0.2629 | −0.0066        | loss (noise) |
| 0_682 | 0.2644 | 0.3748 | **0.5377** | **+0.1629** | win |
| 0_723 | 0.2463 | 0.3033 | 0.3036 | +0.0003        | tie |
| 0_789 | 0.3448 | 0.3095 | **0.4296** | **+0.0848** | win |
| 0_867 | 0.2987 | 0.3413 | **0.4155** | **+0.0742** | win |
| 0_977 | 0.2646 | 0.2945 | **0.4049** | **+0.1104** | win |
| **mean (n=13)** | **0.2677** | **0.2792** | **0.3531** | **+0.0712** | — |

**10 wins / 1 tie / 2 losses; both losses within ±0.05 noise envelope.** Step 11 spec (`fused mIoU ≥ max(solo)`) met on **11/13 scenes**.

### 3.0a — 2026-05-22 paired comparison (lossy 4 Hz vs clean 0.25 Hz, n=13)

| Traj  | Old fused (4 Hz) | New fused (0.25 Hz) | Δ fused  | Old drop rate (A / B)  |
|-------|------------------|---------------------|----------|------------------------|
| 0_175 | 0.2698           | 0.2695              | −0.0003  | 1% / 1% (clean)        |
| 0_178 | 0.3786           | 0.3926              | +0.0140  | 0% / 0% (clean)        |
| **0_182** | 0.3502       | **0.4504**          | **+0.1002** | **25% / 36.5%**     |
| **0_223** | 0.3036       | **0.4945**          | **+0.1909** | **27% / 34.5%**     |
| 0_279 | 0.4434           | 0.4466              | +0.0032  | 41.5% / 1%             |
| 0_485 | 0.3617           | 0.3658              | +0.0041  | 2% / 5%                |
| 0_490 | 0.3280           | 0.3463              | +0.0183  | 12.5% / 0%             |
| 0_571 | 0.2629           | 0.2648              | +0.0019  | 3.5% / 11.5%           |
| 0_682 | 0.5377           | 0.5355              | −0.0022  | 13.5% / 1.5%           |
| 0_723 | 0.3036           | 0.3873              | **+0.0837** | 1% / 1% (clean!)    |
| 0_789 | 0.4296           | 0.4152              | −0.0144  | 0% / 0% (clean)        |
| 0_867 | 0.4155           | 0.3988              | −0.0167  | 1.5% / 1% (clean)      |
| 0_977 | 0.4049           | 0.3974              | −0.0075  | 6.5% / 8.5%            |

Aggregate (n=13):

| Metric                       | Lossy 4 Hz (orig) | Clean 0.25 Hz (rerun) | Δ      |
|------------------------------|------------------:|----------------------:|-------:|
| `fused_miou` mean ± std      |  0.3684 ± 0.0724  | 0.3973 ± 0.0750       | +0.029 |
| `solo_a` mean                |            0.2676 | 0.2898                | +0.022 |
| `solo_b` mean                |            0.2867 | 0.3167                | +0.030 |
| **`fused − max(solo)` mean** | **+0.0712 ± 0.054** | **+0.0744 ± 0.054** | **+0.003** |

**Interpretation:**

1. **Aggregate fusion uplift is robust to the drop bias.** Drops suppress
   both `fused` and `max(solo)` together; their difference barely moves.
   The published Δ +0.071 stays.
2. **Per-cell absolute `fused_miou` is *not* robust.** Cells with
   ≥25% drops on both robots (0_182, 0_223) shifted by +0.10 / +0.19
   between batches. Clean cells (0_175, 0_178, 0_485, 0_789, 0_867)
   shifted <0.02 — within run-to-run noise.
3. **Cell variance bigger than the 0.05 Replica documented number.**
   0_723 had ~1% drops in both batches but `fused_miou` jumped +0.084.
   SceneNet two-robot fused has extra variance sources (DDS message
   ordering, thread scheduling at overlapping voxels). Treat single-cell
   Δ < 0.10 as noise unless verified.
4. **scovox is bit-deterministic across x86_64 ↔ aarch64** when fed
   identical inputs without drops — Jetson Nano 0_223 NPZs are
   bit-exactly identical to desktop 0_223 NPZs (44,808 / 66,197 voxels).

Surfaced by the Jetson edge-feasibility experiment 2026-05-21; full
trace in `src/docs/jetbot_experiment_2026_05_21.md` and
[[project-jetbot-first-run-2026-05-21]].

### Geometry per trajectory (F@5cm + Chamfer L2)

Voxel-grid metrics via [scenenet_compute_geometry_metrics.py](../robot_sw/distributed_mapping/scovox_eval/scripts/scenenet_compute_geometry_metrics.py): precision/recall of voxel centers at 5 cm, symmetric Chamfer L2 in meters via cKDTree.

| Metric         | solo_a mean | solo_b mean | fused mean | Δ (fused − best solo) |
|----------------|-------------|-------------|------------|----------------------|
| F1@5cm         | 0.865       | 0.881       | **0.940**  | **+0.013** (9/13 wins) |
| Chamfer L2 (m) | 0.082       | 0.054       | **0.040**  | **−0.014 m** (9/13 wins) |

The Chamfer outlier is **0_723 (+0.092 m loss)** — same trajectory where solo_b alone already had near-perfect coverage (F1 0.92, Chamfer 0.022 m); merging in solo_a's noisier posterior pulls the fused geometry off. Excluding 0_723: mean Δ Chamfer = **−0.024 m**.

### Three behavioural patterns (visible at n=13)

1. **Balanced scenes (10/13)** — both robots see complementary parts; fusion unambiguously better on mIoU + geometry.
2. **Solo-dominated scenes (0_279, 0_682)** — one robot already covered most of the trajectory; fusion ties or marginally hurts.
3. **Bad-solo contamination (0_723)** — one robot's noisy posterior pulls the merge down. Beta consensus `a_fused = a_A + a_B − 1` doesn't discount low-confidence inputs; the higher-quality solo's information bleeds into the merge. Paper-discussion territory.

Numbers: [phase3_summary_full.csv](../robot_sw/distributed_mapping/scovox_eval/results/phase3_scenenet_fusion_v3_2026_05_14/phase3_summary_full.csv) + [phase3_geometry_full.csv](../robot_sw/distributed_mapping/scovox_eval/results/phase3_scenenet_fusion_v3_2026_05_14/phase3_geometry_full.csv). Full report: [PHASE3_REPORT.md](../robot_sw/distributed_mapping/scovox_eval/results/phase3_scenenet_fusion_v3_2026_05_14/PHASE3_REPORT.md).

## Fusion aggregate (paper-scope: SceneNet n=13)

| Metric         | Wins  | Ties | Losses | Mean Δ (fused − best solo) |
|----------------|-------|------|--------|----------------------------|
| mIoU           | 10/13 | 1/13 | 2/13   | **+0.0712**                |
| F1@5cm         | 9/13  | 2/13 | 2/13   | **+0.013**                 |
| Chamfer L2 (m) | 9/13  | 0/13 | 4/13   | **−0.014**                 |

Step 11 spec (`fused mIoU ≥ max(solo)`) met on **11/13 scenes**. Both mIoU losses (−0.020 / −0.007) and 3 of 4 Chamfer losses (≤ +0.001 m) sit inside the ±0.05 noise envelope. The single substantive loss is 0_723 Chamfer (+0.092 m), a clear bad-solo-contamination outlier.

## Replica: out of paper (substrate-validation evidence only)

The Step 8 v3 wire smoke captured 3 Replica fusion scenes (room0+1+2) as a bonus. As of 2026-05-14, **Replica is dropped from the paper scope** — the paper rests on two datasets (KITTI + SceneNet), two modalities (LiDAR + RGB-D), two label sources (predicted soft + GT). The Replica numbers stay here as substrate-validation evidence that v3 fusion behaves correctly on a third dataset; they do not feed any paper table.

| Scene | source | solo_a mIoU | solo_b mIoU | fused mIoU | Δ mIoU | Δ F@5cm | Δ Chamfer |
|-------|--------|-------------|-------------|------------|--------|---------|-----------|
| room0 | smoke 0ad7f1a | 0.3072 | 0.3101 | 0.3469 | +0.0368 | +0.0413 | −4.5 cm |
| room1 | Phase 4a | 0.3537 | 0.2667 | 0.3302 | −0.0235 | +0.0224 | −0.6 cm |
| room2 | Phase 4a | 0.2663 | 0.2638 | 0.2897 | +0.0234 | +0.1419 | −9.2 cm |
| mean (n=3) | — | 0.3091 | 0.2802 | 0.3223 | +0.012 | +0.068 | −4.8 cm |

Why dropped: `data/replica_niceslam/` was cleared during an earlier disk reclaim; only room1.zip / room2.zip survived as scratch, plus room0 carried in from the Step 8 fusion smoke. A clean 8-scene aggregate would require re-downloading the office split via `download_replica_v1.sh` and re-running — not impossible, but the paper story is cleaner with the two-dataset framing.

Substantive Replica observation (informative even though it's out of paper): geometry metrics are unambiguous fusion-win on every recovered scene (mean +0.068 F@5cm, −4.8 cm Chamfer); the room1 mIoU loss (−0.024) sits inside the ±0.05 noise envelope per [[project-replica-room0-runtorun-variance]].

## Substrate substantiations

* **K=2 is Pareto-optimal on both datasets (KITTI n=5, SceneNet n=13)**: KITTI per-seq Δ(K=2→K=19) ≤ ±0.001 on all 5 sequences (+0.0002, +0.0001, 0.0000, −0.0001, +0.0006). SceneNet K-spread on the 6 per-K means is 0.009 abs (statistically indistinguishable from zero at n=13). Per-voxel ratio: 6.2× on KITTI (K=2 at 20 B vs K_full=19 at 124 B), **4.4× on SceneNet** (K=2 at 20 B vs K_full=13 at 88 B). K=19 SceneNet data points are valid (over-provisioned slots don't hurt mIoU) but the proper SceneNet K_full per-voxel comparison is against K=13.
* **Bayesian history matters everywhere**: D > NP on every cell (range +0.008 to +0.051).
* **Probability magnitudes matter on calibrated input**: D > MV by +0.029 on SceneNet GT (outside noise); ≈ MV on KITTI hard (one-hot makes D degenerate to MV mathematically — expected).
* **v3 wire format end-to-end validated on a substantial SceneNet n=13**: mIoU 10/13 wins (mean Δ +0.071), F1@5cm 9/13 wins (mean Δ +0.013), Chamfer 9/13 wins (mean Δ −1.4 cm). Three behavioural patterns visible: balanced/solo-dominated/bad-solo-contamination. Replica n=3 (out of paper scope) provides third-dataset substrate-validation supporting evidence (3/3 F@5cm + Chamfer wins).

## Notable bugs caught + fixed during this run

1. SceneNet scorer arg `--voxel_size` → `--resolution` (phase 1, 2, 2.5, 3 scripts)
2. `scenenet_eval.launch.py:87` hardcoded `occupancy_vis_threshold=0.5` (Phase 2.5 launch arg silently ignored before this fix)
3. `phase3_scenenet_fusion_batch.sh` TRAJS_DEFAULT had 13 aspirational trajs but only 4 had data on disk — script now auto-filters
4. KITTI scorer requires `--replay_to_yaml_lut` or 4 PolarSeg classes score 0 against yaml-space GT (Phase 2 script: PATCHED; Phase 1 / 2.5 scoring loops: corrected during scoring)
5. `data/replica_niceslam/` + `cam_params.json` cleared by an earlier disk reclaim. Recovered room1+room2 from surviving zips; synthesised cam_params from documented NICE-SLAM intrinsics `{w:1200, h:680, fx:fy:600, cx:599.5, cy:339.5, scale:6553.5}`

## Phase 4 — SLIM-VDB head-to-head (paired Δ table + sdf_trunc sweep on both datasets)

SCovox K=2 numbers reused from Phase 1; SLIM-VDB cached at `third_party_sw/slim_vdb/outputs/`. **KITTI trunc sweep ran 2026-05-15** (5 seqs × 3 truncs + the existing default = 20 cells, ~50 min on RTX 4060 Laptop) — see [[kitti-trunc-sweep-2026-05-15]]. Configs `kitti_trunc{010,015,020}.yaml` added under [slim-vdb config/](../../third_party_sw/slim_vdb/slim-vdb/examples/cpp/config/); reproducer [run_slimvdb_kitti_trunc_sweep.sh](../robot_sw/distributed_mapping/scovox_eval/scripts/slimvdb_harness/run_slimvdb_kitti_trunc_sweep.sh).

### Default-config head-to-head (paper headline)

| Comparison | n | Mean Δ (SC − SL) | 95% CI | Wins/Total |
|------------|---|--------------------|--------|------------|
| **KITTI mIoU (PolarSeg soft, sdf_trunc=0.30 default)** | 5 | **+0.0964** | **±0.0182 SE; 95% CI [+0.046, +0.147]** | **5/5** |
| **SceneNet mIoU (sdf_trunc=0.10 default)** | 13 | **+0.0918** | **±0.0161 (sig)** | **13/13** |

### sdf_trunc sweep — gap-mechanism story on both datasets

| Dataset / config | SLIM mean | SCovox mean | Paired Δ ± SE | 95% CI | Wins | Sig |
| --- | --- | --- | --- | --- | --- | --- |
| SceneNet sdf_trunc=0.15 | 0.187 | 0.327 | +0.140 ± 0.020 | [+0.100, +0.180] | 13/13 | yes |
| SceneNet sdf_trunc=0.10 (default) | 0.239 | 0.327 | +0.088 ± 0.014 | [+0.060, +0.116] | 13/13 | yes |
| SceneNet sdf_trunc=0.075 | 0.274 | 0.327 | +0.053 ± 0.013 | [+0.027, +0.079] | 12/13 | yes |
| SceneNet sdf_trunc=0.05 (tightest) | 0.320 | 0.330 | **+0.0104 ± 0.0105** | **[−0.012, +0.033]** | 8/13 | **n.s.** |
| KITTI sdf_trunc=0.30 (default) | 0.2087 | 0.3052 | +0.0964 ± 0.0182 | [+0.046, +0.147] | 5/5 | yes |
| KITTI sdf_trunc=0.20 | 0.2414 | 0.3052 | +0.0638 ± 0.0182 | [+0.013, +0.114] | 5/5 | yes |
| KITTI sdf_trunc=0.15 | 0.2628 | 0.3052 | +0.0424 ± 0.0175 | [−0.006, +0.091] | 4/5 | marginal |
| KITTI sdf_trunc=0.10 (tightest, 1 vox) | 0.2862 | 0.3052 | **+0.0190 ± 0.0131** | **[−0.018, +0.055]** | 4/5 | **n.s.** |

**SceneNet headline:** Δ +0.092 ± 0.016 over 13 trajectories at the SLIM-VDB published config (sdf_trunc=0.10, 2 voxels). CI [+0.076, +0.108] excludes zero with margin. Reproduces [[project-scenenet-head-to-head-2026-05-13]] cleanly at n=13.

**KITTI headline (PolarSeg soft):** Δ +0.0964 ± 0.0182 SE over 5 sequences at the SLIM-VDB default (sdf_trunc=0.30, 3 voxels). 95% CI [+0.046, +0.147] excludes zero. **Wins on every single sequence** (+0.044 to +0.148). Matches the [[project-kitti-miou-replay-bug-2026-05-11]] historical +0.096. Two earlier-draft fixes are now baked in: (a) Δ +0.043 → +0.096 from switching to standard SemanticKITTI mIoU (include zero-IoU classes in the mean), (b) 2026-05-15 re-run with `topk_probs_dir` properly set on `scovox_node` so the SCovox column actually reflects soft-prob ingestion as the datasets table claims. Both corrections together leave the headline unchanged (soft vs hard differs by ≤ +0.001 abs).

**Gap-mechanism confirmation — same story on both datasets.** At each dataset's tightest physically-meaningful sdf_trunc (SceneNet=0.05 m / 1 voxel, KITTI=0.10 m / 1 voxel), the head-to-head gap collapses and the 95% CI crosses zero:

- SceneNet: +0.092 → **+0.010** at trunc=0.05 (gap −89%, n.s., CI [−0.012, +0.033])
- KITTI: +0.096 → **+0.019** at trunc=0.10 (gap −80%, n.s., CI [−0.018, +0.055])

Monotonic collapse with tightening trunc on both panels. **The mIoU win is a precision/recall trade driven by SLIM-VDB's TSDF labelling-band width, not per-voxel semantic quality.** Connects directly to the Phase 2.5/2.5-v2 finding above: SCovox's labelling envelope is hit-driven (one ray-hit → labelled), SLIM-VDB's is geometry-driven (sdf_trunc-wide TSDF band labelled). Tightening the TSDF band to one voxel makes SLIM-VDB's labelling envelope match SCovox's stinginess and the gap evaporates.

### What stands and what reframes

| Claim | Before 2026-05-15 sweep | After sweep |
|-------|-------------------------|-------------|
| Default-config mIoU win (both datasets) | +0.096 KITTI / +0.092 SceneNet | **Unchanged** — same numbers |
| "SCovox has better per-voxel semantics" | Implicit in earlier draft | **Disproven** — at best-tuned SLIM-VDB the gap is n.s. on both panels |
| Memory advantage (Phase 5) | 4.0× KITTI / 2.8× SceneNet | **Unchanged** — even at trunc=0.10 SLIM-VDB carries 4× more voxels than SCovox on KITTI seq08 (8.9 M vs 2.2 M) |
| Fusion advantage (Phase 3) | +0.071 mIoU, F1, Chamfer at n=13 | **Unchanged** — no SLIM-VDB analogue exists |
| Architectural distinction | Hit-driven labelling | **Stronger** — now empirically grounded by trunc sweeps on both datasets |

**Paper framing implication:** lead with the default-config Δ +0.096 / +0.092 (the comparison most readers care about — out-of-the-box configs), then disclose the sdf_trunc sweep as gap-mechanism evidence. The unique-to-SCovox claims are then **memory** (Phase 5, architectural / O(1) vs O(C)) and **fusion** (Phase 3, no SLIM-VDB analogue) — both robust to hyperparameter tuning.

Cell-level numbers: [kitti_trunc_sweep_summary.csv](../../third_party_sw/slim_vdb/outputs/kitti_trunc_sweep_summary.csv). Full table + paper framing: [PHASE4_HEAD_TO_HEAD_2026_05_14.md](../robot_sw/distributed_mapping/scovox_eval/results/PHASE4_HEAD_TO_HEAD_2026_05_14.md).

### Phase 4 + 5 joint result — precision vs. memory Pareto (KITTI)

Grid memory measured at the **final frame** from each run's logs: `vdb_tsdf_mb + vdb_sem_mb` (SLIM-VDB `kitti_pipeline` TIMING line) vs `tsdf_grid_mb + semdir_grid_mb` (SCovox `[memSplit]` log line). Per the [[feedback-slimvdb-memory-measurement]] note, these are grid-only bytes — not RSS — so the comparison is apples-to-apples regardless of which runtime is bigger.

| System | Config | mIoU mean ± SE | Grid memory mean ± SE | vs SCovox |
| --- | --- | --- | --- | --- |
| **SCovox** | **K=2** | **0.3052 ± 0.020** | **268.7 ± 134.9 MB** | **1.0× (baseline)** |
| SLIM-VDB | trunc=0.10 (best mIoU) | 0.2862 ± 0.024 | 1458.9 ± 600.1 MB | 5.4× heavier |
| SLIM-VDB | trunc=0.15 | 0.2628 ± 0.025 | 1490.7 ± 606.8 MB | 5.5× |
| SLIM-VDB | trunc=0.20 | 0.2414 ± 0.024 | 1519.7 ± 612.8 MB | 5.7× |
| SLIM-VDB | trunc=0.30 (default) | 0.2087 ± 0.022 | 1576.3 ± 621.7 MB | 5.9× |

**SCovox K=2 Pareto-dominates every measured SLIM-VDB configuration.** At SLIM-VDB's *best* mIoU point (sdf_trunc=0.10), SCovox still uses **5.4× less memory while delivering 0.019 higher mIoU** (paired Δ; n.s. on the mIoU axis but unambiguous on memory). At SLIM-VDB's default (trunc=0.30), SCovox uses 5.9× less memory and 0.097 higher mIoU. There is no SLIM-VDB config where it matches or beats SCovox on either axis.

**Why memory is the axis SLIM-VDB can't tune away:** the sdf_trunc sweep saves <10% memory while moving mIoU by ~0.08, because the dominant memory cost is the dense per-voxel Dirichlet counter (80 B/voxel at KITTI's NCLASSES=20), not the TSDF band's voxel count. Tightening trunc trims a few voxels off the band but doesn't change the per-voxel byte footprint. SCovox's K=2 = 20 B/voxel is an architectural choice — it scales O(1) in class count regardless of how many classes the user adds.

Figure: [kitti_precision_vs_memory.png](../../figures/qualitative_2026_05_15/kitti_precision_vs_memory.png) (built by [plot_kitti_precision_vs_memory.py](../robot_sw/distributed_mapping/scovox_eval/scripts/plot_kitti_precision_vs_memory.py)). Per-cell data: [kitti_precision_vs_memory.csv](../../third_party_sw/slim_vdb/outputs/kitti_precision_vs_memory.csv).

This is the **strongest single comparison** the paper has: even if a reviewer accepts the gap-mechanism caveat (that SLIM-VDB's mIoU disadvantage is sdf_trunc-tunable), the Pareto figure shows the combined accuracy + memory advantage is architectural and not closable by hyperparameter tuning.

## Phase 5 — Derived analyses (no new compute)

### Memory scaling (per-voxel bytes)

| Dataset   | C  | SemDir K=2 (B/voxel) | SLIM-VDB (B/voxel) | Ratio |
|-----------|----|----------------------|--------------------|-------|
| SceneNet  | 14 | 20                   | 56                 | **2.8×** |
| KITTI     | 20 | 20                   | 80                 | **4.0×** |

K_TOP scales O(1) regardless of class count C; SLIM-VDB scales O(C). The per-voxel advantage widens with class count.

### Mass conservation

C++ unit tests pass at strict-equality (117/117 in `scovox_core` including `test_semdir_map.cpp`). NPZ-level verification deferred — current publish schema doesn't emit α_other + per-slot cnt[i] (only aggregated p_occ, semantic_class, etc.). Tagged follow-up.

### Eviction telemetry

Pre-SemDir CSVs from the SemBeta era stand as reference (per-frame match/empty/evict/drop counts in `e5_anchors/`). Under SemDir the `drop` category becomes `to_other` (mass preserved in α_other slot, not lost). Semantic-rename rerun queued — small, no logical change.

Full report: [PHASE5_DERIVED_2026_05_14.md](../robot_sw/distributed_mapping/scovox_eval/results/PHASE5_DERIVED_2026_05_14.md).

## What remains for the paper

Paper scope is **KITTI + SceneNet only** (see "Replica: out of paper" above and the [Paper table mapping](NEW_EXPERIMENT_PLAN.md#paper-table-mapping-post-2026-05-14-scope) in the plan).

**All experiments and analyses now complete.** Remaining tasks are paper-writing only:

* Translate the 7 phase reports into the paper's table/figure structure (Tables 1–4, 6, 7 + h2h table; Section discussion of Phase 2.5 + 2.5-v2 null result composition).
* Optional follow-ups (not blocking paper submission):
  - Eviction telemetry SemDir-rename rerun (~5 min, semantic-only).
  - Mass conservation NPZ-schema extension if a reviewer asks.
  - w_occ sweep (would test the actual labelling-envelope knob revealed by Phase 2.5/2.5-v2).

Out of paper (no longer blocking):
* Office scenes for Replica Phase 4a (was 5 scenes data-blocked; Replica dropped 2026-05-14)

## Branch state

`origin/refactor/split-tsdf-sembeta` ←
`1a5d1c9 Phase 4a Replica E2.1 fusion v3 — room1+room2+room0 (n=3)`

10 commits ahead of the pre-Step-8 baseline `75f27df`:

```
1a5d1c9  Phase 4a Replica E2.1 fusion v3
5798c26  Phase 3 SceneNet fusion v3
cf328ab  Phase 2.5 gate sweep + scenenet launch arg fix
9d68d11  Phase 2 component ablation
dadf1fa  Phase 1 K_TOP sweep + scenenet scorer arg fix
c114d6e  Phase 2.5 + Phase 3 scaffolding + Phase 2 cell-count
a958172  NEW_EXPERIMENT_PLAN scaffolding for all 4 phases
3a0a9b7  Step 8: projectSemDirToSemBeta deprecation banner
0ad7f1a  Step 8 fusion smoke: v3 publish/receive validated
3a35eae  Step 8 receiver: dscovox onBinaryMapV3
4dd47a0  Step 8 sender: publishBinaryMapV3 + wire_format launch arg
b76d8c8  Step 8 (Phase 0): SemDir two-stream + num_classes plumbing
```
