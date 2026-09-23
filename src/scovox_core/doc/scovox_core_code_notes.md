# scovox_core — code comment notes

Long comments from files in the `scovox_core` package, moved out of the code on 2026-09-23 so the sources carry short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word. Files with many moved comments have their own notes doc next to this one.

One section per source file, in file order. Each entry names the function (or section) the comment sat in, the line of code it was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [CMakeLists.txt](#cmakeliststxt) — 1
- [include/scovox/beta_voxel.hpp](#includescovoxbeta_voxelhpp) — 2
- [include/scovox/binary_serializer.hpp](#includescovoxbinary_serializerhpp) — 6
- [include/scovox/consensus_merge.hpp](#includescovoxconsensus_mergehpp) — 3
- [include/scovox/dir_voxel.hpp](#includescovoxdir_voxelhpp) — 8
- [include/scovox/map_interface.hpp](#includescovoxmap_interfacehpp) — 8
- [include/scovox/marching_cubes.hpp](#includescovoxmarching_cubeshpp) — 4
- [include/scovox/mesh_labelling.hpp](#includescovoxmesh_labellinghpp) — 2
- [include/scovox/refinement_regions.hpp](#includescovoxrefinement_regionshpp) — 2
- [include/scovox/semantics.hpp](#includescovoxsemanticshpp) — 2
- [include/scovox/sembeta_voxel.hpp](#includescovoxsembeta_voxelhpp) — 4
- [include/scovox/tsdf_map.hpp](#includescovoxtsdf_maphpp) — 5
- [include/scovox/tsdf_voxel.hpp](#includescovoxtsdf_voxelhpp) — 1
- [include/scovox/uncertainty.hpp](#includescovoxuncertaintyhpp) — 5
- [include/scovox/voxel.hpp](#includescovoxvoxelhpp) — 3
- [src/sem_split_map.cpp](#srcsem_split_mapcpp) — 10
- [src/uncertainty.cpp](#srcuncertaintycpp) — 5
- [test/test_binary_serializer.cpp](#testtest_binary_serializercpp) — 2
- [test/test_consensus_merge.cpp](#testtest_consensus_mergecpp) — 2
- [test/test_fine_tsdf.cpp](#testtest_fine_tsdfcpp) — 3
- [test/test_scovox_map_split.cpp](#testtest_scovox_map_splitcpp) — 2
- [test/test_sparse_add.cpp](#testtest_sparse_addcpp) — 1
- [test/test_tsdf_map.cpp](#testtest_tsdf_mapcpp) — 3
- [test/test_uncertainty.cpp](#testtest_uncertaintycpp) — 2
- [test/test_voxel_layouts.cpp](#testtest_voxel_layoutscpp) — 1

## CMakeLists.txt

### cmake-fine-tsdf-test-scope

**What test_fine_tsdf covers** — in `Tests (GTest, no ROS dependencies)`, attached to `ament_add_gtest(test_fine_tsdf test/test_fine_tsdf.cpp)` (line 93)

```text
Fine TSDF band (fine_tsdf_band_dbh_2026_07_30.md): region registry gate,
gated two-lattice integration, per-scan anchor re-registration (drift
absorption), raw-return fine-only path (refineHit), rev-7 fine wire
stream + merge. Uses the DBH circle fit (dbh_fit.hpp, a post-processing
utility the mapping runtime never calls) as its accuracy metric.
```

## include/scovox/beta_voxel.hpp

### beta-prior-single-source

**Occupancy prior constants as single source** — in `File scope`, attached to `constexpr float kBetaOccPrior  = 1.0f;` (line 71)

```text
Shipped split-substrate occupancy prior: symmetric **Beta(1,1)** (uniform /
Bayes–Laplace) → prior `p_occ = 0.5`. SINGLE SOURCE OF TRUTH for the split
occupancy prior: allocation (`SemSplitMap`), the consensus merge's
prior-subtraction (`mergeBeta`), the receiver's at-prior detection
(`isPriorBeta`), the sender's emit gate, and the SSMI unobserved baseline
all reference these constants, so sender and receiver stay consistent — the
prior is a compile-time constant, NOT carried on the wire. Decoupled from
the semantic `(num_classes, α₀)` because occupancy and semantics are
independent priors. See docs/occupancy_prior.md (incl. the Jeffreys
`Beta(0.5,0.5)` runner-up and the conditions to switch).
```

### beta-prior-factory-first-touch

**Beta prior factory at every allocation** — in `defaultBetaVoxel`, attached to `inline BetaVoxel defaultBetaVoxel(float occ_prior = 1.0f,` (line 84)

```text
Beta prior factory. **Required at every allocation**: Bonxai's pool
allocator zero-initialises new leaf blocks, leaving `a_occ = a_free = 0`.
Without this, the first integration would increment from `Beta(0,0)`
instead of from the prior, silently mis-weighting the posterior forever
(the same first-touch invariant as `defaultSemBetaVoxel` /
`defaultSemDirVoxel`).

The factory is prior-agnostic. The 1.0/1.0 default IS the shipped symmetric
Beta(1,1) occupancy prior (`p_occ = 0.5`), which `SemSplitMap` passes
explicitly via `kBetaOccPrior` / `kBetaFreePrior`. Pass `occ_prior = C·α₀`,
`free_prior = α₀` to reproduce the old calibrated unified-Dirichlet marginal
(`p_occ = C/(C+1)`) as an ablation. See docs/occupancy_prior.md.
```

## include/scovox/binary_serializer.hpp

### wire-max-num-classes

**Why num_classes has a ceiling** — in `BinarySerializer — declarations`, attached to `static constexpr uint16_t MAX_NUM_CLASSES = 4096;` (line 152)

```text
Sane upper bound on the header-supplied class count. num_classes drives the
reconstructed Dirichlet OTHER prior (num_classes − K_TOP)·alpha_0 in
consensus_merge; a forged u16 near 65535 with alpha_0≈0.01 injects ~650
pseudo-counts of "unknown" mass per voxel, swamping real evidence (and two
re-sent snapshots from the same bad sender agree, so the per-frame equality
check can't catch it). No real taxonomy approaches this ceiling.
```

### wire-format-version-bumps

**Codec revision history** — in `BinarySerializer — declarations`, attached to `static constexpr uint8_t  FORMAT_VERSION = 8;` (line 159)

```text
Blob codec revision (distinct from the ROS envelope `version`=5 that routes
to this codec). Bumped 5→6 for the block-run coordinate coding + u16
payload quantization (comms_design_2026_07_30.md Part 1); 6→7 for the
fine-TSDF band (fine_ratio_log2 header byte + trailing fine stream,
fine_tsdf_band_dbh_2026_07_30.md); 7→8 for u8 sqrt-companded evidence
payloads + u8 class ids (see the revision-8 block in the file header).
Any layout change means a mixed-revision fleet fails loud (deserialize
rejects the VERSION byte and the frame is dropped with a warning) instead
of silently misparsing.
```

### wire-frame-quant-step

**The Frame quant_step field** — in `Frame — declarations`, attached to `float    quant_step  = 0.0f;` (line 182)

```text
u16 quantization step for Beta/Dir payloads. The sender sets
evidence_saturation / 65535; 0.0 (the default) keeps payloads f32.
```

### wire-validate-header-priors

**Validating header prior parameters** — in `deserialize`, attached to `if (f.num_classes < static_cast<uint16_t>(K_TOP)) {` (line 367)

```text
Validate the header-supplied prior parameters before any voxel is
reconstructed from them. consensus_merge rebuilds the Dirichlet
(semantic) prior as (num_classes − K_TOP)·alpha_0 with per-slot alpha_0,
so a corrupt header — num_classes < K_TOP, or a non-finite/non-positive
alpha_0 — would silently inject negative or NaN mass into the live map
(and two re-sent snapshots from the same bad sender agree with each
other, so the per-frame equality check cannot catch it). The OCCUPANCY
Beta prior is the symmetric constant kBetaOccPrior, independent of these
fields.
```

### wire-count-dos-guard

**Bounding record counts before reserve** — in `deserialize`, attached to `if (tsdf_count > r.remaining() / 20)` (line 408)

```text
Bound the reserve by the bytes actually present: each record is a fixed
20 B on the wire, so a count claiming more records than the remaining
buffer can hold is a truncated/forged frame. Validate BEFORE reserve so
an attacker-controlled count (e.g. 0xFFFFFFFF) raises the documented
runtime_error rather than a multi-GB length_error/bad_alloc.
```

### wire-read-block-stream

**Reading block runs** — in `BinarySerializer — declarations`, attached to `template <typename PayloadReader>` (line 630)

```text
Read block runs until exactly `count` records have been consumed.
`read_payload(coord)` pulls its own payload bytes from the reader.
Structural violations — bad mode byte, empty block, out-of-range or
non-ascending indices, a block overshooting `count` — all throw, so a
corrupt frame is dropped whole rather than half-integrated.
```

## include/scovox/consensus_merge.hpp

### merge-beta-prior

**mergeBeta and the constant occupancy prior** — in `mergeBeta`, attached to `inline BetaVoxel mergeBeta(const BetaVoxel& a,` (line 76)

```text
Per-voxel BetaVoxel merge under the symmetric Beta(1,1) occupancy prior.
The `num_classes` / `alpha_0` params are retained for call-site symmetry with
`mergeDir` but are UNUSED for occupancy: the prior is now the decoupled
constant `kBetaOccPrior` = `kBetaFreePrior` = 1.0 (docs/occupancy_prior.md),
not the old calibrated `C·α₀`. Sender and receiver share this compile-time
constant, so the prior-subtraction below stays consistent across nodes.
```

### merge-beta-floor-at-prior

**Flooring fused Beta evidence at the prior** — in `mergeBeta`, attached to `f.a_occ  = std::max(occ_prior,  a.a_occ  + b.a_occ  - occ_prior);` (line 90)

```text
Floor each fused α at its prior. Below-prior inputs are NORMAL, not just
corruption: the split-path evidence saturation (SemSplitMap::
applyBetaSaturation) rescales BOTH α by cap/s_total, which pushes the
minority bucket below its prior on any saturated lopsided voxel (e.g.
cap=1000, a_occ≈999 ⇒ a_free≈0.999 < 1). The floor then binds with a fused
p_occ error bounded by the prior itself (~1e-4 at cap 1000). It also still
guards a corrupt source from producing a negative α and an out-of-[0,1]
p_occ that would poison downstream entropy/EIG.
```

### merge-dir-slot-order

**Deterministic slot order in mergeDir** — in `mergeDir`, attached to `auto orderBefore = [](const Entry& x, const Entry& y) {` (line 138)

```text
Deterministic, fold-order-invariant ordering: count desc, then class id
asc as a tie-break. The secondary key is what makes the order independent of
source iteration: without it, two classes with equal counts at the K_TOP
truncation boundary would be kept-or-dumped depending on which source we
folded first, making the fused slots (and dominantClass) nondeterministic
across runs/rehashes. NOTE: this makes the merge DETERMINISTIC but not fully
order-INDEPENDENT — a class dumped to OTHER in one pairwise fold cannot climb
back into a slot in a later fold. True commutativity requires accumulating
every source's evidence before a single truncation; callers needing that
must fold in a fixed source order.

Hand-rolled insertion sort rather than std::sort: n ≤ 2·K_TOP, and at -O3 an
inlined std::sort drags in its 16-element introsort fallback whose dead
`__first + 16` access trips a -Warray-bounds false positive when mergeDir is
inlined into a hot caller (the dscovox refold path). Output is identical —
the comparator is a strict total order (all cls distinct after upsert).
```

## include/scovox/dir_voxel.hpp

### dirvoxel-track-qmax

**The SCOVOX_TRACK_QMAX build flag** — in `File scope`, attached to `#ifndef SCOVOX_TRACK_QMAX` (line 50)

```text
Per-slot confidence track for the max-probability eviction comparator.
Build-time only (`-DSCOVOX_TRACK_QMAX=1`), following the `SCOVOX_K_TOP`
one-install-tree-per-variant convention, because it changes
`sizeof(DirVoxel)` (4 + 6·K_TOP → 4 + 8·K_TOP; 16 B → 20 B at K_TOP=2) and
we must be able to quote the shipped build's bytes unchanged.

Never serialized: binary_serializer.hpp emits only `other`, `cnt[]` and
`cls[]`, so a `SCOVOX_TRACK_QMAX=1` sender stays wire-compatible with a
`=0` receiver. The comparator only changes *which* class wins a slot, and
that outcome is already fully visible in the `cls`/`cnt` that do go out.
```

### dirvoxel-layout-size

**DirVoxel size invariants** — in `File scope`, attached to `constexpr std::size_t kDirSlotBytes = SCOVOX_TRACK_QMAX ? 8u : 6u;` (line 97)

```text
Layout invariants. K_TOP=2 (production / paper default):
  sizeof == 4 (other) + 4·K_TOP (cnt) + 2·K_TOP (cls) = 4 + 12 = 16.
General K_TOP: 4 + 6·K_TOP, rounded up to 4-byte alignment for the trailing
uint16_t pair.
```

### dirvoxel-default-prior

**The default Dirichlet prior** — in `defaultDirVoxel`, attached to `inline DirVoxel defaultDirVoxel(uint16_t num_classes = 14,` (line 123)

```text
Default symmetric prior. The top-K slots and OTHER carry the same per-dim
prior `α₀` that the unified `SemDirVoxel` uses, minus the FREE dimension:
  - each `cnt[i] = α₀`  (K_TOP slot placeholders)
  - `other = (C − K_TOP) · α₀`  (the collapsed out-of-K dimensions)
Total class prior `= C · α₀`, equal to `SemDirVoxel::s_occ()` at prior and
to `BetaVoxel`'s `a_occ` prior (`C·α₀`) — keeping the split consistent with
the live path at the prior.
```

### dirvoxel-sparse-add-class

**Heavy-hitter sparse-add into DirVoxel** — in `sparse_add_class`, attached to `inline void sparse_add_class(float*    cnt,` (line 142)

```text
Heavy-hitter sparse-add into the occupied-class Dirichlet, parametrised by
the per-dim prior `α₀`. Routes `inc` into one of the top-K_TOP class slots
(Space-Saving / Metwally 2005) or the OTHER bucket — never lost. Preserves
the strict mass invariant `Δ(other + Σcnt) == inc`.

Direct port of `sparse_add_unified` (semdir_map.cpp), with `alpha_other`
renamed `other` and no FREE interaction (FREE is in the Beta grid).
`q` is the deposit's own class probability in [0,1] (for a Dirichlet update
that is `inc / class_share`). It is *not* evidence and never changes what is
deposited — it only feeds the optional confidence eviction comparator, and
is ignored unless `qmax != nullptr` (which requires SCOVOX_TRACK_QMAX).
```

### dirvoxel-sentinel-class

**Routing class id 0xFFFF to OTHER** — in `sparse_add_class`, attached to `if (c == 0xFFFF) {` (line 166)

```text
(0) Sentinel guard. `0xFFFF` is the empty-slot marker in `cls[]`, so a real
observation of class id 0xFFFF (e.g. a 65535-class taxonomy, or a classifier
whose argmax index hits 0xFFFF) must NOT be written into a slot: it would
fill `cls[i] = 0xFFFF` with real mass yet still read as EMPTY, so the next
add re-fills the slot from scratch (losing the prior inc) and isPriorDir /
dominantClass mis-treat it as unfilled — breaking the strict invariant
Δ(other + Σcnt) == inc. Route the (untrackable) sentinel class straight to
OTHER, which both conserves mass and keeps every slot's sentinel meaning intact.
```

### dirvoxel-evict-clamp

**Clamping evicted evidence at zero** — in `sparse_add_class`, attached to `const float raw_evicted = cnt[min_i] - alpha_0;` (line 202)

```text
Clamp at 0. A filled slot should always hold >= α₀ (prior + observed
evidence), but an evidence-saturation rescale can erode it below α₀; an
unclamped `cnt[min_i] − α₀` would then be NEGATIVE and turn the
`*other += evicted_evidence` below into a mass SUBTRACTION (driving OTHER
negative, breaking the Δ(other + Σcnt) == inc invariant). The saturation
path also floors filled slots at α₀, so this is normally a no-op safety net.
```

### dirvoxel-eviction-comparator

**Evidence vs confidence eviction** — in `sparse_add_class`, attached to `const bool evict_now = track ? (q_fx > qmax[min_i])` (line 210)

```text
Comparator choice changes only WHICH deposits win a contested slot, never
how much mass moves: both branches below conserve Δ(other + Σcnt) == inc.
Shipped rule weighs accumulated evidence, so a class that arrives often
beats one that arrives certain; the confidence rule inverts that, letting a
single high-probability observation displace a pile of low-probability
ones. On noisy real-camera labels the latter measured better.
```

### dirvoxel-dominant-class

**dominantClass on DirVoxel** — in `dominantClass`, attached to `inline uint16_t dominantClass(const DirVoxel& v,` (line 233)

```text
Argmax of the top-K class slots by observed evidence (`cnt − α₀`). Returns
0xFFFF if no slot is filled, or if `OTHER`'s *observed* evidence exceeds
every slot's evidence (the bulk of the class mass is on out-of-K classes,
so committing to a tracked class would be misleading). Mirrors the
`SemDirVoxel` overload in mesh_labelling.hpp, restricted to the
occupied-class Dirichlet.

The slot key subtracts each slot's `α₀` placeholder; the OTHER key must
subtract OTHER's own prior `(C − K_TOP)·α₀` (the collapsed out-of-K
dimensions, set by `defaultDirVoxel`) for an apples-to-apples comparison —
otherwise OTHER's prior mass alone (0.12 at C=14, α₀=0.01) spuriously
dominates a legitimately-observed slot and the only seen class is hidden.
`num_classes` defaults to 14 (matching `defaultDirVoxel`); the residual is
clamped at 0 to match `defaultDirVoxel`'s `num_classes < K_TOP` convention.
```

## include/scovox/map_interface.hpp

### params-dir-leaf-bits

**Leaf bits for the semantic grid** — in `Params — declarations`, attached to `uint8_t dir_leaf_bits = 2;` (line 35)

```text
Leaf block bits for the SEMANTIC (Dirichlet) grid only — see
SemSplitMap::Params::dir_leaf_bits. The Dir grid is ~18× sparser than the
Beta grid (hit-only + gated) so it wastes far more of an 8³ block; 2 →
4×4×4 = 64 voxels/block. Clamped to <= leaf_bits (never coarser than the
occupancy grid), so a leaf_bits=1 LiDAR config is unaffected.
```

### params-occupancy-weights

**Occupancy weights from the sensor model** — in `Params — declarations`, attached to `float w_free = 1.0f;   ///< Evidence added per free-space traversal` (line 43)

```text
Sensor-physics derivation (Beta-Bernoulli equivalent of OctoMap log-odds).
From Beta(1,1) prior + inverse sensor model (prob_hit, prob_miss):
  w_occ  = (2·prob_hit  − 1) / (1 − prob_hit)
  w_free = 1/prob_miss − 2

Calibration table (per sensor):
  RGB-D conservative (OctoMap defaults: prob_hit=0.7, prob_miss=0.4):
    w_occ  = (1.4−1)/0.3 = 1.33
    w_free = 1/0.4 − 2   = 0.5
    Why: noisy stereo / structured-light depth at range — many spurious
    hits at object edges, frequent missed returns on dark/specular
    surfaces. Low confidence per observation; rely on accumulation.

  RGB-D moderate (current defaults: prob_hit≈0.75, prob_miss≈0.33):
    w_occ=2.0, w_free=1.0
    Why: typical depth camera (RealSense, Azure Kinect) at indoor range
    with clean optics — slightly more confident than OctoMap's worst-case.

  LiDAR (prob_hit≈0.9, prob_miss≈0.15):
    w_occ  = (1.8−1)/0.1 = 8.0
    w_free = 1/0.15 − 2  ≈ 4.67
    Why: time-of-flight LiDAR has very low FP rate (returns are physical
    reflections) and low FN rate in clear air — each ray is high-evidence,
    so weights are large and the map converges fast.

Refit from data only if calibration logs show drift. Ratio w_occ/w_free
matters more than absolute scale for steady-state p_occ.
```

### params-carve-wall-guard

**Wall protection during free-space carving** — in `Params — declarations`, attached to `float carve_skip_occ_threshold = 0.0f;` (line 73)

```text
Wall protection during free-space carving — DISABLED by default (<= 0).
We trust the most recent LiDAR scan: a beam that reached its return proves
every voxel it traversed is free NOW, so a previously-occupied (moved/stale)
obstacle in its path should clear, not block the carve. The batched carve
path (the live pipeline) ignores this guard entirely; a positive value only
re-enables it for the immediate (unbatched) path — offline tools/ablations.
(Formerly 0.7; the joint ray-cast reach_prob variant was reverted for a
~6 mIoU cold-start tax on indoor RGB-D — see docs/exp_ablations.md.)
```

### params-production-knobs

**Why the two production knobs carry mIoU** — in `Params — declarations`, attached to `uint16_t evidence_saturation = 1000;       ///< Cap on (a_occ, a_free, sem_cnt). 0 = disabled.` (line 89)

```text
Together these restore OLD-pipeline mIoU within noise (verified by
res_step_bayesian: 0.3488 vs OLD baseline_newgate_05 0.3481). The four
pre-cleanup knobs (smc, smooth-gate sigmoid) were proved redundant —
these two carry all the load. See docs/exp_ablations.md.
```

### params-top-k

**top_k and the heavy-hitter guarantee** — in `Params — declarations`, attached to `int   top_k  = K_TOP;` (line 99)

```text
Number of explicitly tracked semantic classes per voxel.

The sparse representation is equivalent to the Space-Saving /
Misra-Gries streaming heavy-hitter algorithm (Metwally et al.
ICDT 2005; Misra & Gries 1982).

Per Cormode (2016, Encyclopedia of Algorithms): any class with
true frequency > n/k is guaranteed to be tracked, and the
additive count error on any tracked class is at most n/k,
where n = total observations at that voxel and k = top_k.

Calibration: at 10 fps with ~5 s voxel residence (n ~ 50),
K=2 tracks everything above ~25% observation share; K=4 tracks
above ~12%. Indoor scenes rarely have >2 genuine classes per
voxel (interior + boundary), so K=2 is sufficient for most
deployments; K=4 provides margin for noisy classifiers.
```

### params-consensus-deprecated

**Deprecated consensus fusion fields** — in `Params — declarations`, attached to `float consensus_kl_threshold  = 5.0f;` (line 119)

```text
Deprecated 2026-05-03: both fields are no-ops. consensusMerge is now a
pure Beta–Dirichlet conjugate update (additive under conditional
independence). The KL-conflict gate was never read by any caller and
the post-merge occupancy gate on semantics was non-Bayesian. Fields
retained so launch files that set them remain compatible.
```

### params-sdf-trunc

**TSDF truncation distance** — in `Params — declarations`, attached to `float sdf_trunc          = 0.0f;` (line 136)

```text
Truncation distance in metres. 0 disables TSDF integration. The node
sets this from `sdf_trunc_voxels * resolution` so it scales with
resolution; a fixed metre default would silently break across launch
files at different resolutions (5 cm vs 10 cm).
```

### params-band-only-integration

**Band-only TSDF integration mode** — in `Params — declarations`, attached to `bool  band_only_integration = false;` (line 147)

```text
When true, fused_integrate_ray_static walks only the truncation band
  [posToCoord(hit - sdf_trunc * u), posToCoord(hit + sdf_trunc * u)]
instead of the full ray from origin. This matches SLIM-VDB / VDBFusion's
band-only DDA and skips per-voxel Beta-free carving along the long part
of the ray. The Beta occupancy posterior degrades to "carved only inside
the band"; downstream consumers that rely on free-space outside the band
(planner, frontier detection) will see no carved free voxels there.
Default off — the full-ray walk is the SCovox feature that distinguishes
it from a TSDF-only system; this flag is for benchmarking the integration
cost of that feature against SLIM-VDB on matched workloads.
```

## include/scovox/marching_cubes.hpp

### mc-extract-mesh-voxel

**extractMesh on the fused Voxel grid** — in `extractMesh`, attached to `inline TriangleMesh extractMesh(` (line 402)

```text
Extract a triangle mesh from the TSDF zero-crossing via marching cubes.

Iterates every active voxel, treats it as the (0,0,0) corner of a cube
whose 8 corners are the voxel and its 7 positive-offset neighbours.
Cubes with any corner below `min_weight` are skipped.

Vertex positions are sub-voxel interpolated along edges where a sign
change occurs, matching the standard marching cubes algorithm used by
Open3D and SLIM-VDB.

@param grid       Bonxai VoxelGrid containing Voxel with tsdf_distance/tsdf_weight.
@param min_weight Minimum tsdf_weight for a corner to count as valid.
@param resolution Voxel edge length in metres.
@return           Triangle mesh with per-triangle semantic labels.
```

### mc-extract-zero-crossing-voxel

**extractZeroCrossing on the fused Voxel grid** — in `extractZeroCrossing`, attached to `inline std::vector<SurfacePoint> extractZeroCrossing(` (line 522)

```text
Extract zero-crossing surface points by checking 3 positive-axis
neighbours per active voxel. Cheaper than full marching cubes —
produces sub-voxel interpolated points wherever the TSDF sign flips.

Semantic label is taken from the positive-side voxel (the one "in
front" of the surface, where the sensor observed the hit).

@param grid       Bonxai VoxelGrid containing Voxel with tsdf_distance/tsdf_weight.
@param min_weight Minimum tsdf_weight for both neighbours to be valid.
@param resolution Voxel edge length in metres.
@return           Vector of sub-voxel surface points with semantic labels.
```

### mc-extract-point-cloud-voxel

**extractPointCloud on the fused Voxel grid** — in `extractPointCloud — SLIM-VDB style: emit voxel centers with labels`, attached to `inline std::pair<std::vector<Eigen::Vector3f>, std::vector<uint16_t>>` (line 581)

```text
Extract a point cloud of voxel centres where tsdf_weight exceeds the
threshold. This matches SLIM-VDB's `ExtractPointCloud` — every
sufficiently-observed voxel emits one point at its centre with the
argmax semantic label.

@param grid       Bonxai VoxelGrid containing Voxel with tsdf_distance/tsdf_weight.
@param min_weight Minimum tsdf_weight for emission.
@param resolution Voxel edge length in metres (used for centre offset).
@return           Pair of (positions, labels).
```

### mc-tsdf-extractors-centre-offset

**TsdfVoxel extractors and the centre offset** — in `extractMesh`, attached to `inline TriangleMesh extractMesh(` (line 615)

```text
These mirror the three legacy extractors above but operate on
`Bonxai::VoxelGrid<TsdfVoxel>`, which has no semantic fields. Labels
come via free functions in `mesh_labelling.hpp` (Step 4) after
extraction, looking up the SemBeta grid by coord.

Half-voxel centre offset: SLIM-VDB / VDBFusion convention is to
interpolate between voxel CENTRES, not lower corners. The Voxel-typed
extractors above use lower corners (Bonxai's `coordToPos`); these new
overloads add `+0.5*resolution` per axis to match SLIM-VDB exactly.
```

## include/scovox/mesh_labelling.hpp

### labelling-mesh-centroid

**Labelling triangles by centroid** — in `labelMesh`, attached to `inline std::vector<uint16_t> labelMesh(` (line 50)

```text
Per-triangle labels. The anchor voxel is the cube's positive-side
(in-front-of-surface) corner, identified via TSDF sign in `tsdf_grid`.
Returns a vector aligned with `geom.triangles` (one label per triangle).

The implementation walks each triangle, recovers the anchor voxel coord
by mapping the triangle's centroid back to the TSDF grid, then queries
SemBeta. This is approximate (a triangle's centroid doesn't always land
exactly in the anchor voxel) but matches what `Map::extractMesh` already
does for the legacy fused Voxel grid.
```

### labelling-dirvoxel

**DirVoxel labelling overloads** — in `labelMesh`, attached to `inline std::vector<uint16_t> labelMesh(` (line 109)

```text
Same anchor-via-centroid strategy as the SemBeta overloads above; argmax uses
the occupied-class `dominantClass(const DirVoxel&, alpha_0)` from
dir_voxel.hpp, which refuses to commit when `other` exceeds every top-K
slot's observed evidence. Occupancy is *not* consulted here — in the
split substrate the surface geometry comes from the TSDF grid and per-point
occupancy from the Beta grid; this function answers only "which class".
```

## include/scovox/refinement_regions.hpp

### refinement-add-slot-stability

**Slot stability in RefinementRegions add** — in `RefinementRegions — declarations`, attached to `int add(const RefinementCylinder& cyl, float margin) {` (line 54)

```text
Register (or replace, keyed on `cyl.id`) a region. `margin` widens the
gate radius beyond the model radius. Returns the slot index. Slot
indices are stable for the registry's lifetime (removal tombstones,
re-registration of the same id reuses its slot) so per-slot scan
buffers stay valid.
```

### anchor-fit-contract

**The cylinder anchor-shift fit** — in `fitCylinderAnchorShift`, attached to `inline std::optional<Eigen::Vector2f> fitCylinderAnchorShift(` (line 231)

```text
Fit the 2-DoF horizontal shift Δ minimising
  Σ_i ρ_huber( ‖p_i + Δ − c‖ − r )
over one scan's in-region endpoint XY positions `pts`, against the
canonical cylinder centre `c = (cx, cy)` and radius `r`.

Returns std::nullopt when the fit must not be trusted: too few points,
a degenerate normal matrix, or a converged shift beyond `max_shift`
(wrong association / not actually the trunk). Callers then integrate
uncorrected — losing a scan's correction, never inventing one.
```

## include/scovox/semantics.hpp

### semantics-naive-baseline

**NAIVE ablation baseline** — in `naive_update_semantics`, attached to `inline void naive_update_semantics(Voxel* v,` (line 66)

```text
NAIVE mode — intentional ablation baseline.

"Last observation wins": every update wipes prior semantic state and
stores the argmax class with a fixed count of 1.0. By design this:
  - ignores `quality`, `kappa0`, and `p_occ` (no weighting),
  - cannot accumulate confidence over repeated observations,
  - returns the same `semanticEntropy` regardless of how many times the
    cell was observed.
`apply_semantics` still applies a hard `p_occ > 0.5` cutoff for NAIVE
and MAJORITY_VOTE (so they only fire on occupied voxels). This is used
as a contrast against the Dirichlet mode in ablations and should NOT
be "improved" without removing it from the ablation suite.
```

### semantics-majority-vote-baseline

**MAJORITY_VOTE ablation baseline** — in `majority_vote_semantics`, attached to `inline void majority_vote_semantics(Voxel* v,` (line 94)

```text
MAJORITY_VOTE mode — intentional ablation baseline.

Each admitted observation contributes a single +1 vote to the argmax
class. Like NAIVE, this ignores `quality`, `kappa0`, and `p_occ`;
unlike NAIVE it accumulates votes across observations. A hard
`p_occ > 0.5` cutoff is applied at `apply_semantics`,
so MAJORITY_VOTE only fires on occupied voxels — the only ablation
variable vs DIRICHLET is the accumulation rule.
```

## include/scovox/sembeta_voxel.hpp

### sembeta-voxel-layout

**SemBetaVoxel byte layout** — in `SemBetaVoxel — declarations`, attached to `struct SemBetaVoxel {` (line 33)

```text
24-byte SCovox semantic + Beta voxel. K_TOP=2 sparse Dirichlet slots
+ Beta(a_occ, a_free) + a_unk residual mass.

Layout (with K_TOP=2):
  offset 0:   a_occ      (float, 4 B)
  offset 4:   a_free     (float, 4 B)
  offset 8:   a_unk      (float, 4 B)
  offset 12:  sem_cnt[0] (float, 4 B)
  offset 16:  sem_cnt[1] (float, 4 B)
  offset 20:  sem_cls[0] (uint16, 2 B)
  offset 22:  sem_cls[1] (uint16, 2 B)
  total: 24 B, no internal padding.
```

### sembeta-size-invariant

**SemBetaVoxel size across K_TOP** — in `File scope`, attached to `constexpr std::size_t kSemBetaExpectedSize =` (line 82)

```text
At K_TOP=2 (production / paper / wire format default): sizeof == 24.
At other K_TOP values (P6.1/P6.2 K_TOP sweep): the layout still has
  12 B (a_occ + a_free + a_unk) + 4*K (sem_cnt) + 2*K (sem_cls)
rounded up to 4-byte alignment for the trailing uint16_t pair.
2026-05-10: relaxed from K=2-only to allow K_TOP sweeps. K=2 remains the
production / paper default; only solo non-fusion runs are exercised at K!=2.
(The older wire formats that also pinned K=2 were removed when the wire
format was unified; SemBetaVoxel now survives only as a projection / viz type.)
```

### sembeta-default-factory

**Why the default factory is required** — in `defaultSemBetaVoxel`, attached to `inline SemBetaVoxel defaultSemBetaVoxel() noexcept {` (line 116)

```text
Beta(1,1) prior + sentinel-empty slots. **Required at every allocation**:
Bonxai's pool allocator zero-initialises new leaf blocks, which leaves
`a_occ = a_free = 0` and `sem_cls = {0,0}`. Without this factory, the
first `integrateHit` call would increment from 0 instead of from the prior,
silently mis-weighting the posterior forever. See Q5 of the 2026-05-08
design review.

Bonxai accessors take a default value to apply on first-touch, e.g.
`acc.value(coord, defaultSemBetaVoxel())`. Where the API doesn't accept
a default-constructor argument, callers must memcpy the result of this
function into the freshly-allocated voxel before any other field write.
```

### sembeta-default-slot-loop

**Sentinel loop over K_TOP slots** — in `defaultSemBetaVoxel`, attached to `for (int i = 0; i < K_TOP; ++i) {` (line 132)

```text
2026-05-10: loop over K_TOP slots so K!=2 builds (P6.1/P6.2 sweep)
don't write past the array end. The original `v.sem_cls[1] = 0xFFFF`
hardcode silently clobbered the adjacent voxel's a_occ at K=1
(since sem_cls[K=1] is a 1-element array and Bonxai's leaf pool
packs voxels contiguously) — symptom was a_occ=NaN everywhere in
the K=1 cells. Loop emits to all K slots regardless.
```

## include/scovox/tsdf_map.hpp

### tsdf-sdf-trunc

**sdf_trunc must be positive** — in `Params — declarations`, attached to `float   sdf_trunc     = 0.15f;` (line 54)

```text
Signed-distance truncation in metres. Voxels with `|sdf| > sdf_trunc`
behind the surface are skipped; in front, behaviour depends on
`space_carving`. Must be > 0; values <= 0 are clamped at construction
(TSDF disabled is not supported by this class — use a different
pipeline if you don't want a TSDF).
```

### tsdf-integrate-ray

**Curless-Levoy update along one ray** — in `TsdfMap — declarations`, attached to `void integrateRay(const Eigen::Vector3f& origin,` (line 79)

```text
Curless–Levoy update along the truncation band of one ray.

`origin`, `endpoint` are world-space positions in metres. The voxel-set
touched is identical to SLIM-VDB's openvdb DDA range up to the
acknowledged `RayIterator` parity gaps documented in §1.1 of
`docs/design/slimvdb_like_tsdf_mapping_plan.md`.

Per-voxel update inside the band:
    sdf       = sign((vc - origin) · (endpoint - vc)) · ‖endpoint - vc‖
    w         = weight_fn(sdf)
    d_clamped = clamp(sdf, -sdf_trunc, +sdf_trunc)
    d_new     = (d_old · w_old + d_clamped · w) / (w_old + w)
    w_new     =  w_old + w
Voxels with `sdf <= -sdf_trunc` (behind the surface, past the band)
are skipped. Voxels with `w == 0` are skipped (no allocation).

All touched coords are appended to the internal touched-set buffer
for `drainTouched()` (Q7).
```

### tsdf-clear-touched

**clearTouched versus drainTouched** — in `TsdfMap — declarations`, attached to `void clearTouched() noexcept { touched_.clear(); }` (line 110)

```text
O(n) clear of the touched buffer without sort+unique. Use on the
no-publisher path (e.g. dataset-mode runs without ~/scovox_bin
subscribers) where the drained coords would be discarded anyway —
drainTouched()'s sort+unique cost at Replica res 0.05 / 320×240
stride 1 is ~1 s/frame; clearTouched() is ~µs.
```

### tsdf-fused-walker-api

**Fused-walker per-voxel TSDF API** — in `TsdfMap — declarations`, attached to `void applyBandUpdate(const CoordT& c, float sdf, const WeightFn& weight_fn);` (line 181)

```text
`ScovoxMapSplit::integrateHitFused` walks the union DDA once and feeds
pre-computed SDFs to both grids. `applyBandUpdate` is the per-voxel
tail of `visit()` after the SDF math: SLIM-VDB band gate, ±trunc clamp,
weight check, Curless–Levoy update, touched-buffer push. Exposed so
the shared walk doesn't recompute SDF on the TsdfMap side.
```

### tsdf-apply-band-update

**Per-voxel TSDF band update** — in `TsdfMap — declarations`, attached to `void applyBandUpdate(const CoordT& c, float sdf, const WeightFn& weight_fn);` (line 188)

```text
Per-voxel TSDF band update at `c` for a pre-computed signed distance
`sdf` (positive = before endpoint, negative = past). Drops voxels
with `sdf <= -trunc` (SLIM-VDB band gate) and `weight_fn(sdf) <= 0`.
Otherwise clamps to ±trunc, runs Curless–Levoy weighted average, and
pushes `c` to the touched buffer.
```

## include/scovox/tsdf_voxel.hpp

### tsdf-voxel-default

**Default unobserved TSDF voxel** — in `defaultTsdfVoxel`, attached to `inline TsdfVoxel defaultTsdfVoxel() noexcept { return {0.0f, 0.0f}; }` (line 40)

```text
Default-constructed TSDF voxel: distance=0, weight=0 (= unobserved).
Matches Bonxai's pool zero-init, so existing leaf-block construction
already produces this state. Provided for explicit-construction sites
(e.g. unit tests) and as the canonical "what does an unobserved voxel
look like?" reference.
```

## include/scovox/uncertainty.hpp

### uncertainty-sembeta-overloads

**SemBetaVoxel uncertainty overloads** — in `variance`, attached to `float variance(const SemBetaVoxel& v);` (line 27)

```text
SemBetaVoxel-typed (split-grid 24-byte struct) overloads — D6 from
the resume-grilling pass. Bodies are identical to the Voxel versions
(both structs expose a_occ / a_free / a_unk / sem_cls[] / sem_cnt[]
with the same semantics); separate overloads are added rather than
templating the .cpp definitions to keep the existing library symbol
surface stable. Only the helpers actually consumed by the dscovox_node
+ scovox_node publishers are added now (variance, expectedInformationGain);
the rest can be added incrementally if a downstream caller needs them.
```

### uncertainty-distinct-classes

**Distinct-class lower bound** — in `File scope`, attached to `template <typename V>` (line 39)

```text
Lower-bound estimate of distinct classes ever observed at this voxel.
Uses only existing fields — zero extra memory.
Underestimates m, which makes the Hutter floor conservative (less
unknown mass). Acceptable if noted as a lower bound in the paper.

Template form (D6): both Voxel and SemBetaVoxel expose `sem_cnt[K_TOP]`
and `a_unk` fields with identical semantics, so the body is shared.
```

### uncertainty-hutter-escape

**Hutter adaptive escape mass** — in `hutterEscapeMass`, attached to `inline float hutterEscapeMass(int m, float N) {` (line 56)

```text
Hutter (AISTATS 2013, §3) adaptive escape mass for the sparse Dirichlet.

β* = m / [2 ln((N+1)/m)]

m: distinct classes ever observed (or lower bound)
N: total semantic observations (sum of all sem_cnt + a_unk)

Returns a floor for a_unk that gives the residual a principled
interpretation as a Dirichlet escape probability — the posterior
mass allocated to untracked classes — rather than a dump bucket
for evicted evidence.
```

### uncertainty-hutter-clamp

**Clamping the escape mass to N** — in `hutterEscapeMass`, attached to `if (ratio <= 1.f) return N;  // degenerate: m ≥ N+1, escape mass capped at N` (line 70)

```text
The escape mass is a pseudo-count for untracked classes and is bounded
above by the total observation count N — you cannot allocate more unknown
evidence than evidence actually seen. The Hutter formula assumes N ≫ m; as
ratio → 1⁺ (N small relative to m) the raw m / (2 ln ratio) term diverges
(e.g. m=1, N≈1e-4 → ratio≈1.0001 → ~5000), which is physically impossible.
Clamp to N in both the degenerate (ratio ≤ 1, i.e. m ≥ N+1) and the
near-singleton regimes.
```

### uncertainty-effective-residual

**Query-time effective residual** — in `File scope`, attached to `template <typename V>` (line 81)

```text
Effective a_unk with Hutter floor applied.

Use this at QUERY TIME (entropy, class prediction, visualization)
instead of raw v.a_unk. Do NOT use in update paths — the raw
accumulation in sparse_add and dirichlet_update_semantics must
remain unmodified so evidence is conserved exactly.

Template form (D6): shared body for Voxel and SemBetaVoxel. Both
expose `a_unk` and `sem_cnt[K_TOP]` with identical semantics.
```

## include/scovox/voxel.hpp

### voxel-k-top

**K_TOP build-time constant** — in `File scope`, attached to `#ifndef SCOVOX_K_TOP` (line 11)

```text
Number of tracked class slots per voxel. **Ship value 2** — the paper /
production configuration; every default build is byte-identical to the
hard-coded constant this replaced.

Overridable at *build* time only (`-DSCOVOX_K_TOP=n`), for the S1
sufficiency sweep (experiments/PLAN.md §3 S) which needs K ∈ {1,2,3,full}
as four separate builds. It is a struct-layout constant: every translation
unit in the workspace must see the same value, so a sweep build must pass
the flag to `colcon build` as a whole and install into its own base — never
mix objects across values.
```

### voxel-dirichlet-prior

**Default Dirichlet prior value** — in `File scope`, attached to `constexpr float kDefaultDirichletPrior = 0.01f;` (line 27)

```text
Default symmetric Dirichlet prior `α₀` applied per underlying class
dimension. **Recommended ship value `0.01`** — matches the "Beta starts
near zero" behaviour of the legacy code and minimises behavioural drift
across the SemBeta / unified-SemDir / split-Beta+Dir substrates, all of
which share this default. The launch-file knob `dirichlet_prior` exposes it
for the one-shot Jeffreys-prior ablation (`1 / (C + 1)`).
```

### sparse-add-swap-test

**The Space-Saving swap test** — in `sparse_add`, attached to `if (inc > sem_cnt[min_i]) {` (line 118)

```text
Posterior-predictive swap test (Dirichlet-Multinomial model).

The question: "Should incoming class c (with evidence `inc`) replace
tracked class j (with evidence `sem_cnt[min_i]`)?"

Under a symmetric Dirichlet prior (α₀ equal for all classes), the
posterior predictive probability of class i is:

  P(next = i) = (α_i) / (Σα)

where α_i = sem_cnt[i] + α₀ for tracked classes. Swapping c into
the tracking set is optimal when:

  (inc + α₀) / (Σα + inc) > (sem_cnt[min_i] + α₀) / (Σα)

For small inc relative to Σα (typical: inc ~ 1-2, Σα ~ 10-50),
this simplifies to:

  inc > sem_cnt[min_i]

This is exactly the Space-Saving criterion (Metwally et al. 2005),
which is near-optimal for heavy-hitter tracking (Cormode 2016).

Strict `>` (not `>=`) is a deliberate stability choice: a tied
newcomer is dropped to a_unk rather than allowed to evict. This
prevents thrashing under noisy classifiers emitting equal-weight
observations. The trade-off is a first-arrival bias for exact
ties — tolerable because exact ties are rare once any meaningful
evidence has accumulated.

The residual a_unk receives a principled interpretation at query
time via the Hutter (2013) adaptive escape mass — see
effectiveResidual() in uncertainty.hpp.
```

## src/sem_split_map.cpp

### carve-batched-path

**Batched free carve inside a frame** — in `SemSplitMap::applyCarveUpdate`, attached to `if (carve_frame_open_) {` (line 241)

```text
Batched path (a carve frame is open — the live pipeline): stage the
strongest free vote for this voxel and defer the write to flushCarveFrame.
No grid read, no wall guard: a scan trusts its own beam — every voxel it
traversed to reach a return is free NOW (see class docs). One write per
voxel per scan, block-ordered at flush.
```

### carve-occupied-wins-gate

**Occupied-wins gate on the persistent path** — in `SemSplitMap::applyHitUpdateOn`, attached to `if (carve_frame_open_ && touched_beta) carve_hits_.insert(c);` (line 359)

```text
Occupied-wins: a PERSISTENT surface return in this scan must not be carved
free even if another ray grazes through it. Gate on `touched_beta` — it is
non-null only on the persistent path (the transient/dynamic path passes
nullptr). A dynamic endpoint routes its occupancy to the transient grid and,
by the is_dynamic contract, the persistent grid stays free there, so it must
NOT suppress another ray's legitimate persistent free carve of that voxel.
```

### hit-rgbd-bki-spread

**RGB-D to LiDAR kernel spread dispatch** — in `SemSplitMap::applyHitUpdateOn`, attached to `if (prof && prof->kernel_radius > 0.f) {` (line 367)

```text
RGB-D→LiDAR BKI spread: a semantics-only source with a kernel radius spreads
its class onto nearby LiDAR-occupied voxels instead of committing at the lone
endpoint voxel `c` (which the LiDAR downsample almost never leaves occupied).
Deposits only into the Dir grid `dacc`; reads LiDAR occupancy from the
PERSISTENT Beta grid inside the helper. A no-label point (sem_probs null)
contributes nothing — a semantics-only source must not touch geometry.
```

### hit-per-source-overrides

**Per-source hit weight overrides** — in `SemSplitMap::applyHitUpdateOn`, attached to `const float w_occ     = prof ? prof->w_occ              : params_.w_occ;` (line 381)

```text
Per-source overrides (fusion). Null prof => the map's global params_ — the
single-sensor path, byte-identical. A semantics-only source passes w_occ=0
(RGB-D "pure LiDAR authority"), so Stream A is skipped by the existing
`w_occ_share > 0` guard and this voxel's occupancy stays 100% LiDAR-built;
Stream B then gates on that LiDAR occupancy. kappa0/min_p_occ are per-source
too; evidence_saturation / alpha_0 remain global (per-grid caps / priors).
```

### band-semantic-slim-vdb

**SLIM-VDB flat semantic band write** — in `SemSplitMap::applyBandSemantic`, attached to `void SemSplitMap::applyBandSemantic(const CoordT&             c,` (line 451)

```text
SLIM-VDB's Integrate (VDBVolume.cpp) walks [depth−sdf_trunc, depth+sdf_trunc]
per point because that is what the TSDF update needs, and folds the semantic
write into the same DDA iteration:

    if (sdf > -sdf_trunc_) { ...tsdf/weight...; alpha[label] += 1; }

This is that write. The caller (ScovoxMapSplit::integrateHitFused) owns the
|sdf| ≤ band gate and the endpoint exclusion; by the time we are here the
voxel has already been chosen. Keep this function branch-light: it runs once
per band voxel per point, which on KITTI is ~5 extra calls per return.
```

### band-no-label-no-other

**Band voxels skip unlabelled returns** — in `SemSplitMap::applyBandSemantic`, attached to `if (!sem_probs || sem_probs->empty()) return;` (line 466)

```text
No class signal ⇒ nothing to pool. Unlike the endpoint path we must NOT
fall through to `other += class_share` here: a bare geometric return
carries no opinion about its neighbours' classes, and dumping prior mass
into every band voxel would dilute exactly the evidence this is meant to
concentrate.
```

### band-lidar-authority-gate

**Band gate on existing LiDAR occupancy** — in `SemSplitMap::applyBandSemantic`, attached to `const BetaVoxel* b = beta_acc_.value(c, /*create_if_missing=*/false);` (line 484)

```text
LiDAR authority, read-only: no Beta voxel here means no beam has ever
stopped near this cell, so it is free space in front of the surface (or
the unobserved interior behind it) and takes no label.
`create_if_missing=false` is load-bearing — allocating would grow the
Beta grid along every ray.
```

### band-ungated-flat-weight

**Ungated band mirrors SLIM-VDB** — in `SemSplitMap::applyBandSemantic`, attached to `class_share = kappa0 * quality;` (line 495)

```text
Faithful SLIM-VDB mirror: no occupancy model, no gate, flat weight. This
is `alpha[label] += 1` with kappa0 as the unit. Skipping the Beta read is
not just a shortcut — an ungated band voxel may have no Beta entry at all,
so there is no p_occ to weight by, and inventing one (say 1.0) would
quietly re-introduce a different rule again.
```

### bki-kernel-spread

**S-BKI kernel semantic spread** — in `SemSplitMap::applyHitUpdateKernel`, attached to `void SemSplitMap::applyHitUpdateKernel(const CoordT&             c,` (line 514)

```text
For a semantics-only source (RGB-D: w_occ=0, geometry_off) with
`prof->kernel_radius = l > 0`, the class is not committed at the single
endpoint voxel `c` but spread to its neighborhood, following the Semantic
Bayesian Kernel Inference update (Gan et al., RA-L 2020, Eq. 9):

    α*ᵏ  +=  k(d) · (κ₀ · p_occ · q)      for every voxel within radius l

with the Melkumyan–Ramos compactly-supported sparse kernel (Eq. 10, σ₀=1),
which is exactly zero at d ≥ l so the neighborhood is finite:

    k(d) = ⅓(2+cos(2π d/l))(1 − d/l) + (1/2π)·sin(2π d/l),   d < l.

Pure LiDAR authority is preserved: a neighbor receives a label ONLY if it is
occupied in the PERSISTENT Beta grid (`p_occ ≥ dirichlet_min_p_occ`), so
RGB-D can never paint a voxel LiDAR hasn't confirmed as surface. `p_occ` also
weights the deposit, so weakly-occupied voxels get proportionally less label.
```

### dir-saturation-slot-floor

**Saturation floor on filled slots** — in `SemSplitMap::applyDirSaturation`, attached to `if (d->cls[i] != 0xFFFF && d->cnt[i] < alpha_0) d->cnt[i] = alpha_0;` (line 682)

```text
A FILLED slot must never scale below its α₀ prior: a slot conceptually
holds α₀ + observed evidence, and eroding α₀ makes sparse_add_class read a
negative evicted_evidence (cnt − α₀ < 0) and subtract mass from OTHER. Floor
FILLED slots only — flooring empty slots (cnt ≈ k·α₀) would re-inflate
s_class back above the saturation cap.
```

## src/uncertainty.cpp

### uncertainty-beta-diff-entropy

**entropy() is Beta differential entropy** — in `entropy`, attached to `const float a = v.a_occ;` (line 41)

```text
NOTE: this is the Beta *differential* entropy (closed form below), kept
verbatim for the legacy fused `scovox::Map` path so existing fused-path consumers
and tests retain their historical scale. It is UNBOUNDED BELOW: for a
near-point-mass voxel (e.g. a_occ=100, a_free=alpha_0=0.01, which survives
the a<=0/b<=0 guard) it returns a large NEGATIVE value, NOT a bounded
[0, ln2] occupancy uncertainty. Do NOT treat the result as Bernoulli/Shannon
entropy. Consumers that need a bounded occupancy-uncertainty signal must
compute Bernoulli H(p_occ) = -p ln p - (1-p) ln(1-p) at the call site
(as expectedInformationGain does for its H_y term, and as occupancy
map-stats mean-entropy aggregators do); we intentionally do not
change this function's semantics to avoid a silent behavior regression.
```

### uncertainty-eig-clamp

**Clamp expected information gain at zero** — in `expectedInformationGain`, attached to `return std::max(0.f, H_y - E_H);` (line 78)

```text
EIG is a mutual information and must be >= 0. Near saturation (p -> 0 or 1)
the bounded H_y term is clamped to 0 at the 1e-7 boundary while E_H stays
strictly positive, so the raw H_y - E_H goes slightly negative (e.g.
~-8e-3 at Beta(1000,1)). A negative EIG mis-ranks saturated voxels in the
next-best-view / frontier scorers and poisons mean_eig. Clamp to 0, matching
the bernoulliKL noise clamp and the EIGAlwaysNonNegative test contract.
```

### uncertainty-semantic-entropy

**Semantic entropy is bounded Shannon** — in `semanticEntropy`, attached to `float alphas[K_TOP + 1];` (line 88)

```text
Discrete categorical (Shannon) entropy over the mean Dirichlet
probabilities p_i = alpha_i / a0, bounded in [0, ln(K)].

We deliberately do NOT return the Dirichlet *differential* entropy here:
like the Beta differential entropy in entropy() it is unbounded below and
dives to large negatives on concentrated/heavily-observed voxels (e.g.
sem_cnt={1000,0} → Dirichlet(1001, …) gives a strongly negative value),
which is meaningless as a per-voxel "semantic uncertainty" and poisons any
map-level mean. The plug-in mean-probability Shannon entropy is the bounded
categorical uncertainty downstream consumers (labelling/ranking) expect and
is monotone in how peaked the categorical is, matching the documented
contract of the existing semanticEntropy tests.
```

### uncertainty-variance-matches-entropy

**Semantic variance matches semanticEntropy** — in `semanticVariance`, attached to `float a_k = 0.f;` (line 127)

```text
Match semanticEntropy: the Voxel stores raw evidence and the Dirichlet
+1 prior is added at query time (see voxel.hpp::defaultVoxel doc). The
categorical includes only observed slots plus the unknown bucket — the
same K used by semanticEntropy. Without this, entropy and variance
disagreed on under-observed cells (entropy used Dirichlet(c+1, ...)
while variance used Dirichlet(c, ...)) and downstream consumers got
mutually inconsistent uncertainty signals.
```

### uncertainty-sembeta-overloads-2

**SemBetaVoxel overloads mirror Voxel ones** — in `variance`, attached to `float variance(const SemBetaVoxel& v) {` (line 187)

```text
SemBetaVoxel-typed overloads (D6 from the resume-grilling pass) —
bodies are byte-identical to the Voxel versions above; both structs
expose a_occ / a_free with identical semantics. Separate overloads
rather than templating to keep the existing library symbol surface
stable for any downstream linker that explicitly resolves these.
```

## test/test_binary_serializer.cpp

### serializer-forged-count-test

**Forged record counts are rejected** — in `BinarySerializer.ForgedRecordCountIsRejected`, attached to `TEST(BinarySerializer, ForgedRecordCountIsRejected) {` (line 158)

```text
Regression for the forged-record-count DoS (review finding 38 / 8): a frame
with a valid MAGIC/VERSION/header but a record count (tsdf/beta/dir) far
larger than the bytes actually present must be REJECTED, not silently
accepted nor allowed to drive an unbounded reserve(). We forge each of the
three counts in turn (with an otherwise-empty body) and assert deserialize
throws so the receiver can drop the frame instead of integrating garbage.
All three stream guards validate the count against the remaining byte budget
BEFORE the reserve, so the forged count surfaces as the documented
runtime_error; std::exception is asserted as the (looser) contract.
```

### serializer-emit-size-breakdown

**Expected emit size breakdown** — in `TEST`, attached to `if (scovox::K_TOP == 2) {` (line 251)

```text
Both Beta records — coords (1,2,3), (4,5,6) — share block (0,0,0), bits
83 < 302, n=2 → mode 1 index list. Same for the Dir records.
Header: 4 + 1 + 4 + 2 + 1 + 4 + 4 + 1 = 21 B     (rev 7: +fine_ratio_log2)
TSDF:   4 (count) + 2 × 20 = 44 B                        (flat records)
Block:  12 (coord) + 1 (mode) + 2 (n) + 2·2 (idx) = 19 B
Beta:   4 (count) + 19 + 2 × 8  = 39 B                   (f32 payloads)
Dir:    4 (count) + 19 + 2 × 14 = 51 B                   (f32: other +
                             cnt 4·K + cls 1·K, K=2; rev 8 u8 class ids)
Fine:   4 (count, empty here) = 4 B                      (rev 7 tail)
```

## test/test_consensus_merge.cpp

### merge-dir-commutes-under-eviction

**Dir merge commutes under eviction** — in `ConsensusMerge.DirMergeIsCommutativeUnderEviction`, attached to `TEST(ConsensusMerge, DirMergeIsCommutativeUnderEviction) {` (line 105)

```text
E6.3 (order-invariance, pairwise leg). The Beta merge is plain addition, so
`BetaMergeIsSymmetricAndAdditive` above settles occupancy. The Dir merge is
addition PLUS truncation, and truncation is where order can start to matter —
so commutativity has to be pinned separately, and specifically in the regime
that truncates. Both cases below carry more than K_TOP distinct classes
between them, so eviction is live in every assertion.

This holds because mergeDir builds a union dict (its upsert is additive, hence
order-free) and then applies a strict total order — count desc, class id asc.
The class-id tie-break is load-bearing: without it two classes with equal
counts straddling the K_TOP boundary would be kept-or-dumped according to
which source was folded first. `DirMergeTieAtTruncationBoundaryIsCommutative`
pins exactly that case.
```

### merge-other-prior-clamp

**OTHER prior clamp at small num_classes** — in `ConsensusMerge.DefaultDirVoxelClampsOtherPriorAtKTop`, attached to `TEST(ConsensusMerge, DefaultDirVoxelClampsOtherPriorAtKTop) {` (line 161)

```text
num_classes ≤ K_TOP edge (residual_dims ≤ 0). The OTHER prior is
(num_classes − K_TOP)·α₀, which is zero at num_classes==K_TOP and would go
NEGATIVE at num_classes<K_TOP. defaultDirVoxel / mergeDir both clamp it at 0
so the prior subtraction in mergeDir can never become prior INFLATION. These
pin that clamp on the reachable scovox_core path (the receiver-side
projectBetaDirToVoxel / isPriorDir helpers live in the mapping node and are
covered by the explicit mergeFrames reject below + node tests).
```

## test/test_fine_tsdf.cpp

### fine-drift-discriminating-metrics

**Drift test compares RMS and centre** — in `TEST`, attached to `if (fit_r.valid) {` (line 266)

```text
A fixed-direction drift displaces each azimuth's arc by the drift at
its observation time, so the un-anchored surface is a distorted,
TRANSLATED circle: its fitted radius can stay near-true while the
residual and the recovered centre absorb the damage. Those are the
discriminating metrics.
```

### fine-clean-orbit-rms-floor

**Clean-orbit RMS bound** — in `TEST`, attached to `EXPECT_LT(fit.rms, 0.025f);` (line 289)

```text
The clean-orbit RMS floor on this synthetic geometry is ~0.022: the
±60° scan arcs put oblique rays into the band whose projective bias
survives tent-weighting at the band's inner edge. Well under the 0.03
model-refresh gate, and the drift test asserts the anchored map beats
the smeared one on this same metric.
```

### fine-external-model-refresh

**Region model refresh lives outside mapper** — in `TEST`, attached to `scovox::ScovoxMapSplit m(fineParams());` (line 298)

```text
The refresh loop lives OUTSIDE the mapper: a downstream estimator fits
on the fine map and re-publishes the region with the fitted model
(same id → same slot, in-place cylinder update). This contracts the
coarse-detector radius bias out of the anchor fit — see the design
doc's pose section.
```

## test/test_scovox_map_split.cpp

### split-fused-walker-parity

**Fused versus split ray walker parity** — in `Step 12.10 (2026-05-09): fused ray walker — parity vs split path`, attached to `namespace {` (line 117)

```text
The fused walker (`Params::fused_walker = true`, default) runs one
Bresenham DDA over the TSDF band [Hp - sdf_trunc·û, Hp + sdf_trunc·û]
and dispatches per-voxel into both grids. The split walker runs two
independent DDAs. For axis-aligned rays Bresenham reduces to integer
stepping along the major axis and both walkers must produce
bit-identical TsdfMap state. SemBeta state must also match exactly:
the carve subset is `0 < sdf <= carve_band, c != k_hit`, which selects
the same voxels along an axis-aligned ray as the legacy
`SemBetaMap::carveRay [co, Hp)` walk.
```

### split-semdir-jitter-tolerance

**SemDir carve jitter tolerance in fan test** — in `TEST`, attached to `const auto a = m_fused.semdirVoxelCount();` (line 244)

```text
SemDir carve sets agree up to per-ray Bresenham boundary jitter:
the fused walker starts its DDA at `Hp − walk_back·û` while the split
walker starts at `Hp − carve_band·û`, so for oblique rays the two
pick-sequences may pick different voxels at sub-voxel boundaries.
For axis-aligned rays the sets are bit-identical (see the test
above); for the 27-ray off-axis fan with 0.05 m offsets we accept
up to 20 % asymmetric difference, dominated by ±1 voxel/ray jitter
accumulating over ~27 rays in a SemDir set of ~45 voxels.
```

## test/test_sparse_add.cpp

### sparse-add-zero-increment

**Zero-increment contract of sparse_add** — in `TEST`, attached to `Voxel v = makeEmpty();` (line 142)

```text
Pin the zero-increment contract. Tracing sparse_add() on an empty voxel:
  - match loop:  needs sem_cnt[i] > 0.0f, false for every empty slot -> no match.
  - empty loop:  sem_cnt[i] <= 0.0f is true for slot 0 (0.0f) -> the slot is
                 claimed with sem_cls[0] = cls and sem_cnt[0] = inc (== 0.0f).
So a zero increment is NOT dropped: it occupies slot 0 with the given class id
and a zero count, and leaves a_unk untouched (no eviction/drop path is hit).
```

## test/test_tsdf_map.cpp

### tsdf-test-euclidean-sdf-expected

**Expected Euclidean SDF at the probe** — in `TEST`, attached to `EXPECT_NEAR(v->distance, +0.0433f, 1e-3f);` (line 52)

```text
Expected (Euclidean):
  vc - origin   = (0.075, 0.025, 0.025)
  endpoint - vc = (0.025, -0.025, -0.025)
  dist          = ‖endpoint - vc‖ = 0.025·√3 ≈ 0.0433
  proj          = 0.075·0.025 + 0.025·(-0.025) + 0.025·(-0.025)
                = 0.001875 - 0.000625 - 0.000625 = +0.000625 → sign +1
  sdf           = +0.0433 (in front of surface, inside the band)
```

### tsdf-test-centre-sdf-expected

**Expected SDF at the voxel centre** — in `TEST`, attached to `EXPECT_NEAR(v->distance, -0.0354f, 1e-3f);` (line 81)

```text
Expected SDF for vc = (0.075, 0.025, 0.025), endpoint = (0.075, 0, 0):
  endpoint - vc = (0, -0.025, -0.025)
  dist          = 0.0354
  (vc-origin)·(endpoint-vc) = 0.075*0 + 0.025*(-0.025) + 0.025*(-0.025) = -0.00125
  sign          = -1
  sdf           = -0.0354
```

### tsdf-test-lower-corner-alternative

**What a lower-corner sample would give** — in `TEST`, attached to `EXPECT_LT(std::fabs(v->distance + 0.0354f), std::fabs(v->distance - 0.025f));` (line 89)

```text
If the implementation used voxel LOWER CORNER (= 0.05, 0, 0) instead:
  endpoint - corner = (0.025, 0, 0), dist = 0.025
  (corner-origin)·(endpoint-corner) = 0.05*0.025 = +0.00125  → sign +1
  sdf = +0.025
Make sure we are NOT seeing that.
```

## test/test_uncertainty.cpp

### uncertainty-beta-entropy-trap

**The Beta differential entropy trap** — in `Uncertainty.EntropyBetaNearPointMassDivergesNegative`, attached to `TEST(Uncertainty, EntropyBetaNearPointMassDivergesNegative) {` (line 87)

```text
scovox::entropy() is the Beta DIFFERENTIAL entropy, which is unbounded
BELOW and diverges to large negatives on near-point-mass voxels — the
documented "entropy trap". This is exactly the state every occupied
split-substrate voxel reaches: a_occ = C*alpha_0 + accumulated evidence
while a_free stays pinned at the prior alpha_0 (=0.01, C=14 -> occ prior
0.14). Pin the divergence so nobody re-uses entropy() as if it were a
bounded Shannon stat: at Beta(0.14+50, 0.01) the closed form is ~-99
(not in [0, ln2]). The map's "mean Shannon entropy" stat deliberately uses
the bounded Bernoulli form below precisely because this poisons the mean.
```

### uncertainty-bernoulli-entropy-bound

**Bounded Bernoulli entropy used in production** — in `Uncertainty.BernoulliShannonEntropyBoundedOnNearPointMass`, attached to `TEST(Uncertainty, BernoulliShannonEntropyBoundedOnNearPointMass) {` (line 109)

```text
The bounded occupancy-uncertainty stat that production code (e.g. the
occupancy map-stats aggregator, and the H_y term inside
expectedInformationGain) uses INSTEAD of entropy() on the same near-
point-mass voxel: Bernoulli Shannon entropy H(p_occ) with p_occ =
a_occ/(a_occ+a_free). It is bounded in [0, ln2] regardless of how
concentrated the Beta is — the property entropy() above lacks.
```

## test/test_voxel_layouts.cpp

### voxel-layout-zero-init

**Zero-initialised SemBetaVoxel reads p_occ 0.5** — in `TEST`, attached to `scovox::SemBetaVoxel v{};` (line 44)

```text
Bonxai pool zero-init: leaves a_occ = a_free = 0, sem_cls = {0,0}.
p_occ() must return 0.5 (the s == 0 fallback), not NaN.
This is the exact state that defaultSemBetaVoxel() rescues us from
— keeping it under test makes the contrast visible if someone ever
breaks the factory invariant.
```
