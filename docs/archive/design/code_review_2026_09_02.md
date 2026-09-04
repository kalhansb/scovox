# Code-smell review of the mapping stack — 2026-09-02

Five independent Opus 5 reviewers over the whole scovox mapping code: the
Beta/occupancy core, the semantic (Dirichlet / top-K) layer, the ray-traversal
and TSDF layer, serialization + build flags, and the ROS node layer. 63
findings were reported. **Every one was re-checked against source before it was
written down here** — the reviewers' claims are not the record; the verified
subset is.

That check was not a formality. It removed four claims outright and corrected
the scope or the stated mechanism of five more (§5). Confidence ratings in this
note are mine, from that re-check, not the reviewers'.

---

## 1. The headline: no finding moves a published number

The promoted candidate replays with `--evict-by-confidence --sem-band 0.10
--w-occ 1.5 --w-free 1.0 --dump-below-gate-as-unknown --dump-label-gate 0.5`.
It passes neither `--fused-walker` nor `--tsdf-enabled`, so it takes the
defaults in `replay_scenenn.cpp:43-44`: **`fused_walker = 1`, `tsdf_enabled =
0`.**

Those two defaults quarantine the most alarming findings:

- The TSDF band divergence (§4.1, §4.2) is gated behind `tsdf_enabled_` at
  `scovox_map_split.hpp:429`. With `tsdf_enabled = 0` **`applyBandUpdate` is
  never called on any published run.** No TSDF voxel was written; a disagreement
  about which voxels would receive a band value cannot move mIoU.
- The fused/split carve divergence (§4.3) is real, but *every* published run
  used the fused walker. The two walkers disagree with each other, not with
  themselves. No published comparison straddles the boundary.
- The `mergeDir` / `qmax` defect (§3.4) reaches `dscovox_node`, the binary
  serializer and tests. `replay_scenenn` calls none of them.

What the findings *do* invalidate is a comparison nobody has published yet: any
future fused-vs-split A/B is comparing two different models, not two
implementations of one. That is recorded in §4.

---

## 2. Fixed today

Three defects were in code written earlier the same day. All three are fixed,
and each fix is now covered by a test that fails without it.

### 2.1 The lattice step ignored its own scale knob — **fixed**

`beta_voxel.hpp` hard-coded `kBetaLatticeStep = 1.0f/8.0f` in the float branch
while the header 130 lines above advertises other scales ("16 -> 0.0625 /
4095.9, 2 -> 0.5 / 32767.5"). Under `-DSCOVOX_BETA_U16_SCALE=2` a float build
and a u16 build would have snapped the *same* weight to *different* values —
precisely the "two models, not two storage layouts" failure the constant exists
to prevent, committed inside the mechanism meant to prevent it.

The step now derives from `SCOVOX_BETA_U16_SCALE` in both branches, with a
`static_assert` under u16 that it still equals the storage resolution. Measured
across both storage modes at three scales:

| scale | u16 step / snap(1.3) | float step / snap(1.3) |
|---|---|---|
| 8 (shipped) | 0.125 / 1.25 | 0.125 / 1.25 |
| 2 | 0.5 / 1.5 | 0.5 / 1.5 |
| 16 | 0.0625 / 1.3125 | 0.0625 / 1.3125 |

Before the fix the float column read 0.125 / 1.25 on all three rows.

### 2.2 A guarantee the fusion path did not have — **fixed**

The comment claimed `SemSplitMap`'s sanitiser snaps the weights "so it holds by
construction". It does not: `sanitise()` guards `Params`, and `HitWeights`
profiles are handed straight to the integration call. The per-ray reads
(`sem_split_map.cpp:400, 425, 598, 646`) take `prof->w_occ` / `prof->w_free`
raw. The count identity therefore held on the single-sensor path and failed
silently on the fusion path — the worse of the two, because no unit test that
exercises only `Params` can see it.

Added `sanitise(HitWeights&)` and called it from `buildFusionProfiles`
(`scovox_node.cpp`), the only construction site for either profile. Snapping
once at construction leaves the hot path reading raw. The comment now says what
is actually true, including that a third entry point would have to snap for
itself.

Reach was narrower than first reported: `lidar_w_occ`/`lidar_w_free` *default*
to the already-sanitised `P.w_occ`/`P.w_free`, so only an explicit override
could go off-lattice. But `default_params.yaml:46` recommends `w_free = 4.67`
for LiDAR, and 4.67 / 0.125 = 37.36 — off-lattice. The documented
recommendation was the trigger.

### 2.3 A sentinel meaning two opposite things — **fixed**

`beta_max_increments` returned `0.0` both for "float storage, no ceiling" and
for "`w <= 0`, admits nothing". It also had exactly one reference in the tree:
its own definition. It now returns `infinity` for unbounded and `0` for admits-
nothing, and a test asserts the ≈5,460-hit headroom figure that
`count_storage_2026_09_02.md` publishes, so the note and the code cannot drift
apart.

**Verification after all three:** core 178/178; `-DSCOVOX_BETA_U16=1` 181/181
(the `static_assert` compiles and the u16 headroom arm passes, confirming the
5,460 figure); ROS 300 distinct cases, 0 failures.

---

## 3. Confirmed defects — semantic layer

### 3.1 The eviction-rule comment names the wrong default — HIGH
`dir_voxel.hpp:296-297` states "which is why `evict_by_confidence` is on by
default." Every default in the tree is `false`: `sem_split_map.hpp:206`,
`scovox_node.cpp:391`, `:3146`, and `FEATURES.md:117`.

The confusion is a real one worth naming: the *macro* `SCOVOX_TRACK_QMAX` is on
by default (`dir_voxel.hpp:60`, correctly documented), the *parameter* that
reads `qmax[]` is off. The comment conflates them.

This matters more than a typo because the comment also records that the
default arm's measured eviction rate is **0.0000** — "the first K_TOP classes
to arrive own this voxel forever". A reader trusting the comment concludes the
safe arm is the default and does not pass the flag. The promoted candidate
passes `--evict-by-confidence` explicitly, so results are unaffected; the
hazard is for anyone who does not.

### 3.2 The mesh labeller hard-codes 14 classes — HIGH
`labelMesh`/`labelPointCloud`'s DirVoxel overloads (`mesh_labelling.hpp:117-121`)
take no `num_classes`, so `dominantClass` falls back to its default of 14.
`sem_split_map.cpp:1049` and `:1172` pass `params_.num_classes`; the mesh
labeller cannot. On a 20-class config, `other_prior` is computed as 0.12
instead of 0.18, overstating OTHER's observed evidence by 0.06, and every
surface voxel whose winning slot holds less than 0.06 observed evidence is
exported as unknown while `dominantClassAt` on the same voxel returns the
class. The bias is systematic and one-directional.

Does not touch SceneNN results (`replay_scenenn` dumps via `forEachCell`, never
the mesh labeller) but does affect the SemanticKITTI launches, which set
`num_classes: 20`.

### 3.3 Two tie-breaks for one tie — MEDIUM
`dominantClass` (`dir_voxel.hpp:352`) uses strict `>` over slots, so the first
*filled slot* wins a tie — arrival order. `mergeDir`
(`consensus_merge.hpp:155`) breaks the identical tie by `x.cls < y.cls` —
lowest class id. Exact ties are reachable, not theoretical: MAJORITY_VOTE
deposits integer `+1.0f`, so two classes with three votes each sit at exactly
`α₀ + 3`.

Same header also has `best_evidence` initialised to `0.f` under a strict `>`,
so a filled slot sitting at exactly `α₀` reads as unfilled and returns
`0xFFFF`.

### 3.4 `mergeDir` leaves `qmax` zero on filled slots — MEDIUM, latent
`consensus_merge.hpp:116` builds `DirVoxel f{}` and writes only `other`,
`cnt[i]`, `cls[i]`. `dir_voxel.hpp:121-122` states that zero-init "is only ever
the state of an EMPTY slot". With `SCOVOX_TRACK_QMAX` on and
`evict_by_confidence` enabled, the admission test `q_fx > qmax[min_i]` against
a zero `qmax` admits *every* observation — the grid degrades to
last-observation-wins.

Latent today: `dscovox_node` only merges into the fused grid and never calls
`sparse_add_class` on it. Any future path that deposits into a merged or
deserialised grid trips it.

### 3.5 `setTopkTrunc(16)` silently means "no truncation" — MEDIUM
`topk_provider.hpp:49`: `topk_trunc_ = (n > 0 && n < kMaxTrunc) ? n : 0;` with
`kMaxTrunc = 16`. A sweep arm recorded as N=16 runs untruncated, with no
diagnostic — the same silent-misconfiguration class the header's `#error` traps
were added to prevent. Separately, `truncate` keeps every tie at the cut and
probabilities are dequantized from uint8, so "top-1" admits every class whose
probability rounds to the same byte as the argmax.

---

## 4. Confirmed defects — traversal and TSDF

### 4.1 The fused and split TSDF walks are not the same line — HIGH (inert today)
`ExactRayIterator` aims at the **centre** of the destination coord
(`ray_iterator.hpp:39-44`), not at the caller's `end_pos`. The fused walker
starts at `endpoint − max(depth,trunc)·u` — the ray origin — while
`tsdf_map.cpp:93-99` starts at `endpoint − trunc·u`, 0.15 m out. Two different
start points converging on one fixed centre are two different segments.

The reviewer measured 80.9% of oblique rays producing different TSDF voxel sets
(mean symmetric difference 3.09 voxels against a mean band of 9.54), against
2.25% for axis-aligned rays — which is why the axis-aligned parity tests pass
and why `MultiRayBandIdentity` had to be written to tolerate 20% divergence
attributed to "phase jitter". That attribution understates the cause.

**Inert on every published run** (`tsdf_enabled = 0`, §1). It matters for the
SLIM-VDB equivalence claim at `tsdf_map.cpp:2-9`, which holds for
`integrateHitSplit` only — the production row does not walk SLIM-VDB's voxel
set.

Same block: the `walk_back` ternary at `:236-238` is effectively dead, both
arms equalling `depth` for any ray with `depth >= trunc`.

### 4.2 Half-voxel slack where a half-diagonal is needed — MEDIUM (inert today)
`scovox_map_split.hpp:429` gates the band write at `sdf <= trunc + h` with
`h = 0.5·resolution`, but a voxel whose centre the DDA reaches can sit up to
`h·√3` off the ray. At res 0.05 the gate cuts at `trunc + 0.025` where the
reachable bound is `trunc + 0.0433`. The split path has no upper gate at all.
Measured to drop a voxel the split path writes on 4.1% of rays. Also inert
under `tsdf_enabled = 0`.

### 4.3 The origin voxel is carved by one path and not the other — MEDIUM
`carve_band = depth` (`scovox_map_split.hpp:210`) is the whole ray, so the
fused carve gate reduces to `sdf > 0`. `sdf` is measured from the voxel
*centre*, so when the origin lies in the far half of its own voxel the gate
rejects it; measured on 50.8% of rays. The split path
(`sem_split_map.cpp:388-412`) hands `origin` to `ExactRayIterator`, which
calls the body on `coord_from` first, and carves it unconditionally.

One correction to the reported mechanism: for the *carve* the two walks share a
start point — `start_pos = endpoint − depth·u = origin` exactly. They diverge
because they aim at different targets: `centre(k_far)` where `k_far =
coord(endpoint + back_reach·u)`, versus `centre(k_end)` where `k_end =
coord(endpoint)`. The start-point difference in §4.1 is specific to the TSDF
comparison.

Consequence: a persistent per-frame `a_free` deficit at the sensor pose in the
fused row relative to the split row. Since all published runs are fused, this
is an inter-path inconsistency, not a results defect — but it means the
recorded "fused is 10-18% slower than split" compares two different amounts of
work, not two implementations of one.

### 4.4 `tsdf_enabled` and the sanitiser that undoes it — CONFIRMED, documented
`scovox_map_split.hpp:531` (the split branch) has no `tsdf_enabled_` test;
`:429` (the fused branch) does. Already recorded, and the code comment at
`:528-533` cross-references the removal ledger.

The reviewer added one detail worth keeping: `tsdf_map.cpp:24-28` rewrites
`sdf_trunc <= 0` back to `0.15f`, so the obvious way to disable TSDF is a
no-op at the `TsdfMap` level. The node already works around this by threading
`SP.tsdf_enabled = (sdf_trunc_launch_ > 0.f)` (`scovox_node.cpp:108`), and
documents the workaround at `:100-108`. The residual gap is exactly the split
branch.

**One reviewer got this backwards** — see §5.

### 4.5 Degenerate rays: guarded on one path, not the other — LOW
`scovox_map_split.hpp:198-208` returns early on `depth < 1e-4f`. The split path
(`sem_split_map.cpp:362-376`) has no such guard, so it still commits the hit,
the ray-spread (whose direction is numerical noise) and the endpoint carve.
Not reachable through either shipped sensor path — `min_depth` 0.1 and
`min_range` filter first.

### 4.6 Dead traversal code — LOW
`TsdfMap::visit` takes a `trunc` it `(void)`-casts (`tsdf_map.cpp:160`).
`TsdfMap::linear` and `TsdfMap::rangeDecay` have zero callers tree-wide, and
`rangeDecay`'s comment says the weighting "moved to `SemBetaMap`" — a class
that no longer exists. `tsdf_map.hpp:15-24` names `SemBetaMap` three times.
An always-true branch at `tsdf_map.cpp:126` follows the `k0 == k_far` early
return.

---

## 5. Claims removed or corrected during verification

This section exists because the review's value was as much in what it got wrong
as in what it found.

**Removed — the half-voxel offset does not shift mIoU.** A reviewer claimed
`marching_cubes.hpp`'s missing centre offset "shifts mIoU when comparing
against a VDBFusion/SLIM-VDB baseline". `replay_scenenn.cpp` dumps via
`forEachCell` (`:707, :755, :777`) and never calls `extractPointCloud`. The
offset is a real API inconsistency between the `Voxel` and `TsdfVoxel`
overloads; it is not a results defect.

**Removed — the split path does not "now honour `tsdf_enabled`".** One reviewer
concluded the standing note was outdated because `scovox_node.cpp:108` threads
`SP.tsdf_enabled` through. Threading it through is not consulting it: the split
branch at `scovox_map_split.hpp:531` has no such test, and only the fused
branch at `:429` does. The note stands. A second reviewer had this right; the
disagreement is why both were checked.

**Corrected — `mergeDir`/`qmax` scope.** Real, but reaches only `dscovox_node`,
`binary_serializer` and tests, not the single-robot replay path.

**Corrected — profile lattice reach.** `lidar_w_free` defaults to the sanitised
`P.w_free`; only an explicit override bypassed the lattice (§2.2).

**Corrected — the core/node weight defaults.** A reviewer described unit tests
validating 2:1 while production runs 2:1, which says nothing. The ratio is 2:1
in both (core 1.0/0.5, node 2.0/1.0); what differs is absolute concentration,
by 2×. That affects `evidence_saturation` and the `p_occ` convergence rate, not
the promoted config, which passes 1.5/1.0 on the CLI.

**Demoted to hypothesis.** `semantic_band_length` is absent from the
`stageable` exclusion list (`sem_split_map.cpp:594-596`) though
`applyBandSemantic` reads Beta mid-scan. Gate and call graph confirmed; the
size of the delayed set is not measured. Not a finding until it is.

---

## 6. ROS node layer

### 6.1 Blocking TF lookups under the writer lock — HIGH
`onImages` takes `unique_lock(map_mtx_)` at `:1307`, then does its TF lookups
at `:1334` and `:1344`. `onPointCloud` takes it at `:1796` and looks up at
`:1819`/`:1833`. `scovox_bin_min.yaml:75` ships `tf_lookup_timeout_sec: 1.0`,
so a scan whose exact-stamp pose has not arrived blocks for up to 1.0 s (2.0 s
with `tf_require_exact:false`) holding the exclusive lock. Throughout that wait
the viz timer's `shared_lock` cannot be acquired — the exact coupling the
separate `viz_cb_group_` was introduced to remove. The design comment at
`:187-198` asserts TF is in the pre-lock phase. It is not.

`publishPlanningMap` (`:2631`) has the same shape: a 0.05 s `Time(0)` wait
under the lock, where `publishBinaryMap` at `:2218-2226` uses a zero timeout
and documents at length why a nonzero one would be wrong. Same lock, same
semantics, opposite policy.

### 6.2 Parameters that are declared, logged, and read by nothing — HIGH
Five, each verified by exhausting its references:

| Parameter | Declared | Status |
|---|---|---|
| `band_only_integration` | `:360` | Only reader is `scovoxmap.cpp:266` in the legacy `scovox::Map`, which the node never instantiates. **Printed in the startup banner at `:289` as if active.** |
| `grazing_angle_threshold` | `:382` | No reader anywhere. Node default `-1.0` vs core default `0.3f` (`map_interface.hpp:139`); `default_params.yaml` documents it as a live weighting knob. |
| `semantic_top_k` | `:337` | Clamped, warned about, never forwarded into `SP`. The identical-looking clamp in `dscovox_node.cpp:160` *is* live — two nodes, one parameter name, one inert. |
| `robot_id` | `:455` | Declared, member declared, nothing else. |
| `topk_probs_max_k` | `:797` | The K-width never reaches `TopkProvider`. |

`band_only_integration` is the worst of these: a benchmark run with it set to
true logs `band_only=1`, runs the full-ray walk, and reports a band-only
integration cost that was never measured — the identical silent-no-op failure
the comment one line below at `:293-296` was written to prevent.

### 6.3 `min_range`/`max_range` are inert on RGB-D — HIGH
`scovox_node.cpp:1231`: `if (P.range_decay_length > 0 && (rng<P.min_range||rng>P.max_range)) continue;`
`range_decay_length` defaults to `-1.0` and every shipped YAML sets `-1.0`, so
the cull never fires. The LiDAR path applies the same two parameters
unconditionally (`:1685`, `:1749`). An operator raising `max_range` on an RGB-D
config sees no effect and no warning; the same edit on a LiDAR config works.
`default_params.yaml` documents both as unconditional.

The gate is a leftover shape: the comment at `:1229-1230` records that the cull
"has always been gated on `range_decay_length > 0`; it kept that shape when the
decay weight itself was removed."

### 6.4 `range_decay_length` is dead in the core, with three defaults — MEDIUM
`scovox_node.cpp:126` forwards it into `SemSplitMap`, where `sem_split_map.cpp:257`
clamps it and nothing ever reads it. Defaults disagree three ways: `5.0f`
(`map_interface.hpp:136`), `50.0f` (`sem_split_map.hpp:360`), `-1.0` (node).
Its only surviving effect anywhere is the RGB-D gate in §6.3, which is not what
its name or its documentation describes.

### 6.5 Unvalidated divisors — MEDIUM
`scovox_publish_rate` (`:186`) is used as `1.0/sm_rate` at `:200` with no
guard, while the sibling `share_rate_hz` at `:215` and `dscovox_node`'s
`publish_rate_hz` at `:233` are both guarded. Setting it to 0 — the idiom that
disables the other two — yields `+inf`, and converting that to `int64_t`
nanoseconds is UB. `planning_map_resolution` (`:467`) is a divisor in four
expressions and unvalidated, where its sibling `stride` is clamped at declare
time.

### 6.6 RGB-D buffer geometry is unvalidated — MEDIUM
`:1203`/`:1215` `reinterpret_cast` into `depth->data` and `seg->data` with no
check that `data.size() >= H*step` or `step >= W*sizeof(px)`. The LiDAR path
does exactly this at `:1555`, with a comment stating the rationale ("a crash or
silent garbage integration on adversarial / buggy input") that was never
applied to the image path. Separately, `di_` (CameraInfo) supplies intrinsics
at `:1307` with no check that its dimensions match the depth image — a
downscaled depth topic silently reprojects with wrong intrinsics.

### 6.7 Two publishers, one field, two quantities — MEDIUM
`dscovox_node.cpp:806` writes `max_semantic_classes = (uint8_t)top_k_` (the
per-voxel slot width, 2); `scovox_node.cpp:2051` writes `max_sem_` (the label-
space size, 10–20). Both streams carry `class_id` up to `max_sem_-1`, so a
consumer sizing an array from this field on the merged topic indexes out of
bounds.

### 6.8 A wrong-length `gyro_bias` is silently discarded — MEDIUM
`:757` `if (gb.size() == 3) gyro_bias_ = ...;` with no `else`. A two-element
typo leaves the bias at zero and every deskew rotation integrates uncorrected
gyro, with no log line. The same function warns on three other malformed
inputs.

### 6.9 Comments describing code that is not there — LOW
`:349-352` says the split-grid path "can't honor" `enable_tsdf` "so the
constructor warns". There is no such warning anywhere in the file, and `:108`
threads the intent through. The same paragraph describes the deleted legacy
fused path as if live. `:111-112` says `semantic_occ_gate`, `min_range`,
`max_range` and `grazing_angle_threshold` are "node-level sensor filters
consumed BEFORE integrateHit": of the four, `grazing_angle_threshold` has no
reader, `semantic_occ_gate`'s only read is a published message field at
`:2051`, and `min_range`/`max_range` are conditional (§6.3). One of four is
described correctly.

---

## 7. Build flags, serialization, tooling

- **`SCOVOX_E0_COUNTERS` has no `#ifndef` default** — tested with bare `#if` at
  `e0_counters.hpp:33`, `sem_split_map.cpp:112`, `:157`, while all three
  siblings have one (`beta_voxel.hpp:50`, `dir_voxel.hpp:65`,
  `sem_split_map.hpp:62`). This is the undefaulted-macro-compiles-to-zero
  hazard, in the one place the pattern was not applied.
- **`split_memory_demo.cpp:188`'s sanity check is backwards.** `semdirVoxelCount()`
  is hit-sparse and `tsdfVoxelCount()` is band-wide, so hit-sparse < band is
  correct — and the tool prints "FAIL - bug in carve geometry?" and returns 1.
- **`split_memory_demo.cpp:162,164` under-reports.** Per the header's own words
  (`scovox_map_split.hpp:722-723`), voxel count reports the Dir grid while
  bytes report Beta + Dir combined, so the demo's "Bonxai total" gap silently
  contains the entire Beta grid. This tool feeds a paper table.
- **`-Wcomment` at `split_memory_demo.cpp:17`** — a trailing `\` on a `///`
  line swallows line 18. Pre-existing (byte-identical at HEAD); it surfaced
  because the quality removal touched that TU and forced a recompile.
- **Dead:** `dumpEvictStats` exists only in a comment at `voxel.hpp:38`;
  `version.hpp` has no includers; `ssmiOccKL`/`ssmiFreeKL` have zero references.
- **`ScovoxMapSplit::resolution_`** (`scovox_map_split.hpp:115`) is the only
  unsanitised copy of the resolution, and it is the one `resolution()`,
  `extractMesh` and `extractPointCloud` return to the outside world.

---

## 8. Checked and clean

Recorded so the same ground is not re-covered:

- **uint16 saturation in the semantic layer** — `cnt` is float; `qmax` writes
  are short-circuited at 1.0 and NaN-excluded upstream; `nhit` saturates
  explicitly.
- **Class-0 / empty-slot conflation in the DirVoxel path** — the sentinel is
  `0xFFFF`, and class 0 is excluded upstream by both the one-hot (`lbl > 0`)
  and top-K (`j = 1`) paths. The conflation exists only in the legacy
  `semantics.hpp` path, which the split substrate does not use.
- **Mass conservation on eviction** — all four branches of `sparse_add_class`
  conserve `Δ(other + Σcnt) == inc`.
  *(2026-09-04: this finding is now moot rather than merely cleared. The basis
  change removed `sparse_add_class`'s `float* other` parameter — no branch adds
  mass at all, the caller adds it once before branching, and `other` is derived.
  A future reviewer should not re-audit the four branches for this property.
  See dir_total_basis_2026_09_04.md §2.)*
- **Voxel-centre vs lower-corner in the walkers** — the fused walker's inlined
  `coordToPos` is bit-identical to Bonxai's followed by the `+h` narrowing, in
  the same order. Every SDF sample in both paths is taken at the voxel centre.
- **Removed `quality` factor residue** — grepped `src/` and `config/`; the only
  hits are an unrelated TF quality gate and substring matches. The Dirichlet
  increment carries no quality scaling. (`TsdfMap::rangeDecay` survives as dead
  code, §4.6.)
- **`#if` blocks with no build define** — the four removed axes are `#error`
  traps at `dir_voxel.hpp:90-101`, not silently-zero `#if`s. `SCOVOX_E0_COUNTERS`
  is the exception (§7).
- **Direction normalisation, `far_thr` sizing, the `k0 == k_far` branch,
  truncation-band asymmetry, `tsdf_voxel.hpp`, `ray_iterator.hpp`** — no
  findings.

---

## 9. Not fixed

Everything in §3–§7 except the three in §2 is recorded, not repaired. Nothing
there moves a published number (§1), and the standing halt on experiments means
none of it is on the critical path. The ones worth taking first, in order:

1. **§6.2 `band_only_integration`** — it is logged as active and does nothing.
   That is a measurement trap, and it is one line to either wire up or reject.
2. **§3.1 the eviction-rule comment** — one wrong word steering a reader toward
   an arm whose measured eviction rate is zero.
3. **§6.1 TF under the writer lock** — the largest real runtime defect here,
   and the rationale for the fix is already written out at `:2218-2226`.
4. **§6.3 the RGB-D range cull** — two documented parameters that silently do
   nothing on one of the two sensor paths.

---

## 10. Addendum, 2026-09-03 — a defect this review missed

Running the fused-vs-split A/B that §4 motivates surfaced a memory defect none
of the six agents reported, and it is worth recording *why* they missed it.

**`replay_scenenn.cpp` never cleared `TsdfMap::touched_`.** `TsdfMap` appends
every band-written voxel to `touched_` (`tsdf_map.cpp:193`) so a downstream
consumer can publish only the changed cells; the list is emptied solely by
`drainTouched()` / `clearTouched()`. The replay drains the *semantic* lists
each frame (`clearTouchedSemDir()`, `:586`) under a comment that describes this
exact failure mode — "without this call the lists grow monotonically for the
whole run" — but the TSDF grid keeps a **separate** list, and that one was
never touched. It therefore grew for the entire run.

Measured on scene 016 (1300 frames), split walker:

| | peak RSS | s/frame | outcome |
|---|---|---|---|
| before | **10.5 GB** | 0.128 | OOM-killed at frame 800 |
| after `map.clearTouchedTsdf()` | **96 MB** | 0.098 | exit 0, 1 300 057 voxels |

**Why every reviewer missed it, and why it never bit before.** The defect is
invisible in the only configuration anyone had ever run offline. The shipped
replay defaults are `fused_walker=1`, `tsdf_enabled=0` (`:43-44`), and the
fused walker gates its band writes on `tsdf_enabled_`
(`scovox_map_split.hpp:429`) — so on every published run **nothing is ever
appended to the list**. Reaching it requires either `--tsdf-enabled 1` or the
split walker, whose `tsdf_.integrateRay` at `:531` is unconditional. This is
the same quarantine mechanism §1 invokes to argue that no finding moves a
published number, seen from the other side: the defaults that make the review's
TSDF findings inert are also what hid this one. A static review of the source
is not enough to find a bug whose only symptom is unbounded growth in a
configuration the defaults never select.

**Scope: no published number moves.** No published run appended to the list,
and the ROS node clears it every frame
(`scovox_node.cpp:1277`, `:2183`, `:2315`), so live mapping was never affected.
The replay has no consumer of the deltas at all, so clearing changes no map
state — confirmed empirically: the fused arm at band-off reproduces the
published `u16ab` run's occupancy on scene 016 exactly (IoU 0.5580,
n_pred 14 939 in both).

**What it cost.** This is the concrete reason the two walkers had never been
compared on mIoU offline: the split arm could not finish a single scene. A
whole region of the configuration space was unreachable, and the only way to
discover that was to try to enter it.
