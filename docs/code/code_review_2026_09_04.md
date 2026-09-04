# SCovox code review — 2026-09-04

**Scope.** Read-only review of the working tree at commit `d5da6a8` plus its
31 uncommitted source files, covering `src/scovox_core`, `src/scovox_mapping`,
`src/scovox_msgs`, the launch and config files, and the documents that the
code cites. No code was changed. No build or test was run for this review;
the test figure quoted below is the last one recorded in the archived design
notes. The previous review (`docs/archive/code/code-review-2026-06-23.md`)
predates the split-grid refactor and is superseded in full.

**Method.** Every finding was checked against the source at the cited
`file:line`; line numbers are working-tree numbers on 2026-09-04. Companion
document: `docs/scovox_code_structure.md` (best method vs current code).

**Status — updated 2026-09-04, later the same day.** The review was read-only,
but the work it prompted is not, so each finding now carries a status and the
resolved ones say what was done and where. Line numbers in a finding are the
ones it was written against; where a fix moved them, the resolution note gives
the new location. One finding (H4) was added after the review, from evidence a
static read could not have produced: an 8-scene re-score on the current binary.

## Summary

| id | severity | finding | status |
|---|---|---|---|
| H4 | **High** | **The published numbers do not describe the code in the tree** | **open — needs a re-run, not an edit** |
| H1 | High | `scovox_node` defaults are not the promoted configuration | **resolved** — `config/scovox_best_method.yaml`, commit `33aa576` |
| H2 | High | The promoted state is uncommitted (31 files since `d5da6a8`) | **resolved** — `1101e53`, `33aa576` |
| H3 | High | `integrateHitSplit` ignores `tsdf_enabled` | open |
| M1 | Medium | Four parameter structs with four sets of defaults | open |
| M2 | Medium | Six library knobs have no ROS parameter | open — now cross-referenced from the yaml |
| M3 | Medium | `range_decay_length` is dead in the split path but still documented as a weight | open |
| M4 | Medium | One known failing test, `FarCarveBitIdenticalToFullWalk` | open — but re-measured, and a second failure found and fixed |
| M5 | Medium | `dscovox` fused grids ignore `dir_leaf_bits`; first pose wins; fold order matters | open |
| M6 | Medium | Wire block runs hard-code `leaf_bits = 3` | open |
| M7 | Medium | `evidence_saturation` is one knob for two caps | **partly resolved** — trap documented at the point of use |
| M8 | Medium | Experiment write-ups still print the demoted build flag | **resolved** — plus a second error found in the same block |
| L1 | Low | Stale byte-size and type-name comments | open |
| L2 | Low | In-code references to moved or nonexistent documents | **partly resolved** — code swept; `scovox_slot_rules/scripts/` not |
| L3 | Low | README document index broken by the archive move | **resolved** |
| L4 | Low | Legacy voxel / map types still compiled and tested | open |
| L5 | Low | Stale tool and comment text around the wire format | open |
| L6 | Low | `downsample_voxel_size` default described two ways | open |

---

## High

### H4 — The published numbers do not describe the code in the tree

*Added 2026-09-04 after the review, from an 8-scene re-score. A static read
could not have found this one.*

**Where.** `scovox_slot_rules/results_*/scenes/*.json` (every published scene
JSON) against the current `sem_split_map.cpp` / `ray_iterator.hpp`.

**Evidence.** Every baseline in `results_e9` was produced by
`.build/e5/k2_i0_evid/replay_scenenn`, md5 `60e17da907d0`, built
**2026-08-30 04:48**. Three commits after that change what the deposit path
does: `e316f07` (hit batching default-on, `uint16` Beta), `d5da6a8` (the exact
Amanatides & Woo walk replaces Bresenham unconditionally) and `1101e53` (the
count-storage promotion). Re-running the promoted arm on the current binary —
same CLI, same frames, same trajectory — does not reproduce the published
numbers:

| scene 016 | published | current binary |
|---|---|---|
| `n_pred_occupied` | 20 230 | 14 939 |
| precision | 0.6754 | 0.7587 |
| recall | 0.8178 | 0.6784 |
| occupancy IoU | 0.5870 | 0.5580 |
| intersection mIoU | 0.5907 | 0.6070 |
| union mIoU | 0.38293 | 0.38069 |

Scene 015 moves further: union −0.0381, occupancy IoU −0.0618, intersection
+0.0166. Both arms fire byte-identical ray counts (016: 87 197 891 hits,
12 641 878 no-return carves), so the input is identical and the difference is
in what the walk deposits.

**Leading cause: hit batching, not traversal.** `sem_split_map.cpp:606-614`
stages a surface hit and keeps only the frame's strongest ray for that voxel
(`if (!st.staged || w > st.w_occ_share)`), flushing one deposit per scan. A
640×480 frame puts hundreds of rays into one surface voxel, so un-batched
`a_occ` and `cnt[]` counted *pixels* and batched they count *observations*.
Against a fixed `p_occ >= 0.5` gate that is a large, direct reduction in voxels
called occupied — precision up, recall down, which is the signature measured.
The storage-basis change is excluded by magnitude: it is ≤ 1 ULP on the wire
round trip, and 1 ULP cannot produce −0.038 mIoU.

**Impact.** Two things, of different kinds. The published *rankings* survive —
every arm inside a comparison ran on one binary, so no promotion decision is
invalidated. The published *absolute values* do not describe the shipped
mapper, and, more sharply, **any A/B grading a post-2026-08-30 change against
these baselines is confounded**, which retires the cheapest available
regression gate.

**Attribution, measured.** The `--batch-hits 0|1` A/B was run on one binary,
8 scenes, same CLI and frames. **Batching accounts for 91-96 % of every
component of the delta**, and the residual left for traversal plus storage is
small:

| metric | total | batching alone | residual | batching's share |
|---|---|---|---|---|
| `n_pred_occupied` | −11 566 | −10 903 | −663 | 94 % |
| recall | −0.1850 | −0.1744 | −0.0106 | 94 % |
| precision | +0.0690 | +0.0663 | +0.0027 | 96 % |
| intersection mIoU | +0.0318 | +0.0290 | +0.0028 | 91 % |
| occupancy IoU | −0.0399 | −0.0381 | −0.0018 | 95 % |
| union mIoU | −0.0191 | −0.0184 | −0.00070 | 96 % |

The exact-DDA switch is therefore close to free on semantics: its whole residual
on union mIoU is −0.00070, **below the 0.001 material threshold in magnitude**.
It costs ~663 voxels and 0.011 recall per scene — consistent 8/8 in sign, but an
order of magnitude under batching. The correctness fix did not buy a numbers
problem; batching did.

**What batching itself does** (one binary, no confound, `paired_stats` at
MATERIAL 0.001): intersection mIoU **+0.0290** (material, 8/8), precision
**+0.0663** (material, 8/8), recall **−0.1744** (material, 8/8), union mIoU
−0.0184 (**ambiguous**, 4+/4−, dragged by scene 061 at −0.097), occupancy IoU
−0.0381 (ambiguous, 2+/6−). Per-frame wall time is 8.4-13.5 % lower batched,
8/8 — *suggestive only*: the two arms ran in consecutive sessions rather than
interleaved, so that figure is not admissible as a timing result and needs an
interleaved re-run.

**Batching is a reparameterization of `w_occ`, not an independent knob.** The
paragraph above framed this as a trade; a follow-up sweep shows it is a units
change that was never compensated. `w_occ` 1.5 and the fixed `p_occ >= 0.5` gate
were swept while `a_occ` counted pixels; batching changed the unit to
observations without re-tuning either. Sweeping `w_occ` in {3.0, 6.0, 12.0,
24.0} batched, on the same binary, 3 scenes (mean):

| | ship w1.5 | w3.0 | **w6.0** | w12.0 | w24.0 | un-batched w1.5 |
|---|---|---|---|---|---|---|
| union mIoU | 0.3362 | 0.3708 | **0.3797** | 0.3747 | 0.3656 | 0.3800 |
| occupancy IoU | 0.4504 | 0.5022 | **0.5176** | 0.5118 | 0.4981 | 0.5177 |
| recall | 0.5607 | 0.6780 | 0.7592 | 0.8143 | 0.8527 | 0.7572 |

Batched `w_occ` 6.0 reproduces the un-batched arm to four decimals on all three,
with voxel counts within ~1 %. The factor is ~4x rather than the 10-100x a
pixel-footprint argument suggests, because the carve is staged too
(`CarveStage`, per-voxel MAX): `a_occ` and `b_free` shrink together and only the
per-voxel hit:carve imbalance moves.

That collapses the trade. Traversal volume is byte-identical across all arms
(scene 016: 6 493 044 570 voxels on each), so batching's ~11 % is entirely
deposit-side — the cost of the ~10 900 voxels it declines to write — and buying
them back through `w_occ` costs the time back (0.233-0.245 s/frame batched at
6.0 against 0.234-0.237 un-batched). Batching and `w_occ` are two dials on one
speed/completeness axis, and no setting of the pair is simultaneously faster and
equal on mIoU. Which point on that axis to ship is a decision, not a defect, and
it is **not** made in this document; the defaults have not been changed.

**Fix.** Re-run the ablation ring on the current binary; this is a re-run, not
an edit to `RESULTS.md`. `RESULTS.md` carries a provenance banner stating all of
this until the re-run lands. Note that "ambiguous" here is not "inert": at n = 8
an inert verdict needs SE < 0.00042, and these intervals are two orders of
magnitude wider, so the shipped headline is **unknown**, not unchanged.

### H1 — `scovox_node` defaults are not the promoted configuration

**Where.** `src/scovox_mapping/src/scovox_node.cpp` `declareMapParams`
`:322-422`, `declareNodeParams` `:423-811`.

**Evidence.** `resolution` 0.10 (`:325`), `w_occ` 2.0 (`:334`), `kappa0` 2.0
(`:335`), `evidence_saturation` 1000 (`:369`), `max_semantic_classes` 10
(`:387`), `semantic_evict_by_confidence` false (`:391`),
`semantic_band_length` 0.0 (`:393`), `semantic_band_require_occ` true
(`:394`), `enable_tsdf` true (`:354`), `stride` 1 (`:444`), `min_depth` /
`max_depth` 0.1 / 10.0 (`:445`), `trace_no_return_rays` false (`:446`). The
promoted `e5/k2_i0_evid` configuration is `resolution` 0.05, `w_occ` 1.5,
`kappa0` 1.0, `evidence_saturation` 0, 14 classes, evict-by-confidence on,
band 0.10 with `require_occ` false, `tsdf_enabled` 0, stride 2, depth
0.4–4.0, carve-no-return on (replay `Args`,
`scovox_slot_rules/scovox_scenenn/src/replay_scenenn.cpp:40-208`; archived
`docs/archive/design/best_method.md` §3).

**Impact.** Commit `e316f07` is titled "Put the promoted configuration into
the code" but touched the library and replay only. A default launch of
`scovox_mapping_node`, and every launch file in the tree (see M2 / §3.4 of
the structure doc), runs a band-off, evict-by-evidence, `w_occ` 2.0 mapper.
None of the published numbers describe that mapper.

**Fix.** Move the twelve `dp()` defaults above to the promoted values, or add
a `config/scovox_best_method.yaml` and make every RGB-D launch file load it.
Either way, print the effective `SemSplitMap::Params` at startup (the node
already prints the TSDF line at `:288`) so a divergence is visible in the log.

**Resolved (`33aa576`) — second option taken, deliberately.** Moving the twelve
`dp()` defaults would have re-tuned the LiDAR deployments that share them:
`config/lidar_mapping.yaml` derives `w_occ: 8.0` from `prob_hit ≈ 0.9` and runs
at resolution 0.10 on purpose, so the "wrong" defaults are right for that
sensor. The promoted RGB-D configuration is therefore a file you load, not a
default you inherit: `config/scovox_best_method.yaml`, every value transcribed
from the replay `Args` and checked line by line against it rather than copied
from a summary.

The readback half of the fix was taken too: `scovox_node.cpp:309-322` now logs
a `deposit config:` line read out of the *constructed map*, next to the existing
TSDF line, so a binary that predates a knob is visible in the log instead of
silently ignoring the parameter. That check earned itself immediately — the
first rebuild skipped `scovox_node.cpp` on a same-second timestamp, and grepping
the linked binary for the new format string was what caught it. The exit code
and the test count both looked green.

Transcription surfaced one trap worth more than the copy itself, recorded at the
point of use in the yaml: `class_evidence_saturation` has no ROS parameter and
defaults to −1, "share whatever `evidence_saturation` is". The promoted replay
passed 0 explicitly. Those agree **only** while `evidence_saturation` is 0, so
raising it in this file caps both channels, not one — and the class cap is the
one that moves semantics. See M7.

One knob was added to the yaml that the review did not flag, because H4 later
showed it to be the largest single lever in the file: `batch_hits: true`. It is
already the default in all three places, so nothing changes by writing it down —
it is listed so that someone who thinks it is a speed knob reads why it is not.

### H2 — The promoted state is uncommitted

**Where.** `git status`: 31 modified tracked source files, 1 086 insertions /
6 944 deletions including the docs moves; among them `dir_voxel.hpp` (+206),
`scovox_map_split.hpp` (+170), `sem_split_map.hpp/.cpp` (+108/+159),
`beta_voxel.hpp`, and every affected test.

**Evidence.** The `DirVoxel` total basis (`dir_voxel.hpp:156-225`), the
`nhit` removal (`SCOVOX_TRACK_NHIT` default 0, `:118-119`), the
`SCOVOX_BETA_U16` default flip (`beta_voxel.hpp:68-69`) and the
`consensus_merge` / `dscovox_consensus` adjustments all exist only in the
working tree. The archived notes that justify them
(`dir_total_basis_2026_09_04.md`, `nhit_removal_2026_09_04.md`,
`storage_defaults_2026_09_04.md`) are also uncommitted.

**Impact.** The replay in `scovox_slot_rules` builds from this checkout
(`-DSCOVOX_SRC`), so the E9 / deliverable numbers were produced by code that
has no commit hash. A `git stash` or a fresh clone silently reverts to the
16 B / 24 B, float-Beta, `other`-field code.

**Fix.** Commit the working tree as one or two reviewed commits (storage
defaults; total basis + nhit removal) before any further experiment, and
record the hash in `RESULTS.md`'s MANIFEST lines.

**Resolved.** `1101e53` ("Promote the count-storage design; take doc pointers
and results out of code") carries the total basis, the `nhit` removal and the
`SCOVOX_BETA_U16` default flip; `33aa576` carries the config file and a test
fix. The archived design notes are committed with them.

Two things the fix taught, both recorded rather than assumed. `./dev.sh test`
runs 182 core cases and **cannot see `scovox_mapping` at all** — a storage
change has to be graded by `./dev.sh ros-test` (323 cases), and doing so found a
real failure the core suite was structurally blind to (M4). And the promoted
state is now reachable by hash, which is what makes H4 diagnosable at all: the
published binary predates three of these commits.

### H3 — `integrateHitSplit` ignores `tsdf_enabled`

**Where.** `src/scovox_core/include/scovox/scovox_map_split.hpp:587`.

**Evidence.**
```cpp
if (!is_dynamic && !(prof && prof->geometry_off)) tsdf_.integrateRay(origin, endpoint);
```
The fused walker gates its band writes on `tsdf_enabled_` (`:421`); the split
walker does not, and the comment at `:586-587` says so. The archived
`removed_and_untested_2026_09_02.md` §2.3 logged it as "known, not fixed".

**Impact.** Any fused-vs-split A/B at `tsdf_enabled=0` compares "semantics
only" against "semantics plus a full TSDF DDA". The 10–18 % "fused is slower"
timing result was collected with `--tsdf-enabled 1` for that reason; a reader
of the flag alone will not know that.

**Fix.** Add `tsdf_enabled_ &&` to the condition at `:587`, and extend the
`ScovoxMapSplit` timing test to assert the TSDF grid stays empty on both
walkers when disabled.

---

## Medium

### M1 — Four parameter structs, four sets of defaults

**Where.** `scovox::Params` (`map_interface.hpp`), `SemSplitMap::Params`
(`sem_split_map.hpp`), the node's `dp()` calls (`scovox_node.cpp`), the
replay's `Args` (`replay_scenenn.cpp`).

**Evidence.** `w_occ` 2.0 / 1.0 / 2.0 / 1.5; `w_free` 1.0 / 0.5 / 1.0 / 1.0;
`kappa0` 2.0 / 1.0 / 2.0 / 1.0; `evidence_saturation` 1000 / 0 / 1000 / 0;
`range_decay_length` 5.0 / 50 / −1.0 / 50. The node copies from
`scovox::Params` into `SemSplitMap::Params` (`scovox_node.cpp:114-127`), so
the library defaults are dead for the node but live for tests and for any
replay field left untouched.

**Impact.** The same identifier means a different number depending on the
entry point; the replay comment at `replay_scenenn.cpp:429-447` calls this out
for `range_decay_length` and the sibling repo's `FEATURES.md` calls it the
"two-defaults trap". It is the mechanism behind H1.

**Fix.** Give `SemSplitMap::Params` the promoted values and make
`scovox::Params` and the node inherit them (a single `promotedParams()`
factory), so a divergence is a diff against one place. Delete the fields of
`scovox::Params` that nothing reads (`consensus_kl`, `consensus_tau` are
already marked deprecated no-ops).

### M2 — Six library knobs have no ROS parameter

**Where.** `scovox_node.cpp:91-146` sets `SP.semsplit.*` and `SP.*`.

**Evidence.** `hit_flat_share`, `inc_mode`, `inc_thresh`,
`class_evidence_saturation`, `ray_spread` (`SemSplitMap::Params`) and
`far_voxel_fast_paths` (`ScovoxMapSplit::Params`) are never assigned in the
node (grep returns only the `far_voxel_fast_paths` print at `:288`). Every
launch file also omits the band: `scenenet_eval.launch.py` has no
`semantic_band_length` argument; `scenenn_eval.launch.py` and
`semantickitti_eval.launch.py` default `w_occ` 6.0 and `kappa0` 2.0.

**Impact.** The E6 / E7 / §16 ablation arms cannot be reproduced through ROS;
the node is correct only because the struct defaults coincide with the best
method.

**Fix.** Declare the six as parameters with the struct defaults; have the
evaluation launch files take `semantic_band_length`, `w_occ`, `kappa0` and
`semantic_evict_by_confidence` from one shared defaults dict.

### M3 — `range_decay_length` is dead in the split path

**Where.** `sem_split_map.cpp:266-300` (`sanitise` clamps it), nothing else
in `scovox_core` reads it; `scovox_node.cpp:1229`, `:1534` use `> 0` as the
switch for the min/max range cull; `scovoxmap.cpp:67-68`, `:365-366` (legacy
map) are the only exponential-weight readers.

**Evidence.** `src/scovox_mapping/config/default_params.yaml` still
documents it as `w(r) = exp(-r / decay_length)`; the uncommitted
`lidar_mapping.yaml` diff corrected its own comment to ">0 gates the min/max
range cull only".

**Impact.** A user setting `range_decay_length: 5.0` on the node gets a range
cull and no weighting. The E7 sweep of `{50, 10}` recorded "no effect" for a
knob that was never wired.

**Fix.** Either remove the field from `SemSplitMap::Params` and rename the
node parameter to `enable_range_cull`, or implement the weight in
`commitHit`. Update `default_params.yaml` in the same change.

### M4 — One known failing test

**Where.** `src/scovox_core/test/test_scovox_map_split.cpp:886`,
`ScovoxMapSplitFarCarve.FarCarveBitIdenticalToFullWalk` (runs
`runFarCarveIdentity` at band 0.0 and 0.30).

**Evidence.** `dir_total_basis_2026_09_04.md:232`,
`storage_defaults_2026_09_04.md:119-124` and `nhit_removal_2026_09_04.md:184`
all report 180/181 with this single failure, identical under
`-DSCOVOX_BETA_U16=0`, and call it pre-existing. `removed_and_untested_2026_09_02.md:374`
reported 298/298 two days earlier, so the regression entered between
`d5da6a8` and the uncommitted total-basis work.

**Impact.** The test is the bit-identity proof for the far-carve shortcut
that the best method takes on every ray (`far_voxel_fast_paths` 1). Until it
passes, "bit-identical" is a claim, not a result. The replay's
`--far-fast-paths 0` whole-scene differential is the fallback evidence.

**Fix.** Bisect the working-tree diff against `d5da6a8` on this one test;
the `band=0.30` arm exercises the `max(trunc, sem_band)` far end and is the
likely culprit given the total-basis rewrite of `dirichletUpdate`.

**Re-measured, and a second failure found.** The full `./dev.sh ros-test` gate
now reports **323 tests, 2 failures**. The two are not two bugs: colcon
double-counts, since each gtest failure also lands in that package's CTest
`Testing/<date>/Test.xml` aggregate. Two reported failures is **one** distinct
gtest failure — `FarCarveBitIdenticalToFullWalk`, still open, still as described
above.

Running the wider suite for the first time did find a genuinely new failure,
since fixed: `scovox_mapping/test/test_consensus.cpp`,
`SplitRefold.BetaRefoldOrderInvariantToFloatTolerance`. It was **not**
pre-existing — the `SCOVOX_BETA_U16` default flip caused it. The fixture builds
`BetaVoxel c{1.2f, 6.0f}`, and 1.2 is off the 1/8 storage lattice, so fixed
point rounds it to 1.25 and the hand-computed `7.2f − 2·prior` stopped
describing the sum. None of the four order-invariance assertions moved; only the
literal did.

That is worth more than the one test. The byte-identity result behind the
`BETA_U16` promotion was measured on replay, where every parameter is
`prior + w·n` and therefore *on* the lattice by construction. It does not extend
to a value arriving from elsewhere — and a peer with a different weight
configuration handing a refold an off-lattice parameter over the wire is exactly
how that happens. This fixture was, unintentionally, the only test in either
suite covering that case. The assertion now reads the parameters back out of the
source voxels instead of restating the constructor literals, so it tests the
claim that actually holds in both storage modes: the refold is additive with one
prior removed per extra source.

### M5 — `dscovox` fused grids ignore `dir_leaf_bits`; first pose wins; fold order matters

**Where.** `src/scovox_mapping/src/dscovox_node.cpp:90-105` (`SourceGrid`),
`:315-555` (`onBinaryMap`), `consensus_merge.hpp` (`mergeDir`).

**Evidence.** The fused Beta and Dir grids are constructed from
`scovox::Params` defaults (`leaf_bits` 3 for both), so the Dir grid does not
use the `dir_leaf_bits` 2 the sender uses. `T_map_source` is cached from the
first update and never refreshed (`:96-101`, comment: "requires c-slam
disabled … See … C5 in ablations_punch_list.md", a file that does not exist).
`mergeDir` is an insertion-sort fold; refolding in sorted source-key order
makes it deterministic, not order-independent.

**Impact.** Memory shape differs between sender and merger (harmless for
correctness, wrong for the memory claims); a loop closure on any robot
misplaces its whole contribution; a source added later can change the fused
label where slot evictions are close.

**Fix.** Construct the fused grids from the first frame's geometry (the
frame carries `resolution`; add `leaf_bits` / `dir_leaf_bits` to the header),
refresh `T_map_source` from each message's `map_from_source`, and document
the fold-order dependence next to `mergeDir`.

### M6 — Wire block runs hard-code `leaf_bits = 3`

**Where.** `binary_serializer.hpp:518-519` (`kBlockVoxels = 512`,
`kBitmaskBytes = 64`, bit = `(lx<<6)|(ly<<3)|lz`).

**Evidence.** The frame header carries `resolution`, `num_classes`, `K_TOP`,
`alpha_0`, `quant_step`, `fine_ratio_log2` but not `leaf_bits`; the node
declares `leaf_bits` as a parameter (`scovox_node.cpp:327`).

**Impact.** A sender launched with `leaf_bits` ≠ 3 produces frames the
decoder reads as a different geometry; nothing rejects the mismatch.

**Fix.** Either `static_assert` / runtime-refuse `leaf_bits != 3` in the
serializer, or put `leaf_bits` in the header (bumping `FORMAT_VERSION`).

### M7 — `evidence_saturation` is one knob for two caps

**Where.** `sem_split_map.cpp:1112-1143`; node `dp("evidence_saturation")`
`scovox_node.cpp:369`; wire `quant_step` `:2253-2255`.

**Evidence.** `applyDirSaturation` uses `class_evidence_saturation` if ≥ 0,
else the Beta cap; the node never sets `class_evidence_saturation`, and it
also derives the wire quantisation step from the same value. The best method
runs with 0 (off); the node defaults 1000.

**Impact.** Turning the cap off on the node (to match the best method)
switches the wire to f32 payloads and roughly doubles bandwidth; there is no
way to keep u8 companding with an uncapped map.

**Fix.** Separate `wire_evidence_scale` from `evidence_saturation`, and
expose `class_evidence_saturation`.

**Partly resolved — documented, not fixed.** Neither knob was separated; the
coupling is real and still there. What changed is that it is now written down at
the one place a person will meet it, in `config/scovox_best_method.yaml` beside
`evidence_saturation: 0`, stating that the two caps agree here *only* because
the value is 0 and that raising it silently caps semantics as well as occupancy.
The structural fix (a separate `wire_evidence_scale`, plus a
`class_evidence_saturation` ROS parameter) is unchanged and still wanted.

### M8 — Experiment write-ups still print the demoted build flag

**Where.** `scovox_slot_rules/RESULTS.md:31` (deliverable block:
`SCOVOX_EVICT_INHERIT=3  half carry-over on eviction`);
`docs/archive/design/best_method.md:78-81` ("Each voxel uses 24 bytes …
nhit[2]").

**Evidence.** `dir_voxel.hpp:142-144` refuses any `SCOVOX_EVICT_INHERIT`
other than 0; `RESULTS.md:1576` shows the promoted `e5/k2_i0_evid` MANIFEST
with `EVICT_INHERIT=0`; `kDirExpectedSize` is 20 B (`dir_voxel.hpp:236-246`).

**Impact.** The one block a reader is told to copy to reproduce the
deliverable will not compile.

**Fix.** Rewrite the `RESULTS.md` deliverable block to the §1.1 flags of the
structure doc; add a one-line "20 B as shipped" correction to `best_method.md`
or its successor.

**Resolved, and the block held a second error.** The deliverable block now reads
`SCOVOX_EVICT_INHERIT=0   no carry-over on eviction (i3 measured, demoted)`, so
it compiles.

Fixing it exposed a defect of a different kind: the block was not the only place
describing the wrong arm. The **headline table** reported the `cand` (i3) numbers
while the same document demotes i3 fifty lines further down, and the attribution
paragraph named `e5/k2_i3_evid` as the candidate build. Recomputed from the
stored per-scene JSONs with `scripts/paired_stats.py`:

| block | quoted (i3, demoted) | shipped (i0) |
|---|---|---|
| union mIoU | 0.29893, +0.02921 | **0.29809, +0.02837** |
| `with_unknown` mIoU | 0.58702, +0.05396 | **0.58532, +0.05226** |
| occupancy IoU | 0.43926, +0.01682 | **identical** |

Both stay `material` at p = 0.0078, 8/8, so no conclusion turns — the headline
simply overstated the shipped pipeline by 0.00084 union and 0.00170
intersection. Two details are worth keeping. The shipped arm and E4's
`off_evid_i0` attribution control ran on the *same* binary (md5
`60e17da907d0`), which makes that control stronger than it was written to be.
And dropping i3 costs nothing on occupancy in the strongest available sense:
**every field** of the occupancy block — predicted count, intersection,
precision, recall, IoU, phantoms — is identical between the arms on all 8
scenes, which is what a change confined to the class histogram should look like.

Two counts in that section were already inconsistent before this correction and
were **left alone rather than guessed at**: the prose says "six parts" and "six
levels" where the ablation ring has five rows and declares family m = 5.

---

## Low

### L1 — Stale byte-size and type-name comments

| location | says | is |
|---|---|---|
| `sem_split_map.hpp:5` | "alternative to `SemDirMap`" | `SemDirMap` was removed |
| `sem_split_map.hpp:8` | `BetaVoxel` (8 B) | 4 B under the default `SCOVOX_BETA_U16=1` |
| `sem_split_map.hpp:10` | `DirVoxel` (16 B) | 20 B under the default `SCOVOX_TRACK_QMAX=1` |
| `sem_split_map.cpp:690` | "16 B DirVoxel" | 20 B |
| `dir_voxel.hpp:4`, `:26` | "16-byte", "total: 16 B at K_TOP=2" | 20 B (the header corrects itself at `:96`) |
| `beta_voxel.hpp:17-20` | "the 16 B `DirVoxel`" | 20 B |
| `tsdf_voxel.hpp:11` | "`Bonxai::VoxelGrid<scovox::SemBetaVoxel>`; see `sembeta_voxel.hpp`" | the live pair is `BetaVoxel` ∥ `DirVoxel` |
| `tsdf_map.hpp:15-17`, `:49` | "lives in `SemBetaMap`", "same as the SemBetaMap resolution" | `SemBetaMap` was removed; the pair is `TsdfMap` ∥ `SemSplitMap` |
| `binary_serializer.hpp:182-183` | "u16 quantization step … / 65535" | u8 sqrt-companded, `/ 255²` (`scovox_node.cpp:2250-2255`) |

### L2 — In-code references to moved or nonexistent documents

Generated from `grep -rn '\.md' src config scripts README.md` on 2026-09-04.

**Moved to `docs/archive/` by this session** (each needs a `docs/archive/`
prefix; the four design docs now sit under `docs/archive/design/`):

| reference | cited from |
|---|---|
| `design/unified_dirichlet_design_2026_05_13.md`, `design/fine_tsdf_band_dbh_2026_07_30.md`, `design/comms_design_2026_07_30.md`, `occupancy_prior.md` | `README.md:24-27` |
| `docs/distributed_mapping.md`, `docs/publish_scovox_bin_from_bag.md` | `README.md:123`, `:181` |
| `docs/design/fine_tsdf_band_dbh_2026_07_30.md` | `config/scovox_fine_band.yaml:9`, `RefinementRegion.msg:4`, `scovox_core/CMakeLists.txt:138`, `binary_serializer.hpp:163` |
| `comms_design_2026_07_30.md` | `binary_serializer.hpp:161` |
| `efficiency_audit_2026_08_26.md` | `scovox_core/CMakeLists.txt:124`, `carve_stage.hpp:6`, `test_carve_stage.cpp:5` |
| `docs/distributed_mapping_lidar.md` | `config/scovox_lidar_geometric.yaml:17`, `dscovox_multi_robot.launch.py:9,11` |
| `docs/dscovox_multi_robot_run.md`, `docs/user_manual.md` | `dscovox_multi_robot.launch.py:20,159,199` |
| `docs/scovox_bin_manual_bringup.md` | `dscovox_single_robot.launch.py:11`, `scovox_mapping/config/scovox_bin_min.yaml:19` |

**Never existed in this tree:**

| reference | cited from | note |
|---|---|---|
| `docs/design/slimvdb_like_tsdf_mapping_plan.md` | `scovox_core/CMakeLists.txt:96` | |
| `feedback_slimvdb_memory_measurement.md` | `tsdf_map.hpp:151` | |
| `ablations_punch_list.md` | `dscovox_node.cpp:100` | "C5" first-pose-wins caveat |
| `scovox-vertical-overfill.md` | `config/scovox_lidar_raw_deskew.yaml:12` | |
| `experiment_plan.md` | `scovox_node.cpp:572` | |
| `experiments/PLAN.md` | `voxel.hpp:16` | |
| `experiments/results/e82/E82_RESULT.md` | `semantickitti_eval.launch.py:99` | |
| `NEW_EXPERIMENTS_PLAN.md` / `NEW_EXPERIMENT_PLAN.md` | `e0_counters.hpp:3`, `test_sem_split_map.cpp:945`, `scenenet_eval_fusion.launch.py:2,14` | lives in `scovox_slot_rules/`, spelled two ways |
| `docs/exp_ablations.md`, `log_odds_map.hpp` | `map_interface.hpp:84,100`, `:8` | prose mentions, not `.md` links |

`e0_counters.hpp` also cites `scovox_node.cpp:860-863` for the eviction CSV,
which is now `logEvictionDelta` at `:869-879`.

**Partly resolved.** The sweep ran across `src`, `config` and `README.md`: the
24 remaining `.md` references in code were removed or repointed in `1101e53`,
under the standing rule that **code comments cite no document paths and carry no
experiment results** — those live in memory and in `REVIEW_LOG.md`.

**Not swept: `scovox_slot_rules/scripts/*.py|*.sh`**, ~50 references, audited and
deliberately left for a decision. `DESIGN.md` ×12 — the target has never existed
in either repo's git history, so there is nothing to repoint to and the right fix
may be deletion. `FINDINGS.md` ×9 → `archive/FINDINGS.md` and `PLAN.md` ×5 →
`archive/PLAN.md` are simple prefix edits. `a.md` ×3 is a false positive from
`args.md` attribute access.

### L3 — README document index broken by the archive move

`README.md:24-27` lists four design docs by their old paths, and `:123`,
`:181` link `docs/distributed_mapping.md` and
`docs/publish_scovox_bin_from_bag.md`. All six now live under
`docs/archive/`. The README was deliberately left untouched in this session;
the fix is a six-line path edit plus links to
`docs/scovox_code_structure.md` and this file.

**Resolved** in `1101e53`: the six paths were corrected and the index now links
`docs/scovox_code_structure.md` and this document.

### L4 — Legacy voxel / map types still compiled and tested

`voxel.hpp` (`Voxel`, `sparse_add`), `scovoxmap.hpp/.cpp` (`scovox::Map`),
`sembeta_voxel.hpp` (`SemBetaVoxel`) and the naive / majority-vote updaters
in `semantics.hpp` are all built; `test_beta_update` (31 cases) and
`test_consensus` (25) exercise `scovox::Map`, which neither node instantiates
for mapping. `Voxel` is still needed as the `ScovoxMap` message projection
type (`dscovox_consensus.hpp` `projectBetaDirTo*`); the rest is maintenance
surface. Decide whether `scovox::Map` is an ablation baseline worth keeping;
if not, delete it with its two test binaries.

### L5 — Stale tool and comment text around the wire format

`scripts/wire_study/wire_study.py:1-12` documents the v5 frame layout with
20 B Beta and 28 B Dir records; the serializer is rev 8 with block runs and
u8 companding, so the script's re-encodings no longer start from the shipped
bytes. Mark it historical or port it to rev 8.

### L6 — `downsample_voxel_size` default described two ways

Code default is 0.5 (`scovox_node.cpp:754`; the member initialiser at
`:3329` is 0.0 but is overwritten by `dp()`). `config/scovox_lidar_raw_deskew.yaml`
says "0.5 is also the in-code default" (correct);
`dscovox_multi_robot.launch.py` passes 0.1 explicitly (`:144`) and its
surrounding comment reads as if 0.0 were the default. Harmless today, but the
comment should quote `:754`.

---

## Verified as sound (no action)

- `#error` traps for every removed flag and for `SCOVOX_EVICT_INHERIT != 0`
  (`dir_voxel.hpp:133-144`); size `static_assert`s on all three voxel types.
- `sanitise()` snaps `w_occ` / `w_free` onto the ⅛ lattice
  (`sem_split_map.cpp:297-298`) and the node snaps both fusion profiles
  (`scovox_node.cpp:851-852`), so the count identity `a_occ = 1 + w_occ·n`
  holds on both paths.
- The u16 Beta halving at 90 % of `kMax` runs unconditionally before the
  opt-in cap (`sem_split_map.cpp:1097-1102`) and cannot loop.
- `applyDirSaturation` rebuilds `s_total` from the scaled parts and floors
  filled slots only (`:1129-1142`), keeping `other()` non-negative.
- `decayTransient` snaps stalled fixed-point residuals to the prior
  (`:995-1006`) so the prune test terminates under u16 storage.
- `ScovoxMapSplit` refuses `band > 0` with the split walker
  (`scovox_map_split.hpp:129-136`) instead of running endpoint-only under a
  band label; the replay refuses `--evict-by-confidence` on a
  `TRACK_QMAX=0` build (`replay_scenenn.cpp:449-458`).
- `dscovox` uses a dedicated `prior_pinned_` flag rather than
  `num_classes == 0` as the sentinel (`dscovox_node.cpp:388-391`, `:980`) and rejects
  `num_classes == 0` frames.
- `MultiThreadedExecutor(…, 2)` with the viz timer in its own callback group
  and `map_mtx_` as a `shared_mutex`; the `last_pc_pub_ns_` atomic in
  `dscovox` makes the rate limiter race-free under shared locks.
