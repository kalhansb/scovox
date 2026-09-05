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
| H3 | High | `integrateHitSplit` ignores `tsdf_enabled` | **resolved** — gated; dump byte-identical, a whole redundant DDA removed |
| M1 | Medium | Four parameter structs with four sets of defaults | open |
| M2 | Medium | Six library knobs have no ROS parameter | open — now cross-referenced from the yaml |
| M3 | Medium | `range_decay_length` is dead in the split path but still documented as a weight | open |
| M4 | Medium | One known failing test, `FarCarveBitIdenticalToFullWalk` | open — but re-measured, and a second failure found and fixed |
| M5 | Medium | `dscovox` fused grids ignore `dir_leaf_bits`; first pose wins; fold order matters | open |
| M6 | Medium | Wire block runs hard-code `leaf_bits = 3` | open |
| M7 | Medium | `evidence_saturation` is one knob for two caps | **partly resolved** — trap documented at the point of use |
| M8 | Medium | Experiment write-ups still print the demoted build flag | **resolved** — plus a second error found in the same block |
| M9 | Medium | `batch_hits` stages the endpoint only; the semantic band is un-batched | open — mIoU unaffected, but it inverts any confidence read; **deferred action plan recorded, run on request** |
| M10 | Medium | The semantic-uncertainty basis was undefined, and the ROS wire ships the wrong one | **basis resolved** (E12 — Dirichlet aggregation; `calib_alpha0.py:106`) — **but a live defect is now split out as M11**; vacuity is retracted as a functional (it is `-S` exactly) |
| M11 | Medium | The published `semantic_confidence` is a Laplace+Hutter readout, not the posterior marginal | open — live on RViz, the npz exporter and both RPCs; `p_OTHER` inflated 6.6x at delta=20 |
| L1 | Low | Stale byte-size and type-name comments | open |
| L2 | Low | In-code references to moved or nonexistent documents | **partly resolved** — code swept; `scovox_slot_rules/scripts/` not |
| L3 | Low | README document index broken by the archive move | **resolved** |
| L4 | Low | Legacy voxel / map types still compiled and tested | open |
| L5 | Low | Stale tool and comment text around the wire format | open |
| L6 | Low | `downsample_voxel_size` default described two ways | open |
| L7 | Low | `sdf_trunc` does not become 0 when the TSDF is off, and two files said it did | **resolved** — descriptions fixed and the dead tail trimmed by early DDA termination, byte-identical |

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

**Confirmed after the code-review batch, 2026-09-05.** The whole `total` column
above was re-measured from the post-review tree — `cells/goal8.tsv`, the
promoted `abl_inherit` argv and the `e5/k2_i0_evid` flags, all eight scenes —
and it comes back unchanged: `n_pred_occupied` −11 566, recall −0.1850,
precision +0.0690, intersection mIoU +0.0318, occupancy IoU −0.0399, union mIoU
−0.0191, phantom voxels 16 005 → 8 645. The scene-016 column above reproduces
to four decimals as well. That matters twice over: it fixes the size of the H4
gap on the full dataset rather than on two scenes, and it says the review batch
contributed exactly nothing to it — the drift is the three deposit-path commits
and nothing else.

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
observations without re-tuning either. A `w_occ` sweep in {3.0, 6.0, 12.0, 24.0}
batched located the compensating weight at 6.0; the 8-scene arm at that weight
grades **ambiguous on all six metrics** against un-batched `w_occ` 1.5, with
every mean small and the union mean *below* MATERIAL:

| vs un-batched w1.5 | mean Δ | 95 % CI | p_exact | sign |
|---|---|---|---|---|
| union mIoU | **+0.0008** | [−0.0032, +0.0049] | 0.4609 | 5+/3− |
| intersection mIoU | +0.0050 | [−0.0054, +0.0153] | 0.3828 | 6+/2− |
| occupancy IoU | −0.0001 | [−0.0099, +0.0098] | 0.7422 | 4+/4− |
| precision | +0.0032 | [−0.0087, +0.0151] | 0.5469 | 5+/3− |
| recall | −0.0098 | [−0.0241, +0.0046] | 0.1484 | 2+/6− |
| `n_pred_occupied` | −850 | [−2081, +381] | 0.1094 | 1+/7− |

Ambiguous, not inert: the union SE is ~0.0021 against the 0.00042 an inert
verdict needs. The claim is "no material difference detected at n = 8", not
"identical". The compensating factor is ~4x rather than the 10-100x a
pixel-footprint argument suggests, because the carve is staged too
(`CarveStage`, per-voxel MAX): `a_occ` and `b_free` shrink together and only the
per-voxel hit:carve imbalance moves.

Against the *shipped* batched point, `w_occ` 6.0 is a trade and not an upgrade:
occupancy IoU +0.0380 (material, 6+/2−), recall +0.1647 (material, 8/8) and
`n_pred_occupied` +10 053 (material, 8/8), against precision −0.0631 (material,
8/8) and intersection mIoU −0.0241 (material, 8/8) — the mirror of the +0.0290
batching bought. Union, the deciding metric, is **+0.0192 ambiguous (5+/3−)** and
does not separate.

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
runs `scovox_core` only and **cannot see `scovox_mapping` at all** — a storage
change has to be graded by `./dev.sh ros-test` (327 cases at HEAD: `scovox_core`
184, `scovox_mapping` 143, counted from the gtest XML by
`docker/count_tests.py`), and doing so found a
real failure the core suite was structurally blind to (M4). And the promoted
state is now reachable by hash, which is what makes H4 diagnosable at all: the
published binary predates three of these commits.

### H3 — `integrateHitSplit` ignores `tsdf_enabled`

**Where.** `src/scovox_core/include/scovox/scovox_map_split.hpp:648-649`, in
`integrateHitSplit` (which begins at `:629`).

**Evidence, as the defect stood.** The call was
```cpp
if (!is_dynamic && !(prof && prof->geometry_off)) tsdf_.integrateRay(origin, endpoint);
```
The fused walker gates its band writes on `tsdf_enabled_`; the split walker did
not. The archived `removed_and_untested_2026_09_02.md` §2.3 logged it as
"known, not fixed".

**Status at HEAD: FIXED.** `:648-649` now reads
`if (tsdf_enabled_ && !is_dynamic && !(prof && prof->geometry_off))`, and the
`ScovoxMapSplitTsdfDisabled` pair guards it. *Correction 2026-09-05: the three
line pointers this section originally carried — `:587` for the site, `:421` for
the fused gate, `:586-587` for the comment — were all wrong, and `:587` is a
blank line. They were offsets, not reads.*

**Impact.** Any fused-vs-split A/B at `tsdf_enabled=0` compares "semantics
only" against "semantics plus a full TSDF DDA". The 10–18 % "fused is slower"
timing result was collected with `--tsdf-enabled 1` for that reason; a reader
of the flag alone will not know that.

**Fixed** (`:648`). The condition now reads
`tsdf_enabled_ && !is_dynamic && !(prof && prof->geometry_off)`, so the flag
means one thing on both walkers.

This was not a few voxels tacked onto a shared traversal — `TsdfMap::integrateRay`
is a second full DDA of its own, and a split-path run at `tsdf_enabled=0` paid
for it in full to fill a grid the flag declares unread. Scene 016, promoted
flags plus `--sem-band 0 --fused-walker 0` (the band is refused on the split
walker), before and after: both dumps `6a53b84c404e`. Byte-identical, so the
whole second traversal is removed at no cost to the map.

**Guarded.** `ScovoxMapSplitTsdfDisabled.GridStaysEmptyOnBothWalkers`
(`test/test_scovox_map_split.cpp`) integrates a hit and a miss on both walkers
with the flag off and asserts `tsdfVoxelCount() == 0` on each, plus Beta/Dir
grid byte-equality between the two. Before the fix it failed on the split arm
only.

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
in `scovox_core` reads it; `scovox_node.cpp:1264`, `:1571` use `> 0` as the
switch for the min/max range cull; `scovoxmap.cpp:72-73`, `:367-368` (legacy
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

**Where.** `src/scovox_core/test/test_scovox_map_split.cpp:887`,
`ScovoxMapSplitFarCarve.FarCarveBitIdenticalToFullWalk` (runs
`runFarCarveIdentity` at band 0.0 and 0.30).

**Evidence.** `dir_total_basis_2026_09_04.md:232`,
`storage_defaults_2026_09_04.md:119-124` and `nhit_removal_2026_09_04.md:184`
all report 180/181 **for `scovox_core` under `./dev.sh test`** — a core-only
figure, not a whole-tree one — with this single failure, identical under
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
reports **2 failures** on the colcon console. The two are not two bugs: colcon
double-counts, since each gtest failure also lands in that package's CTest
`Testing/<date>/Test.xml` aggregate. Two reported failures is **one** distinct
gtest failure — `FarCarveBitIdenticalToFullWalk`, still open, still as described
above. *Correction 2026-09-05: the case total quoted here was 323, read off that
same console. Counted from the XML instead (`docker/count_tests.py`) it is 326
cases, 1 failure. Prefer the XML count; the console mixes case and aggregate
rows, which is the very double-count this paragraph is about.*

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

**Where.** `binary_serializer.hpp:531` (`kBlockVoxels = 512`,
`kBitmaskBytes = 64`, bit = `(lx<<6)|(ly<<3)|lz`).

**Evidence.** The frame header carries `resolution`, `num_classes`, `K_TOP`,
`alpha_0`, `quant_step`, `fine_ratio_log2` but not `leaf_bits`; the node
declares `leaf_bits` as a parameter (`scovox_node.cpp:362`).

**Impact.** A sender launched with `leaf_bits` ≠ 3 produces frames the
decoder reads as a different geometry; nothing rejects the mismatch.

**Fix.** Either `static_assert` / runtime-refuse `leaf_bits != 3` in the
serializer, or put `leaf_bits` in the header (bumping `FORMAT_VERSION`).

### M7 — `evidence_saturation` is one knob for two caps

**Where.** `sem_split_map.cpp:1112-1143`; node `dp("evidence_saturation")`
`scovox_node.cpp:404`; wire `quant_step` `:2294-2295`.

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

**Evidence.** `dir_voxel.hpp:144-146` refuses any `SCOVOX_EVICT_INHERIT`
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

### M9 — `batch_hits` stages the endpoint only; the semantic band is un-batched

`stageable = batch_hits && !kernel_ray && semantic_spread_radius <= 0 && ray_spread == 0`
(`sem_split_map.cpp:606-608`) governs the **endpoint** deposit. The band deposit
is a separate immediate write issued from inside the walk
(`ScovoxMapSplit::integrateHitFused`, band gate at `scovox_map_split.hpp:493-495`
→ `semBand` `:977-980` → `applyBandSemantic` `sem_split_map.cpp:854-901`), which
staging never sees. Per frame:

| | deposits per voxel | weight |
|---|---|---|
| endpoint | 1 — the frame's strongest ray | `kappa0 · p_occ_post`, gated `p_occ >= min_p_occ` |
| band voxel | one per depth pixel whose ray passes it | flat `kappa0`, no Beta read |

At `fx` 544.47, stride 2, `resolution` 0.05, a fronto-parallel surface puts
~46 rays through one voxel column at 2 m and ~185 at 1 m — all of which band the
voxels either side of the surface, while the surface voxel takes one deposit.
`s_total` in a band voxel therefore grows one to two orders of magnitude faster
than in the voxel actually observed, scaling as 1/d². Geometric estimate, not
measured.

**mIoU is unaffected** and no published number moves: a Dir-only voxel dumps as
`state=1` with `p_occ` set to the Beta prior, and the scorer's occupied set is
`p_occ >= 0.5 && (state == 0 || state == 2)` (`replay_scenenn.cpp:636,663-664`),
which excludes it. The exposure is C3 / E3.1 / E3.2 in `docs/papers/experiment_plan.md`:
read as a Dirichlet concentration, `s_total` is a count of correlated looks in
band voxels, so vacuity is understated there and understated *most* where the
sensor was closest. `semantic_band_require_occ: false` compounds it — the band is
the only path that deposits class evidence with no occupancy gate, so a band
voxel can hold a sharp class posterior on no occupancy evidence at all.

The E10 α₀ calibration study does **not** bear on this. It scores the same set
the mIoU scorer does — `p_occ >= 0.5 && state ∈ {0,2}` intersected with GT — and
a Dir-only band voxel dumps as `state=1`, so every voxel this finding is about
was excluded from it. M9 remains open and unmeasured.

Not acted on. Batching the band, weighting it by `p_occ`, or dividing `kappa0` by
the span would each change the deposited field and so every mIoU number; none is
a free correction, and the band's mIoU value was measured in its present form.
The near-term fix for an uncertainty consumer is to read `state`, not `p_occ`.

**DEFERRED ACTION — run on request, do not start unprompted.** Three candidate
corrections, cheapest first. All three are arms, not fixes: each must be graded
before any of it is adopted.

| # | arm | change | new knob |
|---|---|---|---|
| 1 | `band_batch` | stage the band deposit per (voxel, frame) as the endpoint already is — one deposit per band voxel per scan, strongest ray | `SemSplitParams::batch_band` |
| 2 | `band_pocc` | deposit `kappa0 · p_occ` in the band instead of flat `kappa0`, i.e. `semantic_band_require_occ` without the hard gate | reuse `hit_flat_share`'s shape |
| 3 | `band_norm` | divide `kappa0` by the band's voxel span so one measurement contributes one unit total | none — derived from `semantic_band_length / resolution` |

Arm 1 is the one that targets the defect directly; 2 and 3 attack the
correlation from the weight side and would also change the endpoint/band
balance. They are not composable — grade separately before any pair.

Protocol, unchanged from the rest of the suite: all 8 scenes, per-scene
`--max-frames`, frame order untouched, ONE binary shared with a freshly-run
control arm, graded by `scripts/paired_stats.py` `verdict(deltas, material=0.001)`
on union mIoU and occupancy IoU. Log the result in `REVIEW_LOG.md` whichever way
it lands.

The catch worth stating up front: **mIoU cannot adjudicate this.** The defect is
in a quantity no metric in the suite reads, and the scorer already excludes the
affected voxels. An arm that leaves mIoU ambiguous has not thereby passed — it
has only shown it costs nothing, and the benefit needs the calibration readout
E3.1 / E3.2 in `docs/papers/experiment_plan.md`, which does not exist yet. Build
that first or the ranking has no dependent variable.

---

### M10 — The semantic-uncertainty basis (RESOLVED by E12)

**Status.** The basis question is settled. The measurement that settled it also
retracts this section's own recommendation and exposes a live defect, split out
below as **M11**.

**Where.** `scovox_slot_rules/scripts/calib_alpha0.py:106-180` (the answer);
`src/scovox_core/include/scovox/uncertainty.hpp:55-112` and
`experiments/uncertainty/functionals.py` (deleted by `821374e`, recover with
`git show fcf7c86:experiments/uncertainty/functionals.py`) — the rejected basis.

**The resolution.** By **Dirichlet aggregation**, merging categories of a
Dirichlet gives a Dirichlet whose alphas are the sums, so the
`{slot_0, slot_1, OTHER}` marginal of the symmetric `C`-class posterior is
*unique* — not a modelling choice. `reparam()` already computes it:

```
alpha_i = cnt[i]                        filled slot i   (already alpha0 + e_i)
alpha_O = s_total - sum_filled cnt[i]   = (C - n_f)*alpha0 + e_pool
A       = C*alpha0 + S
H       = -sum p_j ln p_j                       total
E[H]    = psi(A+1) - sum p_j psi(alpha_j+1)     aleatoric
MI      = max(0, H - E[H])                      epistemic
```

The `n_pool = C - n_f` unresident classes are exchangeable under the symmetric
prior, so their identical terms collapse and the cost is `O(K+1)` digammas per
voxel. Mass conserves exactly; no Laplace `+1`, no Hutter floor.

Laplace `+1` on the three cells is not a prior: it implies per-class prior 1 on
residents and `1/(C-K)` on non-residents, the assignment is made *after* seeing
which classes are resident, and it changes on every eviction. The OTHER `+1`
never decays — that is the mechanism behind the 6.6x `p_OTHER` gap below.

**What the two bases do to the numbers** (one class at delta = 20 deposits):

| | `functionals.py` (Laplace + Hutter) | `reparam()` posterior marginal |
|---|---|---|
| resident slot alpha | `cnt + 1` | `alpha0 + evidence` |
| OTHER alpha | `effectiveResidual + 1` | `(C-n_f)*alpha0 + escaped evidence` |
| `H_y` | 0.206 | 0.047 |
| `p_OTHER` | 0.053 | 0.008 |

**Three retractions this section must carry.**

1. *"the floor ships and never runs on the promoted map"* — **false**. See M11.
2. *"vacuity is the only quotable functional"* — **retracted**. Vacuity is
   `C*alpha0/A`, monotone in `S`, so as a ranking it is `-S` exactly: measured
   `AUROC(vacuity) == AUROC(-S)` to **0.00e+00 on all 8 scenes**, mean 0.4752,
   worse than chance. It is invariant to the basis because it is not an
   uncertainty — it is a visit counter.
3. *"E3.2 must fix one alpha basis before any number is quoted"* — done, and
   the numbers are now quoted in `scovox_slot_rules/REVIEW_LOG.md` (E12).

**What the basis bought, measured.** Aleatoric equals total
(`max |AUROC(H) - AUROC(E[H])| = 1.24e-03`, 8 scenes); epistemic MI is dead
(mean AUROC 0.5102, above chance 4/8, `spearman(miss_rate, AUROC_MI) = +0.071,
p = 0.88`) because MI's large-`A` asymptote `(C-1)/(2A)` pins it at `A ~ 1e4`;
`p_OTHER` is the miss detector (mean 0.7781, beating MI by +0.2679 and vacuity
by +0.3029, both 7/8, `p = 0.0234`, material). And the whole family
anti-correlates with scene difficulty:
`spearman(miss_rate, AUROC_H) = -0.9286, exact p = 0.00223`. Full tables, the
tempering sweep that disqualifies `class_evidence_saturation`, and the graded
pre-registered predictions are in `scovox_slot_rules/REVIEW_LOG.md` under E12.

**One trap survives for anyone reviving the deleted file.** It tests residency
with `cnt > 0` and no `cls != EMPTY` guard, but a `DirVoxel` empty slot holds
`cnt = alpha0 = 0.01 > 0` and would read as an observed class
(`calib_alpha0.py:118` guards this). And `hutterEscapeMass` is **clamped** —
`min(., N)` plus `ratio <= 1 -> N` (`uncertainty.hpp:93-94`) — which quotations
of the bare formula routinely drop; at `m=1, N=0.14` the raw term is 3.816
against a shipped 0.140.

---

### M11 — The published `semantic_confidence` uses the rejected basis

**Where.** `src/scovox_mapping/include/scovox/dscovox_consensus.hpp:78`
(`projectBetaDirToSemBetaForViz`) and `:128` (`projectBetaDirToVoxel`);
`src/scovox_mapping/src/dscovox_node.cpp:702, 706, 813`;
`src/scovox_mapping/src/scovox_node.cpp:2928`;
`src/scovox_mapping/include/scovox/node_utils.hpp:246-247`.

**What.** M10 previously recorded that `uncertainty.hpp` "never names
`DirVoxel`" and therefore never runs on the promoted map. The first half is
true and the conclusion does not follow: a projection exists whose entire
purpose is to bridge the two. `projectBetaDirToSemBetaForViz` strips `alpha0`
from every slot and `(C-K)*alpha0` from OTHER, yielding the raw-evidence
convention the legacy helpers expect:

```cpp
out.a_unk    = std::max(0.f, d->other() - other_prior);   // other_prior = (C-K)*alpha_0
out.sem_cnt[i] = std::max(0.f, d->cnt[i] - alpha_0);
```

`dscovox_node.cpp:706` then calls `argmaxClassConfidence` (`node_utils.hpp:298`), whose denominator is
`sum_cnt + n_active + effectiveResidual(v) + 1` (`node_utils.hpp:325-326`) —
Laplace on the residents, Hutter-floored `+1` on OTHER. So the published
`semantic_confidence` PointField, the `pointcloud_to_npz.py` export, the RViz
colouring and both the GetRegion and GetOccupancyGrid RPCs all read the
promoted state through the basis M10/E12 rejects.

**Why it matters.** The `+1` on OTHER never decays, so a well-observed voxel's
residual mass is over-stated without bound in the look count: at delta = 20 the
readout gives `p_OTHER` 0.053 against the posterior marginal's 0.008, and `H_y`
0.206 against 0.047. Confidence is correspondingly depressed, which interacts
with the labelling threshold. Note the projection's own comment records that
leaving the prior in `a_unk` had already made the published confidence disagree
with the RPC for the identical voxel — the same class of bug, fixed once at the
prior and still open at the `+1`.

**Why it was not caught.** Nothing *evaluated* reads it. The mIoU scorer works
off the `.slots` dump and the calibration study off `reparam()`; the confidence
field is wire-only. That is also why the fix is low-risk for the published
numbers and high-value for anything downstream that consumes the field.

**Action.** Replace the `argmaxClassConfidence` denominator with the posterior
marginal (`alpha_best / A`, `A = C*alpha0 + S`) computed directly from
`DirVoxel`, and keep `semanticEntropy` / `semanticVariance` consistent with it
— the comment above `argmaxClassConfidence` explicitly requires all three to share one
categorical. Not done: it changes a wire value and needs its own before/after
on the labelling threshold. Untested, not removed.

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
| `binary_serializer.hpp:40`, `:194` | "u16 quantization step … / 65535" | u8 sqrt-companded, `/ 255²` (`scovox_node.cpp:2294-2295`) |

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

Code default is 0.5 (`scovox_node.cpp:787`; the member initialiser at
`:3329` is 0.0 but is overwritten by `dp()`). `config/scovox_lidar_raw_deskew.yaml`
says "0.5 is also the in-code default" (correct);
`dscovox_multi_robot.launch.py` passes 0.1 explicitly (`:144`) and its
surrounding comment reads as if 0.0 were the default. Harmless today, but the
comment should quote `:754`.

### L7 — `sdf_trunc` does not become 0 when the TSDF is off, and two files said it did

`enable_tsdf: false` makes the node pass `sdf_trunc = 0` (`scovox_node.cpp:389-391`),
but `TsdfMap::sanitise` clamps any `<= 0` back to **0.15 m** (`tsdf_map.cpp:25`)
and the fused walker reads the sanitised value (`scovox_map_split.hpp:213`). The
node documents this correctly at `:100-107`. Two derived descriptions did not:
`config/scovox_best_method.yaml` said "sdf_trunc collapses to 0", and
`scovox_code_structure.md` §1.3 said the walk's far end with `tsdf_enabled=0`
is the semantic band's 0.10 m. Both **fixed**.

The behavioural consequence is small but real. `back_reach = max(trunc, sem_band_)`
(`scovox_map_split.hpp:246`) is `max(0.15, 0.10) = 0.15 m`, while every consumer
of those voxels stops earlier: the band gate is `sdf > −0.10` (`:439`), the carve
gate is `sdf > 0` (`:444`), and the TSDF write is gated off (`:421`). Voxels with
`sdf ∈ [−0.15, −0.10)` are therefore visited by the exact DDA and produce no
write — roughly one voxel per ray at `resolution` 0.05.

**Fixed** — but not the way the sentence above suggests, and the reason is a
property of the DDA that was not recorded anywhere before.

`back_reach` is not only where the walk stops, it is the point the DDA **aims
at**. `ExactRayIterator` steers at the *centre* of `coord_to`
(`ray_iterator.hpp:29-57`, `:55-57`), so shortening `back_reach` rotates the
whole segment and changes which voxels are crossed **in front of** the surface,
where every write actually happens. Tried and refuted by measurement, not by
argument: scene 016 with `back_reach` trimmed to the band gives
`e7f872b6d8e1` against the reference `a76dd502bb5b`. The `back_reach`
expression is therefore left verbatim, with a comment saying why.

The tail is dropped by **early termination** instead (`:280-282`, `:449-456`,
`:580`). `trim_tail = !tsdf_writes && useful_back < back_reach` arms it, where
`tsdf_writes = tsdf_enabled_ && !is_dynamic && !geometry_off` and
`useful_back = band_active ? sem_band_ : 0`. Inside `exact_body`, a voxel with
`sdf < 0` whose exact along-ray offset `t = −(v_point_voxel · u)` has reached
`useful_back` latches `stop_walk` and the iterator callback returns `false`.
Every gate below that point is already decided — the carve needs `sdf > 0`, the
band needs `dist ≤ sem_band_` and `dist ≥ t` — so returning drops nothing, and
`t` is non-decreasing along the walk (each DDA step adds `res·|u_i| ≥ 0`), so
latching drops nothing for the rest of the ray either.

One intermediate version was also wrong and is worth recording: stopping on the
bound `dist² ≥ useful² + 3h²` (half a voxel diagonal) gives `7c3c6d08b560`, not
the reference. The half-diagonal guarantee is against the *aimed* segment while
the test needs the offset from the *true* ray, and the two deviations add. The
offset has to be measured with the dot product, not bounded.

Acceptance was **byte identity of the dumped map**, not equal mIoU: the change
removes work and must therefore remove nothing else. Scene 016 fused, before and
after: both `a76dd502bb5b`.

---

## Verified as sound (no action)

- `#error` traps for every removed flag and for `SCOVOX_EVICT_INHERIT != 0`
  (`dir_voxel.hpp:133-144`); size `static_assert`s on all three voxel types.
- `sanitise()` snaps `w_occ` / `w_free` onto the ⅛ lattice
  (`sem_split_map.cpp:297-298`) and the node snaps both fusion profiles
  (`scovox_node.cpp:851-852`), so the count identity
  `a_occ = kBetaOccPrior + w_occ·n` holds on both paths. Since 2026-09-05 the
  prior in that identity is **0.5**, not 1.0 (see the new item below); `0.5 = 4/8`
  is itself on the lattice, and `sanitise()` now snaps the two prior fields too.
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


## Addendum — 2026-09-05: the occupancy prior moved to Jeffreys `Beta(0.5,0.5)`

`kBetaOccPrior` / `kBetaFreePrior` (`beta_voxel.hpp`) and `defaultBetaVoxel`'s
default arguments are now `0.5f`, promoted from Bayes–Laplace `Beta(1,1)`. E11
measured the switch across the eight SceneNN scenes at the shipped flags:
+0.0000 on union mIoU, intersection mIoU, occupancy IoU, both accuracies and
band mIoU, 0/8 wins, with `n_occupied` (21164.2) and `n_intersect` (12519.1)
identical to the voxel. The reason is algebraic — occupancy admission tests
`p_occ > 0.5`, and for any symmetric `Beta(a,a)` that reduces to
`W_occ > W_free` with `a` cancelling — so no tolerance is being leaned on.

**Three sites this review should now treat as prior-sensitive.**

1. `dscovox_node.cpp:664` publishes on `v.p_occ() >= ot` with `ot = 0.7`. The
   cancellation above does not extend to a non-0.5 threshold: at
   `w_occ=1.5, w_free=1.0` Jeffreys publishes after 2 hits against one miss
   where `Beta(1,1)` needed 3. This is a real change on the ROS wire and is not
   covered by the offline-dump measurement.
2. `carve_skip_occ_threshold` (`sem_split_map.hpp:378`) compares `p_occ`
   against an arbitrary value. It defaults to `0.0f` = off and the batched live
   path has no wall guard at all (`sem_split_map.cpp:452,462-467`), which is
   why the prior change is safe today — but a caller that enables it is
   choosing a regime where the prior is load-bearing.
3. `binary_serializer.hpp:296`, `:466-467` quantize and dequantize `a_occ`
   against `kBetaOccPrior`. Self-consistent for new data; a `.scovox_bin`
   archived under `Beta(1,1)` now decodes 0.5 low.

**Known divergence, deliberate.** The legacy `SemBetaVoxel` / `Voxel` substrate
(`sembeta_voxel.hpp:129`, `voxel.hpp:89`, `map_interface.hpp:50`,
`scovoxmap.cpp:54-55`) hardcodes `1.0f` and never reads the constants. It was left
alone: it is a different substrate, and unifying it is a change no measurement
here covers.

**Sentinel worth noting.** `sanitise()` treats `beta_occ_prior <= 0.f` as
"unset" and substitutes the constant, so Haldane `Beta(0,0)` cannot be reached
through `--beta-occ-prior` / `--beta-free-prior`. That arm is untested, not
rejected.

**Test-suite note.** Seven tests hardcoded a prior of 1.0 and were rewritten in
terms of the constants. One of them,
`BinarySerializer.QuantizedClampsBelowPriorAndAboveCap`, had a latent hazard
worth generalising from: its "below prior" input was the literal `0.62f`, which
is below 1.0 but above 0.5, so under the new prior it would have stopped
exercising the clamp *while still passing*. Prior-relative inputs, not
literals, wherever a test's premise is "below the prior".

**Pre-existing failure, unrelated.**
`ScovoxMapSplitFarCarve.FarCarveBitIdenticalToFullWalk` fails at pristine HEAD
on its `band = 0.30f` arm (verified by stashing the whole working tree:
`scovox_core` 183/184 under `./dev.sh ros-test`,
that test the only failure; it fails identically with the constants reverted).
The fast path stages more carve writes than the exact walk — 2887 vs 2886 on
scan 0, 3043 vs 3040 on scan 3 — so the far-voxel reduction is not exact when
`sem_band > sdf_trunc`: at band 0.30 / trunc 0.15 / res 0.05 `far_thr` is 7
voxels while the walk already reaches 6 behind the hit. The shipped
configuration runs `--sem-band 0.10 < sdf_trunc 0.15`, which is why it went
unnoticed. **The suite baseline is 183/184, not 184/184.**

---

## Addendum — 2026-09-05: the semantic prior `alpha_0` is settled at 0.01

E10 swept `alpha_0` post hoc by reparameterising dumps that were all *built* at
0.01. That leaves one path untested: `evict_by_confidence` compares a
probability, so `alpha_0` can flip an eviction and change which classes hold the
two slots — and slot residency is what `miss_rate` measures. E13 (2026-09-05)
re-ran the mapper on all eight scenes at the Perks level 1/14 against the
shipped 0.01, one binary, promoted configuration, frame order untouched, and
dumped `.bin` and `.slots` from the same runs. Full write-up in
`scovox_slot_rules/REVIEW_LOG.md`.

**The build path fires, and it does not reach the scored set.** Over 19,116,481
dumped voxels the two arms differ in slot class residency on **6** voxels
(0.00003 %), **none** of which pass `p_occ > 0.5`; `cnt` differs by more than
one count on **5** voxels, again none past the gate. The apparent 4.12 % of
voxels with a nonzero `cnt` delta is the prior itself being stored — on every
gate-passing voxel the difference is `1/14 − 0.01 = 0.0614285714` to within the
count quantisation, which is the expected consequence of `cnt` holding
`alpha_0 + evidence` (see the reparameterisation section above, `alpha_i =
cnt[i]`, "already alpha0 + e_i").

**Every map metric is exactly inert** — per-scene delta `+0.000e+00` on 8/8 for
union mIoU, intersection mIoU and occupancy IoU, with `n_pred_occupied` and
`n_intersection` integer-identical. Every E12 uncertainty functional is inert or
mildly negative; `MI_epist` stays below chance on the miss task in both arms.

**Consequence for this review.** The `alpha_0` axis is closed at both ends —
readout (E10) and build (E13) — so the M10 basis work should not be revisited by
moving the prior. `vacuity` remains retracted as a functional: it is
`C·alpha_0/(S + C·alpha_0)`, a monotone function of `S`, and E13 confirms this
directly — `vacuity` and `neg_S` have AUROCs identical to four decimal places in
both arms, because a common factor on `alpha_0` cannot reorder a ranking.

**Slot occupancy, measured.** 5.26 % of gate-passing voxels hold fewer than two
resident classes (011 is the outlier at 15.05 %; the next scene is 5.63 %), so
their query-time categorical is `{one class, OTHER}`. That is too small a share
to explain `MI_epist` reading below chance — 94.7 % of scored voxels have both
slots filled and MI is uninformative there too.

**Provenance caveat.** E13 ran on the `Beta(1,1)` occupancy build, predating the
Jeffreys promotion in the addendum above. Both arms share that binary so the
comparison is internally valid, but its absolute numbers are not comparable to a
post-promotion run.

**E13b — calibration, and the one place the prior is not inert.** The pass above
graded uncertainty with AUROC, a *ranking* measure that no monotone rescaling
can move, so it could not speak to calibration. Scoring Brier / ECE / NLL on the
same dumps splits the effect three ways: cell A (built 0.01, read 0.01), cell C
(built 0.01, read 1/14 — E10's readout counterfactual) and cell B (built 1/14,
read 1/14).

The **build** contribution `B − C` is at most `4.7e-6` absolute on `nll_miss`
and `5.1e-9` on `nll_hit`, Brier and ECE — float noise, sign-split 4/8. So
`B − A` equals `C − A` to five decimals. **Consequence for anyone reading this
review: a post-hoc reparameterisation of a 0.01 dump answers any α₀ question
exactly, and no replay is needed.** That retroactively validates E10's method,
which was the one open methodological doubt about it.

The prior is **not** inert on `nll_miss` — the surprise on voxels whose truth
was in neither slot improves 8/8, 4.8423 → 4.8171, because a miss is scored out
of the pool at `(alpha_0 + o_e/n_pool)/D` and a bigger prior raises that floor.
Three qualifications keep it from being a reason to move: one scene (015, miss
rate 27.9 %) is 93 % of the mean and 100 % of the total-NLL effect; `nll_hit` is
*worse* on 7/8 and rises monotonically in alpha_0 there; and `nll_miss` itself is
monotone in alpha_0 all the way to 1000, so selecting on it selects the uniform
prior. ECE, the metric that most directly asks whether the probabilities are
honest, is best at the incumbent and degrades monotonically upward. `alpha_0`
stays at 0.01.

## Addendum — 2026-09-05: code-smell sweep of the full tree

A maintainability pass over the whole non-vendor, non-test source: **14 725
lines** across 35 files (`find src \( -name '*.cpp' -o -name '*.hpp' -o -name
'*.h' \) | grep -v third_party | grep -v /test/ | xargs wc -l`; an earlier
revision said 14 402 across 33, which was neither the file set nor the count
that command produces). Scope is deliberately different from the rest of this
review — nothing below is a claim about mIoU or timing. `S`-prefixed ids so
they cannot be confused with the `H`/`M`/`L` findings above.

Two things should be said before the list, because they change how the list
reads. **There are zero `TODO`, `FIXME`, `HACK` or `XXX` markers in the entire
tree**, and **zero compiler warnings under `-Wall -Wextra`** at `-O2`, `-O3`,
`-O3 -march=native` and LTO (four separate configures; LTO is the interesting
one, since it is the pass that surfaces cross-TU problems). *Scope correction,
2026-09-05: those four configures build the offline harness, which reaches
`scovox_core` but cannot configure `scovox_mapping` — that package needs ament
and needs a colcon path — `docker/dev.sh ros-test`, or `scovox/docker/
build_and_test.sh`, which also runs colcon. Three warnings were hiding in the
part of the tree the four configures never compiled; they are described at the
end of this section and are now fixed, so the zero holds tree-wide. One of the
three was in `tools/split_memory_demo.cpp`, an unconditional target of
`scovox_core`'s own CMakeLists, so `./dev.sh test` would have shown it too — the
gap was the harness's four-TU subset, not ament.* There is no
`const_cast`, no `goto`, no raw `new`/`delete`, and no owning raw pointer. The
two ROS nodes share **no** duplicated 6-line window between them. The two
standing comment rules hold in `src`: **0** comments cite a `.md` path, and
**0** quote an mIoU or timing number. *Correction 2026-09-05: the second half
was false when written. `map_interface.hpp:36-37` carried "−31.81 ms/frame
t=−11.02, −69.38 MB RSS t=−5.68" in the `leaf_bits` comment — a measured result,
exactly what the rule forbids. Rewritten to state the design choice and its
mechanism, with the numbers left in memory. The `.md` half was true of `src` but
not of `config`, which the same sweep was supposed to cover: three YAML comments
survived it, all with dead targets, now repaired.* So what follows is not a codebase in trouble —
it is a small number of specific, mostly mechanical items.

### S1 — `SCovoxNode` is a god object, and it is the untested 30 %

**Status 2026-09-05: PARTIAL.** The two duplicated idioms inside `publishBinaryMap` were extracted to `node_utils.hpp` and are now unit-tested (S6). The full class decomposition was not attempted — the remaining 40 methods still sit in one translation unit and are still untested.

`scovox_node.cpp` is **3421 lines** — 24 % of the tree in one file — holding one
class with **148 member variables**, **42 methods** and **168 declared ROS
parameters** (138 node + 30 map). Four of its methods are over 170 lines
(`publishBinaryMap` 452, `declareNodeParams` 387, `integrateLidarSnapshot` 286,
the constructor 271).

The sharper half of the finding is where the test boundary falls. **No test
compiles `scovox_node.cpp` or `dscovox_node.cpp`.** Those two files are 4414
lines, 31 % of the tree, and their coverage is zero. The tested/untested line is
exactly the `.hpp` / `.cpp` boundary of the node packages.

This is not an argument for testing a ROS node in place. The project already has
the right answer and has used it three times: pull the logic into a header and
test that. `topk_provider.hpp`, the heartbeat re-emit in `node_utils.hpp` and
`dscovox_consensus.hpp` were each extracted out of a node and each has a suite.
The gap is that the single largest and most-branching function in the codebase,
`publishBinaryMap`, has not had the same treatment — while its exact
counterpart, `BinarySerializer::deserialize`, is 175 lines and *is* tested.

### S2 — `publishPointCloud` hardcodes two semantic slots, and its guard checks the wrong side

**Status 2026-09-05: FIXED.** The slot loops are bounded by `scovox::K_TOP` (`scovox_node.cpp:2117`, `:2201`) and the parameter is clamped to the compiled-in cap with a warning naming the recompile needed (`:372-377`). The load-bearing half of the fix is the pair of `static_assert`s in `publishPointCloud` (`:2807-2810`): that function is the one place in the tree that is not generic over `K_TOP`, so a `K_TOP > 2` build now fails to compile instead of silently publishing a truncated cloud.

`scovox_node.cpp:2815` reads

    static_assert(scovox::K_TOP >= 1, "publishPointCloud requires at least 1 sparse slot");

but the schema immediately below names `sem_cnt0`, `sem_cls0`, `sem_cnt1`,
`sem_cls1` and nothing else, and the copy loop guards only the low side
(`if constexpr (scovox::K_TOP >= 2)`). `K_TOP == 1` is therefore handled
correctly. **`K_TOP >= 3` is not**: slots 2… are silently absent from the
published cloud, with no error, no warning, and a `static_assert` that passes.

The contrast with the wire path is what makes this a defect rather than a
limitation. `binary_serializer.hpp` is K-generic, carries a `K_TOP_wire` byte in
the header, and makes the receiver assert it matches — sender/receiver K
disagreement fails loud by design. `publishPointCloud` is the one place in the
tree that quietly assumes 2.

This blocks a planned experiment: a K sweep needs one build per K, and at K=3 or
4 the point-cloud output would be wrong without saying so. Fix is one line —
`static_assert(scovox::K_TOP == 2, …)` to fail the build honestly, or a loop
over the slots to make it actually generic.

### S3 — The ROS envelope version is an unnamed literal duplicated across two packages

**Status 2026-09-05: FIXED.** `BinarySerializer::ENVELOPE_VERSION` now names it alongside `FORMAT_VERSION`, and both ends read the constant.

Sender, `scovox_node.cpp:2519`:

    bin.version = 5;   // envelope version — dscovox onBinaryMap routes on it

Receiver, `dscovox_node.cpp:314`:

    if (msg->version != 5) { … "expects envelope version 5, got %d (dropping)" … }

Three bare `5`s in two packages, agreeing by hand. The codec revision sitting
right next to it does the same job correctly: `FORMAT_VERSION = 8` is a named
constant in `binary_serializer.hpp`, written by `serialize` and checked by
`deserialize`, so the two ends cannot drift. Bumping the *envelope* version is
the operation with no such protection.

### S4 — `binary_serializer.hpp`'s file header still says the codec revision is 6

**Status 2026-09-05: FIXED.** The header now points at `FORMAT_VERSION` as the single definition instead of restating a number that goes stale on the next bump.

`binary_serializer.hpp:65` — "The blob VERSION byte is the codec revision: now 6
(was 5)…". `FORMAT_VERSION = 8` at `:171`, and the layout block at `:106` says
`[VERSION: u8 = 8]`. Revisions 7 and 8 each appended a new paragraph to the file
header instead of updating that sentence, so the first statement a reader meets
about the wire revision is two bumps stale — in the one file where a wrong
version number is most expensive.

### S5 — `0xFFFF` is written 76 times with no name

**Status 2026-09-05: FIXED.** `kEmptySlot` (`voxel.hpp:55`, `inline constexpr`) replaces the sites that mean *empty class slot* — 67 occurrences on 67 lines (no line carries two), re-counted 2026-09-05; the commit message's "69 sites" is wrong. Eleven bare `0xFFFF` remain outside the tests and `third_party`: the bit-mask, saturation-ceiling and unrelated-sentinel uses, deliberately left alone. Verified by byte-identity of the emitted wire.

The empty-slot sentinel appears as a bare literal 76 times across the non-test
tree, and `65535` (the same value, and also the `nhit` saturation ceiling) a
further 17. The same codebase already names the neighbouring sentinel —
`sem_split_map.hpp:754`, `static constexpr uint32_t kNoHitProbs = 0xFFFFFFFFu` —
so the convention exists and simply was not applied here. A `kEmptySlot` in
`dir_voxel.hpp` is a pure rename with a byte-identity check available.

### S6 — `publishBinaryMap` repeats one traversal idiom four times and one gate idiom twice

**Status 2026-09-05: FIXED.** `emitSnapshotOrTouched` and `gateAndRefresh` in `node_utils.hpp` hold the idiom once, including the `setValue` correctness note, and `test_publish_gate.cpp` covers them (15 cases).

An adversarial review of the extraction confirmed the six call sites are
behaviourally exact and found no lifetime hazard in the drained touched
vector, but landed six coverage and accuracy defects, all since fixed: the
test stand-in for `drainTouched*` copied where the real one swaps (and its
comment claimed otherwise); the `std::ref` guard, the gate's
`create_if_missing == false`, and the composed snapshot tick were each
documented as load-bearing yet passed with the property removed; the snapshot
traversal asserted cardinality but not coord/value pairing; and the asymmetric
predicate's comment claimed to pin an argument order at the node's two
call-site wrappers, which no test target compiles. The one finding not fixed
in code is a history defect: four commits in this series use the helpers
before the commit that defines them, so `git bisect` cannot build across
`f202860..bd45049`.

Inside the 452 lines, this shape appears verbatim for `tsdf`, `fine_tsdf`,
`beta` and `dir`:

    if (snapshot) { grid.forEachCell(emit_X); clearTouchedX(); }
    else { for (const auto& c : drainTouchedX()) if (auto* v = acc.value(c,false)) emit_X(*v,c); }

and this one twice, for `beta` and `dir`, differing only in the changed-since-
emit predicate and the wire transform: two `std::optional` accessors, a
`!snapshot` gate check, `gacc->setValue(c, v)`, and a conditional stamp write.
The tell is the comment. The first copy carries a real correctness note — that
it must be `setValue` and not `*value(c, true) = v`, because the preceding miss
caches a null leaf pointer — and the second copy has to say "see the Beta gate
note above". Load-bearing knowledge restated because the code was copied is
exactly the case for extracting it once.

### S7 — `map_mtx_` is a comment-enforced contract in a 42-method class

**Status 2026-09-05: FIXED, by the tag parameter rather than the annotations.**
All six methods now take a `scovox::MapLockHeld&` (or `MapWriteHeld&` where
the body mutates), a witness with no public constructor that only
`scovox::MapReadLock` / `MapWriteLock` can produce — see
`scovox_mapping/include/scovox/map_lock.hpp`. All seven lock sites now go
through those guards, and `finishScanTail` carries the witness through to
`publishBinaryMap`. The clang route was measured and rejected: neither the
host nor the container image has clang, and `apt-cache policy clang` offers
no candidate, so `-Wthread-safety` attributes would have been carried by gcc
as decoration and checked by nothing. `test/test_map_lock.cpp` asserts the
properties the contract leans on (witness not constructible, not copyable,
exclusive converts to shared and not the reverse).

Six methods document "Caller must hold `map_mtx_` (shared)" or "(unique)" —
`:2072`, `:2780`, `:2963`, `:3026`, `:3054`, and `publishBinaryMap` implicitly.
The lock is real (`std::shared_mutex`, taken at six other sites) but the contract
is prose. In a class this size a caller can be added without ever meeting the
comment. Clang's `-Wthread-safety` annotations, or a tag parameter the caller
can only construct while holding the lock, would make it checkable.

### S8 — one build switch has no default, which is what keeps `-Wundef` off

**Status 2026-09-05: FIXED.** `SCOVOX_E0_COUNTERS` declares `#ifndef … 0` at `e0_counters.hpp:45-47`, so every switch now has a default and `-Wundef` has nothing left to report. The *general form* named at the end of this section is also done — see S9.

Six of the seven compile-time switches declare their own default
(`SCOVOX_BETA_U16` 1, `SCOVOX_BETA_U16_SCALE` 8, `SCOVOX_TRACK_QMAX` 1,
`SCOVOX_TRACK_NHIT` 0, `SCOVOX_K_TOP` 2, `SCOVOX_DEPOSIT_TRACE` 0) via
`#ifndef`. `SCOVOX_E0_COUNTERS` does not — it is only ever tested with a bare
`#if`, so it is the preprocessor's silent 0.

Measured: compiling `sem_split_map.cpp` with `-Wundef` added produces **exactly
three warnings, all of them this one macro**. Giving it the `#ifndef … 0` its
six siblings have would let `-Wundef` be turned on permanently at zero noise,
which makes an undeclared `#if` switch impossible in source from then on.

Two scoping notes so this is not oversold. `-Wundef` does **not** catch a
misspelled `-D` on a build command line — that is a different hazard, and it is
currently guarded only for the four *removed* flags, by the `#error` traps at
`dir_voxel.hpp:135-147` (which are the right pattern and should be kept). The
general form of that guard would be for the binary to report its own compiled-in
switch values so a build manifest records what a binary *is* rather than only
its md5; `replay_scenenn` already refuses `--evict-by-confidence` on a
non-`TRACK_QMAX` build, so the ingredients are there.

### S9 — `version.hpp` is included by nothing

**Status 2026-09-05: FIXED.** Wired, not dropped: `version.hpp` now declares `scovox::buildSwitches()` (implemented in `src/version.cpp` by stringifying the macros the library itself compiled with), and `replay_scenenn` prints it to stderr at startup. This is the command-line half of the S8 hazard — the half `-Wundef` cannot reach.

The only orphan among the 22 public headers. It defines `SCOVOX_VERSION_STRING
"0.1.0"` and four constants, and no file in the tree includes it — while the
wire carries two other, unrelated version numbers (S3, S4) that it has nothing
to do with. Logged rather than deleted: either wire it into the envelope/codec
story or drop it, but do not leave a third version concept lying around unused.

### S10 — 36 comment lines name types that no longer exist (watch item, not a defect)

**Status 2026-09-05: PARTIAL.** `Bresenham` is at 0 mentions in the source tree, on user instruction — the deleted traversal must not read as a still-available option anywhere in the live code, and one launch-file parameter description was actively wrong about which traversal the build uses. The removal log in `docs/archive/design/` keeps the name, by the standing rule that everything removed stays logged. `SemDirMap` (8) and `SemDirVoxel` (14) remain as historical framing.

`SemDirMap` ×5, `SemDirVoxel` ×11, `Bresenham` ×2, "unified" ×18 — all in
comments, none in code (the single `SemDirVoxel` in code is inside a
`static_assert` message). The overall comment ratio is **0.63 comment lines per
code line**, which for this codebase is a feature: the *why* is the deliverable
and the design-choice register lives in the comments by standing instruction.
The corollary is that 5164 comment lines are a maintenance surface, and these 36
are where prose already outlived its subject. Most read as deliberate historical
framing ("de-unifies `SemDirMap` into…"), which is wanted; they are listed so a
future reader can tell that apart from rot such as S4.

---

## Errata — claims made by this review series that did not survive checking (2026-09-05)

The fixes in this series were themselves reviewed, and a pass dedicated to the
*truthfulness* of what the commits and this document assert found twenty
candidate defects. Nineteen were confirmed against source; one was rejected.
They are recorded here rather than silently repaired because several live in
commit messages, which are immutable without a history rewrite this series has
not performed.

### The pattern

`8f7e013` refreshed a batch of stale `file:line` pointers by applying a
**mechanical offset** rather than re-reading the headers. That is why `+3` was
right for two `dir_voxel.hpp` rows and wrong for the `SCOVOX_EVICT_INHERIT`
trap, and why the freshly authored `e0_counters.hpp:42-44` landed on the comment
*above* the `#ifndef` it was meant to name. Every pointer this document carries
was re-read from source on 2026-09-05, and the standing rule is now stated at
the top of `scovox_code_structure.md`: when a pointer disagrees with the tree,
trust the symbol name and re-read, never add an offset.

### Corrected in the documents

| Claim | Where | Actual |
|---|---|---|
| `e0_counters.hpp:42-44` holds the default | this doc, `structure.md:367`, `:404-405` | `:45-47`; `:42-44` is the comment above it |
| `dir_voxel.hpp:145-147` is the `EVICT_INHERIT` trap | this doc | `:144-146` |
| `FORMAT_VERSION = 8` at `:164` / `:168` | this doc `:1126`, `structure.md:512` | `:171`; `ENVELOPE_VERSION = 5` at `:179` |
| S2's slot loops at `scovox_node.cpp:2103`, `:2187`, clamp at `:362-368` | this doc `:1080` | `:2117`, `:2201`, `:372-377` — and the load-bearing part, the `static_assert` pair, was not mentioned at all (`:2807-2810`) |
| `dscovox_node.cpp:661` publishes on `p_occ() >= ot` | this doc `:922`, `structure.md:110` | `:664`; `:661` is a comment |
| `integrateHitFused` is `:183-565`; band deposit at `:439-441` | `structure.md:441`, this doc `:540` | `:183-618`; the band gate is `scovox_map_split.hpp:493-495` → `semBand` `:977-980` → `applyBandSemantic` `sem_split_map.cpp:854-901` |
| `integrateHitSplit` `:575-595` has "no `tsdf_enabled_` test" | `structure.md:449` | `:629-655`, and it *does* test it (`:648`) — `f2d535e` fixed that, and the doc kept asserting the bug after its own fix |
| "Exactly one `assert()` outside the tests" | `structure.md:394` | two: `carve_stage.hpp:78` and `third_party/bonxai/bonxai/grid_allocator.hpp:155` |
| `scovox_mapping` has 8 gtests | `structure.md:329` | 10 (9 in `SCOVOX_TEST_SOURCES` + `test_topk_provider`). `scovox_core`'s "11" is correct |
| Seven line counts | `structure.md:333-335` | six of the seven were stale; only `dscovox_node.cpp` 994 was right |
| "§2 is this repository at commit `33aa576`" | `structure.md:10` | 27 commits behind; re-pinned to `32121f2` |
| `version.hpp` reports "every `SCOVOX_*` build switch" | `version.hpp:16-17` | it reports the seven *live* switches; the four removed-arm `#error` traps (`VICTIM_MEAN`, `VICTIM_QMAX`, `ADMIT_NORM`, `EVICT_INHERIT`) are absent. Wording narrowed, and the omission of `SCOVOX_EVICT_INHERIT` — the one switch the harness puts on its command line — is now stated explicitly |

### Fixed forward in code, because the commit message cannot be amended

- **`e04b76d`: "prints `buildSwitches()` before it does anything else."** It
  printed at `replay_scenenn.cpp:414`, *after* `Args a;` and after the `parse()`
  failure branch that `return 2`s — so the one run that most needs to say which
  binary it is, a run killed by a malformed command line, said nothing. The
  print is now the first statement of `main`, which makes the claim true.
- **`e04b76d`: "the gate for the whole review series."** `verify_fixes.sh` gated
  nothing. It ran `set -uo pipefail` with no `-e`; it *printed* the warning count
  instead of comparing it to the stated zero bar; a package that failed to
  configure or build was `continue`d past; a missing reference dump was
  `continue`d past **without setting `FAIL`**, so with both references absent the
  script would still print "byte-identical on every scene checked" having
  compared nothing; and it ended `echo "DONE"` with exit 0 unconditionally.
  Every one of those is now a `FAIL`, and the script exits nonzero.
- **`db659af`: "make version.hpp the binary's self-report."** The print was added
  to `replay_scenenn` but `version.cpp` was never added to the offline harness's
  `scovox_core` source list, so `replay_scenenn` **did not link** from `db659af`
  until 2026-09-05. It went unnoticed because the review series only rebuilt
  `scovox_core`'s own test targets, and because the harness's standalone CMake
  path is exercised only outside a ROS workspace. This is the first thing the
  hardened `verify_fixes.sh` caught, on its first run.

### Recorded, not fixed

- **`f202860`: "69 sites renamed", "7 bare `0xFFFF` remain".** Re-counted at
  `32121f2`: 67 occurrences on 64 lines were renamed, and **11** bare `0xFFFF`
  remain outside the tests and `third_party`. The rename itself is correct — the
  survivors are all bit-mask, saturation-ceiling and unrelated-sentinel uses, as
  intended — only the arithmetic in the message is wrong.
- **`32967a2`: "replaced three `file:line` pointers".** It replaced **five**
  (`voxel.hpp:39-42`, `scovox_node.cpp:860-863`, `dir_voxel.hpp:250`,
  `sem_split_map.cpp:130-168`, `dir_voxel.hpp:233-236`).
- **Four consecutive commits do not compile.** `f202860`, `eb4f9b3`, `8de5be1`
  and `bd45049` each call `scovox::gateAndRefresh` / `emitSnapshotOrTouched`
  while `node_utils.hpp` still defines them behind a `def=0` guard; the helpers
  only arrive in `f766258`. Verified by `git show <commit>:<file>`. `git bisect`
  across this range will report "cannot build" on four points. Repairing it means
  splitting `f766258` and rebasing, which this series has deliberately not done:
  the commits are local and unpushed, a backup branch
  `backup/review-series-pre-split` exists at `33724d4`, and the rewrite is
  waiting on an explicit decision rather than being taken unilaterally.

### Rejected

- **`eb4f9b3`: "three bare `5`s in two packages."** Checked against the diff:
  the three functional sites are the sender's `bin.version = 5`
  (`scovox_node.cpp`), the receiver's guard `msg->version != 5` and its error
  string (`dscovox_node.cpp`), across `scovox_core` and `scovox_mapping`. The
  claim is accurate as written.

### S11 — three warnings the harness build never compiled, and two tests weaker than their names

**Status 2026-09-05: FIXED.** Found by running the full colcon build rather
than the harness configure. Zero warnings now holds across the whole tree, not
just the part the offline harness compiles.

`split_memory_demo.cpp:17` was a `-Wcomment`: a trailing `\` on a `///` line
splices the next line into it, so the second half of the documented invocation
was never part of the rendered comment. Replaced with alignment; a sweep found
it to be the only instance in the tree.

The other two were `-Wunused-variable`, and in both cases **the variable the
compiler flagged was the assertion the test was named for**:

- `BetaUpdate.NHitsConvergesToExpected` (`test_beta_update.cpp`, `TEST` at `:92`,
  the flagged variable at `:108`), computed
  `expected_a_occ = 1 + 2*N` and then compared nothing to it, passing instead on
  `a_occ > 10` — a bound any `N >= 6` satisfies, so a 3x error in the deposit
  weight would have left it green. Its comment blamed range weighting for the
  slack, but `integrateRay` defaults `range_w` and `angle_w` to `1.0f` and
  `fused_integrate_ray_static` uses what the caller passed;
  `range_decay_length` is consulted only inside `carve_free`. Probed rather
  than argued: `a_occ = 41` exactly, `a_free = 1`, `p_occ = 0.976190448`. The
  count identity is now asserted with `EXPECT_FLOAT_EQ`, together with `a_free`
  and `p_occ`.
- `BetaUpdate.DecayPreservesRatio` (`test_beta_update.cpp`, `TEST` at `:158`,
  `ratio_before` at `:162`; `:155` is inside `DecayMovesTowardPrior`, a different
  test), computed
  `ratio_before` and asserted only `ratio_after > 1.0f`. The upper bound is what
  separates a decay from a rescale: without `ratio_after < ratio_before` the
  test passes for a transform that moves the ratio *away* from the prior.
  Measured 2.0 → 1.95238125; both bounds are now asserted.

A note against a plausible wrong conclusion: the `1.0f` in that formula is not
stale with respect to the Jeffreys promotion. These tests use `scovox::Voxel` /
`defaultVoxel()` from `voxel.hpp`, whose prior is Beta(1,1); the 0.5/0.5
Jeffreys prior belongs to `BetaVoxel` in `beta_voxel.hpp`, a different type on
the split path.

## Addendum — 2026-09-05: round 3, and two mechanisms the tests uncovered

Three reviewers ran in parallel over the gate, the tests, and the factual
claims of these documents. The document corrections are folded into the Errata
table above; the gate rewrite is recorded in `REVIEW_LOG.md`. What belongs
*here* is the two pieces of production behaviour that only surfaced because a
test I had written from first principles disagreed with the binary — and in
both cases the binary was right.

### The carve weight is asymmetric between `integrateRay`'s two branches

`BetaUpdate.FreeCarveAlwaysGoesToPersistent` deposits 1.5488117 of free mass
where its two sibling tests, on identical geometry, deposit 2.0.

- **Dynamic** (`scovoxmap.cpp:186`) calls the two-argument `carve_free`, which
  derives `range_w = exp(-|hit − origin| / range_decay_length)` from the
  geometry itself (`:69-74`, `range_decay_length = 5.0` → `exp(-0.6)` for a
  3 m ray).
- **Static** (`:191`) runs `fused_integrate_ray_static`, whose `carve_w` is the
  `range_w` its *caller* passed (`:260`), defaulting to `1.0f`.

Both are deliberate: the node computes the range weight once from the full
sensor→hit range and hands it down, while the legacy dynamic two-pass path
recomputes it. Nothing is being changed. It is documented because the two
contracts are invisible at the call site — `map.integrateRay(o, h, is_dynamic)`
looks like one function with a flag, and it is two weighting schemes.

### `s_total` counts looks, not labelled looks

`SemSplitMap.EvictionConservesMassToOther` came out **exactly 0.75** above a
sum taken over the labelled hits, and 0.75 is `kappa0 · p_occ_post` for the one
`applyHitUpdate(coord, nullptr)` the test uses to lift `p_occ` over the gate.

`commitHit` gates on `p_occ_post` alone and never inspects `sem_probs`
(`sem_split_map.cpp:719-745`), and `dirichletUpdate` opens with an
unconditional `d->s_total += class_share` before any branch reads the vector
(`:79`) — its own comment states the invariant: *every* branch adds exactly
`class_share` and no more. A hit that names no class still books its whole
share; it lands as uncovered mass, i.e. in the derived `other()`.

Consequence for anything that reads the Dirichlet concentration:
`s_total − C·α₀` is a running count of **observations of an occupied voxel**,
not of **class observations**. Combined with M9 (the band is un-batched, so a
band voxel takes one deposit per depth pixel while the hit voxel takes one per
frame), there are now two independent reasons a raw `s_total` overstates the
semantic evidence behind a voxel. Neither affects mIoU — the scorer excludes
Dir-only voxels by `state` — and both matter to any confidence readout.

---

## Addendum — 2026-09-05: the batch is verified byte-identical on all 8 scenes

Every fix in this review series was claimed to be behaviour-preserving. That
claim is now measured rather than asserted, three independent ways, and the
detail lives in `scovox_slot_rules/REVIEW_LOG.md` under *"Verifying the
code-review batch"*.

- **The gate.** `scripts/verify_fixes.sh` PASSes: clean build at the shipped
  `-O3` flag set with **0 warnings in our tree**, **0** under `-Wundef`,
  `scovox_core` 184 / 1 and `scovox_mapping` 143 / 0. The one red is M4's
  `FarCarveBitIdenticalToFullWalk` at `band = 0.30`, red at HEAD before this
  batch.
- **Byte identity against HEAD itself.** `scripts/pristine_head_replay.sh`
  compiles a `git archive HEAD` snapshot of both trees — submodule `32121f2`,
  toplevel `e04b76d` — at a flag line character-identical to the reviewed
  build, and replays all eight scenes. **8/8 identical**, 458.8 MB of map
  matching byte for byte from two different binaries.
- **The metrics.** `cells/goal8.tsv` regrades the shipped candidate on all
  eight scenes; see H4 above, whose `total` column reproduces unchanged.

The scope this rests on: outside tests, the entire compiled delta of the batch
is one function extraction (`bernoulliEntropy` out of the two
`expectedInformationGain` overloads). `map_interface.hpp`, `version.hpp`,
`scovoxmap.cpp`, `split_memory_demo.cpp` and all three touched YAMLs differ in
**comments only**. Byte identity therefore confirms the scope claim; it does
not validate the extraction, since no offline path calls it — the only callers
are `scovox_node.cpp:2944` and `dscovox_node.cpp:725`. What covers the
extraction is 18 assertions in `test_uncertainty.cpp`, three of them added for
the guard's out-of-range and NaN behaviour, plus three in `test_beta_update.cpp`.
