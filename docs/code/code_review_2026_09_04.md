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

## Summary

| id | severity | finding |
|---|---|---|
| H1 | High | `scovox_node` defaults are not the promoted configuration |
| H2 | High | The promoted state is uncommitted (31 files since `d5da6a8`) |
| H3 | High | `integrateHitSplit` ignores `tsdf_enabled` |
| M1 | Medium | Four parameter structs with four sets of defaults |
| M2 | Medium | Six library knobs have no ROS parameter |
| M3 | Medium | `range_decay_length` is dead in the split path but still documented as a weight |
| M4 | Medium | One known failing test, `FarCarveBitIdenticalToFullWalk` |
| M5 | Medium | `dscovox` fused grids ignore `dir_leaf_bits`; first pose wins; fold order matters |
| M6 | Medium | Wire block runs hard-code `leaf_bits = 3` |
| M7 | Medium | `evidence_saturation` is one knob for two caps |
| M8 | Medium | Experiment write-ups still print the demoted build flag |
| L1 | Low | Stale byte-size and type-name comments |
| L2 | Low | In-code references to moved or nonexistent documents |
| L3 | Low | README document index broken by the archive move |
| L4 | Low | Legacy voxel / map types still compiled and tested |
| L5 | Low | Stale tool and comment text around the wire format |
| L6 | Low | `downsample_voxel_size` default described two ways |

---

## High

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

### L3 — README document index broken by the archive move

`README.md:24-27` lists four design docs by their old paths, and `:123`,
`:181` link `docs/distributed_mapping.md` and
`docs/publish_scovox_bin_from_bag.md`. All six now live under
`docs/archive/`. The README was deliberately left untouched in this session;
the fix is a six-line path edit plus links to
`docs/scovox_code_structure.md` and this file.

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
