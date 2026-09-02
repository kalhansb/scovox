# Removals ledger and untested-feature register — 2026-09-02

A snapshot taken at the flag-removal commit. Two parts, and the second is the
reason the first is safe to read: **nothing was deleted for being untested.**
Code was deleted only where a measurement chose between two implementations and
one lost. Everything that ships without proof is listed in Part 2 and kept.

**Recovery point.** Every deletion below is reachable at tag *and* branch
`pre-flag-removal-2026-09-02`, which exists in **both** repositories — the
`scovox` submodule (`new_experiments`) and the outer `scovox_new_experiments`
(`phase-p-slot-rules`). The submodule gitlink in the outer tag pins the matching
core revision, so checking out the outer tag alone reproduces the whole build.

---

## Part 1 — Removed

### 1.1 The Bresenham ray traversal

`RayIterator` — integer, 26-connected, stepping up to three axes at once — is
gone from `include/scovox/ray_iterator.hpp`. `ExactRayIterator` (Amanatides &
Woo 1987, 6-connected, vendored verbatim from Bonxai under MPL-2.0) is now the
only traversal, unconditional at all six call sites:

| file | site |
|---|---|
| `src/scovox_core/src/tsdf_map.cpp:122` | band walk |
| `src/scovox_core/src/sem_split_map.cpp:406` | `carveRay` |
| `src/scovox_core/include/scovox/scovox_map_split.hpp:467` | fused walker |
| `src/scovox_mapping/src/scovoxmap.cpp:355` | `fused_integrate_ray_static` |
| `src/scovox_mapping/src/scovoxmap.cpp:84`, `:379` | both `carve_free` overloads |

**Why deleted rather than kept behind the switch.** The two are not
superset/subset: the exact walk sees 1.84x more voxels per ray (159.4 vs 86.7 at
res 0.05), and on 82% of random oblique rays Bresenham visited a voxel the true
segment never enters. Upstream states the consequence itself — carving becomes
"weaker and depends on the direction of the ray relative to the grid axes" — and
free-space carving is the measurement most exposed to it, because the skipped
voxels are exactly the free-space evidence that never gets deposited. A
correctness difference of that shape is not an A/B axis. Cost accepted: ~+46%
walker time (split), +49–59% (fused), well under the 1.84x voxel ratio.

**Signature change to watch at any new call site.** `ExactRayIterator` takes a
CONTINUOUS start position plus the resolution, where `RayIterator` took two
integer coords. The continuous position must correspond to the integer start
coord passed, or the walk begins on the wrong voxel boundary. This is why
`fused_integrate_ray_static` gained an explicit `start_pos`: `k0` had been
computed inline from one of two different continuous points depending on
`band_only_integration`.

**One test bound moved with it.** `FineTsdf.CleanOrbitDbhWithinBudget`: the
clean-orbit RMS floor on that synthetic geometry rose ~0.022 → ~0.0255 because
the exact walk visits the oblique band-edge voxels Bresenham was skipping, and
their projective bias survives tent-weighting. The tripwire moved 0.025 → 0.027.
Radius, centre and arc-coverage assertions are untouched and still pass, so the
fitted geometry is as accurate as before — only the residual scatter grew. The
engineering gate behind the tripwire is 0.03.

### 1.2 Environment-variable latches (three)

| removed | what it selected | fate |
|---|---|---|
| `SCOVOX_EXACT_RAY` | Bresenham vs exact DDA | deleted with the loser — nothing left to select |
| `SCOVOX_DISABLE_FAR_SKIP` | far-voxel skip off | **promoted**, see below |
| `SCOVOX_DISABLE_FAR_CARVE` | far-voxel fast carve off | **promoted**, see below |

The two far-voxel switches were NOT removed as features. They select whether to
take a shortcut *around* code that runs either way — there is no second
implementation to delete — and the four differential tests are the only proof
the shortcut is sound, so deleting the switch would have deleted the proof.
They became one explicit field, `ScovoxMapSplit::Params::far_voxel_fast_paths`
(default `true`), read back through `farVoxelFastPaths()` and printed by the
node's startup `TSDF:` line. Hidden `getenv` state is gone; the A/B is not.

Also deleted with them: `envExactRay()`, `envFarSkipDisabled()`,
`envFarCarveDisabled()`, `farSkipDisabled()`, `farCarveDisabled()`, the
`exact_ray_` / `far_skip_disabled_` / `far_carve_disabled_` members across three
headers, and the `ScopedEnv` test helper (replaced by `fullWalk(p)`).

`grep -rn getenv src/` in the core now returns nothing.

### 1.3 Slot-rule build flags (four)

Swept alternatives that lost. Each was a `-D` axis in the rule matrix; the
winning branch is now the only code.

| removed | what it was |
|---|---|
| `SCOVOX_VICTIM_MEAN` | mean-strength victim selection (`cnt/nhit`) instead of min `cnt` |
| `SCOVOX_VICTIM_QMAX` | min-`qmax` victim selection |
| `SCOVOX_ADMIT_NORM` | nhit-normalised admission test |
| `SCOVOX_EVICT_INHERIT` = 1, 2, 3 | evicted-evidence inheritance modes; 0 (no inherit) ships |

**Silent-zero hazard, closed.** An undefined macro is `0`, so a `-D` that never
reaches the compiler produces a clean build and a variant-labelled but
byte-identical binary — the MANIFEST md5 cannot catch it, because the md5 is
correct and the label is the lie. `dir_voxel.hpp:91–100` now carries four
`#error` traps naming this tag, so a stale `-D` fails loudly at configure time.

### 1.4 Collateral, logged and left alone

- `scovox_slot_rules/scripts/build_mvic.sh` passes `-DSCOVOX_VICTIM_MEAN=1` and
  `-DSCOVOX_ADMIT_NORM=1`, so it now fails to configure by design. Left as the
  historical record of that invocation — its paths (`/home/user/quick_cc_work`)
  are already dead on every current machine.
- `scovox_slot_rules/scripts/build_rules.sh` gained an explicit guard: `I` other
  than 0 exits 2 with the tag name rather than reaching the `#error`.
- `scovox_slot_rules/cells/e9.tsv:353` still names `e5/k2_i3_evid`, a cell that
  is now unbuildable. Left because the file is a record of what was run.
- `replay_scenenn`'s provenance line lost its `exact_ray=` field; there is one
  traversal to report.

---

## Part 2 — Ships without proof (kept, by instruction)

Listed so that "we have tests" is never read as "this is covered". None of these
is a deletion candidate.

### 2.1 Compile-time axes no in-tree build exercises

| flag | default | status |
|---|---|---|
| `SCOVOX_BETA_U16` (+ `_SCALE`, 8) | 0 | Halves the Beta grid. `test_sem_split_map.cpp` has `#if SCOVOX_BETA_U16` blocks, but **no build in the tree sets it to 1**, so neither `./dev.sh test` nor `ros-test` ever compiles them. Reachable only via `EXTRA=-DSCOVOX_BETA_U16=1` through `build_rules.sh`. Known constraint: requires `--batch-hits 1`. |
| `SCOVOX_TRACK_NHIT` | 0 in core, **1** in `build_rules.sh` | The unit-test build and the sweep build therefore disagree on the shipped `DirVoxel` layout. The layout `static_assert`s cover both, the *behaviour* under `nhit` tracking is exercised only by the sweep. |
| `SCOVOX_K_TOP` | 2 | The size `static_assert`s at `dir_voxel.hpp:157,160` are guarded on `K_TOP != 2`, so K=4/8 layouts are unasserted. `ktop_map_sweep.sh` is the only consumer. |
| `SCOVOX_DEPOSIT_TRACE` | 0 | Instrumentation. No test. |
| `SCOVOX_E0_COUNTERS` | undefined | Instrumentation (`e0_gate_a.sh`, `build_e0.sh`). No test in the default build. |

### 2.2 Claims resting on measurements that are no longer reproducible

- **The free-space-carve redundancy finding.** The 8/8 tie between carving and a
  Dirichlet evidence threshold was measured on the Bresenham traversal deleted
  in §1.1 — the traversal whose documented weakness is precisely weak carving.
  "The carve adds nothing" and "the traversal never delivered the carve" are
  indistinguishable in that score. Not established on the shipped mapper; must
  be re-run before it is published. No re-run has been done.
- **Every mapping number in `scovox_slot_rules/RESULTS.md` above its final
  section** was produced by a traversal the source no longer implements.
- **The fused-walker performance case.** Measured on four scenes, but not at the
  shipped `space_carving=false`.
- **`band_perf`.** Referenced by `docs/design/fix_plan_2026_08_26.md` and, until
  today, by the far-path tests as their execution proof. It does not exist in
  this repository. Part 3 replaces what it was standing in for.

### 2.3 Known defect, logged and not fixed

`integrateHitSplit` (`scovox_map_split.hpp:534`) calls `tsdf_.integrateRay`
with no `tsdf_enabled_` gate, while the fused walker gates on it at `:431`. So
`--tsdf-enabled 0` is honoured by the fused walker only, and any fused-vs-split
A/B run with it is measuring the flag on one arm and not the other. The fix is
one conjunct; it is not applied here because it changes shipped behaviour and
belongs in its own commit with its own re-run.

### 2.4 A guard this diff thinned without breaking

`FineTsdf.AnchorAbsorbsOdometryDrift` asserts the anchored map beats the raw
map on fit RMS. Under the exact DDA both arms got worse and the *margin between
them* narrowed 3.2x:

| arm | before (Bresenham) | after (exact DDA) |
|---|---|---|
| anchored rms | 0.022518 | 0.025511 |
| raw rms | 0.025801 | 0.026542 |
| **margin** | **0.003283** | **0.001031** |

The assertion still holds and the direction is unchanged, so nothing is failing.
It is logged because this is precisely the backup guard that
`test_fine_tsdf.cpp:297-298` cites when it justifies raising the
`CleanOrbitDbhWithinBudget` tripwire from 0.025 to 0.027 — the diff that moved
the tripwire also thinned the evidence the move leans on. Anyone who narrows
that margin further should treat the drift test as at risk of going
non-discriminating before it goes red.

---

## Part 3 — Newly proven

The far-voxel skip and fast carve had four differential tests asserting **bit
identity** with the full walk. Identity is the absence of a difference, which a
fast path that never armed produces just as reliably as one that armed and was
correct: with `far_thr` mis-sized so nothing ever qualified, all four would have
stayed green over dead code. The tests said so themselves and deferred the
execution proof to `band_perf`, which is not in the tree (§2.2).

`ScovoxMapSplit` now counts the voxels each shortcut actually removed from the
exact float body — `farSkippedVoxels()`, `farCarvedVoxels()`, plain counters
next to `tsdf_ns_` — and all four tests assert on them:

- the arming configs must show a non-zero count, and the `far_voxel_fast_paths=false`
  reference arm must show zero;
- the mutually-exclusive path must show zero (skip off on batched configs, fast
  carve off on unbatched);
- each inertness config must show zero, which turns three tautological tests
  ("both arms run the same path, so of course they agree") into direct
  observations that no shortcut fired.

Verified: `ScovoxMapSplitFarSkip.*` and `ScovoxMapSplitFarCarve.*` — 4/4 pass
with the counter assertions compiled in. Full suite 294/294. The build emits
exactly one warning, a pre-existing `-Wcomment` from the backslash-continued
`///` usage block at `tools/split_memory_demo.cpp:17` — a file this diff does
not touch.
