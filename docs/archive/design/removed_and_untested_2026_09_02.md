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

### 1.5 The per-observation `quality` factor

`quality` was a float threaded through ~60 sites — every `integrateHit` /
`integrateMiss` / `integrateRay` entry point, `HitWeights`' staging record, and
the Dirichlet weight itself. It scaled all three per-observation increments:

```
a_occ       += w_occ  * quality
a_free      += w_free * quality
class_share  = kappa0 * p_occ * quality
```

It is gone. Every admitted observation now contributes a fixed **count**:
`a_occ += w_occ`, `a_free += w_free`, `class_share = kappa0 * p_occ`.

**Why this changed no number this suite has produced.** In ROS, `quality` was
`range_weight * angle_weight`, with `rw = exp(-rng/range_decay_length)` and `aw`
from a grazing-angle normal estimate. Both knobs are gated on a positive
parameter, and both default to **-1.0 (disabled)** in every RGB-D/SceneNN config
in the tree — `scovox_bin_min.yaml`, `scovox_single_robot.launch.py`,
`scenenn_eval.launch.py`, `default_params.yaml`. The only file that sets one is
`semantickitti_eval.launch.py` (`range_decay_length`, launch-arg default 50.0),
a LiDAR benchmark outside this suite. And `replay_scenenn.cpp` — the binary that
produced every mIoU in RESULTS.md — passed the literal `1.0f`. So on every
configuration this repository has ever measured, `quality === 1.0` already, and
the removal is behaviour-neutral by construction rather than by A/B.

**What went with it.**

- The grazing-angle block in `scovox_mapping/src/scovox_node.cpp` — the normal
  cross-product, `aw`, and the `rdZ` depth-sampling lambda that fed it — is
  deleted. It had no consumer once `quality` was gone, and no config in the tree
  enabled it. `rdZ` outlived the first pass and was caught by
  `-Wunused-but-set-variable` on the full ROS rebuild; the no-return-ray
  endpoint list `nr_eps`, which sits beside it, is still live.
- `DirichletUpdate.QualityScalesEvidence` (`test_dirichlet_update.cpp`) is
  deleted: it asserted "higher quality produces more evidence", and its entire
  subject no longer exists.
- `SemanticMode.MajorityVoteIgnoresQuality` would have become a tautology — both
  arms identical — so it is retargeted to `MajorityVoteAddsExactlyOneVote`,
  asserting the invariant that survives: one admitted observation is one vote,
  `EXPECT_FLOAT_EQ(cnt, 1.0f)`, whatever softmax mass the winning class carried.

**Left in place deliberately, not overlooked.**

- `p_occ` stays in `class_share = kappa0 * p_occ`. That is the documented
  Bayesian-soft attribution — marginalising the class observation over the Beta
  posterior — and a separately swept design axis, not a confidence factor.
- The legacy `scovox_mapping` `Map` keeps its `range_w` / `angle_w` parameters
  **in this commit**. They are distinct named arguments weighting occupancy
  rather than semantics, and `test_beta_update.cpp`'s `FarHitsGetLessEvidence`
  still exercises `range_w`. **Scheduled for removal — see 1.5a.**
- `grazing_angle_threshold` is still declared at `scovox_node.cpp:382` and is
  now **read by nothing**. Left as a declared-but-unused ROS parameter so an
  existing launch file that sets it does not fail to load; it has no effect.
- `semantickitti_eval.launch.py` still passes `range_decay_length: 50.0`. On the
  split-map path that value no longer weights anything. The parameter continues
  to gate the min/max range cull (below), which is its only surviving role.

**What the removal bought.** With no per-observation confidence factor, the
Beta parameters became a pure function of two integer counts —
`a_occ = prior + w_occ*n_hit`, `a_free = prior + w_free*n_miss`. That identity
is now enforced (weights are snapped onto a storage lattice at construction) and
asserted by five `BetaCountIdentity` tests. The design, the lattice rule and
its rationale, the `w_occ:w_free` evidence, and the four operations that are
*not* count-expressible are written up in
[count_storage_2026_09_02.md](count_storage_2026_09_02.md).

A five-reviewer code-smell pass over the whole mapping stack, run the same day,
found that the enforcement above was incomplete: `HitWeights` fusion profiles
reach `applyBetaUpdate` WITHOUT passing through `SemSplitMap::sanitise`, so the
identity held on the single-sensor path and failed silently on the fusion path.
Fixed (`sanitise(HitWeights&)`, called from `buildFusionProfiles`), along with
two other defects in the same day's code. That review also extends this ledger
in three places -- it independently confirms the `range_decay_length` death
recorded below, adds five more declared-but-unread ROS parameters, and finds
one macro (`SCOVOX_E0_COUNTERS`) still carrying the undefaulted-`#if` hazard
that §2 was written about. All 63 findings were verified against source before
being recorded; four were withdrawn on that check. See
[code_review_2026_09_02.md](code_review_2026_09_02.md).

### 1.5a Scheduled: `range_w` / `angle_w` on the legacy `Map`

Noted 2026-09-02 for a follow-up commit. These are the last surviving
per-observation weighting factors, and they are the same idea as `quality` under
different names — `a_occ += w_occ * range_w * angle_w`
([scovoxmap.cpp:18](../../src/scovox_mapping/src/scovoxmap.cpp#L18)),
`a_free += w_free * range_w` ([:23](../../src/scovox_mapping/src/scovoxmap.cpp#L23)).
Removing them makes the legacy substrate agree with the split path: one admitted
observation, one count.

**Scope.** ~40 sites. Three public `Map` entry points and four private helpers in
[scovoxmap.hpp](../../src/scovox_mapping/include/scovox/scovoxmap.hpp) (lines
44, 51, 57, 140, 144, 155, 157, 158), their definitions plus the interior
plumbing in `scovoxmap.cpp` (`beta_update_occupied`, `beta_update_free`,
`carve_free`'s `range_w_override`, `update_endpoint*`, and the `w_ray =
range_w * angle_w` fused-walk weight at :226 / :247 / :326).

**Two consequences to decide before doing it, not after.**

1. Unlike the split path, the legacy `Map` is a *real* consumer of
   `range_decay_length`: it computes `range_w` itself at
   [:66-68](../../src/scovox_mapping/src/scovoxmap.cpp#L66-L68) and
   [:362-364](../../src/scovox_mapping/src/scovoxmap.cpp#L362-L364). Removing
   these arguments therefore retires the parameter's last live consumer
   anywhere in the tree, and the node's declaration becomes dead in the same way
   `grazing_angle_threshold` already is.
2. `BetaUpdate.FarHitsGetLessEvidence` exists solely to assert that a distant
   hit accrues less occupancy evidence than a near one. With the mechanism gone
   the two arms become identical and the test becomes a tautology, so it is
   deleted rather than retargeted — the same call made for
   `DirichletUpdate.QualityScalesEvidence` above. `test_tsdf_band.cpp:121-122`
   only mentions the factors in a comment computing an expected value with both
   at 1, so its arithmetic is unaffected; the comment needs rewording.

The `float`-adjacent-to-`bool` sweep in the hazard note below applies to this
removal too: `integrateRay`'s `bool is_dynamic` sits before the weights, so a
dropped argument shifts the remaining floats rather than binding to the bool —
less dangerous than the `quality` case, but the sweep is still the check.

**One hazard this created, and the guard against it.** `quality` sat immediately
before `bool is_dynamic` in the hit signatures, so a leftover float literal at a
call site binds *silently* to `is_dynamic` — it compiles clean and routes the hit
to the transient grid. It surfaced as a denormal `4.5914945e-41` occupancy and a
zero fine-voxel count, not as a compiler error. Patching failures one at a time
missed sites; the sweep that found them all was

```
grep -rn '<changed entry point>' | grep -E '[0-9]\.[0-9]*f'
```

which caught `replay_scenenn.cpp:575` — the experiment binary, which no unit
test covers. Any future parameter removal adjacent to a `bool` should run the
same sweep.

**A near-miss worth recording.** The min/max range cull in `scovox_node.cpp` was
nested *inside* `if (P.range_decay_length > 0) { ... }`, alongside the dead decay
weight. Deleting that block wholesale — the obvious move — would have silently
removed the range filter from the RGB-D path. It is restructured to
`if (P.range_decay_length > 0 && (rng < P.min_range || rng > P.max_range)) continue;`,
preserving the exact pre-existing gate semantics, with a comment saying why it
keeps that shape.

**Collateral.** `verify_core`'s layout check hard-coded `DirVoxel 16 B`, which
is only true with `SCOVOX_TRACK_QMAX` off — but `qmax[]` ships **on**, since the
promoted candidate's `--evict-by-confidence` reads it, making the true width
20 B. The check had been reporting a false FAIL independently of this change.
`verify_core.cpp` now emits `beta_u16` / `track_qmax` / `track_nhit`, and
`verify_core_check.py` recomputes the expected width from them instead of
hard-coding it.

---

## Part 2 — Ships without proof (kept, by instruction)

Listed so that "we have tests" is never read as "this is covered". None of these
is a deletion candidate.

### 2.1 Compile-time axes no in-tree build exercises

| flag | default | status |
|---|---|---|
| ~~`SCOVOX_BETA_U16`~~ (+ `_SCALE`, 8) | **1** | **CLOSED 2026-09-04 — no longer an unexercised axis at all.** It is the shipped default, so `./dev.sh test` now compiles the `#if SCOVOX_BETA_U16` blocks as the primary path and `EXTRA=-DSCOVOX_BETA_U16=0` builds the float comparison arm. The polarity of this row is inverted from what it said before that date. See [storage_defaults_2026_09_04.md](storage_defaults_2026_09_04.md). |
| `SCOVOX_TRACK_NHIT` | **0 everywhere** | **CLOSED 2026-09-04.** The unit-test/sweep layout disagreement recorded here is resolved in favour of the core default: `build_rules.sh` no longer forces it on, so the shipped `DirVoxel` is 20 B in both builds and the 20 B `static_assert` at `dir_voxel.hpp:160` — previously short-circuited true by this very flag — is now live. The field, its guards and the `.slots` `has_nhit` byte remain; `build_e0.sh` still passes `-DSCOVOX_TRACK_NHIT=1` for analysis dumps. Both former readers (`VICTIM_MEAN`, `ADMIT_NORM`) were removed 2026-09-02, so nothing reads it. See [nhit_removal_2026_09_04.md](nhit_removal_2026_09_04.md) and [storage_defaults_2026_09_04.md](storage_defaults_2026_09_04.md). |
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
- **`band_perf`.** Referenced by `docs/archive/design/fix_plan_2026_08_26.md` and, until
  today, by the far-path tests as their execution proof. It does not exist in
  this repository. Part 3 replaces what it was standing in for.

### 2.3 Known defect, logged and not fixed

`integrateHitSplit` (`scovox_map_split.hpp:534`) calls `tsdf_.integrateRay`
with no `tsdf_enabled_` gate, while the fused walker gates on it at `:431`. So
`--tsdf-enabled 0` is honoured by the fused walker only, and any fused-vs-split
A/B run with it is measuring the flag on one arm and not the other. The fix is
one conjunct; it is not applied here because it changes shipped behaviour and
belongs in its own commit with its own re-run.

**Two ROS tests that do not assert their own claim** (pre-existing at HEAD
`9b1740d`, surfaced 2026-09-02 by the first *full* `scovox_mapping` rebuild —
incremental colcon builds do not re-emit warnings for unchanged translation
units, which is why they had gone unreported):

- `BetaUpdate.DecayPreservesRatio` (`test_beta_update.cpp:155`) computes
  `ratio_before` and never uses it. The test named for ratio preservation only
  asserts `ratio_after > 1.0f` — it would pass on any decay rule that keeps the
  voxel occupied, including one that destroyed the ratio.
- `BetaUpdate.NHitsConvergesToExpected` (`:104`) computes
  `expected_a_occ = 1 + 2N = 41` and never compares against it. The test named
  for convergence to an expected value asserts only `p_occ > 0.9` and
  `a_occ > 10` after 20 rays — a quarter of the value it derived, and a bound
  that 6 rays would already clear.

Both are left as-is: neither is touched by the `quality` removal, and tightening
an assertion is a behaviour question for its own commit. Logged so the coverage
they appear to provide is not counted twice.

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
with the counter assertions compiled in.

**Corrected 2026-09-02.** Two claims in the paragraph this replaces were
measured on an incremental build and did not survive re-measurement:

- *"Full suite 294/294"* — the suite is now **298 distinct cases, 0 failures**
  (294 at the time, minus the one test the `quality` removal deleted, plus the
  five new `BetaCountIdentity` tests). Colcon double-counts; the distinct figure
  comes from `count_tests.py`.
- *"The build emits exactly one warning ... a file this diff does not touch"* —
  both halves are wrong now. The `quality` removal **does** touch
  `tools/split_memory_demo.cpp` (two `integrate*` call sites), which is why the
  pre-existing `-Wcomment` at `:17` re-appeared; line 17 is byte-identical at
  HEAD `9b1740d`, so the warning itself is not new. And the build does not emit
  *exactly* one warning — two `-Wunused-but-set-variable` warnings (§2.3) are
  still latent in `test_beta_update.cpp` and simply were not recompiled.

**Process note, which is the durable finding here.** A warning count taken from
an incremental colcon build is not a measurement. Unchanged translation units
are not recompiled and their diagnostics are not re-emitted, so the count
reports "what changed since last time", not "what this tree warns about". Both
of the corrections above trace to that single mistake, as did the original
discovery of the two dead test variables. Only a clean build gives a
trustworthy count.
