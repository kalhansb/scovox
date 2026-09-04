# The Dirichlet cell stores the TOTAL, not the residual — 2026-09-04

`DirVoxel`'s offset-0 word changed from `float other` (the unattributed
remainder) to `float s_total` (all class evidence). `other()` is now derived as
`s_total − Σ cnt`. The struct is the same size, holds the same information, and
serialises to the same bytes; what moved is *which* of the two the arithmetic
accumulates into.

This document is the authority for that choice. `dir_voxel.hpp` (file header and
the `sparse_add_class` doc) and `sem_split_map.cpp::dirichletUpdate` all point
here by name.

Scope: this is a numerical-representation change inside the semantic cell. It
does not touch the eviction rule, the deposit model, the wire format, or the
occupancy cell. It is independent of the two other 2026-09-04 storage changes
(`SCOVOX_BETA_U16`, `SCOVOX_TRACK_NHIT`), which are documented in
`storage_defaults_2026_09_04.md` and `nhit_removal_2026_09_04.md`.

---

## 1. The old basis could not represent what the shipped pipeline computes

Under the promoted `--hit-share flat`, a look deposits a flat `kappa0` — the
endpoint's `class_share` no longer scales with `p_occ_post`. So for the shipped
build the total class evidence is

```
S  =  C·α₀  +  kappa0 · (number of looks)
```

with `kappa0 = 1`, which makes `S − C·α₀` an **integer**: the count of times the
voxel has been looked at. That is a quantity the cell should be able to hold
exactly. In the old basis it could not, because `S` was never stored — it was
recovered on each read as `other + Σ cnt`, the sum of three floats that had each
been accumulated separately from many small increments.

The failure is not bounded rounding noise. `other` holds ~88% of the mass and
takes the most adds, so each small deposit is added to a large running value and
the low bits are truncated away. The drift grows without bound relative to the
increment.

> **Correction, 2026-09-04 (review).** An earlier revision of this paragraph and
> of the `dir_voxel.hpp` header called the drift "a *systematic* one-sided
> undercount ... downward". That is wrong. Re-measured with signs on the
> soft-deposit pattern, the old basis errs **−0.003, −2.8, +44, −771** looks at
> 2k / 30k / 120k / 330k — it changes sign as the running total crosses binades.
> The magnitude claim below, which is what motivates the change, is unaffected;
> only the direction claim was false. Independently reproduced by an adversarial
> re-derivation (+50.6 at 120k on a slightly different deposit pattern).

Measured against exact rational arithmetic (`fractions.Fraction`), in units of
looks, at the look counts the SceneNN suite actually produces:

| looks | error, `other + Σcnt` | error, single accumulator | ratio |
|---:|---:|---:|---:|
| 100 | 0.000046 | 0.000001 | 76× |
| 2 000 | 0.009263 | 0.000015 | 632× |
| 30 000 | 0.384141 | 0.000625 | 615× |
| 90 000 | 35.319688 | 0.000625 | 56 512× |
| 330 000 | **2 236.64  (0.68%)** | 0.015 | 149 109× |

Those are not extrapolations past the working range. Look counts read off the
real flat-arm slot dumps: **p50 1 917, p90 29 046, p99 87 852, max 332 622**
(scene 243; scene 273 reaches 321 077). The busiest voxels in the suite sit in
the last row of that table.

At n = 330 000 the split is `other ≈ 290 480`, `Σcnt ≈ 37 390` — confirming
where the error comes from. A single accumulator adding `class_share` once per
look is exact to 0.015 looks over the whole range: five orders of magnitude
better at the top end, and never worse anywhere.

`s_total` is not a *more precise* float. It is the same 32 bits reached by far
fewer additions.

## 2. The mass invariant stops being a discipline and becomes arithmetic

`sparse_add_class` used to take a `float* other` and write to it on three of its
five branches (sentinel, evict, drop). The invariant `Δ(other + Σcnt) == inc`
was therefore a property that every branch had to actively maintain, and every
branch carried a comment arguing that it did. Adding a sixth branch, or getting
one of the five wrong, silently created or destroyed mass.

The `float* other` parameter is now gone. The function only **attributes**:
whatever it does not put into `cnt[]` remains in the derived `other()` by
subtraction. There is no branch that *can* leak mass, because no branch touches
the total.

The equivalence is branch-by-branch, and is exact in the sense that each branch
reproduces the old `other` **to the precision the old basis itself had** — two
rows carry caveats, noted under the table.

| branch | Δ`Σcnt` | derived Δ`other` | old explicit write |
|---|---|---|---|
| sentinel (`c == 0xFFFF`) | 0 | `+inc` | `*other += inc` |
| match | `+inc` | 0 | none |
| fill empty | `+inc` | 0 | none |
| evict | `+inc − evicted_evidence` | `+evicted_evidence` | `*other += evicted_evidence` |
| drop | 0 | `+inc` | `*other += inc` |

Every row reproduces the old semantics. Note the evict row in particular: the
victim's accumulated evidence falls back into `other()` automatically, because
the newcomer gets a fresh `α₀ + inc` counter and the total did not move. That
used to require a second write kept in step with the first.

Two rows are exact only up to a bounded re-attribution, in both bases:

- **fill empty.** The slot is seated at `α₀ + inc`, but an empty slot is only
  *exactly* `α₀` if no saturation rescale has scaled it to `k·α₀`. When it has,
  seating re-attributes `α₀·(1 − k)`.
- **evict.** `raw_evicted = cnt[min] − α₀` can be negative if a rescale eroded
  the victim below `α₀`; the code returns 0 rather than negative evidence, again
  re-attributing `α₀·(1 − k)`.

Both are bounded by `α₀` per event and are reachable only with the saturation
cap on. Under the OLD basis that excess was *created from nothing*; under
`s_total` it correctly comes out of the derived `other()`.

`dirichletUpdate` now makes the deposit on its own first line, before any
branching, and its five `d->other +=` sites are gone. That includes the two
*residual* lines — `class_share − deposited` under THRESH and
`class_share · (1 − covered)` under SOFT — which were the only two places where
the remainder was computed by one route and the deposits by another. That
mismatch is where the SOFT path's `covered` rounding used to leak.

The caller contract is one line: **add the deposit to the total exactly once,
whatever branch is taken.** Three callers honour it — `dirichletUpdate`
(`+= class_share`), `naiveUpdate` and `majorityVoteUpdate` (`+= 1.0f`).

## 3. `s_class()` is O(1)

It is now a field read rather than a `K_TOP`-length sum. The live library
call sites are the share gate (`scovox_node.cpp:2166,2167`) and the saturation
gate (`sem_split_map.cpp:1119`); the only site that actually *divides* by it is
the replay tool's readout (`replay_scenenn.cpp:718`). `dominantClass` compares
`cnt[i] − α₀` and never divides. So this is the smallest of the three reasons —
it was not on its own sufficient to justify the change, and an earlier draft of
this section overstated it as "every classification divides by it".

## 4. What this costs

`other()` becomes the derived quantity and inherits the accumulated error of
`Σ cnt` — which the table in §1 shows is the *much* smaller of the two. The
error did not disappear; it moved onto the term that can carry it:

- `other` is by definition the bucket of mass that could not be attributed;
- it is read only as `a_unk` and by the prior gate (`isPriorDir`);
- it is never a denominator.

`other()` is deliberately **not clamped at zero**. A negative *far exceeding
`ulp(s_total)`* means a caller added less to `s_total` than it attributed to
`cnt[]` — a pairing bug — and clamping would hide it. (A negative within an ulp
is expected and is not a bug; see §1's correction and `other()`'s own doc.)

There are **four** `a_unk` sites, not one, and only two of them clamp:

| site | clamps? |
|---|---|
| `dscovox_consensus.hpp:95` | yes, `std::max(0.f, … − other_prior)` |
| `dscovox_consensus.hpp:143` | yes |
| `scovox_node.cpp:2079` | **no**, and does not subtract `other_prior` |
| `scovox_node.cpp:2851` | **no**, and does not subtract `other_prior` |

The two unclamped sites are the exact pattern `dscovox_consensus.hpp:89`
documents as wrong. They predate this change and this change does not make them
worse — a sub-ulp negative is negligible against `a_unk`'s scale — but they are
the reason "the one consumer that must not see a negative" was false as written.

The full reader set is also wider than `a_unk` and `isPriorDir`: `dominantClass`,
`mergeDir`, the serialiser, `decayDir`, `applyDirSaturation` and the `at_prior`
gate (`scovox_node.cpp:2394`) all read the residual or the total.

One measured instance of the new cost is in the test suite:
`FrameMergeAcceptsNumClassesEqualKTop` expects a derived `other()` of exactly
zero and gets **−9.3e-09**. `EXPECT_FLOAT_EQ` degenerates to a near-exact test
at zero, so that assertion was relaxed to `EXPECT_NEAR(…, 1e-6f)`. The value is
recorded here so it is not later mistaken for a regression.

## 5. Why `float s_total` and not an integer look count `n`

`S − C·α₀` is an integer under `flat`, so a `uint32 n` would be exact by
construction and the same width. It was rejected on interface grounds:
reconstructing `s_class()` from `n` needs `(num_classes, α₀, kappa0)` at every
reader. `s_class()` takes no arguments today. The threading cost is smaller than
this section first claimed: `binary_serializer.hpp`, `dscovox_consensus.hpp`,
`consensus_merge.hpp` and `scovox_node.cpp` already have `num_classes` and
`alpha_0` in hand, so only `kappa0` would actually need threading, and
`scovox_node.cpp` would need nothing. `s_total` is *numerically equal* to what the old
basis computed, keeps the 4 B field and the 20 B struct, needs no parameter
threading, and — unlike `n` — stays correct for any `hit_share` mode, including
the non-flat arms that remain buildable.

## 6. The two rebuild sites, and why they are not `s_total *= k`

Two functions rescale a voxel rather than depositing into it. Both must rebuild
the total instead of scaling it, for different reasons.

**`decayDir`** decays `other` toward its prior at `rate`. It reads the residual,
shrinks it, and re-adds `Σcnt` *after* the slot loop has decayed and possibly
released each slot. The sum runs over **all** `K_TOP` slots, not just filled
ones: a released slot still carries an α₀ placeholder, and a prior saturation
rescale can leave an empty slot holding something other than exactly α₀.

**`applyDirSaturation`** cannot use `s_total *= k`. The α₀ floor below the
rescale puts mass *back* into `cnt[]` with no matching change to the residual,
so a floored slot would eat into the derived `other()` and could drive it
negative. Scaling the residual separately (`other_scaled = other() * k`) and
rebuilding from `other_scaled + Σcnt_after_floor` preserves the function's
pre-existing behaviour exactly — including the pre-existing quirk that a fired
floor leaves the voxel marginally above `cap`.

## 7. The wire format is unchanged; the round trip is no longer bit-exact

`binary_serializer.hpp` writes the **derived** `other()` (quantised or raw,
unchanged) and reconstructs `s_total` with `set_other()` after reading `cnt[]`.
The on-wire record is still `other` then `cnt[]`, in the same order and the same
widths, so a new sender stays compatible with an old receiver in both
directions. The format is unchanged.

**The reconstruction is not.** An earlier revision of this section claimed "no
`.slots` dump, ROS message, or frame consumer sees any difference". That
overclaims. `other` and `cnt[]` round-trip bit-exactly, but `s_total` does not:
the sender emits `fl(fl(s − c0) − c1)` and the receiver recomputes
`fl(fl(o + c0) + c1)`. Over 400 000 random triples these differ in **71 087
(18%)** of cases, worst 0.125 absolute at `s_total ≈ 1.2e6` — at most 1 ulp, and
always in the last place. Federated fusion folds repeatedly, so the perturbation
is per-fold, not once.

Two consequences worth holding on to:

* The **quantised** branch, not the raw-float branch, is the ROS default —
  `scovox_node.cpp:369` reads `evidence_saturation` with a default of 1000, so
  `quant_step = 1000/65025 ≈ 0.0154`. The raw-float branch is what the offline
  replay and the core's own `Params` default (`evidence_saturation = 0.0f`)
  select. The 18% figure above is the raw-float branch.
* Quantisation clamps negatives, so a voxel whose `other()` has gone slightly
  negative (§ the `other()` note in `dir_voxel.hpp`) is silently repaired in
  transit on the deployed path but preserved on the replay path.

## 8. HAZARD: writing `cnt[]` directly now moves `other()`

This is the one place the change is not transparent to a reader of existing
code. A raw `cnt[i] = …` no longer leaves `other()` alone — it moves it down by
exactly what was written, because the total did not change.

Most live `cnt[]` writes are correct without help — `naiveUpdate`, `decayDir`,
`applyDirSaturation`, `defaultDirVoxel`, `mergeDir` and the deserialiser all
write `cnt[]` directly and all pair or rebuild the total themselves. But "live
code never hits this", as this section first read, was false, and the word
*never* is what let a real bug through: `scovox_node.cpp`'s `wireDir` wrote
`cnt[0] = α₀ + share_binarize_evidence_` with no matching `s_total` add, driving
the derived `other()` to `−19.88` at the shipped default `E = 20.0`. It is fixed
(paired `+=` on both), and it was dormant — `share_gate_mode` defaults to
`"significance"` and the scoring binary never references it — so no published
number moves. The hazard bites hardest in **test fixtures and tools that
fabricate an observation history** by assigning slots directly. Such a
fixture must finish with `set_other()` (or the `seat()` helper the test files
now define, which restores "slots exactly as written, OTHER at its class
prior").

Three merge tests failed in exactly this way when the basis changed, and one of
them was `ConsensusMerge.DirMassConservation` — the mass-conservation test
itself. The failures were in the fixtures, not the code: `a.cnt[0] = kAlpha + 5`
drained `other()` to −8.88 instead of leaving it at the 0.12 prior. All 30-odd
direct-`cnt[]` fixture sites across four test files were swept, not just the
three that happened to fail.

`set_other()` must be called **after** `cnt[]` is final, for the same reason.

`inherit_invariant.cpp` was rewritten for the same reason, and its assertion
changed: `Δ(other + Σcnt) == inc` would now pass *by construction* whatever the
eviction rule did, so it tests nothing. The property that can still fail is
**over-attribution** — `sparse_add_class` putting more into `cnt[]` than was
deposited — so that is what it now measures.

## 9. Acceptance

- `./dev.sh test` — **181/182**. The one failure is
  `ScovoxMapSplitFarCarve.FarCarveBitIdenticalToFullWalk`, which is pre-existing
  *relative to this change* — it comes from the uncommitted far-fast-paths work
  in the same tree, so `git stash && ./dev.sh test` gives a clean pass, and it is
  not a defect inherited from an earlier commit. It is unrelated on its merits
  too: it diverges only in the **Beta** grid (flushCarveFrame 2886 vs 2887, beta
  touched-list 2950 vs 2951); the Dir comparison never fires.
- `./dev.sh test` builds only `src/scovox_core`. The **122** `scovox_mapping`
  cases — including `test_consensus.cpp`, `test_heartbeat.cpp` and
  `test_dirichlet_update.cpp` — need `./dev.sh ros-test` and are **not yet run**
  for this change.
- `./dev.sh verify` — **13/13**, reporting `BetaVoxel 4 B / DirVoxel 20 B at
  K_TOP=2 [beta_u16,qmax]`.
- Wire format: unchanged by construction (§7), not by measurement.
- **Still pending: the 8-scene re-score.** Unlike the `BETA_U16` and `nhit`
  changes, this one *moves numbers* — it changes the floating-point value of
  `s_class()` and hence of every `p_c`. It cannot be accepted on a byte-compare
  and needs a full 8-scene run graded against the 0.001 mIoU MATERIAL threshold.
  The expectation is a null result at that threshold in *both* directions: the
  drift removed is real but sub-mIoU-relevant on most voxels.

## 10. What is not claimed

- Not a memory saving. Same 4 B word, same 20 B struct.
- Not a speed claim. `s_class()` gets cheaper and `other()` gets more expensive;
  the two roughly trade, and no timing was run.
- Not a claim that the old drift was corrupting results. It was 0.68% of the
  total on the busiest voxels in the suite and far less on typical ones; §9's
  re-score is what would establish whether it mattered, and it has not run.
- Not a change to the deposit model, the eviction rule, or `flat`.

## 11. Statements this document overrides

1. **`dir_voxel.hpp`'s layout block.** At HEAD its line 21 reads
   `///   offset 0:   other  (float, 4 B)  — lumped OTHER / evicted mass`.
   Offset 0 is now `s_total`; `other` is derived and is not stored anywhere.
   (An earlier draft of this item quoted that line as "lumped α for the
   C − K_TOP untracked classes", which is not what HEAD says.)

2. **`dir_voxel.hpp`'s mass-conservation paragraph**, which stated the invariant
   as a property of coordinated writes — "every increment lands somewhere …
   never lost", defended per branch. The invariant is now structural: one write
   to the total, and `other ≡ s_total − Σ cnt` by definition.

3. **The per-branch comments in `sparse_add_class`** arguing that each `*other
   +=` kept the books balanced. There are no such writes; the branches argue
   about *attribution* only.

4. **`sparse_add_class`'s signature** as documented anywhere else: the
   `float* other` parameter no longer exists. Callers pass 5–10 arguments with
   no `other` among them.

5. **`e0_counters.hpp:57`**, which described the `dirichletUpdate` early returns
   as "the `d->other += class_share` early returns". They are now simply the
   no-signal / all-zero-softmax early returns; the deposit has already been made
   before either is reached.

6. **Any claim that `s_class()` is a sum over slots.** It is a field read.

7. **The residual lines in `dirichletUpdate`'s THRESH and SOFT paths** as part
   of the algorithm. They were identities of the stored total and are deleted;
   the uncovered mass is unattributed rather than explicitly re-added.
