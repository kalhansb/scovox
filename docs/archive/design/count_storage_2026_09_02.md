# Counts, and the lattice they need — 2026-09-02

Written after the `quality` removal, which is what makes any of this true.

> **Status update 2026-09-04.** `SCOVOX_BETA_U16` was an opt-in when this was
> written; it is now the **shipped default**, on the strength of the byte-identity
> measurement in the last section. Wherever this document treats uint16 storage
> as the hypothetical arm and float as the baseline, the polarity is now
> reversed. See [storage_defaults_2026_09_04.md](storage_defaults_2026_09_04.md).

## The identity

Every admitted observation now contributes a fixed increment. There is no
per-observation confidence factor, so the accumulated Beta parameters are a
pure function of two integer counts:

```
a_occ  = kBetaOccPrior  + w_occ  * n_hit
a_free = kBetaFreePrior + w_free * n_miss
p_occ  = a_occ / (a_occ + a_free)
```

The map stores accumulated evidence, not the counts themselves, but under this
identity the two carry the same information: given the weights, `n_hit` is
recoverable from `a_occ` by one subtraction and one division. That is asserted
directly rather than assumed — `BetaCountIdentity.HitCountIsRecoverableFromAOcc`
in `test_sem_split_map.cpp`.

Before the removal the identity did not hold. `a_occ += w_occ * quality` made
the parameter a function of the ray's range and incidence angle as well as its
existence, and no count could reconstruct it.

## A correction to an earlier claim

An intermediate version of this argument held that storing raw integer counts
would avoid a quantisation error that `SCOVOX_BETA_U16` suffers. **That is
wrong**, and the mistake is worth recording because it inverts the conclusion.

At the shipped scale of 8, the weights are already whole numbers of lattice
units:

| value | eighths | exact? |
|---|---|---|
| `kBetaOccPrior` / `kBetaFreePrior` = 1.0 | 8 | yes |
| `w_free` = 1.0 (candidate) | 8 | yes |
| `w_occ` = 1.5 (candidate) | 12 | yes |
| `w_free` = 0.5 (core-test default) | 4 | yes |

So `BETA_U16` **is** integer-count storage, denominated in eighths, and it
accumulates these weights with zero error. A separate raw-count representation
would buy no precision at all.

What it would buy is headroom, and only that:

| scheme | per voxel | ceiling |
|---|---|---|
| `BETA_U16` (2 × uint16 of 1/8) | 4 B | 8191.875 evidence ≈ **5 460 hits** at `w_occ` 1.5 |
| raw counts (2 × uint16 of 1) | 4 B | **65 535** observations |
| float32 | 8 B | unbounded in practice |

`applyBetaSaturation` halves both parameters at 90 % of the ceiling, which
preserves `p_occ` exactly but *does* end the count identity for that voxel —
after a halving, `n_hit` is no longer recoverable.

> **Answered, 2026-09-04.** This paragraph originally closed "whether ~4 900
> hits on a single voxel is reachable on these scenes is a measurement nobody
> has taken". It has been taken, by this same document's final section, and the
> answer is **no**. Byte identity between the float and uint16 builds is only
> possible if `applyBetaSaturation` never fired: one halving would have moved
> the uint16 voxel and not the float one. The ceiling is a documented limit and
> is not reached on SceneNN at this resolution and frame count. The two sections
> contradicted each other for two days; this one was the stale half.

## The requirement: weights on the lattice

The identity holds only while every increment is exactly representable. This is
**not** a fixed-point-only concern, which was the second thing the first draft
got wrong: binary floating point cannot represent `1.3` either, so
`a_occ += 1.3f` drifts off `prior + 1.3·N` by accumulated rounding just as
surely as fixed point rounds it. A test asserting the identity with `w = 1.3`
fails under *float* storage.

So the lattice is defined **identically in both storage modes** —
`kBetaLatticeStep`, whole eighths — and `SemSplitMap`'s parameter sanitiser
snaps `w_occ` and `w_free` onto it at construction.

Quantising in both modes rather than only under fixed point is the deliberate
part. If the weights were snapped only when `SCOVOX_BETA_U16` was on, then
turning the flag on would silently move the model, and any A/B across the flag
would be comparing two *models* rather than two *storage layouts*. Uniform
quantisation makes the flag a pure storage choice, which is the only way its
measurement means anything.

Whole eighths are dyadic rationals: exact in float, exact on the fixed-point
lattice. One rule covers both, and it is the tightest requirement either
storage imposes.

**In practice this changes nothing that is configured today.** Every shipped
and candidate weight is already a whole eighth. The rule bounds what a future
config can silently do.

## `w_occ : w_free` — what the data says

The ratio is the one free parameter in the identity, so it belongs here.
[RESULTS.md](../../../scovox_slot_rules/RESULTS.md) E7 Family 1, all 8 scenes,
pre-registered m = 4:

| ratio | union mIoU | vs shipped 6:1 |
|---|---|---|
| 12:1 | 0.27554 | −0.00663 |
| 6:1 (shipped) | 0.28217 | — |
| 3:1 | 0.28825 | +0.00608 |
| 2:1 | 0.29116 | +0.00899 |
| **1.5:1** | **0.29263** | **+0.01046** |

`R* = {1.5:1}` is a singleton, unanimous across all 8 leave-one-scene-out
folds, with measured LOSO optimism exactly 0.000000. **2:1 misses the band by
0.001473**, above the 0.001 MATERIAL threshold — so 2:1 is not the best swept
level and is not within the promotion band of the one that is.

The response is monotone toward more free weight across the whole swept range,
and 1.5:1 is `best-at-edge` — the most extreme level tested. E7's
pre-registered boundary rule, fixed before the deciding cell ran, requires this
be reported as a bound rather than an optimum: **the true optimum lies at or
beyond 1.5:1 and this sweep did not locate it.** It is therefore not 2:1, but
where it actually is remains unmeasured.

Both `1.5` and `1.0` are whole eighths, so the promoted ratio satisfies the
lattice rule with room to spare. So does any ratio built from halves, quarters
or eighths — `1:1`, `1.25:1`, `1.5:1`, `2:1`. A ratio like `1.3:1` does not,
and would be snapped to `1.25:1` at construction rather than accumulating
crooked.

## What is *not* expressible as a count

Four operations write the Beta parameters without adding a fixed weight. They
bound how far the count model can be taken.

1. **Transient decay** — `SemSplitMap::decayTransient`
   (`sem_split_map.cpp:957`) contracts geometrically toward the prior,
   `x = target + (x − target)·rate`. Irreducibly fractional. It touches only
   `transient_beta_grid_`; the persistent grid never decays, so a count-backed
   persistent grid beside a float transient grid is coherent — at the cost of
   two storage types.
2. **`evidence_saturation`** — `a *= cap/s`. RESULTS E1 selected `C* = NONE`,
   so this is dormant in every promoted config, but it is not removed.
3. **The `BETA_U16` high-water halving** — `a *= 0.5f` at 90 % of the ceiling.
   Exists solely to work around the fixed-point ceiling; a raw-count scheme
   would not need it.
4. **Per-source weights** — `HitWeights` lets each source carry its own
   `w_occ` / `w_free`. A voxel observed by two sources at different weights
   cannot be reconstructed from one count. **This is the real architectural
   constraint on the count model.** It holds today only because this suite runs
   a single RGB-D source; a multi-source deployment needs either one count per
   source or an explicit single-weight restriction.

The consensus merge is *not* on this list: `a.a_occ + b.a_occ − prior`
(`consensus_merge.hpp:98`) is already count arithmetic — the counts add and the
double-counted prior comes back off.

## What was implemented

- `kBetaLatticeStep`, `beta_lattice_snap`, `beta_lattice_exact`,
  `beta_max_increments` in `beta_voxel.hpp`, with the identity documented at
  the head of the block.
- `sanitise()` in `sem_split_map.cpp` snaps `w_occ` / `w_free` onto the lattice
  at construction.
- Five tests in `test_sem_split_map.cpp` under `BetaCountIdentity`: the shipped
  weights are on the lattice; `a_occ` and `a_free` equal `prior + N·w` exactly
  after N observations; `n_hit` is recoverable; and an off-lattice weight is
  snapped rather than accumulated crooked.

Verified 176/176 under float storage and 179/179 under
`-DSCOVOX_BETA_U16=1` — the identity tests pass in both, which is the claim
the uniform lattice exists to support.

## Measured: `SCOVOX_BETA_U16` is exactly neutral

The identity predicts something falsifiable and sharp. If every weight is a
whole eighth, then uint16-of-eighths accumulates the *same number* float does —
not a near number. So the two builds should produce not similar maps but
identical ones.

A/B on SceneNN, promoted candidate config
(`--evict-by-confidence --sem-band 0.10 --w-occ 1.5 --w-free 1.0
--dump-below-gate-as-unknown --dump-label-gate 0.5`, inherit i0), same frame
order, `base` = float storage, `u16` = `-DSCOVOX_BETA_U16=1`:

| scene | dump | metric fields differing |
|---|---|---|
| 014 | byte-identical (83 066 756 B) | 0 / 254 |
| 015 | byte-identical (42 752 660 B) | 0 / 254 |
| 016 | byte-identical (31 201 196 B) | 0 / 254 |
| 057 | byte-identical (37 453 436 B) | 0 / 254 |
| 061 | byte-identical (37 595 444 B) | 0 / 254 |
| 243 | byte-identical (72 615 836 B) | 0 / 254 |

**1 524 metric fields compared, 0 differ.** Not within the 0.001 MATERIAL
threshold — identical, which is a stronger and more useful statement: there is
no accuracy question to trade off against the memory saving, because there is
no accuracy difference to measure.

Two things follow.

- The saving is free at these weights. `BetaVoxel` goes 8 B → 4 B; on the
  ~19 M-voxel maps this suite builds that is ~72.6 MB off the persistent grid,
  for byte-identical output.
- The ceiling was never reached on any of these scenes. Had
  `applyBetaSaturation` fired even once, that voxel's parameters would have
  halved under u16 and not under float, and the dumps would have diverged.
  Byte identity is therefore also evidence that ~5 460 hits on a single voxel
  is not reachable on SceneNN at this resolution and frame count — the one
  open question the headroom table above could not answer by inspection.

The result is contingent on the weights staying on the lattice, which is
exactly what `sanitise()` now guarantees. A config with `w_occ = 1.3` would
break the identity, and the two builds would then differ — which is the reason
the snap happens at construction rather than being left to the operator.
