# The shipped storage layout — 2026-09-04

**Authority.** This document is the single source of truth for the two voxel
storage widths the mapper ships with. Where an earlier document states a
different default, **this one overrides it**; §6 lists every such statement by
file and line so none of them is left to be rediscovered as a contradiction.

| cell | was | **now ships** | flag |
|---|---|---|---|
| `BetaVoxel` (occupancy) | 8 B, two `float` | **4 B, two `uint16` of ⅛** | `SCOVOX_BETA_U16` default `0` → **`1`** |
| `DirVoxel` (semantics, `K_TOP=2`) | 24 B | **20 B** | `SCOVOX_TRACK_NHIT` forced `1` in `build_rules.sh` → **left at the core default `0`** |

Both are **byte-identity** changes: the scored dump is bit-for-bit what it was.
Neither is a mIoU trade, so neither needs a mIoU grading — the acceptance gate
is stronger than the 0.001 MATERIAL threshold, not weaker.

---

## 1. Why the occupancy cell, and why first

The two changes are not equal, and the ordering was not a preference. A
`BetaVoxel` is allocated for **every voxel a ray touches**; a `DirVoxel` only
for a voxel that took a semantic deposit. Census over all 8 SceneNN scenes at
the shipped `--hit-share flat`, counting a Dirichlet cell as one with at least
one seated class:

| scene | Beta cells | Dir cells | Dir as % | Beta @ 8 B | Dir @ 24 B |
|---|---|---|---|---|---|
| 011 | 3 239 817 | 148 248 | 4.58 % | 24.7 MiB | 3.39 MiB |
| 014 | 3 461 114 | 86 328 | 2.49 % | 26.4 MiB | 1.98 MiB |
| 015 | 1 781 360 | 115 331 | 6.47 % | 13.6 MiB | 2.64 MiB |
| 016 | 1 300 049 | 52 266 | 4.02 % | 9.9 MiB | 1.20 MiB |
| 057 | 1 560 559 | 101 503 | 6.50 % | 11.9 MiB | 2.32 MiB |
| 061 | 1 566 476 | 103 751 | 6.62 % | 12.0 MiB | 2.37 MiB |
| 243 | 3 025 659 | 90 797 | 3.00 % | 23.1 MiB | 2.08 MiB |
| 273 | 3 181 447 | 88 687 | 2.79 % | 24.3 MiB | 2.03 MiB |
| **total** | **19 116 481** | **786 911** | **4.12 %** | **145.8 MiB** | **18.0 MiB** |

**One byte off `BetaVoxel` is worth 24× the same byte off `DirVoxel`.** Layout
work that starts at the semantic cell is optimising 4 % of the map. That is the
design rule this document registers, and it is why `SCOVOX_BETA_U16` is now a
default rather than an opt-in switch nobody flipped.

Measured savings, same census:

- **Beta 8 B → 4 B: 72.9 MiB over the suite, 9.1 MiB/scene.**
- Dir 24 B → 20 B: 3.0 MiB over the suite, 0.4 MiB/scene.

## 2. `SCOVOX_BETA_U16 = 1` — free, not a trade

The saving costs no accuracy, and that is a falsifiable claim rather than a
tolerance argument. Every weight this repo ships is a whole eighth
(`kBetaOccPrior` = `kBetaFreePrior` = 1.0, `w_occ` = 1.5, `w_free` = 1.0), so
uint16-of-eighths accumulates *the same number* float does — not a near number.
`sanitise()` snaps `w_occ`/`w_free` onto `kBetaLatticeStep` at construction in
**both** storage modes, so the flag is a pure storage choice and a future
off-lattice config cannot silently break it.

The prediction is that the two builds produce not similar maps but identical
ones, and that is what was measured — see
[`count_storage_2026_09_02.md`](count_storage_2026_09_02.md) §"Measured":
**6 scenes byte-identical, 1 524 metric fields compared, 0 differing.**

Two further properties make this safe to promote:

- **Saturation never fires.** Byte identity is itself the proof: had
  `applyBetaSaturation` halved even one voxel, that voxel would have halved
  under uint16 and not under float, and the dumps would have diverged. The
  ~5 460-hit ceiling is not reachable on SceneNN at this resolution and frame
  count. This **answers** the open question left at
  `count_storage_2026_09_02.md` §"A correction to an earlier claim" ("a
  measurement nobody has taken") — that paragraph and the §"Measured" section
  of the same document contradicted each other, and the latter is correct.
- **Independent of the semantic deposit model.** Occupancy IoU is bit-identical
  across the `flat`/`hard`/`counts` arms on all 8 scenes, so the result carries
  to any `--hit-share` setting.

**`--batch-hits 1` is a hard prerequisite** and is the shipped default.
Un-batched, `a_occ` counts depth *pixels*: the largest value measured on this
suite is 756 508, 23× past the widest `uint16` range any usable scale reaches.
Batched, a voxel takes at most one deposit per scan and the ceiling is
`prior + w_occ × frames` ≈ 1 951. A build that sets `SCOVOX_BETA_U16=1` with
`--batch-hits 0` will saturate and is not a supported configuration.

**Building the float arm for comparison:** `EXTRA=-DSCOVOX_BETA_U16=0`. The
polarity is inverted from what every pre-2026-09-04 document says.

## 3. `SCOVOX_TRACK_NHIT` left at 0 — removing residue, not a feature

`nhit[K_TOP]` had exactly two readers, `SCOVOX_VICTIM_MEAN` (mean-strength
victim selection, `cnt/nhit`) and `SCOVOX_ADMIT_NORM` (nhit-normalised
admission). **Both were removed on 2026-09-02** as swept alternatives that lost.
Every surviving reference is a write; no mapping or scoring decision reads the
field. Full site table and the reasoning:
[`nhit_removal_2026_09_04.md`](nhit_removal_2026_09_04.md).

The build configuration had not caught up: `build_rules.sh` forced
`-DSCOVOX_TRACK_NHIT=1` into `TRACKS` for every build, on the stated grounds
that `dir_voxel.hpp` static-asserts the mean/norm rules will not compile without
it. **That reason expired with those rules.** The field was being written on
every deposit and read by nothing.

This is a **build-configuration correction to match the core default**, not a
code deletion. The field, its `#if` guards, and the `.slots` `has_nhit` header
byte all remain, so an analysis build is one `-D` away — and `build_e0.sh`
already passes it explicitly.

**The contrast with `qmax`, which stays.** The two fields look symmetric — both
2 B/slot behind a `SCOVOX_TRACK_*` flag — and are not. E9's single-variable arm
(`cand` vs `abl_admit`: the *same binary*, differing only by
`--evict-by-confidence`) is **+0.0462 intersection mIoU and +0.0196 union, 8/8
scenes, at bit-identical occupancy**. `qmax` earns its 2 B; `nhit` is read by
nothing.

## 4. Acceptance

| gate | status |
|---|---|
| `./dev.sh test`, u16 default on | **180/181** |
| same, control build `-DSCOVOX_BETA_U16=0` | **180/181, identical single failure** |
| byte-identity of scored dumps, 8 scenes, both flags | *pending — §5* |
| `./dev.sh verify` | *pending* |

The single failure is `ScovoxMapSplitFarCarve.FarCarveBitIdenticalToFullWalk`,
a **pre-existing** far-carve fast-path divergence from uncommitted
`--far-fast-paths` work. Running the float control build and getting the
*identical* single failure is what establishes it is not a regression from
either promotion.

A `static_assert` re-arms as a side effect. `dir_voxel.hpp` —

```cpp
static_assert(!SCOVOX_TRACK_QMAX || SCOVOX_TRACK_NHIT || K_TOP != 2
              || sizeof(DirVoxel) == 20, ...)
```

— was short-circuited true by `SCOVOX_TRACK_NHIT`, i.e. disabled in the very
build it describes. With the flag off it becomes live and pins the 20 B layout.
The neighbouring 16 B assert is still named the *"Production K_TOP=2
invariant"* while describing a build that is not shipped; renaming it is logged,
not done here.

## 5. What is not claimed

- **No speed improvement.** Neither change is justified on throughput. The ray
  walk dominates: E-W10 counted ~5.86 G voxel steps against ~83 M hits on a
  single scene. Any timing claim must come from arms interleaved inside one
  session (the standing cross-session drift rule), and none has been run.
- **No mIoU improvement.** Byte identity means there is nothing to grade.
- **The `nhit` saving is small** — 0.4 MiB/scene, 4 % of the Beta saving. It is
  justified by the field being unread, not by its size.

## 6. Statements this document overrides

Every one of these was correct when written and is wrong now. Corrected in
place where the document describes *current* state; annotated in place where the
document is a dated record of a past measurement, because rewriting a lab
notebook to match today's defaults would destroy the provenance the notebook
exists for.

| file:line | said | disposition |
|---|---|---|
| `removed_and_untested_2026_09_02.md:275` | `BETA_U16` — "no *default* build sets it to 1, so `./dev.sh test` does not compile the `#if SCOVOX_BETA_U16` blocks; reachable via `EXTRA=-DSCOVOX_BETA_U16=1`" | **corrected** — inverted on both counts |
| `removed_and_untested_2026_09_02.md:276` | `TRACK_NHIT` — "0 in core, **1** in `build_rules.sh`… the unit-test build and the sweep build disagree on the shipped `DirVoxel` layout" | **corrected** — the disagreement is closed in favour of the core default |
| `FEATURES.md:121` | `TRACK_NHIT` in-matrix **1**, "Required by VICTIM_MEAN and ADMIT_NORM (both static_asserted)" | **corrected** — both readers removed 2026-09-02 |
| `best_method.md:16` | `SCOVOX_TRACK_NHIT` `1` in the shipped build table | **corrected** — the shipped build no longer sets it |
| `best_method.md:108` | "`SCOVOX_TRACK_NHIT=1` … is analysis instrumentation, not part of the method" | **corrected** — still true in spirit; the build no longer carries it |
| `scripts/slots_io.py:22` | "Every build in every MANIFEST carries `-DSCOVOX_TRACK_NHIT=1`" | **corrected** — the reader already branches on `has_nhit`; only the comment was stale |
| `scripts/build_rules.sh` (`TRACKS`) | forced `-DSCOVOX_TRACK_NHIT=1`, justified by the mean/norm `static_assert`s | **corrected** — flag dropped, rationale replaced |
| `count_storage_2026_09_02.md` §"A correction…" | "Whether ~4 900 hits on a single voxel is reachable on these scenes is a measurement nobody has taken" | **annotated** — answered by the same document's §"Measured"; see §2 above |
| `RESULTS.md:1295` | "Every build in every `MANIFEST` carries `-DSCOVOX_TRACK_NHIT=1`" | **annotated** — true of every MANIFEST written before 2026-09-04 |
| `RESULTS.md:2552`, `NEW_EXPERIMENTS_PLAN.md:1276` | `TRACK_QMAX`/`TRACK_NHIT` "**held on** in every build" | **annotated** — `TRACK_QMAX` still held on; `TRACK_NHIT` no longer |
| `RESULTS.md:3236` | "Enable it per build with `EXTRA=-DSCOVOX_BETA_U16=1`" | **annotated** — polarity inverted; `=0` now builds the comparison arm |
| `RESULTS.md:3298` | "Kept, and why: … `TRACK_NHIT` … survives its two former readers" | **annotated** — the *field* survives; the build flag does not |
| `RESULTS.md:3437` | "the `SCOVOX_BETA_U16` blocks no in-tree build compiles, the `TRACK_NHIT` layout disagreement…" | **annotated** — both conditions resolved |

`scovox_cleanup.md:139/154/383` proposed the `TRACK_NHIT=0` deployment build as
future work. It is now done; those entries are the origin of the change, not a
contradiction of it.

`archive/FINDINGS.md` is not corrected. It is archived.
