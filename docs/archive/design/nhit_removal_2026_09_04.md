# Dropping `nhit` from the production build — 2026-09-04

**Decision (TL;DR).** The production build stops passing
`-DSCOVOX_TRACK_NHIT=1`. `DirVoxel` goes from **24 B to 20 B** at `K_TOP=2`.
**No source changes anywhere** — every `nhit` site is already `#if`-guarded and
the flag already defaults to `0`. The field, the guards, and the `.slots`
`has_nhit` header byte all stay, so the analysis build is one `-D` away.

Acceptance is **byte-identity of the scored dump**, and unusually that is
provable before running anything: no code reads the field, and a value that is
never read cannot change what is written.

This is not the removal of a feature. It is the removal of **residue** — the
field outlived the two readers that were deleted on 2026-09-02, and the
production build has been carrying it since.

---

## 1. What `nhit` is

`nhit[K_TOP]` in [`dir_voxel.hpp`](../../src/scovox_core/include/scovox/dir_voxel.hpp)
is a per-slot count of deposits absorbed since the slot was last filled. It
saturates at 65535 rather than wrapping, and resets to 1 on fill and on
eviction, so it always counts deposits of the class currently in `cls[i]` and
never a predecessor's.

It is distinct from `cnt[i]`, which is a Dirichlet α — accumulated soft-label
mass, not a tally. `cnt` and `nhit` together give mean deposit confidence
(`cnt/nhit`); that quotient is the only thing `nhit` was ever for.

## 2. Why it is residue

`nhit` had exactly two readers in core, and both were removed on 2026-09-02 as
swept alternatives that lost — see
[`removed_and_untested_2026_09_02.md`](removed_and_untested_2026_09_02.md) §1.3:

| removed flag | what it read `nhit` for |
|---|---|
| `SCOVOX_VICTIM_MEAN` | mean-strength victim selection, `cnt/nhit` instead of min `cnt` |
| `SCOVOX_ADMIT_NORM` | nhit-normalised admission test |

Both now carry `#error` traps at `dir_voxel.hpp:91` and `:97` naming the
recovery tag. With the readers gone, every remaining reference is a **write**:

| site | operation |
|---|---|
| `dir_voxel.hpp:234` | `if (nhit && nhit[i] != 65535) ++nhit[i]` — bump |
| `dir_voxel.hpp:267` | `nhit[i] = 1` — on fill |
| `dir_voxel.hpp:319` | `nhit[min_i] = 1` — on evict |
| `sem_split_map.cpp:42` | passes `d->nhit` through |
| `sem_split_map.cpp:88` | `pre_nhit[i]` snapshot, for the deposit trace |
| `replay_scenenn.cpp:691, :911` | copies into the `.slots` dump |

The eviction comparator does not appear in that list. It reads `qmax`:

```cpp
// dir_voxel.hpp:298
const bool evict_now = use_q ? (q_fx > qmax[min_i])
                             : (inc > evicted_evidence);
```

So in the shipped mapper `nhit` is **write-only**: a branch, a saturation
check and a store per deposit, feeding nothing but offline instrumentation.

## 3. The contrast with `qmax`, which stays

The two fields look symmetric — both are 2 B/slot confidence tracks compiled in
behind a `SCOVOX_TRACK_*` flag — and they are not. `qmax` has a live reader and
a measured effect on the shipped metric.

E9 contains a clean single-variable test: `cand` and `abl_admit` are the **same
binary** (`e5/k2_i3_evid`), differing only by `--evict-by-confidence`.

| metric (mean of 8 scenes) | `cand` (qmax) | `abl_admit` (cnt) | delta | sd | wins |
|---|---|---|---|---|---|
| semantic mIoU, intersection | 0.5870 | 0.5409 | **+0.0462** | 0.0134 | **8/8** |
| semantic mIoU, union | 0.2989 | 0.2793 | **+0.0196** | 0.0076 | **8/8** |
| occupancy IoU | 0.4393 | 0.4393 | +0.0000 | 0.0000 | — |

Occupancy is identical to the last digit, which is the check that the arm is
what it claims to be: eviction is purely semantic and must not move occupancy.

`qmax` is therefore worth **+0.046 intersection mIoU, 8/8 scenes** — far above
the 0.001 materiality threshold. `nhit` sits in the same struct and is read by
nothing.

**Evidence note.** Do not substitute the phase-P sim's rule rankings
(`archive/phase_p/results/p4_front.json`) for the table above. That sim scored
retention, never mIoU, and was archived 2026-08-29 precisely because its
rankings did not transport to the C++ mapper. The decision here rests on the E9
C++ measurement and on byte-identity, not on any sweep ordering.

## 4. The change

One line, in the outer repository:

```
scovox_slot_rules/scripts/build_rules.sh:86
- TRACKS="-DSCOVOX_TRACK_QMAX=1 -DSCOVOX_TRACK_NHIT=1"
+ TRACKS="-DSCOVOX_TRACK_QMAX=1"
```

`TRACKS` is the single point where both tracks enter the production build. They
were coupled there for no reason beyond having been added together; the
production binary needs one and the analysis binary needs both.

Analysis builds (`build_e0.sh:41`, and any invocation wanting `--dump-slots`
with a populated `nhit[]`) keep `-DSCOVOX_TRACK_NHIT=1` explicitly.

**No `scovox_core` change is required.** Verified: every site is already
`#if SCOVOX_TRACK_NHIT` guarded with an `#else` fallback (`nullptr`, `0`) —
`sem_split_map.cpp:41` and `:87`, `replay_scenenn.cpp:690` and `:886` — and the
header's own default is already `#define SCOVOX_TRACK_NHIT 0`.

## 5. What stays, and why

- **The field and its guards.** Recovering mean-victim selection means restoring
  its reader from tag `pre-flag-removal-2026-09-02` anyway; deleting the storage
  as well would add a second thing to restore for no gain.
- **The `has_nhit` header byte** in the `SCVXSL01` slot dump. The format already
  makes the trailing `nhit[K]` block conditional, and `scripts/slots_io.py`
  reads byte 21 and branches. A `has_nhit=0` dump is a valid dump.
- **`SCOVOX_TRACK_NHIT` itself**, defaulting to 0. This is a build-configuration
  correction, not a code deletion, so nothing needs a ledger recovery tag.

## 6. Acceptance

**Gate: the scored `.bin` must be byte-identical.** This is stronger than "no
material mIoU loss" and is the right bar, because the change cannot alter what
is written — it can only alter how much is stored beside it.

1. Build both arms from identical source, one with the `-D` and one without.
2. Replay one scene per arm at the shipped flags, `--out` to distinct paths.
3. `md5sum` the two dumps. They must match. Repeat across all 8 scenes.
4. `./dev.sh test` and `./dev.sh verify` must pass on the new arm.
   `scripts/verify_core_check.py:63-68` recomputes the expected `DirVoxel` size
   from the flags the binary reports, so it adapts with no edit.

**A static_assert re-arms.** `dir_voxel.hpp:160` —

```cpp
static_assert(!SCOVOX_TRACK_QMAX || SCOVOX_TRACK_NHIT || K_TOP != 2
              || sizeof(DirVoxel) == 20, ...)
```

— is currently short-circuited true by `SCOVOX_TRACK_NHIT`, i.e. disabled in the
build it describes. With the flag off it becomes live and pins the 20 B layout.
The neighbouring assert at `:157` is named the *"Production K_TOP=2 invariant"*
and asserts 16 B; after this change it remains disabled by `SCOVOX_TRACK_QMAX`,
which is correct but means its name still describes a build that is not shipped.
Renaming it is out of scope here and is logged rather than done.

## 7. Consequences for `.slots` consumers

| consumer | status |
|---|---|
| `scripts/slots_io.py` | branches on `has_nhit` (byte 21). Unaffected. |
| `scripts/verify_core_check.py` | recomputes size from reported flags. Unaffected. |
| `archive/phase_p/*` | reads `nhit`; **archived**, superseded by the C++ mapper. |
| ROS wire (`binary_serializer.hpp`) | never carried `nhit`. Unaffected. |

Slot dumps produced *before* this change keep their `nhit` block and keep
reading correctly — the header byte is what disambiguates them, which is why it
was made a header field rather than an implicit convention.

## 8. What is not claimed

**No speed improvement is claimed here.** The change removes a branch and a
store per deposit and 4 B per Dirichlet voxel, but the ray walk dominates: E-W10
counted ~5.86 G voxel steps against ~83 M hits on a single scene. Any timing
effect must be measured under the standing rule that a comparison is admissible
only between arms interleaved inside one session. The justification for this
change is that it removes a field nothing reads — not that it is faster.

## 8a. Acceptance — result

| gate | result |
|---|---|
| `./dev.sh test`, `TRACK_NHIT` off (new default) | **180 / 181** |
| `./dev.sh test`, control build `-DSCOVOX_BETA_U16=0` | **180 / 181, identical single failure** |
| byte-identity of scored dumps, 8 scenes | *pending* |
| `./dev.sh verify` | *pending* |

The single failure is `ScovoxMapSplitFarCarve.FarCarveBitIdenticalToFullWalk` —
a **pre-existing** far-carve fast-path divergence (one extra voxel at band 0.30,
scan 4) from uncommitted `--far-fast-paths` work. It is not a regression from
this change: a control build that differs only in Beta storage width reproduces
the *identical* single failure, so the failure is independent of both storage
promotions made on this date.

The `static_assert` predicted in §6 did re-arm as described, and the build is
clean, which is the compile-time confirmation that `DirVoxel` is now 20 B.

## 9. Rollback

Restore the `-D` in `build_rules.sh:86` and rebuild. There is no state, no
format migration and no source change to revert. Dumps written under either
setting remain readable by the same reader.

## 10. Ledger

`removed_and_untested_2026_09_02.md` §2.1 recorded `SCOVOX_TRACK_NHIT` as
"0 in core, **1** in `build_rules.sh`", with the note that the unit-test build
and the sweep build therefore disagree on the shipped `DirVoxel` layout. This
change closes that disagreement in favour of the core default, and **§2.1 has
been updated** to say so.

This document covers one of the two storage defaults changed on 2026-09-04. The
other is `SCOVOX_BETA_U16`, which moved the opposite way (opt-in → default).
[`storage_defaults_2026_09_04.md`](storage_defaults_2026_09_04.md) is the
authority for both, carries the memory census that ranks them, and lists every
older statement the pair overrides.
