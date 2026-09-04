# Unified Dirichlet voxel design — collapsing Beta + Dirichlet into one SemDir grid

**Date:** 2026-05-13
**Status:** Design doc — implementation not yet started
**Branch target:** `refactor/split-tsdf-sembeta` (slots between Step 7 and Step 8)
**Replaces:** `SemBetaVoxel` + the separate Beta/Dirichlet update + consensus paths
**Wire format:** forces a `FORMAT_VERSION` bump from 2 → 3
**Carries through:** mass-conservation invariant (E5.1) becomes exact equality, not `≥ 0`

## Motivation

The current SCovox voxel ([sembeta_voxel.hpp:45-80](src/robot_sw/distributed_mapping/scovox_core/include/scovox/sembeta_voxel.hpp#L45-L80))
holds two separately-updated posteriors per voxel:

- a **Beta** over occupancy, parameterised as `(α_occ, α_free, α_unk)` with `α_unk`
  acting as a residual mass slot;
- a **sparse Dirichlet** over the top-`K_TOP` semantic classes, parameterised
  as `(sem_cls[k], sem_cnt[k])` pairs.

These were originally split for staging reasons (Beta was the substrate; Dirichlet
got added on top so the `K_TOP` ablation could vary the semantic head without
perturbing the occupancy story). Three things now argue for collapsing them into
a single Dirichlet over `{class_1, …, class_C, FREE}`:

1. **Mass conservation becomes exact** rather than `≥ 0`. Today's E5.1 invariant
   (`a_unk + Σ sem_cnt ≥ 0`) is a one-sided check; the unified scheme makes it
   a strict equality `ΔΣα == Σ Δ inputs` per update.
2. **K_TOP=1 stops being a special case.** The K=1 uninit-memory bug fixed at
   [sembeta_voxel.hpp:137-139](src/robot_sw/distributed_mapping/scovox_core/include/scovox/sembeta_voxel.hpp#L137-L139)
   was a class of bug that only exists because Beta and Dirichlet have separate
   update paths and the empty-slot sentinel is special-cased. Under unified
   Dirichlet there's a single update function and the OTHER bucket absorbs
   non-top-K mass — no special case to forget.
3. **Wire format shrinks** from 37 B/voxel (v2 SemBeta block) to ~20 B/voxel
   at K_TOP=2, because `α_unk` collapses into OTHER and the per-stream framing
   header disappears.

What is **explicitly preserved** is the admission gate semantics. SCovox's
precision advantage on SceneNet head-to-head
([scenenet-scovox-vs-slim-vdb-gap-mechanism-2026-05-13](../../memory/project_scenenet_gap_mechanism_2026_05_13.md))
comes from the Beta admission gate being geometrically stingier than SLIM-VDB's
TSDF truncation band. That gate is `p_occ ≥ τ ∧ Var[p_occ] ≤ σ` — both quantities
have identical closed forms under the unified Dirichlet, so the labelling
envelope (the actual mechanism of the SceneNet win) does not move.

## Theoretical framing

Let `α ∈ ℝ^(K_TOP + 2)` per voxel:

- `α[k]` for `k ∈ 1…K_TOP` — top-`K_TOP` semantic class slots, paired with class IDs `cls[k]`
- `α[FREE]` — free-space evidence (carved ray hits, no-return rays, miss bookkeeping)
- `α[OTHER]` — lumped evicted class mass (everything outside the top-`K_TOP` slots)

The true generative model is `Dir(α_1, …, α_C, α_FREE)` over `C + 1` categories
where `C` is the dataset's class count (e.g. 14 for NYU13 / SceneNet, 19 for
SemanticKITTI). Marginalising over collapsed dimensions is itself a Dirichlet,
so the truncated representation
`Dir(α_{c_1}, …, α_{c_K}, α_OTHER, α_FREE)` with `α_OTHER = Σ_{c ∉ top-K} α_c`
is the proper marginal — every quantity of interest is preserved exactly,
the only thing lost is the per-class breakdown *outside* the top-K. Same lossy
contract as today, but now mathematically clean.

### Priors

The symmetric Dirichlet prior `α_0 ∈ ℝ_+` is applied per underlying dimension:

| Slot       | Prior mass | Reason                                                 |
| ---------- | ---------- | ------------------------------------------------------ |
| `α[k]`     | `α_0`      | One Dirichlet dim                                      |
| `α[FREE]`  | `α_0`      | One Dirichlet dim                                      |
| `α[OTHER]` | `(C − K_TOP − 1) · α_0` | Sufficient statistic for `C − K_TOP − 1` collapsed dims |

Recommended default: `α_0 = 0.01`. This matches "Beta starts near zero,
Dirichlet starts near zero" behaviour from the existing code and minimises
behavioural drift in the refactor. A one-line toggle for Jeffreys-style
`α_0 = 1 / (C + 1)` should be plumbed for a single-shot ablation.

### Marginal queries

| Query                                  | Closed form                                                |
| -------------------------------------- | ---------------------------------------------------------- |
| `S_occ` (total occupied evidence)      | `α[OTHER] + Σ_{i=1..K_TOP} α[i]`                          |
| `S` (total)                            | `S_occ + α[FREE]`                                          |
| `p_occ`                                | `S_occ / S`                                                |
| `Var[p_occ]`                           | `S_occ · α[FREE] / (S² · (S + 1))`                         |
| argmax in-top-K class                  | `cls[ argmax_i α[i] ]`                                     |
| `p(c* \| occupied)` (confidence)       | `α[i*] / S_occ`                                            |
| `p(out-of-K \| occupied)`              | `α[OTHER] / S_occ`                                         |
| `Var[p(c)]` for top-K class            | `α[i_c] · (S − α[i_c]) / (S² · (S + 1))`                   |

The mesh-labelling rule gains a knob: prefer `argmax α[i]` only when
`α[i*] > α[OTHER]`. Otherwise label with the cross-grid-absent sentinel
(`0xFFFF`), making "evidence is spread across rare classes, no single mode"
a queryable state instead of silently picking whichever rare class happened
to land in slot 0.

## Struct layout

```cpp
namespace scovox {

/// 20-byte (at K_TOP=2) unified Dirichlet voxel.
///
/// Layout (K_TOP=2):
///   offset 0:   alpha_free  (float, 4 B)  — α[FREE]
///   offset 4:   alpha_other (float, 4 B)  — α[OTHER], lumped evicted mass
///   offset 8:   cnt[0]      (float, 4 B)  — α for slot 0
///   offset 12:  cnt[1]      (float, 4 B)  — α for slot 1
///   offset 16:  cls[0]      (uint16, 2 B) — class id at slot 0 (0xFFFF=empty)
///   offset 18:  cls[1]      (uint16, 2 B) — class id at slot 1
///   total: 20 B at K_TOP=2.
///
/// General formula:
///   sizeof = ((8 + 6·K_TOP + 3) / 4) * 4  (rounded up to 4-byte alignment)
struct SemDirVoxel {
  float    alpha_free;
  float    alpha_other;
  float    cnt[K_TOP];
  uint16_t cls[K_TOP];

  inline float s_occ() const noexcept {
    float s = alpha_other;
    for (int i = 0; i < K_TOP; ++i) s += cnt[i];
    return s;
  }
  inline float s_total() const noexcept { return s_occ() + alpha_free; }
  inline float p_occ() const noexcept {
    const float s = s_total();
    return (s > 0.f) ? (s_occ() / s) : 0.5f;
  }
};

constexpr std::size_t kSemDirExpectedSize =
    ((8u + 6u * static_cast<std::size_t>(K_TOP) + 3u) / 4u) * 4u;
static_assert(sizeof(SemDirVoxel) == kSemDirExpectedSize);
static_assert(K_TOP != 2 || sizeof(SemDirVoxel) == 20);
static_assert(std::is_trivial_v<SemDirVoxel>);
static_assert(std::is_standard_layout_v<SemDirVoxel>);
}
```

The factory `defaultSemDirVoxel()` initialises `alpha_free = α_0`,
`alpha_other = (C − K_TOP − 1) · α_0`, every `cnt[i] = α_0`, every
`cls[i] = 0xFFFF`. **Required at every allocation** for the same reason as
the SemBeta factory: Bonxai's pool allocator zero-inits leaf blocks, so
absent the factory the first update would increment from 0 instead of from
the prior.

The constant `C` (total class count) is a per-mapping parameter, plumbed via
`SemDirMap::Params::num_classes` and serialised in the wire-format v3 header.
This is **the one new mapping parameter** vs. today.

## Update rules

### Hit (point return, soft-prob ingest)

Input: `K_TOP_input` pairs `(cls, p)` plus a residual `r = 1 − Σ p` representing
softmax mass not in the top-`K_TOP_input` slot. `quality` scales the increment.

```text
function update_hit(voxel, hit_topk, quality):
  voxel.alpha_free += quality · hit_topk.residual_free

  merged = small_dict()
  for i in 0..K_TOP-1:
    if voxel.cls[i] != 0xFFFF:
      merged[voxel.cls[i]] = voxel.cnt[i]
  for (c, p) in hit_topk.pairs:
    merged[c] = merged.get(c, α_0) + quality · p
    # Note: when a class re-enters from OTHER, its prior α_0 reappears.
    # OTHER's prior compensation: subtract α_0 from voxel.alpha_other
    # if c was previously OTHER-bucketed. Net prior conservation: ✓.

  sorted_entries = sort(merged.items(), key=value, descending)
  for i in 0..K_TOP-1:
    if i < len(sorted_entries):
      voxel.cls[i], voxel.cnt[i] = sorted_entries[i]
    else:
      voxel.cls[i], voxel.cnt[i] = 0xFFFF, α_0
  for j in K_TOP..len(sorted_entries)-1:
    voxel.alpha_other += sorted_entries[j].value
    # Subtract one α_0 per slot if this class was already in a top-K slot
    # (its prior was double-counted on entering `merged`)
```

**Cost.** `merged` holds ≤ `K_TOP_voxel + K_TOP_input` entries (4 at K=2).
A small insertion sort on 3–4 entries is cheaper than today's
`sparse_add` branch tree. No allocation.

### Miss (no-return ray, carve through free)

```text
function update_miss(voxel, quality):
  voxel.alpha_free += quality
  # cls/cnt and alpha_other untouched
```

A scalar bump. Same shape as the current Beta `α_free` increment, and
preserves the "miss does not update occupancy classes" invariant pinned by
[test_sembeta_map.cpp](src/robot_sw/distributed_mapping/scovox_core/test/test_sembeta_map.cpp)'s
`MissDoesNotUpdateOccupancy` test.

### Carve-skip wall threshold

Today's `carve_skip_occ_threshold` (don't carve free through already-occupied
voxels) is preserved: gate becomes `s_occ() / s_total() < carve_skip_occ_threshold`
before applying `update_miss`. Same single comparison as today.

### Evidence saturation

Today's `evidence_saturation` cap on `α_occ + α_free + α_unk` becomes
`S = s_total() ≤ evidence_saturation`. One cap, one comparison. When the cap
is hit, all increments scale uniformly so the distribution shape is preserved.

## Consensus merge (between robots)

Two voxels `A` and `B` for the same coordinate, both in `(K_TOP, OTHER, FREE)`
form:

```text
fused.alpha_free  = A.alpha_free  + B.alpha_free  - α_0
fused.alpha_other = A.alpha_other + B.alpha_other - (C - K_TOP - 1) · α_0

# Sparse class slots: union, sum coinciding, re-truncate
union = small_dict()
for i in 0..K_TOP-1:
  if A.cls[i] != 0xFFFF: union[A.cls[i]] = A.cnt[i]
for i in 0..K_TOP-1:
  if B.cls[i] != 0xFFFF:
    union[B.cls[i]] = union.get(B.cls[i], α_0) + B.cnt[i] - α_0
    # The α_0 subtraction is the Dirichlet-conjugate equivalent of the
    # Beta consensus formula a_fused = a_A + a_B - 1: each coinciding class
    # carries its prior on both sides, so we subtract one prior once.

sorted_entries = sort(union.items(), key=value, descending)
for i in 0..K_TOP-1:
  if i < len(sorted_entries):
    fused.cls[i], fused.cnt[i] = sorted_entries[i]
  else:
    fused.cls[i], fused.cnt[i] = 0xFFFF, α_0
for j in K_TOP..len(sorted_entries)-1:
  fused.alpha_other += sorted_entries[j].value
```

This is the same `a_fused = a_A + a_B - 1` Beta-consensus formula
([consensus_merge_v2.hpp](src/robot_sw/distributed_mapping/scovox_core/include/scovox/consensus_merge_v2.hpp))
generalised across all K_TOP + 2 dimensions — `α_0` (or `(C - K_TOP - 1) · α_0`
for OTHER) is the per-dim prior to subtract once for the shared baseline.
Bit-identical to what the consensus node already does, just on a single
vector instead of `Beta + Dirichlet` separately.

## Wire format v3

| Field                            | Type   | Size                  | Notes                            |
| -------------------------------- | ------ | --------------------- | -------------------------------- |
| magic                            | u32    | 4 B                   | unchanged from v2                |
| version                          | u8     | 1 B                   | `= 3`                            |
| resolution                       | f32    | 4 B                   | unchanged                        |
| `num_classes` (C)                | u16    | 2 B                   | **new** — needed to recompute OTHER prior on deserialise |
| `K_TOP`                          | u8     | 1 B                   | **new** — pinned, deserialise asserts match |
| tsdf section                     | (as v2)| variable              | unchanged (still 20 B/voxel)     |
| **semdir section**               |        |                       |                                  |
|   count                          | u32    | 4 B                   |                                  |
|   per-voxel record               |        | `12 + 6·K_TOP` B      | coord (12 B) + struct (8 + 6·K_TOP B) |
|     coord                        | i32×3  | 12 B                  |                                  |
|     alpha_free                   | f32    | 4 B                   |                                  |
|     alpha_other                  | f32    | 4 B                   |                                  |
|     cnt[K_TOP]                   | f32×K  | `4·K_TOP` B           |                                  |
|     cls[K_TOP]                   | u16×K  | `2·K_TOP` B           |                                  |

At K_TOP=2 with NYU13 (C=14): per-record body is **20 B** vs v2's **37 B**
for the SemBeta section → ~46 % wire-size reduction on the semantic stream.

`share_tsdf` option default remains `false`; when false the TSDF section is
empty (length = 0).

Backward compat: v1 and v2 readers stay in tree (no deletion). Deserialise
routes on the version byte.

## Mapping to existing ablations

| Ablation | Effect under SemDir |
| -------- | ------------------- |
| K_TOP sweep (B1, P6.1, P6.2) | Single knob: K_TOP slots. K=1 has the same OTHER bucket as K=C. No K=1 special case. |
| Soft-prob ingest pipeline (`[[softprob-pipeline-2026-05-04]]`) | Untouched — the `(cls, p)` pairs feed straight into `update_hit`. |
| Voxel-flip churn (B6) | Argmax remains `argmax_i α[i]` over top-K. Variance bound `S_occ` is now a single number — clean. |
| Carve-skip (C4 admission bias) | Same threshold gate, single comparison. |
| Evidence saturation | Single cap on `S = s_total()`. |
| Mass conservation (E5.1) | Becomes strict equality, not `≥ 0`. |
| No-return ray bias (`[[exp5-exp7-no-return-bias]]`) | Unchanged — `update_miss` is the same operation on `alpha_free`. |
| Eviction telemetry (match/empty/evict/drop) | Same categories; "drop" mass is now retained in OTHER instead of dropped — telemetry needs renaming to reflect that. |

## Mapping to existing files

| Existing file | Change |
| ------------- | ------ |
| [sembeta_voxel.hpp](src/robot_sw/distributed_mapping/scovox_core/include/scovox/sembeta_voxel.hpp) | Replace with `semdir_voxel.hpp` defining `SemDirVoxel` + `defaultSemDirVoxel()`. Keep static_assert layout invariants. |
| [sembeta_map.hpp](src/robot_sw/distributed_mapping/scovox_core/include/scovox/sembeta_map.hpp), `sembeta_map.cpp` | Replace with `semdir_map.hpp`/`.cpp`. Same `Params`/`integrateHit`/`integrateMiss` API surface — internal update logic changes; carve-skip / saturation gates stay. |
| [scovox_map_split.hpp](src/robot_sw/distributed_mapping/scovox_core/include/scovox/scovox_map_split.hpp) | Composer's second grid type changes from `SemBetaMap` to `SemDirMap`. Accessor names rename. |
| [binary_serializer_v2.hpp](src/robot_sw/distributed_mapping/scovox_core/include/scovox/binary_serializer_v2.hpp) | Stays as v2 reader (backward compat). |
| **new** `binary_serializer_v3.hpp` | v3 framing per the table above. |
| [consensus_merge_v2.hpp](src/robot_sw/distributed_mapping/scovox_core/include/scovox/consensus_merge_v2.hpp) | Stays. |
| **new** `consensus_merge_v3.hpp` | `mergeSemDir` per the consensus pseudocode. `mergeTsdf` reused from v2. |
| [mesh_labelling.hpp](src/robot_sw/distributed_mapping/scovox_core/include/scovox/mesh_labelling.hpp) | Update `labelMesh` to read `SemDirVoxel` (argmax top-K vs. OTHER comparison). |
| [voxel.hpp](src/robot_sw/distributed_mapping/scovox_core/include/scovox/voxel.hpp) — legacy fused `Voxel` | Untouched during transition. Retire after smoke run. |

## Test deltas

Existing 87 scovox_core tests / 176 full-tree tests assume `SemBetaVoxel`. The
ones that pin Beta-specific invariants need ported:

| Existing test | Status under SemDir |
| ------------- | ------------------- |
| `FirstObservationStartsFromBeta11Prior` | Re-pin: `defaultSemDirVoxel()` returns `Dir(α_0, …, α_0)`. |
| `MissDoesNotUpdateOccupancy` | Re-pin: `update_miss` only touches `alpha_free`. |
| `WallBlockingViaCarveSkip` | Re-pin: same threshold gate on `s_occ()/s_total()`. |
| `EvidenceSaturationCap` | Re-pin: cap on `s_total()`. |
| `SemBetaMap` Beta layout tests | Replace with `SemDirMap` layout tests. |
| All `Voxel` legacy tests | Untouched (legacy path stays during transition). |

**New tests** (10 minimum):

1. `SemDirVoxelLayout` — `sizeof == 20` at K_TOP=2; offsets pinned.
2. `DefaultSemDirVoxelPrior` — every dim equals `α_0` (FREE / cnt) or `(C-K-1)·α_0` (OTHER).
3. `MassConservationStrictEquality` — `ΔΣα == Σ Δ inputs` per update (the upgraded E5.1).
4. `UpdateHitTopKMergeAndEviction` — feed 4 distinct classes, verify top-2 kept by count, evictees lumped in OTHER with correct mass.
5. `UpdateHitClassReentryFromOther` — class re-enters top-K from OTHER; verify OTHER decrements by the class's prior `α_0`, not its full historical mass (since historical mass is unrecoverable — known lossy contract).
6. `OccupancyMarginalMatchesBetaAtK0` — at K_TOP=0 (only OTHER + FREE), `p_occ` and `Var[p_occ]` match the closed-form Beta values.
7. `ConsensusV3MergeIdempotent` — merging A with `defaultSemDirVoxel()` returns A.
8. `ConsensusV3MergeSymmetric` — `merge(A, B) == merge(B, A)`.
9. `WireFormatV3RoundTrip` — serialize + deserialize identity.
10. `WireFormatV3BackwardCompatRoute` — v2 stream still decodes via v2 reader path.

## Open questions for the design review

1. **Prior `α_0` default** — `0.01` (minimise drift) or `1 / (C+1)` (Jeffreys)?
   Recommend ship with `0.01` and a launch-file toggle, run a one-shot ablation
   on Replica room0 + SemanticKITTI seq08 to decide.
2. **OTHER prior on re-entry** — when a class re-enters top-K from OTHER, the
   "historical evidence" is unrecoverable (lossy compression). Spec says
   subtract only the per-dim prior `α_0`. Acceptable? Alternative is to
   track a per-OTHER class-count histogram (defeats the K_TOP compression).
3. **Should `num_classes` (C) be a compile-time constant or runtime parameter?**
   Runtime is cleaner for cross-dataset transfer; compile-time is simpler.
   Recommend **runtime** — it costs 2 B in the wire header and one extra
   `Params` field. Matches the existing `K_TOP` convention.
4. **Slot ordering vs. Step 8 (ROS-node wiring)**. SemDir refactor lands cleanly
   *before* Step 8 so the ROS plumbing wires straight to the new grid. Lands
   cleanly *after* Step 10 (legacy retirement) as a separate pass. Lands
   awkwardly in between (would force Step 8 to plumb a doomed `SemBetaMap` API).
   Recommend **before Step 8**, marking Step 7 (consensus_merge_v2) as
   bypassed-then-superseded-by-v3.
5. **Variance vs. preserving the 87 just-landed tests**. ~15 of them (those
   pinning Beta-specific invariants) need rewriting against SemDir. The rest
   port unchanged. Acceptable scope or stage-gate first via a `SemBeta → SemDir`
   shim layer? Recommend **rewrite directly** — the shim has no production
   user and adds technical debt.

## Risks

1. **Performance regression on the hit update.** The merge-sort-evict path
   touches more memory than the current Beta + Dirichlet pair (which short-
   circuits on the common case "class already in slot 0"). Mitigation: profile
   on Replica room0 100 frames before declaring done. Budget: ≤ 5 % wall-clock
   regression vs. current SemBeta path.
2. **`α_0` shifts numerical conditioning.** At `α_0 = 0.01` and K_TOP = 2,
   C = 14, the OTHER prior is `11 · 0.01 = 0.11`. This is the "I haven't seen
   anything yet" baseline. Need to verify `p_occ` at a fresh voxel stays at
   the symmetric 0.5, not shifted by OTHER's larger prior. (It does: OTHER
   contributes to `S_occ`, free contributes to `S_total - S_occ`; symmetric
   priors on both sides → `p_occ = 0.5`.) Pin this via unit test.
3. **Wire format v3 invalidates v2 NPZs / .scovox blobs on disk.** Any
   in-flight experiments using v2 wire format need either rerunning or
   keeping the v2 reader perpetually in tree. Recommend keep the v2 reader
   in tree (it's read-only and ~150 LOC).
4. **The mass-conservation strict equality is *exact* under the unified
   scheme**. Any floating-point drift now shows up as a test failure rather
   than passing under a `≥ 0` slack. Mitigation: tolerance budget in the test
   (`abs(ΔΣα - Σ Δ inputs) < 1e-5 · Σ Δ inputs`) rather than bit-exact.

## Effort estimate

| Phase | Files | LOC | Tests | Wall-clock |
| ----- | ----- | --- | ----- | ---------- |
| Voxel + factory | `semdir_voxel.hpp` | ~150 | 2 | 1 h |
| Map | `semdir_map.hpp`/`.cpp` | ~400 | 8 | 4 h |
| Composer rewire | `scovox_map_split.hpp` | ~50 | (existing 6 ported) | 1 h |
| Wire format v3 | `binary_serializer_v3.hpp` | ~250 | 3 | 3 h |
| Consensus v3 | `consensus_merge_v3.hpp` | ~180 | 4 | 2 h |
| Mesh labelling | `mesh_labelling.hpp` patch | ~30 | (existing 5 ported) | 0.5 h |
| Existing test ports | ~15 tests | -30/+60 | n/a | 2 h |
| Profiling + tuning | n/a | n/a | n/a | 2 h |
| **Total** | | **~1 000** | **17 new + 15 ported** | **~16 h** |

A full day of focused work plus a half-day of testing. Slot it as a new Step
"7.5" between consensus_merge_v2 (Step 7) and ROS-node wiring (Step 8) so the
ROS plumbing built in Step 8 reads the new API and is not thrown away.

## Decision needed before implementation

See **Open questions for the design review** above. The five questions resolve
to a single scope choice — "land SemDir as Step 7.5 (recommended)" vs. "stage
behind a SemBeta shim" vs. "defer until after Step 10 retirement" — once that
choice is made, the rest of the design is deterministic and the LOC budget
above stands.
