# SCovox code structure — best method and current code (2026-09-06)

This document has two halves that deliberately do not agree with each other.

- **§1 The best method** is the promoted configuration `e5/k2_i0_evid` — the
  mapper the experiments campaign selected and the one every published number
  comes from. It is defined by the offline replay driver in the sibling
  repository (`scovox_slot_rules/scovox_scenenn/src/replay_scenenn.cpp`) and
  by the archived write-up `docs/archive/design/best_method.md`.
- **§2 The current code** is this repository at commit `76b8962`, re-pinned
  and re-audited on 2026-09-06. It was first written against `33aa576` and the
  pin was left behind for 27 commits, during which the code-review series moved
  most of the addresses below. A second audit on 2026-09-05 re-read every
  `file:line` against source and found the first pass had NOT: about thirty
  pointers were off, several by the ±1 / ±10 signature of an offset rather than
  a read. A third audit on 2026-09-06 found the second had left every
  `scovox_node.cpp` address in §2.5 and §3.1 33-36 lines low (the startup log
  block grew under it), and five commits since `32121f2` (`5aae015`, `8535425`,
  `91b0892`, `703eb7b`, `a062908`) had moved `scovox_map_split.hpp` by up to
  96 lines and `sem_split_map.cpp` by about ten. Every pointer below was
  re-read against `76b8962`. Treat any pointer here as an address that
  drifts with the next edit — the symbol name beside it is the durable half, and
  `grep -n` on that name is the way to re-find it.
  Historically: the `DirVoxel` total-basis change, the `nhit` removal and the
  `SCOVOX_BETA_U16` default flip were committed in `1101e53`; the
  promoted-configuration file and an off-lattice test fix in `33aa576`.
- **§3 The gap** lists every place where the code has not been brought up to
  the best method. The library defaults and the replay already match; the
  ROS node, its launch files and its config files do not.

Everything in §2 and §3 was read from source, not from older docs. Line
numbers are line numbers at commit `76b8962` (2026-09-06) and will
drift; the function names will not. The working tree at the time of this audit
already carried an uncommitted change (the `sem_top_k` ingest cap: `sem_obs.hpp`,
`sem_split_map.hpp/.cpp`, `scovox_map_split.hpp`), and the three pointers that
describe it — `prepareRayObs`, `HitStage::probs_cap`, `probs_argmax` in §2.3 —
are working-tree addresses from that change, not `76b8962` ones. When a pointer here disagrees with the
tree, trust the name and not the number — and do not repair it by adding an
offset, which is exactly how the 2026-09-04 set rotted. The previous version of this document is at
`docs/archive/scovox_code_structure.md` and describes the pre-`d5da6a8` tree.

---

## 1. The best method (`e5/k2_i0_evid`)

### 1.1 Build

```
-O3 -DSCOVOX_K_TOP=2 -DSCOVOX_EVICT_INHERIT=0 -DSCOVOX_TRACK_QMAX=1 -DSCOVOX_E0_COUNTERS=1
```

`SCOVOX_TRACK_NHIT` is no longer set (the counter was write-only; the dump is
byte-identical with or without it). `SCOVOX_BETA_U16` is on by default in the
headers and needs no flag. All four values are already the header defaults
except `E0_COUNTERS`, whose default is `0` — it is a diagnostics-only counter
set, and the flag turns it on. Any build that
sets `SCOVOX_EVICT_INHERIT` to a non-zero value, or any of the removed
`SCOVOX_VICTIM_MEAN` / `SCOVOX_VICTIM_QMAX` / `SCOVOX_ADMIT_NORM` flags, is
refused at compile time by `#error` traps in
`src/scovox_core/include/scovox/dir_voxel.hpp:135-146`.

### 1.2 Run configuration

The replay's `Args` defaults **are** the best method; the command line only
needs the dataset paths. Written out in full, with the field each flag lands
in:

| replay flag | value | lands in | note |
|---|---|---|---|
| `--resolution` | 0.05 | `ScovoxMapSplit::Params::resolution` (and `semsplit.resolution`) | voxel edge, metres |
| `--stride` | 2 | replay only | depth subsample; must match the stored top-k payload |
| `--min-depth` / `--max-depth` | 0.4 / 4.0 | replay only | Asus Xtion usable band |
| `--num-classes` | 14 | `semsplit.num_classes` | sets the OTHER bucket as `C − K_TOP` |
| `--alpha0` | 0.01 | `semsplit.alpha_0` | Dirichlet prior per class. Settled: E10 swept it at readout, E13 (2026-09-05) re-ran the mapper so the `evict_by_confidence` path was live, and 1/14 (Perks) is exactly inert — union/inter mIoU and occupancy IoU delta `+0.000e+00` on 8/8, slot class residency differing on 6 voxels in 19.1 M and none of them past the `p_occ > 0.5` gate. E13b adds the calibration half: the BUILD contribution is float noise (≤ 4.7e-6), so a post-hoc reparameterisation of a 0.01 dump answers any α₀ question exactly and **no replay is needed**. The one non-inert metric is `nll_miss` (8/8 better at 1/14), but it is monotone in α₀ to 1000, concentrated on one scene, and paid for on `nll_hit` (worse 7/8); ECE is best at the incumbent |
| `--w-occ` | **1.5** | `semsplit.w_occ` | Stream A hit weight; on the ⅛ lattice |
| `--w-free` | 1.0 | `semsplit.w_free` | full-ray carve weight |
| `--beta-occ-prior` | **0.5** | `semsplit.beta_occ_prior` | Beta prior `a_occ` at first touch; `0.5 = 4/8`, on the ⅛ lattice |
| `--beta-free-prior` | **0.5** | `semsplit.beta_free_prior` | Beta prior `a_free`; the pair defaults to the shipped Jeffreys `Beta(0.5,0.5)`. `<= 0` means "unset" to `sanitise()`, so Haldane `Beta(0,0)` is **not** reachable through these flags |
| `--kappa0` | 1.0 | `semsplit.kappa0` | Stream B deposit scale |
| `--min-p-occ` | 0.5 | `semsplit.dirichlet_min_p_occ` | Stream B gate on posterior occupancy |
| `--evict-by-confidence` | **on** | `semsplit.evict_by_confidence` | needs `SCOVOX_TRACK_QMAX=1`; replay refuses otherwise |
| `--sem-band` | **0.10** | `semsplit.semantic_band_length` | metres of along-ray deposit around the hit |
| `--band-require-occ` | **false** | `semsplit.semantic_band_require_occ` | flat `kappa0` band deposit, no Beta read |
| `--evidence-saturation` | 0 | `semsplit.evidence_saturation` | off (Beta u16 still halves at 90 % of its ceiling) |
| `--class-evidence-saturation` | −1 | `semsplit.class_evidence_saturation` | −1 = share the global cap |
| `--inc-mode` / `--inc-thresh` | 0 / 0.10 | `semsplit.inc_mode` / `inc_thresh` | soft deposit over the softmax |
| `--hit-share` | pocc (`hit_flat_share=false`) | `semsplit.hit_flat_share` | endpoint share = `kappa0 · p_occ_post` |
| `--ray-spread` | 0 | `semsplit.ray_spread` | surface voxel only |
| `--spread-radius` | 0 | `semsplit.semantic_spread_radius` | no BKI kernel |
| `--semantic-mode` | 0 (Dirichlet) | `semsplit.semantic_mode` | |
| `--range-decay-length` | 50 | `semsplit.range_decay_length` | **written, never read** in the split path |
| `--batch-hits` | 1 | `semsplit.batch_hits` | strongest ray per voxel per scan |
| carve-no-return | on (`--no-carve-no-return` to disable) | replay only | carves free space along no-return pixels |
| `--fused-walker` | 1 | `ScovoxMapSplit::Params::fused_walker` | the band exists only here |
| `--tsdf-enabled` | 0 | `ScovoxMapSplit::Params::tsdf_enabled` | no mesh in the replay; Beta/Dir unaffected |
| `--far-fast-paths` | 1 | `ScovoxMapSplit::Params::far_voxel_fast_paths` | bit-identical shortcuts around the exact body |
| `--dump-label-gate` | 0.5 | replay dump only | readout gate on `p_occ` |
| `--dump-below-gate-as-unknown` | on | replay dump only | below-gate voxels scored as unknown, not dropped |

Library-side constants that are part of the method but not flags: `leaf_bits`
3, `inner_bits` 2, `dir_leaf_bits` 2, `batch_free_carve` true, `carve_skip`
0, Beta lattice step ⅛ (`SCOVOX_BETA_U16_SCALE` 8), TSDF truncation 3 fine
voxels when the fine band is on (it is off here).

The Beta prior is a flag as of the E11 sweep, but its default is still
`kBetaOccPrior` / `kBetaFreePrior`, so the promoted configuration passes neither
flag. **Those constants are Jeffreys `Beta(0.5,0.5)` as of 2026-09-05**,
promoted from Bayes–Laplace `Beta(1,1)`; E11 measured the switch as +0.0000 on
all six metrics, 0/8, with `n_occupied` and `n_intersect` identical to the
voxel. What a *symmetric* prior `Beta(c,c)` can reach is bounded before
any run: `p_occ >= 0.5` iff `a_occ >= a_free` iff `c + w_occ·n_hit >=
c + w_free·n_miss`, and `c` cancels. Both gates in the pipeline — the deposit
gate `dirichlet_min_p_occ` and the scorer's occupied set — sit at exactly 0.5,
so the occupied voxel set, `n_pred_occupied` and occupancy IoU are invariant
under any symmetric prior. Exactly one channel is left live, and it is narrower
than it looks: the **endpoint** class deposit weight `kappa0 · p_occ_post`
(`sem_split_map.cpp:740-741`, `hit_flat_share` false). The band path does *not*
share it — with `semantic_band_require_occ` false the band takes the flat
SLIM-VDB weight `class_share = kappa0` and never reads a Beta voxel
(`:890-897`) — and the BKI kernel is off (`semantic_spread_radius` 0). That one
channel is a transient: a smaller prior lets one look reach a higher `p_occ`
(at `w_occ` 1.5 a first look is `2.5/3.5 = 0.714` under `Beta(1,1)` and
`2.0/2.5 = 0.800` under the shipped Jeffreys) and both tend to the same limit,
so the prior re-weights early observations against settled ones and nothing
else.

The cancellation argument above is confined to the **0.5** gate. Where a
threshold is not 0.5 the prior is load-bearing: `dscovox_node.cpp:664`
publishes at `p_occ() >= 0.7`, where Jeffreys admits a voxel after 2 hits
against one miss and `Beta(1,1)` needed 3. The carve wall guard
`carve_skip_occ_threshold` is the other such site, and is off by default.

### 1.3 Per-ray algorithm (what one depth pixel does)

1. **Ray setup.** Origin `O`, endpoint `E`, direction `u`, depth `d`. The fused
   walker runs one exact DDA over `[E − max(d, trunc)·u, E + max(trunc, band)·u]`
   (`scovox_map_split.hpp:183-714`). `tsdf_enabled=0` does **not** make `trunc`
   0: the node passes `sdf_trunc = 0` but `TsdfMap::sanitise` (`tsdf_map.cpp:25`)
   clamps any `<= 0` back to 0.15 m, and the walker reads the sanitised value.
   So `back_reach` (`:264`) is `max(0.15, 0.10) = 0.15 m`, not the band's 0.10 m,
   and the segment handed to the DDA is the full 0.15 m even in the promoted
   configuration.

   That far end is **also the DDA's aim point**, which is why it is not simply
   shortened. `ExactRayIterator` steers at the *centre* of `coord_to`
   (`ray_iterator.hpp:38-43`; the aim itself at `:53-55`), so `k_far` sets the direction of the whole
   segment rather than only where it stops: moving it rotates the walk and
   changes which voxels are crossed **in front of** the surface too, where every
   write happens. A shortened `back_reach` changes the dumped map.

   The dead tail is dropped by **ending the walk early** instead (`:289-291`,
   `:540-546`). When the TSDF cannot write — `!tsdf_enabled_`, a dynamic source,
   or `geometry_off` — the semantic band is the only surviving behind-surface
   write, so `useful_back = sem_band_` (0 with the band off). Inside
   `exact_body`, a voxel behind the surface whose exact along-ray offset
   `t = −(v_point_voxel · u)` has reached `useful_back` latches `stop_walk` and
   the DDA callback returns `false`. `t` is non-decreasing along the walk (each
   DDA step adds `res·|u_i| ≥ 0`) and `dist ≥ t` always, so every remaining
   voxel is decided: the carve needs `sdf > 0`, the band needs `dist ≤ sem_band_`.
   Verified as **byte identity** of the dumped map, not as equal mIoU.

   The **near** end is shortened the other way, and only when the free-space
   carve is off. `--w-free 0` does not shorten anything by itself: it removes
   the carve WRITE (`applyCarveUpdate` returns on `w_inc <= 0` before touching
   state) while `walk_back` keeps traversing the whole ray to write nothing —
   which is why a carve-off arm still walks from the sensor. When the write is a
   guaranteed no-op for the ray, the walker instead **seeds the DDA in front of
   the surface** (`scovox_map_split.hpp`, `seed_carve_off_walk`), so carve-off
   integrates over the same window SLIM-VDB does.

   Unlike the far end, moving the near end is safe: `ExactRayIterator`'s visited
   sequence depends only on the start point, the direction and the `t ≤ 1`
   bound, so seeding at a point **on the same segment** yields a true SUFFIX of
   the full walk rather than a different walk. `exact_body` is a pure function of
   `(c, origin, endpoint)` — it never reads where the walk began — so with the
   carve off no latch (`carve_blocked`, `ring_left`, `k_hit_visited`,
   `stop_walk`) can be set by a dropped prefix voxel. The kept front window is

       front_reach = max(tsdf ? trunc + h : 0, band ? sem_band : 0)
                   + 2.7320508 · res

   where the `√3·res` term is **two** half-diagonals — a dropped voxel's centre
   may sit `h·√3` off the segment, and the aim point `centre(k_far)` may sit
   `h·√3` off `end_pos` — and the remaining `1·res` is headroom. This is also
   verified as byte identity, against the same arguments on the pre-change
   library.
2. **Far voxels** (further than `trunc + h` before the hit) take the far-skip
   or far-carve shortcut (`:397-401`, `:443-448`): carve staged into
   `CarveStage`, no per-voxel float body. Both shortcuts are asserted
   bit-identical to the exact body by `test_scovox_map_split`. `far_skip` arms
   on three separate ways the far body can be a no-op; the third, carve-off, is
   the only one that does not need a carve frame open, and `far_carve` stands
   down for it so the two remain mutually exclusive.
3. **Near voxels** run `exact_body` (`:498-592`). `sdf` is **not** the along-ray
   offset and **not** a true signed distance: `dist = |endpoint − voxel_centre|`
   with the sign taken from `(voxel_centre − origin)·(endpoint − voxel_centre)`
   (`:510-532`), so it is the straight-line distance from the voxel centre to
   *this ray's hit point*, positive between camera and surface. Every `|sdf| ≤ L`
   gate is therefore a sphere of radius `L` about the hit, intersected with the
   walked voxels — and because a walked voxel's centre sits up to half a voxel
   diagonal off the ray line (0.043 m at res 0.05), that offset counts toward
   `dist` and closes the band marginally early at both ends. TSDF write skipped
   (`tsdf_enabled=0`, gate at `:564`); if the voxel is not the hit voxel and
   `−band < sdf ≤ band`, a **band deposit** is staged (`:582-584`); otherwise
   the voxel is **carved** (`:587-591`).
4. **Hit voxel, Stream A** (`SemSplitMap::commitHit`, `sem_split_map.cpp:685-753`):
   `a_occ += w_occ` (1.5 → 12 lattice units). Under `batch_hits` only the
   strongest ray per voxel per scan reaches here (`flushStagedHits` `:536-575`).
5. **Hit voxel, Stream B**: gate `p_occ_post ≥ 0.5`; deposit
   `class_share = kappa0 · p_occ_post` into the `DirVoxel` through
   `dirichletUpdate` (`:60-231`): `s_total += class_share` once, then the
   softmax is spread over the K=2 slots by `sparse_add_class`
   (`dir_voxel.hpp:313-438`) with outcomes match / fill / evict / drop.
   Eviction compares the arrival's confidence against the weakest slot's
   `qmax`; an evicted slot's evidence falls back into the derived
   `other()` (`SCOVOX_EVICT_INHERIT=0`).
   The gate reads `p_occ_post` and **nothing else** — `commitHit` never
   inspects `sem_probs`, and `dirichletUpdate` books `s_total += class_share`
   before any branch looks at the vector. A hit carrying no labels therefore
   still deposits its whole share, as uncovered mass in `other()`. So
   `s_total − C·α₀` counts *looks*, not *labelled looks*; anything reading it
   as a count of class observations is over-counting by the unlabelled hits.
6. **Band voxels, Stream B only** (`applyBandSemantic` `:860-907`): with
   `band_require_occ=false` a flat `kappa0` deposit with no Beta read, so a
   band voxel can hold a class before it holds occupancy evidence. **Not
   batched** — `batch_hits` stages the endpoint only (`:622-624`), and this is
   an immediate write from inside the walk, so a band voxel takes one deposit
   per depth pixel passing it while the hit voxel takes one per frame. `s_total`
   therefore grows far faster off the surface than on it; harmless for mIoU (the
   scorer excludes Dir-only voxels by `state`) and material for anything reading
   `s_total` as a Dirichlet concentration. See M9 in `archive/code/code_review_2026_09_04.md`.
7. **Carved voxels**: `a_free += w_free` per voxel, written once per voxel per
   scan at `flushCarveFrame` (`:499-534`), block-ordered; occupied hits win
   over carves in the same scan.
8. **Saturation** (`applyBetaSaturation` `:1104-1126`, `applyDirSaturation`
   `:1128-1158`): the opt-in cap is off; the unconditional u16 halving at 90 %
   of `BetaCount::kMax` (`kMax` = 65535/8 ≈ 8 192 on the ⅛ lattice, so
   ≈ 4 915 hits at `w_occ` 1.5) is the only ceiling.
9. **Readout** (replay dump, not the library): argmax over the K slots plus
   `other()`; voxels with `p_occ < 0.5` are written as unknown.

#### Two things called "band", and four called "neighbour"

Two lengths share the word *band* and are routinely conflated:

| | knob | value | what it sets | active? |
|---|---|---|---|---|
| TSDF truncation band | `sdf_trunc` | 0.15 m | the shell around the surface where a TSDF value would be stored, and — because `back_reach = max(trunc, sem_band)` — how far past the hit the DDA walks | TSDF **writes off**; the length still sets the walk |
| semantic band | `semantic_band_length` | 0.10 m | how far along the ray the class is deposited | **on** |

They are independent numbers on the same axis. The semantic band borrows the
TSDF band's *shape* (it is SLIM-VDB's `alpha[label] += 1` over `sdf > −trunc`)
but writes into the Dirichlet grid, and is gated separately.

Both are written **along the ray** — only voxels the DDA steps through — while
the length each one compares against is **radial from the hit point**, not an
along-ray offset (step 3 above). For one ray a band is a line segment; the shell
around the surface that the name suggests is the union of those segments over
many rays.

The radial measure is inherited, not invented: `exact_body`'s `sdf` is SLIM-VDB's
`ComputeSDF` (`slim-vdb/src/slimvdb/slimvdb/VDBVolume.cpp:60-69`) transcribed —
same `v_voxel_origin` / `v_point_voxel` / `dist` / `proj` decomposition, same
sign convention. scovox computes it in float rather than double, and guards
`proj == 0` explicitly where the original divides `0/0` and is rescued only by
`NaN > -sdf_trunc` being false. **`semantic_band_length` 0.10 is SLIM-VDB's own
`sdf_trunc`** (`examples/cpp/config/scenenet.yaml`, alongside `voxel_size` 0.05,
`space_carving` False, `min_weight` 20.0): the band was set to that value so the
two mappers deposit semantics over the same footprint, which is what makes
`band_require_occ=false` a mirror of `alpha[label] += 1` rather than an
approximation of it. Where the two genuinely part is the ray range — SLIM-VDB
walks `[depth − sdf_trunc, depth + sdf_trunc]` and so never enters free space,
while scovox always walks from the sensor origin and carves it.

One correction to that inheritance, from E-W17: 0.10 is SLIM-VDB's *published*
`sdf_trunc`, not its best one on SceneNN. Swept on these eight scenes, SLIM-VDB's
own optimum is **0.04** — 0.10 costs it 0.0555 union mIoU. scovox's 0.10 was
separately re-selected by its own held-out sweep (E3/E4), so the value is earned
here rather than borrowed; but the sentence "0.10 because SLIM-VDB uses 0.10" is
no longer an argument for it, and whether scovox's band optimum is also below
0.10 has not been re-tested since the deposit rule changed.

**E-W33 (2026-09-07) settles the provenance question this raises.**
`semantic_band_length` 0.10 is the driver default (`replay_scenenn.cpp:191`),
as are the other three flags the sweep scripts pass on the scovox arm —
`w_occ` 1.5 (:113), `evict_by_confidence` true (:201), `dump_gate` 0.5 (:150).
None of them is an override. Together with the fact that shipped `w_occ` 1.5 is
*not* the union-mIoU argmax (w6.0 scores 0.2982 against 0.2790), the scovox
operating point cannot be read as fitted to this benchmark.


Four mechanisms involve a voxel's neighbours, and exactly one is promoted:

| mechanism | shape | site | promoted |
|---|---|---|---|
| semantic band | a segment of the ray | `applyBandSemantic` (`sem_split_map.cpp:860-907`) | **yes**, 0.10 m |
| BKI ball | a sphere | `applyHitUpdateKernel` (`:959-992`) | no — `semantic_spread_radius` 0 |
| three-voxel ray spread | ±1 voxel along the ray | `raySpreadDeposit` (`:775-844`) | no — `ray_spread` 0 |
| spatial readout | 6-connected relaxation | `scripts/slot_readout.py`, scoring time | no |

The first three are mutually exclusive by construction (`sem_split_map.cpp:281-289`):
a non-zero spread radius zeroes the band length, and either zeroes `ray_spread`.
Only the fourth is a *vote* in the literal sense — a voxel reading its
neighbours' state; the other three are one measurement written to several
voxels, and no voxel ever reads another.

### 1.4 Storage

| grid | voxel | bytes | fields |
|---|---|---|---|
| occupancy | `BetaVoxel` | **4** (`SCOVOX_BETA_U16=1`) | `a_occ`, `a_free` as `BetaCountU16` (⅛ lattice, ceiling 8191.875) |
| semantics | `DirVoxel` | **20** (K=2, `TRACK_QMAX=1`, `TRACK_NHIT=0`) | `s_total` f32, `cnt[2]` f32, `cls[2]` u16, `qmax[2]` u16; `other()` derived |
| geometry | `TsdfVoxel` | 8 | `distance`, `weight` — grid stays empty at `tsdf_enabled=0` |

`best_method.md` §5 still says "24 bytes ... nhit[2]"; that describes the
build that produced the E9 numbers, and the doc's own caveat says the dump is
byte-identical without `nhit`. 20 B is the shipped size.

### 1.5 Scores (SceneNN, 8 scenes, from `best_method.md` / `RESULTS.md`)

| metric | value |
|---|---|
| intersection mIoU | 0.5853 |
| union mIoU | 0.2981 |
| occupancy IoU | 0.4393 |
| precision / recall | 0.5236 / 0.7272 |
| phantom voxels | 16 005 |
| throughput | 6.2–6.7 fps |

Union mIoU is the number to use for cross-mapper comparison: it is the only one
of these that charges a mapper for the voxels it invents, and the two mappers
sit at very different points on the precision/recall axis.

> **Against SLIM-VDB, quote the tuned arm, not the published one.** SLIM-VDB at
> its published SceneNN configuration (`sdf_trunc` 0.10, `min_weight` 20) loses
> union mIoU 8/8, and that is a real number but not an informative one: tuning
> its only two knobs — `(sdf_trunc, min_weight)`, which is the complete CLOSED-mode
> surface, because `min_weight` is a readout filter, `fill_holes` is inert above
> zero and `p_threshold` is OPEN-only — is worth **+0.1073 union mIoU** to it on
> these eight scenes, more than the whole margin. At its tuned optimum
> `(0.04, 800)` it scores union 0.2919 against the shipped scovox point's 0.2790,
> and the paired verdict is **ambiguous** (p_exact 0.3828, 3/8). What scovox still
> holds materially is precision (+0.0914, 7/8) and half the phantom voxels.

> **Amended by E-W33 (2026-09-07): quote BOTH arms, and say why they differ.**
> The published config does not fail for want of a sweep — it fails because
> `min_weight` is a raw observation count (`weighting_function` returns 1.0,
> `scenenet_pipeline.cpp:133`; accumulated `VDBVolume.cpp:193`; gated once at
> readout `:518`), and a count threshold does not transfer across sequence
> length. At 1300–3800 frames a surface voxel accumulates thousands of hits, so
> 20 admits nearly everything. scovox's gate is a probability and is scale-free.
> Report defaults-against-published (**+0.0946 union, 7/8**) with that
> explanation, and tuned-against-tuned (**−0.0127, 3/8**) as the conservative
> bound. Two caveats travel with the first: the tuned SLIM-VDB cell survives
> leave-one-scene-out (0.2878 held-out vs 0.2919 oracle), so it is a shippable
> constant rather than an oracle fit; and against scovox `w_occ` 6.0 the
> published config is ambiguous on all five metrics, so much of the margin is
> operating-point distance rather than mapper quality.
> Placed at a matched operating point — scovox `w_occ` 6.0 — the two mappers are
> **ambiguous on every metric measured**, union at p_exact 1.0000. See
> `scovox_slot_rules/REVIEW_LOG.md` E-W17, which supersedes E-W16's headline.

> **Confirmed on the current binary, and extended to the fast arm (E-W32,
> 2026-09-07).** An 8-scene paired sweep on today's build reproduces E-W17's
> verdict to four decimals — shipped scovox union **0.2792** (E-W17: 0.2790)
> against tuned SLIM-VDB's **0.2919** (E-W17: 0.2919), **3 win / 5 loss**, still
> ambiguous. Two things are new. First, the **carve-off** arm (`--w-free 0`,
> 26.85 fps) loses union mIoU **0/8** against that same tuned cell, mean
> −0.0385 — it is a 3.464x speed lever that costs semantic quality on every
> scene, and is not a shipping default on this evidence. Second, intersection
> and union **disagree in sign** on the shipped-vs-tuned comparison (∩ 6/2 for
> scovox, ∪ 5/3 against), which is the clearest demonstration to date of why
> only the union column is quotable across mappers. Speed is uniform: SLIM-VDB
> at its own best cell is **2.560x** faster than scovox's best cell on 8/8
> scenes. What survives 8/8 is memory — **5.79x** less grid and **2.87x** less
> peak RSS — and it is not map shrinkage: the carve-off arm predicts *more*
> occupied voxels than SLIM-VDB on every scene.

> **These scores do not describe the code in §2.** They were produced by
> `.build/e5/k2_i0_evid/replay_scenenn`, md5 `60e17da907d0`, built
> **2026-08-30**. Three commits since then change what the deposit path does:
> `e316f07` (hit batching default-on, `uint16` Beta), `d5da6a8` (the exact walk
> replaces the earlier approximate traversal) and `1101e53` (count storage). Re-running the promoted arm
> on the current binary, same CLI and same frames, moves occupancy IoU by −0.029
> on scene 016 and −0.062 on 015: `n_pred_occupied` falls ~26 %, precision
> rises, recall falls, intersection mIoU rises, union mIoU falls. Ray counts are
> byte-identical between the two, so the input is the same and the difference is
> in what the walk deposits.
>
> The cause is **hit batching** (§1.3 step 4), not traversal — now measured, not
> inferred. Batched, `a_occ` counts observations; un-batched it counted pixels,
> and a 640×480 frame puts hundreds of pixels in one surface voxel, so the two
> reach the fixed `p_occ ≥ 0.5` gate at completely different rates. The
> `--batch-hits 0|1` A/B on one binary, 8 scenes, attributes **91–96 % of every
> component** of the shift to batching: `n_pred_occupied` −10 903 of −11 566,
> recall −0.1744 of −0.1850, precision +0.0663 of +0.0690, intersection mIoU
> +0.0290 of +0.0318, union mIoU −0.0184 of −0.0191.
>
> The residual — exact DDA plus count storage together — is small and, on
> semantics, **immaterial**: union mIoU −0.00070, below the 0.001 threshold in
> magnitude; occupancy IoU −0.0018; recall −0.011. The traversal correctness fix
> did not cost the numbers; batching moved them.
>
> **Re-measured on all eight scenes, 2026-09-05** (`cells/goal8.tsv`: the
> promoted `abl_inherit` argv and the `e5/k2_i0_evid` flags, replayed from the
> post-code-review tree). This is what the table above becomes on the current
> binary:
>
> | metric | published (E9) | current tree | Δ | scenes |
> |---|---|---|---|---|
> | intersection mIoU | 0.5853 | 0.6172 | **+0.0318** | 8/8 up |
> | union mIoU | 0.2981 | 0.2790 | **−0.0191** | 5/8 down |
> | occupancy IoU | 0.4393 | 0.3993 | **−0.0399** | 6/8 down |
> | precision | 0.5236 | 0.5926 | +0.0690 | 8/8 up |
> | recall | 0.7272 | 0.5422 | −0.1850 | 8/8 down |
> | phantom voxels | 16 005 | 8 645 | −7 360 | — |
> | `n_pred_occupied` | 32 730 | 21 164 | −11 566 | — |
>
> Every one of those deltas is the *total* column of the `--batch-hits` A/B
> above, to the digit — −0.1850 recall, +0.0690 precision, +0.0318 intersection,
> −0.0191 union, −11 566 predicted-occupied — and the two per-scene occupancy
> figures quoted earlier land at −0.0291 (016) and −0.0618 (015). The shift is
> the one already attributed, reproduced end to end; the code review added
> nothing to it, which is the point of running it.
>
> Batching is a **reparameterization**, not a separate optimization: sweeping
> `w_occ` batched recovers the un-batched result. Batched `w_occ` 6.0 against
> un-batched `w_occ` 1.5 grades **ambiguous on all six metrics at n = 8**, union
> mIoU +0.0008 (below MATERIAL), occupancy IoU −0.0001, recall −0.0098 — no
> material difference detected, though the intervals are too wide for *inert*.
> The ~4x factor is small because the carve is staged as well, so `a_occ` and
> `b_free` shrink together. Traversal volume is byte-identical across every arm,
> so batching's speed edge is the ~10 900 voxels it declines to write, and
> re-tuning `w_occ` costs it back: `batch_hits` and `w_occ` are two dials on one
> speed/completeness axis, and there is no free direction along it — only a
> precise sparse map (shipped: union 0.2790, precision 0.5926, recall 0.5422) or
> a complete imprecise one (w6.0: 0.2982 / 0.5295 / 0.7068), with union
> ambiguous between them.
>
> Rankings are unaffected: every arm inside a published comparison ran on one
> binary. Absolute values are stale until the ablation ring is re-run. Full
> finding: `docs/archive/code/code_review_2026_09_04.md` §H4.

The `RESULTS.md` deliverable block (`scovox_slot_rules/RESULTS.md:66-74`) has been corrected to
`SCOVOX_EVICT_INHERIT=0`, along with the headline table, which had been
reporting the demoted `i3` arm.

---

## 2. The current code

### 2.1 Repository layout

```
scovox/
├── compose.yaml              persistent dev container (ROS 2 Jazzy, GPU passthrough)
├── docker/
│   ├── Dockerfile            osrf/ros:jazzy-desktop + rosdep over the manifests
│   └── build_and_test.sh     one-shot colcon build + test in a throwaway container
├── scripts/
│   ├── launch_scovox.sh      in-container launcher for the raw-Ouster LiDAR path
│   └── wire_study/           offline scovox_bin re-encoding studies (header is stale, v5)
├── config/                   run configs for the real robot (LiDAR, fused, share, fine band)
├── docs/                     this file, code_review_and_updates.md, docs/archive/ (code reviews under archive/code/), docs/img/, docs/papers/
└── src/
    ├── scovox_core/          zero-ROS mapping library + 11 gtest binaries + split_memory_demo
    ├── scovox_msgs/          5 msgs, 3 srvs
    ├── scovox_mapping/       scovox_node, dscovox_node, scovoxmap lib, 9 launch files, 10 gtests
    └── seg_pipeline/         Python Mask2Former (Mapillary) → 14 outdoor classes
```

Line counts at `76b8962`: `scovox_node.cpp` 3437, `sem_split_map.cpp`
1229, `dscovox_node.cpp` 994, `scovox_map_split.hpp` 1116, `sem_split_map.hpp`
852, `binary_serializer.hpp` 712, `dir_voxel.hpp` 477. Bonxai is vendored at
`src/scovox_core/include/third_party/bonxai/`.

### 2.2 Build and run

- **Docker (the supported path).** `docker/build_and_test.sh` builds the
  image `scovox:jazzy` and runs `colcon build && colcon test` with the repo
  bind-mounted at `/scovox`. `compose.yaml` gives the same image as a
  persistent container (`docker compose up -d`, `docker compose exec scovox
  bash`). No build artefacts exist in the working tree right now.
- **Standalone `scovox_core`.** `src/scovox_core/CMakeLists.txt` builds
  without ament (commit `b13ca55`); tests register through `ament_add_gtest`
  under ROS or plain `add_executable` otherwise (`:83-85`).
- **The replay** in `scovox_slot_rules/` compiles `scovox_core` from this
  checkout with `-DSCOVOX_SRC=<this repo>` (its `CMakeLists.txt:55-58`) and is
  run through `scovox_slot_rules/docker/dev.sh`. That is where the best method
  is actually exercised; nothing in this repository runs it end to end.
- **Real robot.** `scripts/launch_scovox.sh raw` →
  `lidar_mapping.launch.py` with `config/scovox_lidar_raw_deskew.yaml`.

### 2.3 `scovox_core` — the library

#### Compile-time flags

| macro | default | effect |
|---|---|---|
| `SCOVOX_K_TOP` | 2 | slots per `DirVoxel` (`voxel.hpp:20-21`) |
| `SCOVOX_TRACK_QMAX` | 1 | per-slot confidence `qmax[]`; +4 B; required by evict-by-confidence (`dir_voxel.hpp:111-112`) |
| `SCOVOX_TRACK_NHIT` | 0 | per-slot deposit counter; write-only; +4 B (`dir_voxel.hpp:121-122`) |
| `SCOVOX_BETA_U16` | 1 | 2×u16 fixed-point Beta counters (`beta_voxel.hpp:69-70`) |
| `SCOVOX_BETA_U16_SCALE` | 8 | lattice step ⅛ (`beta_voxel.hpp:79-80`) |
| `SCOVOX_DEPOSIT_TRACE` | 0 | per-deposit trace sink; null unless one is installed (`sem_split_map.hpp:63-64`) |
| `SCOVOX_E0_COUNTERS` | 0 | admission/eviction counters (`e0_counters.hpp:45-47`) |
| `SCOVOX_SPARSE_BRANCH_COUNTERS` | 0 | the four `sparse_add` outcome counters; `SCOVOX_SPARSE_BUMP` is `(void)0` unless set (`voxel.hpp:92-104`; §4.6) |
| `SCOVOX_WALK_MARGIN_VOX` | 2.7320508f | `kWalkMarginVox`, the DDA's front/back margin in voxels (`scovox_map_split.hpp:349-352`) |
| `SCOVOX_EVICT_INHERIT` | must be 0 | any other value → `#error` (`dir_voxel.hpp:144-146`) |
| `SCOVOX_VICTIM_MEAN`, `SCOVOX_VICTIM_QMAX`, `SCOVOX_ADMIT_NORM` | removed | `#error` (`dir_voxel.hpp:135-143`) |

Size invariants are `static_assert`ed: `DirVoxel` 20 B at K=2 with QMAX
(`dir_voxel.hpp:251-253`), 16 B without; `BetaVoxel` 4 B under u16, 8 B float
(`beta_voxel.hpp:274-277`); `TsdfVoxel` 8 B (`tsdf_voxel.hpp:31`).

#### What the build system itself sets (nothing)

The table above is the whole switch story, because **no `CMakeLists.txt` in the
tree defines a single compile macro**. Every switch above takes its value from an
`#ifndef` default in a header unless a `-D` overrides it on the command line, so
"the default build" and "the header defaults" are the same object. Read back
from the generated `build.ninja`, the compile line for every core and mapping TU
is exactly:

```
-O3 -DNDEBUG -std=gnu++17 -Wall -Wextra
```

`CMAKE_CXX_FLAGS` is empty; both C++ package `CMakeLists.txt` files
(`scovox_core` and `scovox_mapping` — `scovox_msgs` is message generation only
and sets neither) default `CMAKE_BUILD_TYPE` to `Release` (with the reason
written down) and add `-Wall -Wextra`. There is no `-march`, no `-ffast-math`,
and no LTO — the last two are graded choices, not omissions (E14). Four
consequences worth knowing:

Before any of them, a scoping fact that has bitten this project once already:
**several build paths exist and they compile different subsets of the tree.**
`scovox_slot_rules/docker/dev.sh` alone has four C++-building branches — `test)`
(plain cmake over `scovox_core`), `scenenn)`, `verify)` (`verify_core` with
`-DSCOVOX_SRC`) and `ros-test)` (colcon) — and `scovox/docker/build_and_test.sh`
is a fifth, also colcon. An earlier revision of this section said there were two
and that `dev.sh ros-test` was the only path with ament; both were wrong.

What matters is the SUBSET, not the count. The offline harness
(`scovox_slot_rules/scovox_scenenn`, the one every replay and every
byte-identity check goes through) compiles four `scovox_core` translation units
plus the harness, and cannot configure `scovox_mapping` at all. `scovox_core`
non-test, non-vendored source is 8766 lines against `scovox_mapping`'s 5939, so
even a whole-`scovox_core` claim covers about **three fifths** of the C++ — and
the harness compiles well under that, since it takes 4 TUs, not the package.
An earlier revision said "roughly two thirds"; the arithmetic did not support
it. Three warnings lived in the gap until 2026-09-05 — and note that
`./dev.sh test`, which builds all of `scovox_core`'s targets including
`tools/split_memory_demo.cpp`, would have shown one of them, so "only the ROS
build could see them" is also too strong.

* **`-DNDEBUG` costs almost no checking.** Two `assert()`s exist outside the
  tests, one of them ours (`carve_stage.hpp:78`) and one vendored
  (`third_party/bonxai/bonxai/grid_allocator.hpp:155`); every other invariant is
  a `static_assert`, which `NDEBUG` does not touch.
* **`-march` is left off deliberately, not by omission, and the flag set has
  been graded.** The x86-64 baseline has no FMA, so although `-std=gnu++17`
  leaves `-ffp-contract=fast` enabled, no contraction actually happens.
  `objdump -d` counts 76 `vfmadd`/`vfmsub` instructions in the `-march=native`
  build and zero in `-O3`, `-O2` and LTO — so `native` really does change float
  results, and byte-identity of the output map across builds is a contract this
  project leans on. Measured on two scenes × two reps, interleaved
  (`scovox_slot_rules` E14):

  | arm | pooled wall vs `-O3` | map bytes |
  |---|---:|---|
  | `-O2` | +2.1 % | identical |
  | `-O3 -march=native` | −6.3 % | **different** (FMA) |
  | `-O3 -flto=auto` | −2.3 % | identical |

  Within a single arm the run-to-run spread reaches 11.5 %, which is larger than
  the LTO effect and larger than its sign flip between the two scenes, so **only
  `native`'s effect is bigger than the noise — and `native` is the one arm that
  is inadmissible.** `-O3` stands. See `REVIEW_LOG.md` E14 for the full table
  and the conditions under which `native` would become worth re-scoring for.
* **Every switch declares its own `#ifndef` default**, so `-Wundef` can be
  carried permanently at zero noise and an undeclared `#if` switch is impossible
  in source. `SCOVOX_E0_COUNTERS` was the last exception and now declares `0` at
  `e0_counters.hpp:45-47`.
* **A misspelled `-D` on a build command line is a different hazard**, which no
  warning flag catches and no md5 distinguishes from an intended change. The
  binary reports its own compiled-in switch values instead:
  `scovox::buildSwitches()` (`version.hpp` / `src/version.cpp`) returns one
  `NAME=value` line, `replay_scenenn` prints it to stderr at startup and the
  ROS node logs it once in its constructor (`scovox_node.cpp:277`), so a
  run log records what the binary *is* rather than only that two builds differ.
  The line names seven switches plus `NDEBUG` (`version.cpp:41-49`); the two
  newest, `SCOVOX_SPARSE_BRANCH_COUNTERS` and `SCOVOX_WALK_MARGIN_VOX`, are not
  on it yet (G15 in `docs/code_review_and_updates.md`).

Note that `-O2` cannot be selected by passing it in `CMAKE_CXX_FLAGS`: cmake
emits `CMAKE_CXX_FLAGS_RELEASE` (`-O3 -DNDEBUG`) after it and the last `-O` on
the line wins. Override `CMAKE_CXX_FLAGS_RELEASE` instead —
`../scovox_slot_rules/scripts/build_flags.sh` (the harness repo, a sibling of
this one) does this and prints each arm's
actual compile line, read back from `build.ninja` rather than the one intended.

#### Voxel types

| header | type | status |
|---|---|---|
| `tsdf_voxel.hpp` | `TsdfVoxel` 8 B | live (TSDF grid) |
| `beta_voxel.hpp` | `BetaVoxel`, `BetaCountU16`, lattice helpers `beta_lattice_snap` / `beta_max_increments`, priors `kBetaOccPrior = kBetaFreePrior = 0.5` (Jeffreys) | live (occupancy grid) |
| `dir_voxel.hpp` | `DirVoxel` (total basis: `s_total`, derived `other()`, `set_other()`), `sparse_add_class`, `dominantClass` | live (semantic grid) |
| `voxel.hpp` | legacy unified `Voxel` (`a_occ,a_free,a_unk,sem_cnt[K],sem_cls[K]`), `sparse_add`, the four `g_sparse_*_count` atomics | legacy: still the projection type for `ScovoxMap` messages and the substrate of `scovox::Map` |
| `sembeta_voxel.hpp` | `SemBetaVoxel` 24 B | legacy: viz/projection only |
| `semantics.hpp` | naive / majority-vote updaters | legacy ablation modes, still selectable via `semantic_mode` |

#### `ScovoxMapSplit` (`scovox_map_split.hpp`) — the three-grid façade

- **Params** (`:40-101`): `resolution`, `inner_bits`, `leaf_bits`,
  `dir_leaf_bits`, nested `tsdf` (`TsdfMap::Params`) and `semsplit`
  (`SemSplitMap::Params`), `fused_walker` (default true, `:60`),
  `tsdf_enabled` (true, `:68`), `far_voxel_fast_paths` (true, `:82`), fine
  band `fine_ratio_log2` 0 / `fine_sdf_trunc_voxels` 3 /
  `fine_region_margin` 0.15 / `fine_anchor_enable` true / `AnchorFitParams`.
- **Ctor** (`:103-151`): copies shared geometry into both sub-maps; aborts if
  `semantic_band_length > 0 && !fused_walker` (`:129-136`).
- **`integrateHitFused`** (`:183-714`): the single exact DDA described in
  §1.3. `band_active` (`:232-234`) requires band > 0, not dynamic, not
  geometry-off, no BKI kernel, and semantic probabilities present. Far-skip
  and far-carve (`:397-401`, `:443-448`) are gated on
  `far_voxel_fast_paths_ && !space_carving && carveFrameOpen()`.
  `exact_body` is `noinline` (`:498-592`) so the shortcuts can be diffed
  against it.
- **`integrateHitSplit`** (`:725-751`): two DDAs; calls `tsdf_.integrateRay`
  gated on `tsdf_enabled_ && !is_dynamic && !geometry_off` (`:744`). Only
  reached when `fused_walker=false`. The `tsdf_enabled_` term was missing until
  `f2d535e`, so a split-path run paid in full for a TSDF the flag declared
  unread; any fused-vs-split comparison taken before that commit with the flag
  off had one walker doing that work and one not.
- **Fine TSDF band** (`fine_ratio_log2 > 0`): a second `TsdfMap` at
  `resolution / 2^k`, written only inside registered refinement cylinders
  (`refinement_regions.hpp`; `RefinementRegion.msg`), with optional per-scan
  2-DoF anchor re-fit. Off in every evaluation path.
- Accessors used by the node and tests: `farSkippedVoxels()`,
  `farCarvedVoxels()`, `exactBodyVoxels()`, `farVoxelFastPaths()`,
  `resetTiming()`, per-walker nanosecond counters.

#### `SemSplitMap` (`sem_split_map.hpp` / `.cpp`) — Beta ∥ Dir substrate

- Owns the persistent Beta and Dir grids, transient Beta/Dir grids (per-frame
  decay, `decayTransient` `:998-1067`), a `fallback_dir_grid_` for
  `ray_spread` mode 4, a `CarveStage` (`carve_stage.hpp`: block-keyed,
  per-voxel max `w_free`) and a `HitStage` + `hit_obs_` pool for
  `batch_hits`.
- **The deposit reads a prepared observation, not a raw softmax**
  (`sem_obs.hpp`). `SemObs` is the dense per-class vector with its zeros
  dropped, its `norm` already applied and its argmax already located, held in
  ascending class order so the deposit sequence a voxel sees is unchanged.
  `prepareRayObs` (`:467-470`) builds it **once per ray** into `sem_obs_`;
  every internal deposit signature then takes `const SemObs&`. The two public
  `integrateHit` overloads still take `const std::vector<float>*` — the
  preparation happens behind them, so the caller-facing contract is unchanged.
  The staging pool `hit_obs_` stores `SemObsEntry` rather than a fixed
  `num_classes`-wide row; because a prepared observation's width varies from
  ray to ray, `HitStage::probs_cap` (`:777`) records what a voxel's block
  actually holds so a restage reuses it instead of appending, and
  `probs_argmax` (`:778`) is carried to the flush rather than recomputed on
  the normalised copy.
- **Params defaults** (`sem_split_map.hpp`): `w_occ` 1.0, `w_free` 0.5,
  `beta_occ_prior` / `beta_free_prior` = `kBetaOccPrior` / `kBetaFreePrior`
  (both **0.5**, Jeffreys, since 2026-09-05), `kappa0` 1.0, `dirichlet_min_p_occ` 0.5, `hit_flat_share` false,
  `evidence_saturation` 0, `class_evidence_saturation` −1,
  `evict_by_confidence` false, `inc_mode` 0, `inc_thresh` 0.10,
  `semantic_spread_radius` 0, `semantic_band_length` 0,
  `semantic_band_require_occ` **true**, `ray_spread` 0,
  `carve_skip_occ_threshold` 0, `batch_free_carve` true, `batch_hits` true,
  `range_decay_length` 50, `num_classes` 14, `alpha_0` 0.01,
  `sem_top_k` **`K_TOP`** (§4.9; `0` disables the cap, and `sanitise` collapses
  a cap at or above `num_classes`, above `SemObs::kMaxTopK`, or negative, to
  `0` rather than leaving a setting that silently does nothing).
- **`sanitise(Params&)`** (`.cpp:267-308`): clamps, snaps `w_occ`/`w_free`
  and `beta_occ_prior`/`beta_free_prior` onto the ⅛ lattice — the prior
  accumulates into the same counters the weights do, so off-lattice it breaks
  the count identity `a_occ = prior + w_occ·n_hit` under `SCOVOX_BETA_U16`; a
  non-positive prior is not a valid Beta and falls back to the shipped
  constant rather than being clamped to an epsilon — clamps `dir_leaf_bits ≤ leaf_bits`,
  enforces band / BKI ball / ray-spread mutual exclusion. A second
  `sanitise(HitWeights&)` snaps the fusion profiles the node builds.
- **Per-ray entry** `integrateHit` (`:372-397`, two overloads) → `carveRay` (`:409-445`,
  staged or direct via `applyCarveUpdate` `:451-486`) → hit staged
  (`batch_hits`) or `applyHitUpdateOn` (`:599-680`) → `commitHit`
  (`:685-753`). Stageable only when no kernel, no spread, no ray_spread.
- **Frame protocol**: `beginCarveFrame` / `flushCarveFrame` (`:499-534`)
  around each scan; `flushStagedHits` (`:536-575`) runs first so occupied
  wins over carve for the same voxel in the same scan.
- **Other deposit modes**: BKI kernel `applyHitUpdateKernel` (`:959-992`,
  `spreadTable` `:933-957`), `raySpreadDeposit` (`:775-844`, modes 1-4),
  `applyBandSemantic` (`:860-907`).
- **Queries / drains**: `getBetaVoxel`, `getDirVoxel`, `dominantClassAt`
  (`:1202-1224`), `drainTouchedBeta` / `drainTouchedDir` (`:1184-1196`,
  sort-unique, swap-scratch).

#### `TsdfMap` (`tsdf_map.hpp` / `.cpp`)

Curless–Levoy over `TsdfVoxel`; `Params::sdf_trunc` 0.15 clamped positive,
`space_carving` false; `integrateRay` (band-only DDA), `applyBandUpdate` (the
per-voxel tail the fused walker calls), weighting factories `constant` /
`linear` / `exponential` / `rangeDecay`, `drainTouched` / `clearTouched`,
`forEachVoxel` (centres, not corners). The header still names the
deleted `SemBetaMap` (`:15-17`, `:49`). The nonexistent
`feedback_slimvdb_memory_measurement.md` citation was removed in `1101e53`;
what remains near `:162` are pointers to `slimvdb_pipelines/*.cpp`, which are
source files, not documents.

#### Wire format (`binary_serializer.hpp`, `lz4_codec.hpp`)

- `BinarySerializer::FORMAT_VERSION = 8` (`:171`; the separate ROS envelope
  `ENVELOPE_VERSION = 5` is at `:179`). Payload: TSDF deltas,
  Beta deltas, Dir deltas, optional fine-TSDF deltas, block-run coordinate
  coding assuming 8×8×8 leaf blocks (`kBlockVoxels = 512`, `:531`, i.e. `leaf_bits = 3`).
- Evidence is u8 sqrt-companded when the sender sets `quant_step =
  evidence_saturation / 255²` (`scovox_node.cpp:2301-2302`); `quant_step = 0`
  keeps f32 payloads. Class ids are u8 when `num_classes ≤ 255`.
  `Frame::quant_step`'s comment still describes the older u16 scheme, in two
  places (`binary_serializer.hpp:40` and `:194`).
- `MAX_NUM_CLASSES` 4096, `MAX_FINE_RATIO_LOG2` 8.
- `lz4_codec.hpp`: 4-byte big-endian original-size header + LZ4 block,
  256 MB decode cap.
- ROS envelope `ScovoxMapBinary`: `version` (the node writes 5,
  `scovox_node.cpp:2510`), `little_endian`, `map_from_source` transform,
  `data`.

#### Everything else in `scovox_core`

| file | role |
|---|---|
| `consensus_merge.hpp` | `mergeBeta` (floors at the prior), `mergeDir` (insertion-sort fold, not fully order independent) |
| `uncertainty.hpp/.cpp` | Beta / Dirichlet entropy and variance helpers (21 tests). **Never names `DirVoxel`** — `estimateDistinctClasses` / `effectiveResidual` are templated on `sem_cnt[K_TOP]` + `a_unk`, fields only `Voxel` and `SemBetaVoxel` carry. It reaches the promoted map anyway: `dscovox_consensus.hpp:78` projects `DirVoxel` → `SemBetaVoxel` (stripping `α₀` and `(C−K)·α₀`), and `dscovox_node.cpp:706` / `scovox_node.cpp:2932` call `argmaxClassConfidence` on the result, so the published `semantic_confidence` **is** a Laplace + Hutter readout of the promoted state — on the basis E12 rejects. See M11 in `docs/archive/code/code_review_2026_09_04.md`. |
| `mesh_labelling.hpp`, `marching_cubes.hpp` | mesh extraction + per-vertex labels for `ExtractMesh` |
| `refinement_regions.hpp`, `dbh_fit.hpp` | fine-band cylinders, 2-DoF anchor fit, DBH fit |
| `carve_stage.hpp` | block-keyed staged carve |
| `e0_counters.hpp` | admission / eviction counters behind `SCOVOX_E0_COUNTERS` |
| `ray_iterator.hpp` | exact DDA iterator (the only traversal since `d5da6a8`) |
| `map_interface.hpp` | `scovox::Params` (the node's map-parameter struct), `SemanticMode`, `HitWeights` |
| `version.hpp` | `0.1.0` |
| `tools/split_memory_demo.cpp` | prints grid memory for the three-grid layout |

### 2.4 `scovox_msgs`

| type | fields |
|---|---|
| `ScovoxMapBinary.msg` | `header`, `version` u8, `little_endian`, `map_from_source` (Transform), `data` u8[] |
| `ScovoxMap.msg` | `header`, `resolution`, `origin`, `occupancy_threshold`, `semantic_threshold`, `max_semantic_classes` u8, `ScovoxVoxel[] voxels` |
| `ScovoxVoxel.msg` | `position` Point32, `a_occ`, `a_free`, `a_unk`, `ScovoxSemanticEvidence[]` |
| `ScovoxSemanticEvidence.msg` | `class_id` u16, `evidence_count` f32 |
| `RefinementRegion.msg` | `id`, `x`, `y`, `base_z`, `radius`, `remove` |
| `GetRegion.srv` | AABB → `ScovoxMap` |
| `GetOccupancyGrid.srv` | `z_min`, `z_max`, `resolution_2d` → `nav_msgs/OccupancyGrid` |
| `ExtractMesh.srv` | `min_weight`, `output_path` → counts + `ply_path` |

### 2.5 `scovox_mapping`

#### `scovox_node.cpp` (`scovox_mapping_node`, class `SCovoxNode`)

**Lifecycle.** Ctor `:67-345` → `declareMapParams` (`:357-457`, fills
`scovox::Params P`) → `declareNodeParams` (`:458-851`) → build
`ScovoxMapSplit::Params SP` from `P` and node members (`:93-148`) →
`buildFusionProfiles` (`:860-903`) → `setupSubscribers` (`:920+`) →
`setupPublishers` (`:997-1036`). `main` (`:3399-3437`) spins a
`MultiThreadedExecutor` with 2 threads; the viz timer sits in its own
callback group; the map is guarded by `map_mtx_` (`std::shared_mutex`).

**The `map_mtx_` contract is a parameter, not a comment.** Six helpers —
`publishScovoxMap`, `publishPointCloud`, `publishTSDFPointCloud`,
`publishFineTSDFPointCloud`, `onRefinementRegion`, `publishBinaryMap` —
deliberately do not take the lock, because the viz timer holds one lock across
four of them and `std::shared_mutex` is non-recursive. Each takes a
`scovox::MapLockHeld&` (or `MapWriteHeld&` where it mutates): a witness with no
public constructor, produced only by `scovox::MapReadLock` / `MapWriteLock`
(`scovox_mapping/include/scovox/map_lock.hpp`), which are the only two ways the
node takes `map_mtx_`. A caller that has not locked cannot name the argument,
so the contract is checked by the shipped gcc build rather than by a reader.

**Map parameters and their `dp()` defaults** (`declareMapParams`):

| parameter | default | line |
|---|---|---|
| `resolution` | 0.10 | 360 |
| `inner_bits` / `leaf_bits` / `dir_leaf_bits` | 2 / 3 / 2 | 361-362, 368 |
| `w_free` / `w_occ` | 1.0 / 2.0 | 369 |
| `kappa0` | 2.0 | 370 |
| `enable_tsdf` / `sdf_trunc_voxels` | true / 3 | 389-391 |
| `semantic_occ_gate` | 0.5 | 396 |
| `batch_hits` | true | 399 |
| `evidence_saturation` | 1000 | 404 |
| `dirichlet_min_p_occ` | 0.5 | 413 |
| `range_decay_length` | −1.0 | 415 |
| `semantic_mode` | "dirichlet" | 418-421 |
| `max_semantic_classes` | 10 | 422 |
| `semantic_evict_by_confidence` | false | 426 |
| `semantic_spread_radius` | 0 | 427 |
| `semantic_band_length` | 0.0 | 428 |
| `semantic_band_require_occ` | true | 429 |
| `min_range` / `max_range` | 0.3 / 10.0 | 416 |

**Node parameters** (`declareNodeParams`, selected): `base_frame` base_link,
`integration_frame` odom, `depth_topic`, `stride` 1 (`:479`), `min_depth` /
`max_depth` 0.1 / 10.0 (`:480`), `trace_no_return_rays` false (`:481`),
`carve_band` −1, `mode` "rolling" (`:488`), `occupancy_vis_threshold` 0.7,
`publish_planning_map` true, the `share_*` wire knobs, `fused_walker` true
(`:710`), `num_classes` 14 (`:717`), `dirichlet_prior` 0.01 (`:718`),
`tsdf_dump_path` "" (`:728`), `deskew_mode` "auto" + IMU knobs (`:737-749`),
`tf_lookup_timeout_sec` 0.2 / `tf_require_exact` false / `rgbd_tf_timeout_sec`
0.2 (`:758-766`), `downsample_voxel_size` **0.5** (`:787`),
`startup_tf_stable_sec` 2.0 / jump thresholds (`:795-807`), the localizer
reject gate (`:818-821`), `topk_probs_dir` (`:829`), `eviction_stats_csv`
(`:833`). `dataset_queue_depth` 1000 (`:980`).

**How `P` reaches the library** (`:93-148`): geometry and TSDF from `P`;
`SP.tsdf_enabled = (sdf_trunc_launch_ > 0)` (`:110`); `w_free`, `w_occ`,
`kappa0`, `carve_skip_occ_threshold`, `batch_free_carve`, `batch_hits`,
`evidence_saturation`, `dirichlet_min_p_occ`, `range_decay_length`,
`semantic_mode` from `P`; `evict_by_confidence`, `semantic_spread_radius`,
`semantic_band_length`, `semantic_band_require_occ`, `num_classes`,
`alpha_0`, `fused_walker` and the fine-band fields from node members.
**Not set anywhere in the node**: `hit_flat_share`, `inc_mode`, `inc_thresh`,
`class_evidence_saturation`, `ray_spread`, `far_voxel_fast_paths` — they stay
at `SemSplitMap::Params` / `ScovoxMapSplit::Params` defaults.

**Input paths.**
- RGB-D: `onImages` (`:1342-1413`) → `DepthSnapshot` →
  `integrateDepthSnapshot` (`:1219-1278`, per-pixel `integrateHit`, no-return
  carve at `:1274` when `trace_no_return_rays`) → `finishScanTail`
  (`:1297-1334`).
- LiDAR: `onPointCloud` (`:1833+`) → `LidarSnapshot` →
  `integrateLidarSnapshot` (`:1545-1830`): gyro deskew (`:1417-1519`
  helpers), optional translation deskew, medoid voxel downsample
  (`:1703-1781`), per-point label / top-k path (`:1782-1825`), one carve frame
  per scan (`:1702`, `:1826`).
- Fusion (`fuse_lidar_rgbd`): both streams, LiDAR profile = global weights,
  RGB-D profile `w_occ 0 / w_free 0 / geometry_off / min_p_occ 0.55` (`:860-903`).
- Gates: `tfGatePass` (`:1110-1152`: startup stability + runtime jump),
  localizer reject gate, `admitFrame` (`:1187`).

**Outputs.** `publishScovoxMap` (`:2095-2137`), `publishBinaryMap`
(`:2216-2635`: change gate, heartbeat, state-flip / binarize modes, chunk
interleave, byte budget), `publishPlanningMap` (`:2644-2770`,
terrain-relative), `publishPointCloud` (`:2779-2965`, 14/16 fields),
`publishTSDFPointCloud` (`:2973-3021`), `publishFineTSDFPointCloud`
(`:3059-3086`), services `GetRegion`, `GetOccupancyGrid`, `ExtractMesh`
(`onExtractMesh` `:3088-3121`, ASCII PLY), `onRefinementRegion`
(`:3033-3055`). Memory / perf line via `scheduleMemUsage` (`:1909-2004`),
optional TSDF dump to `tsdf_dump_path`.

#### `dscovox_node.cpp` (`dscovox_mapping_node`, class `DSCovoxNode`)

One `SourceGrid` per `header.frame_id` (`:84-102`), stored in the map frame
via the carried `map_from_source` — first pose wins and is never refreshed
(`:96-99`, "requires c-slam disabled"). `onBinaryMap` (`:313-554`) decodes
rev-8 frames, checks envelope version 5 and endianness, snapshot-replaces the
source grid, then reset-and-refolds each touched cell into
`split_fused_beta_` / `split_fused_dir_` in sorted source-key order
(`dscovox_consensus.hpp`: `isPriorBeta/Dir`, `refoldBeta/Dir`,
`projectBetaDirTo*`). The fused grids are built from `scovox::Params`
defaults, so the Dir grid uses `leaf_bits` 3 rather than `dir_leaf_bits` 2.
Outputs: `publishPointCloud` (`:623-730`, 11 fields), `publishFusedMap`
(`:834-851`, latched `ScovoxMap`), `GetRegion` (`fillRegion`),
`GetOccupancyGrid` (`occupancyGridOnGrid` `:858-919`). Parameters:
`occupancy_vis_threshold` 0.7, `semantic_occ_gate` 0.5, `publish_rate_hz`
1.0, `pointcloud_min_interval_s` 0.1, `share_roi_z_*`.

#### `scovoxmap.hpp/.cpp` (`scovoxmap` library)

Legacy `scovox::Map` over unified `Voxel`; the only consumer of
`range_decay_length` as an exponential weight (`scovoxmap.cpp:72-73`,
`:367-368`). Not instantiated by either node's mapping path; exercised by
`test_beta_update` and `test_consensus`.

`integrateRay`'s two branches weight the carve differently, and both are
deliberate. The **dynamic** branch calls the two-argument `carve_free`, which
derives `range_w = exp(-|hit − origin| / range_decay_length)` itself from the
geometry it was handed. The **static** branch runs `fused_integrate_ray_static`,
which carves with `carve_w = range_w` — the value the *caller* supplied
(`scovoxmap.cpp:260`), defaulting to `1.0f`, because the node computes that
weight once from the full sensor→hit range and hands it down. Identical
geometry therefore deposits different free mass on the two paths: a 3 m ray
carves `1.0` per voxel when static and `exp(-0.6) = 0.5488` when dynamic. This
is the only place in the tree where the same call weights the same ray two
ways.

#### Launch files (`src/scovox_mapping/launch/`)

| file | map defaults it sets |
|---|---|
| `scenenn_eval.launch.py` | res 0.05, stride 2, depth 0.4–4.0, NYU 41 classes, `w_occ` 6.0, `w_free` 1.0, `kappa0` 2.0, `enable_tsdf` true, `fused_walker` true, `batch_hits` true; no band, no evict-by-confidence |
| `scenenet_eval.launch.py` | res 0.05, stride 1, depth 0.1–10, 14 classes, `w_occ` 6.0, `kappa0` 2.0, `semantic_evict_by_confidence` arg default "false", **no `semantic_band_length` arg at all** |
| `scenenet_eval_fusion.launch.py` | as above with `semantic_occ_gate` 0.6; fusion profiles |
| `semantickitti_eval.launch.py` | res 0.05, `carve_band` 0.1, `w_occ` 6.0, `kappa0` 2.0, 20 classes, range 5–30, depth 1–80, `evidence_saturation` 1000 |
| `scovox_single_robot.launch.py` | res 0.10, `w_occ` 2.0, `kappa0` 2.0, `evidence_saturation` 1000, `max_semantic_classes` 10, stride 1 |
| `scovox_multi_robot.launch.py` | wrapper over the single-robot launch |
| `lidar_mapping.launch.py` | params file driven (`config/*.yaml`) |
| `dscovox_single_robot.launch.py`, `dscovox_multi_robot.launch.py` | LiDAR: res 0.10, `w_occ` 8 / `w_free` 4, `enable_tsdf` false, `mode` rolling, `downsample_voxel_size` 0.1 |

#### Config files

`config/` (repo root): `scovox_lidar_raw_deskew.yaml`,
`scovox_lidar_geometric.yaml`, `scovox_fused_lidar_rgbd.yaml`,
`scovox_robot_share.yaml`, `scovox_fine_band.yaml`,
`exploration_fused_bag.yaml` — all LiDAR-centric (`w_occ` 8, `w_free` 4, res
0.10, `enable_tsdf` false). `src/scovox_mapping/config/`:
`default_params.yaml` (parameter reference mirroring the node's `dp()`
defaults), `lidar_mapping.yaml`, `dscovox_params.yaml`, `scovox_bin_min.yaml`.

#### Tests

`scovox_core` (11 binaries): `test_carve_stage`, `test_mesh_labelling`,
`test_fine_tsdf`, `test_binary_serializer`, `test_consensus_merge`,
`test_tsdf_map`, `test_voxel_layouts`, `test_scovox_map_split` (incl. the
far-path bit-identity suite `ScovoxMapSplitFarCarve` and the
`ScovoxMapSplitTsdfDisabled` pair that guards the `tsdf_enabled=0` walk), `test_sparse_add`,
`test_uncertainty`, `test_sem_split_map`. `scovox_mapping` (10 — nine in
`SCOVOX_TEST_SOURCES` at `scovox_mapping/CMakeLists.txt:73-83` plus the separate
`ament_add_gtest(test_topk_provider …)` at `:96`, which is why a reader counting
only the list gets 9 and this document previously said 8):
`test_beta_update`, `test_consensus`, `test_topk_provider`, `test_tsdf_band`,
`test_marching_cubes`, `test_semantic_audit`, `test_dirichlet_update`,
`test_heartbeat`, `test_publish_gate`, `test_map_lock`. Last recorded run (archived `storage_defaults_2026_09_04.md`,
`dir_total_basis_2026_09_04.md`): **180/181 under `./dev.sh test`** — the
archive attributes that figure to the core-only path, so it is a `scovox_core`
count, not a whole-tree one. The one failure is
`ScovoxMapSplitFarCarve.FarCarveBitIdenticalToFullWalk`
(`test_scovox_map_split.cpp:887`). For the current whole-tree figures see §3.6.

### 2.6 `seg_pipeline`

Python node `seg_node.py`: subscribes colour (compressed) + depth + camera
info, runs Mask2Former (Mapillary Vistas) and publishes a colour-coded
segmentation image, a re-stamped depth image and camera info on
`/scovox/segmentation/colored`, `/scovox/depth/image_raw` (`:67-72`,
`:112-118`). `outdoor_palette.py` maps Mapillary to the 14 outdoor classes.
Runs in its own container (`src/seg_pipeline/Dockerfile`, `compose.yaml`).

### 2.7 The three parameter structs (why defaults disagree)

| field | `scovox::Params` (`map_interface.hpp`) | `SemSplitMap::Params` | node `dp()` | replay / best method |
|---|---|---|---|---|
| `w_occ` | 2.0 | 1.0 | 2.0 | **1.5** |
| `w_free` | 1.0 | 0.5 | 1.0 | 1.0 |
| `kappa0` | 2.0 | 1.0 | 2.0 | **1.0** |
| `evidence_saturation` | 1000 | 0 | 1000 | **0** |
| `range_decay_length` | 5.0 | 50 | −1.0 | 50 (inert) |
| `dirichlet_min_p_occ` | 0.5 | 0.5 | 0.5 | 0.5 |
| `semantic_band_require_occ` | — | true | true | **false** |
| `evict_by_confidence` | — | false | false | **true** |
| `semantic_band_length` | — | 0 | 0 | **0.10** |
| `resolution` | 0.05 | — | 0.10 | 0.05 |

The node copies from `scovox::Params` into `SemSplitMap::Params`, so the
node's values win at runtime; `SemSplitMap::Params` defaults are only in
force for direct library users (tests, the replay's untouched fields).

---

## 3. Where the code has not caught up to the best method

### 3.1 Node defaults (`scovox_node.cpp`, `declareMapParams` / `declareNodeParams`)

| knob | node default | best method | line |
|---|---|---|---|
| `resolution` | 0.10 | 0.05 | 325 |
| `w_occ` | 2.0 | 1.5 | 334 |
| `kappa0` | 2.0 | 1.0 | 335 |
| `semantic_band_length` | 0.0 (band off) | 0.10 | 393 |
| `semantic_band_require_occ` | true | false | 394 |
| `semantic_evict_by_confidence` | false | true | 391 |
| `evidence_saturation` | 1000 | 0 | 369 |
| `max_semantic_classes` | 10 | 14 (`num_classes` is already 14 at `:717`) | 422 |
| `enable_tsdf` → `tsdf_enabled` | true | 0 (evaluation) | 354-357, 108 |
| `stride` | 1 | 2 | 444 |
| `min_depth` / `max_depth` | 0.1 / 10.0 | 0.4 / 4.0 | 445 |
| `trace_no_return_rays` | false | carve-no-return on | 446 |

Commit `e316f07` ("Put the promoted configuration into the code") changed the
library and the replay; the node's `dp()` defaults above are unchanged from
before it. Anyone launching `scovox_mapping_node` without a params file gets a
band-off, evict-by-evidence, `w_occ` 2.0 mapper that no published number
describes.

**Addressed in `33aa576`, without moving these defaults.** They are shared with
the LiDAR deployments, which run a coarser, more confident sensor model on
purpose (`lidar_mapping.yaml` derives `w_occ: 8.0` from `prob_hit ≈ 0.9` at
resolution 0.10), so re-tuning them here would silently re-tune those. The
promoted RGB-D configuration is a file you load instead:

```
ros2 run scovox_mapping scovox_mapping_node --ros-args \
    --params-file src/scovox_mapping/config/scovox_best_method.yaml
```

Every value in it was transcribed from the replay `Args` and checked against
that struct line by line. The table above therefore still describes the *bare*
node and is still the reason the config file has to exist — it is not stale.

To verify the file took effect, read the node's log rather than the launch
arguments: `scovox_node.cpp:336-343` prints a `deposit config:` line beside the
existing TSDF line at `:298` (and a `semantic deposit:` line at `:310-318`), all
read out of the **constructed map**. A binary
that predates a knob accepts the parameter and ignores it, and only the readback
shows that.

### 3.2 Knobs the node cannot set at all

`hit_flat_share`, `inc_mode`, `inc_thresh`, `class_evidence_saturation`,
`ray_spread` (`SemSplitMap::Params`) and `far_voxel_fast_paths`
(`ScovoxMapSplit::Params`) have no `declare_parameter`. Their defaults happen
to equal the best method, so the node is correct by accident; none of the
ablation arms behind them can be reproduced through ROS. The node prints
`far_voxel_fast_paths` (`:298`) as if it were configurable.

`class_evidence_saturation` is the one that bites. Its default is −1, meaning
"share whatever `evidence_saturation` is", while the promoted replay passed 0
explicitly. Those agree **only** while `evidence_saturation` is 0 — so raising
`evidence_saturation` in `scovox_best_method.yaml` does not cap one channel, it
caps both, and the class cap is the one that moves semantics. Capping occupancy
alone requires exposing the parameter first. The trap is written into the yaml
at the line where someone would spring it.

### 3.3 No readout policy in the node

`--dump-label-gate 0.5 --dump-below-gate-as-unknown` are replay-side. The node
has `occupancy_vis_threshold` 0.7 for visualisation and `semantic_occ_gate`
0.5 for the wire, neither of which is the "score below-gate as unknown" rule.
If a node-produced map is ever scored, the dump policy must be re-implemented
in the consumer.

### 3.4 Launch and config files

- `scenenet_eval.launch.py` and `scenenn_eval.launch.py` default `w_occ` 6.0
  and `kappa0` 2.0, never pass a band, and default evict-by-confidence off;
  `scenenet_eval.launch.py` has no `semantic_band_length` argument to pass.
- `scovox_single_robot.launch.py` hard-codes `w_occ` 2.0, `kappa0` 2.0,
  `evidence_saturation` 1000, `max_semantic_classes` 10.
- `default_params.yaml` documents the node defaults (so it is internally
  consistent with §3.1) and describes `range_decay_length` as exponential
  range weighting, which the split path never applies (the node uses it only
  as the on/off switch for the range cull, `:1271`, `:1578`;
  `lidar_mapping.yaml` was already corrected to say so in the uncommitted
  diff).
- The LiDAR configs (`w_occ` 8, `w_free` 4, res 0.10) are a different
  operating point by design, not a gap; they are listed so nobody reads them
  as the best method.
- `scovox_best_method.yaml` (added `33aa576`) is the promoted RGB-D
  configuration, under a `/**:` wildcard so it applies at any node namespace or
  name. The launch files above still do **not** load it — passing
  `--params-file` is manual for now, and wiring it into the RGB-D launch files
  is the remaining half of §3.1.
- The yaml lists `batch_hits: true` explicitly even though it is already the
  default in all three places (library, node, replay). It is spelled out because
  it is the largest single lever on the numbers in §1.5 and the one most likely
  to be flipped by someone who reads it as a speed knob: it sets whether a unit
  of evidence is an observation or a pixel.
- **Three RGB-D operating points are recorded, not one.** `batch_hits` and
  `w_occ` are a single axis, so the pair is fixed together in each file, and the
  three differ from one another in exactly one key each (asserted by comparing
  the parsed parameter maps — the key *sets* are identical):

  | file | `batch_hits` | `w_occ` | evidence unit |
  |---|---|---|---|
  | `scovox_best_method.yaml` | true | 1.5 | observation, uncompensated — promoted |
  | `scovox_rgbd_complete.yaml` | true | 6.0 | observation, compensated |
  | `scovox_rgbd_unbatched.yaml` | false | 1.5 | depth pixel |

  The promoted file is unchanged and remains the default answer; the other two
  exist because §1.5 shows the axis has no free direction, so the choice is an
  application decision rather than a tuning result. `w_occ` 6.0 is a whole
  eighth, which is required — `beta_lattice_snap` silently moves anything else
  and the uint16 and float builds then diverge.
- `scovox_node` logs `batch_hits` in its `deposit config` line. Without it the
  un-batched file — which differs from the promoted one in that key alone —
  produces log output identical to the promoted configuration, so a binary that
  ignored the parameter would be indistinguishable from one that honoured it.

### 3.5 Code defects that touch the method

- `integrateHitSplit` ran the TSDF DDA regardless of `tsdf_enabled` until
  `f2d535e` (2026-09-04); the gate now sits at `scovox_map_split.hpp:744`.
  Invisible under the default `fused_walker`, but it invalidates any
  fused-vs-split timing A/B run at `tsdf_enabled=0` taken before that commit.
- `range_decay_length` is written into `SemSplitMap::Params` and read by
  nothing in the split path; the E7 "swept, no effect" result for it is
  therefore not a measurement.
- `evidence_saturation` caps Beta and, through `class_evidence_saturation
  = −1`, Dir as well; the node exposes only the shared value.
- One known failing test, `FarCarveBitIdenticalToFullWalk`. Re-measured on the
  full `./dev.sh ros-test` gate. Counted from the gtest XML that gate leaves in
  `.build/ros/build/*/test_results/` (`scovox_slot_rules/docker/count_tests.py`): **327 cases,
  1 failure** — `scovox_core` 184 / 1, `scovox_mapping` 143 / 0, `scovox_msgs` 0.
  Earlier revisions of this document said "325 tests, 2 failures"; the 2 was one
  failure counted twice, because colcon reports both the gtest case and the
  package's CTest aggregate, and the total was a colcon-console reading rather
  than an XML count. It is still the only failure.
- The numbers in §1.5 predate three commits to the deposit path and no longer
  describe this code. See the box in §1.5 and `code_review_2026_09_04.md` §H4.
  The box now carries what the promoted arm *does* score on the current binary,
  measured on all eight scenes (`cells/goal8.tsv`, 2026-09-05), so the size of
  the gap is no longer an open question. Promoting those figures to the headline
  table is still not a doc edit: the campaign selected the arm on the old deposit
  rule, and a re-base means re-running the ablation ring against the new one.

### 3.6 Stale text inside the code

`sem_split_map.hpp:8,10` ("8 B" Beta, "16 B" Dir, and `SemDirMap` at `:5`);
`sem_split_map.cpp:705` ("16 B DirVoxel"); `dir_voxel.hpp:4,26` ("16-byte",
"16 B at K_TOP=2" — corrected further down at `:99`); `beta_voxel.hpp:17-20`
("16 B DirVoxel"); `tsdf_voxel.hpp:11` (`SemBetaVoxel`);
`binary_serializer.hpp:40,194` (u16 quantisation); `wire_study.py:1-12` (v5
layout, 28 B Dir records).

The `.md` citations are a separate matter. `1101e53` swept `src` and
`README.md` under the standing rule that code comments cite no document paths
and carry no experiment results — those live in memory and `REVIEW_LOG.md` —
but it did **not** finish `config`: three YAML comments survived it
(`scovox_fine_band.yaml:9`, `scovox_lidar_raw_deskew.yaml:12`,
`scovox_lidar_geometric.yaml:17`), and all three pointed at paths that no longer
resolved. Repaired 2026-09-05: two now cite their `docs/archive/` locations and
the third states the measurement instead of naming a deleted file. `src` is
clean — `grep -rnE '[A-Za-z0-9_/.-]+\.md' --include=*.cpp --include=*.hpp src`
returns nothing outside `third_party`. Roughly 50 citations remain in
`scovox_slot_rules/scripts/*.py|*.sh` and are audited but not edited:
`DESIGN.md` ×12 (never existed in either repo's history — deletion may be the
right fix, not a repoint), `FINDINGS.md` ×9 and `PLAN.md` ×5 (simple
`archive/` prefix edits). See `docs/archive/code/code_review_2026_09_04.md` §L2.

### 3.7 What already matches

Compile defaults (`K_TOP` 2, `TRACK_QMAX` 1, `TRACK_NHIT` 0, `BETA_U16` 1,
`EVICT_INHERIT` pinned to 0), `fused_walker` and `far_voxel_fast_paths`
defaults, `batch_hits` / `batch_free_carve`, `dir_leaf_bits` 2, `alpha_0`
0.01, `dirichlet_min_p_occ` 0.5, `num_classes` 14, `semantic_mode`
Dirichlet, the exact DDA as the only traversal, the `DirVoxel` total basis,
and every replay `Args` default. The library, built as-is, is the best
method; only the ROS surface around it is behind — and since `33aa576` that gap
is bridgeable by loading `config/scovox_best_method.yaml`, with the node's
`deposit config:` readback (`:336-343`) to prove it took.

Two caveats on "the library is the best method". First, "best method" here means
the configuration the campaign selected, which is not the same as the
configuration the published numbers were measured under: see the box in §1.5.
Second, the storage state is verified by `./dev.sh ros-test` (327 cases), not by
`./dev.sh test` — the core-only path cannot see `scovox_mapping`'s 143 cases at
all, and a storage change graded only by it will pass while broken.

---

## 4. What the best method costs (measured 2026-09-06)

§1–§3 describe what the code computes. This section records what it costs, so a
reader weighing a change against "faster without losing mIoU" starts from
measured proportions rather than intuition. Method, statistics and caveats are
in `scovox_slot_rules/REVIEW_LOG.md` E-W18; the numbers are quotable because
`--tsdf-enabled 1` was proven to leave the semantic dump byte-identical (scene
016 at tsdf 0, tsdf 1 and the goal8 reference all md5
`483483456709d3ff76932aaeda009ac1`).

Eight SceneNN scenes, single-threaded, both mappers instrumented inside their
own process by one shared header. Mean of 8:

| | scovox tsdf on | scovox tsdf off | SLIM-VDB 0.04/800 | SLIM-VDB 0.10/20 |
|---|---|---|---|---|
| mapping, ms/frame | 164.2 | 148.3 | **19.9** | 35.8 |
| grid memory, MB | **21.8** | 17.9 | 40.0 | 42.4 |
| peak RSS @ integration, MB | **45.4** | 33.2 | 54.6 | 57.0 |

Scovox is **4.6–8.3× slower and holds half the map**, both unanimous over the
eight scenes (p 0.0078, the exact signed-rank floor at n = 8).

### 4.1 Where the time goes

Scene 016, `--tsdf-enabled 1`, per-voxel traversal accounting:

| term | voxels | share |
|---|---|---|
| `fused_far` (all of it carve) | 3 216 401 557 | 49.5% |
| `carve_dda` | 1 708 177 814 | 26.3% |
| `fused_exact` | 1 568 465 199 | 24.2% |
| `band_dda` | 0 | — |
| **total** | **6 493 044 570** | |

**Free-space carve is 75.8% of all traversal.** `skip` reads 0 and structurally
must: the fused walker's far-voxel skip requires `!batch_free_carve`, and the
shipped pipeline batches, so its counterpart `far_carve` takes every one of
those 3.2 B voxels instead. That is not a bug and should not be "fixed" by
reading the counter as a failure.

### 4.2 The two mappers build the same surface

Scovox's Dirichlet voxel count against SLIM-VDB's at `sdf_trunc` 0.10, per
scene: −1.63, −2.11, −0.45, −0.15, +0.24, +2.33, −0.10, +0.82 %. All eight
within ±2.4%. What scovox additionally holds is the Beta free-space grid —
1.3–3.5 M voxels against 53–150 k semantic, a factor of 15 to 39 — which
SLIM-VDB does not build at all (`space_carving` off means `VDBVolume.cpp:176`
starts each ray at `depth − sdf_trunc`).

Per ray, scovox costs 2.115 µs against SLIM-VDB's 0.302/0.545 µs while walking
an estimated 12–31× more voxels, so **the walker is roughly 3–4× cheaper per
voxel than the baseline's and loses on volume, not on efficiency.** Micro-
optimising the walk is therefore the small lever; the free-space model is the
larger one — but **not the whole gap.** §4.4 removes free-space carve from the
pipe entirely, traversal included, and finds SLIM-VDB still 1.92x faster at a
matched band. Read the 75.8% as a share of *voxels*, which is what it measures,
and not as a share of time. **§4.8 measures the share of time** — with
`perf` and with flag ablation — and it is 38.1% of instructions and 55.8%
of wall clock, not 75.8%.

### 4.3 What must not be quoted

`s_wall` excludes argument parsing and the trajectory read. `s_dump` also spans
the `--slots` and `--e0-counters` side files, so it varies with flags. The
whole-process RSS peak (127.4 MB) measures the driver's dump buffer — one record
per voxel, 2.4 M of them — not the map. Quote `s_map` and `peak_rss_kb_map`.

### 4.4 What removing free-space carve is actually worth (measured 2026-09-06)

§4.1 found free-space carve to be 75.8% of all voxel traversal. This section
records what happens when it is removed from the pipe — not just its write, but
its traversal — so that scovox walks the same `depth ± trunc` window SLIM-VDB
walks. Method, gates and statistics are in `scovox_slot_rules/REVIEW_LOG.md`
E-W21; all seven arms ran back to back in one session, with the scovox
carve-off arms bracketing the SLIM-VDB pair so drift is shared rather than
loaded onto one mapper.

`--w-free 0` alone does not do this. `applyCarveUpdate` returns on `w_inc <= 0`
so the write disappears, but `walk_back = max(depth, trunc)` never reads
`w_free`, so the walker still traverses the whole ray to write nothing. The
carve-off pipeline seeds the exact DDA a short window in front of the surface
instead — see §1.3 for why seeding is a true suffix of the same walk where
shortening `k_far` would not be.

Frames per second (`frames / s_map`), mean of the eight scenes:

| shipped | carve write off | carve off the pipe | ditto, band 0.10 | SLIM-VDB 0.10 | SLIM-VDB 0.04 |
|---|---|---|---|---|---|
| 6.56 | 9.34 | **14.68** | **16.44** | 31.52 | 56.05 |

| comparison | ratio | scenes | p |
|---|---|---|---|
| short walk vs full walk, carve off | 1.574x | 8/8 | 0.0078 |
| carve-off pipeline vs shipped | 2.242x | 8/8 | 0.0078 |
| **SLIM-VDB vs carve-off scovox, band matched at 0.10** | **1.919x** | 8/8 | 0.0078 |

**Traversal is not the remaining gap.** Removing the carve cuts voxels
traversed 5.04x (1 509 616 973 to 299 285 094 on the 300-frame gate; `carve_dda`
407 777 267 to 0) and walker time only 1.54x; nanoseconds per traversed voxel
rise 19.52 to 66.13. The carve's voxels were the cheap ones, a Beta miss
increment on a grid with no semantics. With the free-space model gone entirely
and the band matched, SLIM-VDB is still 1.92x faster — 60.8 ms/frame against
31.7. The residual is per-surface-voxel model cost: the Dirichlet deposit —
established by ablation in §4.5, which is where that claim's evidence lives.

Memory runs the other way, and by more:

| mean of 8 | shipped | carve off | band 0.10 | SLIM-VDB 0.10 |
|---|---|---|---|---|
| grid MB | 21.8 | **10.9** | **10.9** | 42.4 |
| peak RSS @ mapping, MB | 45.4 | 31.4 | **28.8** | 57.0 |

**This is a switch, not a default.** At the shipped readout the carve-off map
loses intersection mIoU −0.0417 (0/8, p 0.0078) and union −0.0258 (3/8,
ambiguous). E-W19c shows most of the intersection number is the readout gate
rather than the model — matched to the shipped map's occupied count it falls to
−0.0106, 2/8, CI straddling zero — but ambiguous-at-a-0.99-threshold is not
"fast without losing mIoU", so the default keeps the carve.

### 4.5 Why scovox is slower than SLIM-VDB (measured 2026-09-06)

§4.4 asserts the residual gap is the per-surface-voxel deposit. That assertion
had a confound. The carve-off seed does not start at the deposit window; it
starts `kWalkMarginVox = 2.7320508` voxels in front of it, two half-diagonals
plus a voxel of headroom, so the walk is a guaranteed superset of every voxel
the deposit window can touch (§1.3). At band 0.10 that margin is **38% of every
ray**:

| band-matched arm, res 0.05, trunc 0.10, band 0.10 | m | voxels |
|---|---|---|
| useful front window | 0.1250 | 2.50 |
| exactness margin — traversed, never deposits | 0.1366 | **2.73** |
| back window | 0.1000 | 2.00 |
| **scovox span** | **0.3616** | **7.23** |
| **SLIM-VDB span** (`depth ± trunc`) | **0.2000** | **4.00** |

1.808x the span against a 1.919x time ratio is what a traversal-bound mapper
looks like, so §4.4's conclusion could not stand on the E-W21 arms alone.

E-W22 settles it. `SCOVOX_WALK_MARGIN_VOX` is a guarded macro — `#ifndef` /
`#define`, never bare, because an unpassed `-D` compiles to 0 and would silently
break the suffix — and the default build is md5-identical to the binary E-W21
timed, so the refactor is provably inert. A `margin0` build sets it to 0 purely
to price those voxels. **It is a timing probe, not a candidate**: it destroys
the suffix guarantee and its maps DIFFER from the control on all three scenes.

Three scenes, 8200 frames, scovox arms bracketing SLIM-VDB:

| | span (vox) | FPS | ms/frame |
|---|---|---|---|
| control (real binary) | 7.23 | 14.78 | 67.7 |
| margin removed | 4.50 | 15.99 | 62.5 |
| SLIM-VDB @ 0.10 | 4.00 | 27.90 | 35.8 |

**Cutting 38% of every ray's traversal buys 8.2% of time**, where a
traversal-bound mapper would give 1.61x. At a near-matched span — 4.50 against
4.00 — SLIM-VDB is still **1.745x** faster.

The margin voxels reach `exact_body` and fail every deposit gate, so they price
reach-and-gate directly: `(67.7 − 62.5) / (7.23 − 4.50)` = 1.905 ms/frame per
span-voxel. Assuming a useful voxel costs the same to reach and gate as a margin
one, traversal and gating is 4.50 x 1.905 = **8.6 ms of 62.5, or 13.7%**, and
scovox's non-traversal work alone (53.9 ms/frame) is **1.51x SLIM-VDB's entire
frame**. That last step is an extrapolation, but the directly measured 8.2%
already caps how much traversal can be worth.

**What the per-voxel work is.** SLIM-VDB (`VDBVolume.cpp:180-206`): voxel
centre, SDF, one gate, two accessor reads, Curless–Levoy, `alpha[label] += 1`
into a **dense `VecXIGrid<14>`** where `label` is an argmax taken upstream,
three `setValue`s. scovox (`exact_body` → `applyBandSemantic` →
`dirichletUpdate`): coordToPos, norm, dot, degeneracy guard, `trim_tail`,
`applyBandUpdate` into a **separate** TSDF grid, `getOrAllocateDirOn`, then
two passes over the 14-entry probability vector and one `sparse_add_class` — a
`K_TOP` scan with eviction, plus a locked atomic branch counter — per non-zero
class, then `applyDirSaturation` and a `touched_dir_.push_back`.

(An earlier draft of this list also named a per-band-voxel Beta read and a third
`dirichletUpdate` pass. Neither runs: `semantic_band_require_occ` is false —
see §4.2 line 74 — so the band takes the flat-`kappa0` branch, and the argmax
pass is `SCOVOX_E0_COUNTERS`-only. The TSDF write listed here runs in the
benchmark, which passes `--tsdf-enabled 1`, but not in the shipped ROS config,
which sets `enable_tsdf: false`. All three corrections make the deposit
*cheaper* than stated and none touches the 8.2% traversal ceiling.) Measured on the SceneNN `.topk` blobs, the mean number
of non-zero classes per pixel is **2.79**.

So: ~2.8 sparse evicting insertions plus ~42 vector iterations of
normalise/argmax, against one dense integer increment.

**scovox is slow because it maintains a richer model, not because it walks
badly.** Three ablations, each removing a different traversal term, agree:
carve out of the pipe is 5.04x fewer voxels for 1.54x time; the margin is 38% of
the ray for 1.08x; and at matched span the gap is still 1.745x. Speed work
belongs on the deposit — the per-non-zero-class loop, the eviction scan, the
per-deposit atomic branch counters, the `touched_dir_` push — not on the walk.
The branch counters have since been cut (§4.6) and the soft-vs-argmax deposit
has since been priced (§4.7). The rest is untested.

### 4.6 The `sparse_add` branch counters are compiled out (measured 2026-09-06)

**Design choice, and the default is the choice.** `SCOVOX_SPARSE_BRANCH_COUNTERS`
defaults to **0**, so the four process-wide `std::atomic<uint64_t>` counters in
`sparse_add` / `sparse_add_class` are declared but never incremented. Nine sites
switch (`voxel.hpp:172,179,221,224`, `dir_voxel.hpp:356,366,378,430,435`). The
declarations stay unconditional so every reader still compiles.

**Why they were costing anything.** On x86-64 an atomic read-modify-write is a
`lock`-prefixed instruction whatever the memory order — `relaxed` relaxes the
compiler's reordering, not the CPU's. It costs tens of cycles and drains the
store buffer, so it does not overlap with the work either side of it. That was
negligible when a deposit happened once per hit ray, which is what the old
comment at this declaration assumed. It is not the shape of the path now: the
semantic band deposits once per **non-zero class** per **band voxel**, and the
SceneNN `.topk` blobs carry a mean 2.79 non-zero classes per pixel, so a single
ray issues roughly fourteen of these and a frame issues them by the million.

**Why it is inert.** Nothing on any code path reads a counter to make a
decision, and they are members of no voxel struct, so they reach no dump. Proved
twice: the counters-ON build is md5 `8bddc9bbb999`, byte-identical to the E-W22
control binary, so the macro rewrite emits the program it replaced; and scene
016 at full length dumps byte-identically with counters on and off, in both the
shipped and the carve-off configs.

**What it is worth.** Repeated measures on scene 016, carve config, 18 paired
reps with the arm order flipped every rep: **+2.31 ms/frame saved**, 95% CI
[+0.62, +4.00], exact signed-rank p 0.0208, faster in 13/18. Real but small —
3.2% of the 66 ms carve frame and only **1.45% of the 152 ms shipped frame**,
which is under this walker's own ~2% layout noise floor. Against the ~135
ms/frame deficit to SLIM-VDB at its tuned optimum (§4.5) it is not a step
toward parity; it is kept because it is free.

**Two honesty guards, and they are load-bearing.** A counters-off build reads
zero because the increments are gone, not because no slot ever overflowed, and
nothing in the number distinguishes those. So `scovox_node.cpp:840` warns that a
requested `eviction_stats_csv` will be all zeros, and `:3431` prints "not
counted in this build" instead of `0`. `SCOVOX_E0_COUNTERS` forces the gate back
on, because `dumpEvictStats()` reads these four as a cross-check against its own
outcome tally and a cross-check that silently reads zero is worse than none.

Build harnesses that reset and read these globals directly must set
`SCOVOX_SPARSE_BRANCH_COUNTERS=1` for the **whole** build: `sparse_add` and
`sparse_add_class` are inline, so a mixed setting is an ODR violation rather
than a partial measurement.

### 4.7 Argmax on ingest is a runtime flag, and it is worth 3.5% (measured 2026-09-06)

SLIM-VDB's closed-set `Integrate()` argmaxes the per-pixel class distribution
before its timer starts: `alpha[labels[idx] & 0xFFFF] += 1`, one increment per
pixel into a dense grid. scovox deposits every non-zero class — mean 2.79 on
the SceneNN `.topk` blobs — so a ray issues roughly fourteen `sparse_add_class`
calls where SLIM-VDB issues one per surface voxel. §4.5 and §4.6 leave that
asymmetry standing as the last untested part of the gap.

**scovox already implements the same choice.** `SemSplitMap::Params::inc_mode`
(`sem_split_map.hpp:248-263`) selects what one observation deposits:

| value | flag | behaviour |
|---|---|---|
| 0 | `--inc-mode soft` | shipped: every class with `p > 0` gets `class_share * p_i` |
| 1 | `--inc-mode hard` | the argmax takes `class_share` whole; runner-ups stay in `other()` |
| 2 | `--inc-mode thresh` | soft, with a per-class probability floor `inc_thresh` |

The hard branch is `sem_split_map.cpp:193-201`. Every deposit site passes
`params_.inc_mode` — band `applyBandSemantic`, the batched endpoint in
`commitHit`, `raySpreadDeposit`, `applyHitUpdateKernel` — and
`IncMode.AllThreeModesConserveDirMassUnderEviction` covers all three.

**`inc_mode` cannot move occupancy, by construction.** It enters neither
Stream A, nor the carve, nor the `p_occ >= 0.5` label gate. Measured
consequence: the entire occupancy block of the score JSON is bit-identical to
the shipped arm on 8/8 scenes for `hard`, `flat` and `counts` alike. Only the
label on an already-occupied voxel can change.

**What it costs and what it buys.** Semantics, 8 scenes, shipped config: union
mIoU −0.00006 (SE 0.00055, 4/8), intersection −0.00026 (SE 0.00103, 3/8) —
below materiality at the mean, with an interval that still reaches it. Speed,
scene 016 under the §4.6 paired-rep protocol with one binary and two argv,
n=18: **+5.27 ms/frame saved, SE 0.92, 95% CI [+3.33, +7.20], p = 0.0001,
faster in 16/18** — +7.43% of the carve frame, **+3.46% of the 152 ms shipped
frame**, which clears the ~2% layout noise floor.

Why the label barely moves is the same argument that bounds the whole
candidate: `K_TOP = 2` already truncates to two tracked classes, and the
readout argmaxes over those two `cnt` values without consulting `other()`.
A runner-up deposit can only fill an empty second slot, win an eviction, or be
dropped — so most of the soft distribution was already discarded downstream.
`hard` discards it earlier, before the deposit loop pays for it.

**Design choice: `inc_mode` stays 0, and stays out of the ROS interface.** It
is deliberately harness-only — the three config yamls document the default
rather than exposing a parameter — because the measurement above does not meet
the promotion bar. It is faster by a real margin with no *detected* mIoU loss,
but at n=8 the interval cannot exclude a material one, and re-running cannot
tighten it. Exposing a ROS parameter is the prerequisite if that ever changes.

**What this prices.** Adopting SLIM-VDB's exact deposit model closes 3.9% of
the ~135 ms/frame gap at matched accuracy. The number of classes deposited was
never the dominant cost; §4.5's per-surface-voxel model cost and the walk
still hold the rest.

### 4.8 The frame, block by block (measured 2026-09-06)

§4.1 counts voxels. This section counts milliseconds, with two instruments that
answer different questions and are meant to disagree.

Scene 016, 400 frames, shipped config, **run natively on the host** — proved
byte- and timing-equivalent to the container (identical dump md5, 136.3 vs
136.0 ms/frame). Native execution is what makes `perf` available; it is not
installed in the container image.

**Baseline: 154.42 ms/frame** over 5,059,914 voxels walked = 30.52 ns/voxel.

#### `perf` self time — what the CPU is executing (partition, sums to 100%)

| block | self % | ms/frame |
|---|---:|---:|
| free-space carve (`applyCarveUpdate`, `carveRay`, `flushCarveFrame`) | 38.13% | 58.88 |
| fused walker DDA (`integrateHitFused` + per-voxel lambda) | 32.25% | 49.80 |
| semantics — Dirichlet deposit | 13.88% | 21.43 |
| TSDF band | 5.21% | 8.05 |
| input decode (libpng, libz, `TopkImage::fill`) | 2.50% | 3.86 |
| walker timing brackets (clock reads) | 1.48% | 2.29 |
| hit staging map | 1.32% | 2.04 |
| per-pixel driver loop (`main`) | 1.04% | 1.61 |
| other / unattributed | 2.03% | 3.13 |

`SemSplitMap::applyCarveUpdate` alone is **33.76%** of the frame.

#### Flag ablation — what we get back if the mechanism does not exist

4 reps, arm order rotated per rep, paired by rep index.

| mechanism off | flag | ms/frame | saved | 95% CI | walk changed? |
|---|---|---:|---:|---|---|
| free-space / occupancy stream | `--w-free 0` | 68.26 | +86.16 | [+81.27, +91.04] | **yes, −80.2%** |
| semantic band | `--sem-band 0` | 127.80 | +26.62 | [+21.57, +31.66] | no (identical) |
| TSDF band write | `--tsdf-enabled 0` | 146.43 | +7.99 | [+1.58, +14.39] | no (−0.17%) |
| soft → argmax deposit | `--inc-mode hard` | 151.32 | +3.09 | [−2.95, +9.14] | no (identical) |

**Calibration.** For the TSDF band the two instruments agree to 0.8% — perf
8.05 ms, ablation 7.99 ms. That is the check that licenses reading the rest of
the table.

**Read `--w-free 0` carefully.** It is the only arm that changes the walk, and
it changes more than the carve: the far segment stops carving and starts
skipping (5.06× fewer voxels), and the Beta occupancy grid collapses from
545,723 voxels to 10,465. It is a timing probe with a large accuracy
consequence (§4.4, §4.5), not a candidate.

#### The number that ranks the remaining work

Subtracting the carve-off arm from the baseline splits the frame by voxel kind:

| | ms/frame | voxels/frame | ns/voxel |
|---|---:|---:|---:|
| retained (surface voxels) | 68.26 | 999,904 | **68.27** |
| removed (free-space voxels) | 86.16 | 4,060,010 | **21.22** |

**A surface voxel costs 3.22× a free-space voxel.** Free-space voxels are 80.2%
of the walk but 55.8% of the time. Deleting all of them still leaves 68.26 ms
against SLIM-VDB's ~32 ms at a matched band and ~18 ms tuned — so the residual
is the per-surface-voxel model cost, which is what §4.5 argued from ratios and
this prices in nanoseconds.

#### The frame is instruction-bound

`perf stat` over the same run: IPC **2.45**, L1-dcache miss **0.28%**
(130.4 M / 46.8 G), branch miss **0.92%** (250 M / 27.1 G), and **266
instructions per voxel** (156.0 G instructions / 586.5 M voxels), 108.6
cycles/voxel.

Cache and branch prediction are both healthy, so **layout changes,
cache-blocking and prefetching have nothing to recover here.** A candidate has
to remove instructions. `perf annotate applyCarveUpdate` shows where they are:
the function is not inlined, pays a full prologue and a stack-canary load on
~3.7 M calls per frame, and its hot instructions are flat and dominated by
shifts and masks recomputing the block key.

#### The walker's timing brackets are ungated

`scovox_map_split.hpp` contains ten `clk::now()` sites (`clk` is
`std::chrono::steady_clock`, :193) and none is behind a macro — the header's
only `#if` is `SCOVOX_WALK_MARGIN_VOX` (:349). On the shipped fused path each
hit ray pays two clock reads (`integrateHitFused` :194/:695, plus the
degenerate-ray early return :205), each no-return ray two more (`integrateMiss`
:758/:760), and the frame two in `flushCarveFrame` :790/:793; the split
walker's three (`integrateHitSplit` :731/:746/:748) are not reached under the
default `fused_walker`. This is compiled into the ROS node, not only the
replay harness.

At 153,602 clock reads per frame and 16.89 ns per isolated
`steady_clock::now()` on this host, that is 2.59 ms/frame; perf's clock symbols
independently total 1.48% = 2.29 ms/frame. The arithmetic is an upper bound and
perf a lower one, so the cost is **2.3–2.6 ms/frame, 1.5–1.7% of the frame** —
the same class and size as the branch counters §4.6 compiled out, and gateable
the same byte-identical way, since a clock read cannot change a map byte.

Also note what the printed split is **not**: on the fused path
`integrateHitFused` brackets the *whole* ray into `tsdf_ns_` by design
(:694-713), so the log's `walker Xs (tsdf Ys + sem Zs)` is not a TSDF-versus-
semantics split. The `--sem-band 0` ablation demonstrates it directly: it takes
26.6 ms/frame off the frame and every one of those milliseconds leaves the
`tsdf` bucket (50.14 s → 39.67 s over 400 frames) while `sem` does not move
(8.89 s → 9.02 s). Quote `s_map`, not the bracket split.

### 4.9 The ingest cap: `sem_top_k` defaults to `K_TOP` (measured 2026-09-06)

§4.7 asked what a *harder* deposit rule is worth and left `inc_mode` at 0. This
section asks the narrower question that flag does not: a voxel can hold
`K_TOP = 2` classes, so why does ingest offer it more?

It used to offer everything. `dirichletUpdate` walked the whole per-pixel
softmax and handed every class with `p > 0` — mean **2.96 non-zero classes** on
scene 016's `.topk` blobs — to `sparse_add_class`, which fills, matches,
evicts or drops it against two slots. `SemSplitMap::Params::sem_top_k` now
defaults to `K_TOP`, so ingest keeps the `K_TOP` largest and stops.

**Two changes, priced separately.** They carry different evidence and must not
be quoted as one number:

| | change | evidence | scene 016, 400 frames |
|---|---|---|---|
| 1 | the observation is prepared once per ray (`sem_obs.hpp`) | byte-identical | **−7.48 ms/frame, −5.12%**, CI [−7.95, −7.02], 5/5 reps |
| 2 | ingest capped at `K_TOP` | changes the map | **−3.65 ms/frame, −2.63%**, CI [−4.62, −2.67], 5/5 reps |
| | both | | **−11.13 ms/frame, −7.61%**, CI [−12.14, −10.12], 5/5 reps |

Arm means, three arms interleaved inside each rep after a discarded warm-up:
pristine HEAD 146.21 ± 0.28, prepared-but-uncapped 138.73 ± 0.37, shipped
default 135.08 ± 0.87 ms/frame.

**Why the representation is free.** The old deposit re-derived three things at
every voxel a ray touched — an O(C) sum over the positive entries, the `norm`
built from it, and under HARD a second O(C) argmax pass. None depends on the
voxel, and §4.8 counts ~1.17 M voxel visits per frame against ~66 k labelled
pixels, so the work repeated ~17.7× more often than it had inputs. `SemObs`
holds the zeros-dropped, already-normalised, already-argmaxed observation in
**ascending class order** — the order the dense scan visited them in — so the
deposit sequence a voxel sees is unchanged. With `--sem-top-k 0` the new binary
reproduces a `git archive HEAD` build byte for byte, at 400 frames and over the
full 1300-frame scene (`483483456709`), and a cap of 13 — at or above every
observed non-zero count — agrees as the second off position.

**Why the cap is not free by construction.** It is tempting to argue the extra
class was going to be evicted anyway. The code says otherwise:
`sparse_add_class` (`dir_voxel.hpp:313`) fills an empty slot in **arrival**
order, and arrival order is ascending class id, not descending probability, so
an unlikely class can take a slot early and hold it against a likelier one
arriving later. The cap therefore changes *which* class holds a slot, and the
full-scene dump moves accordingly (`2bdf66f53c2c`).

**What that motion costs.** Scene 016, 1300 frames, against `gt_5cm.npz`:
intersection mIoU 0.6070 → 0.6078 (**+0.0008**), union 0.3807 → 0.3811
(**+0.0004**) — both *up*, both below the 0.001 materiality bar. The occupancy
block is bit-identical (IoU 0.5580, precision 0.7587, recall 0.6784), which is
the same structural argument §4.7 makes for `inc_mode`: the cap enters neither
Stream A, nor the carve, nor the `p_occ >= 0.5` gate, so only the label on an
already-occupied voxel can change.

**Scope.** Scene 016 only. Scene 015 (2100 frames) agrees in timing direction
(179.48 → 169.89 ms/frame, −5.3%) but was not scored. One scene cannot produce
a paired verdict, so the honest reading of the mIoU column is "no loss detected
where we looked", not "no loss"; confirming across 8 scenes belongs with the
next full sweep.

**The dropped mass is not renormalized.** `norm` only ever shrinks a sum above
1, so a capped observation reads as "these classes, and I decline to guess
about the rest" and the remainder lands in `other()`. This is
`TopkProvider::truncate`'s existing rule (ROS `semantic_topk_trunc`), applied
where it also shortens the per-voxel deposit loop; renormalising instead would
make a truncated observation read as a *more confident* version of the same
distribution, a claim the sensor did not make.

**The escape hatch and its guard rails.** `--sem-top-k 0` restores the uncapped
ingest and is how byte-identity is checked. `sanitise` collapses to `0` any cap
that is negative, above `SemObs::kMaxTopK`, or `>= num_classes` — a cap the
observation can never reach is indistinguishable from no cap, and a setting
that silently does nothing is worse than one that says so. Both walkers are
capped: `integrateHitFused` prepares once per ray, and the split path prepares
inside `SemSplitMap::integrateHit`. The ROS node inherits the default because
it assigns fields into a default-constructed `SP.semsplit` and never named this
one; `semantic_topk_trunc` (ROS default `0`) now only matters for a cap
*tighter* than `K_TOP`.

### 4.10 Where the fps comparison stands after the cap (measured 2026-09-06)

§4.9 shipped the ingest cap. This is what it leaves against SLIM-VDB. Scene
016, **full 1300 frames**, four arms interleaved in one session with the two
scovox arms bracketing the two SLIM-VDB arms; identical container flags, both
mappers single-threaded.

| arm | ms/frame | fps | grid MB | peak RSS at map, MB |
|---|---:|---:|---:|---:|
| **scovox shipped (capped)** | **150.97** | **6.63** | **11.12** | **24.8** |
| scovox HEAD (uncapped) | 162.74 | 6.15 | 11.12 | 24.8 |
| SLIM-VDB 0.10 / w20 (published) | 36.99 | 27.04 | 34.58 | 48.6 |
| SLIM-VDB 0.04 / w800 (tuned) | 22.03 | 45.43 | 33.49 | 47.1 |

Paired per-rep ratios: SLIM 0.10 is **4.081x** faster [3.970, 4.195], SLIM 0.04
is **6.855x** [6.522, 7.204]. The cap is worth 1.078x [1.041, 1.116] — about
**8% of the ~142 ms gap**, consistent with §4.9's −7.61% at 400 frames.

**Read the two columns together.** scovox holds a third of the grid and half
the mapping RSS, and is the only one of the two that builds a free-space model
at all. The comparison is a trade — ~4.1x the frame time for ~0.33x the grid —
not a deficit, and quoting the fps column alone misstates it. §4.5 locates the
residual in per-surface-voxel model cost, and §4.4 shows that removing the
entire free-space carve still leaves SLIM-VDB 1.919x ahead at a matched band,
so no deposit-side change is going to close an order of magnitude.

**Two protocol facts this run established, both cautionary.** Full-scene and
400-frame numbers are **not** interchangeable: the same uncapped binary is
163.6 ms at 1300 frames and 146.2 at 400, because the map grows. And the
bracketing ordering protects the middle arms while leaving the **last** one
exposed — one rep's final arm ran 21% slow when the host went busy, with a
clean `CPUCHECK` (`real=244.2 user=243.5`) throughout, because a
single-threaded arm can lose turbo and memory bandwidth to other cores without
`real >> user+sys` ever showing it. Check host load as well as CPUCHECK, and
treat the last arm of a bracket as the one to re-run.

### 4.11 The frame, step by step, against SLIM-VDB (measured 2026-09-06)

§4.10 gives one number per mapper. This is the same comparison split into steps
that mean the same thing on both sides, so the question "which step consumes the
time" has an answer rather than an inference. Scene 016, 1300 frames,
`--tsdf-enabled 0` (the shipped ROS setting), `--sem-band 0.10`, ingest capped
at `K_TOP`. ms/frame, mean of 3 reps × 2 passes.

| step | SCovox shipped | SCovox carve off the pipe | SLIM-VDB 0.10 | SLIM-VDB 0.04 |
|---|---|---|---|---|
| load — PNG + `.topk` decode | 4.11 | 4.08 | 3.87 | 3.85 |
| prep — deproject + label | 6.88 | 4.75 | 2.83 | 2.80 |
| **walk — traverse + deposit** | **143.36** | **47.53** | **32.78** | **16.87** |
| drain — touched lists | 0.00 | 0.00 | — | — |
| **map = prep + walk** | **150.23** | **52.28** | **35.61** | **19.67** |
| grid | 11.12 MB | 4.43 MB | 34.58 MB | 33.49 MB |
| peak RSS at map | 24.5 MB | 16.1 MB | 47.3 MB | 46.0 MB |

**Which step is which.** The rows are matched by what the code does, not by
name:

| SCovox | SLIM-VDB |
|---|---|
| `loadDepthPng` / `loadTopk` | the same two functions — one shared harness |
| deproject + `TopkImage::fill` → `SemObs` truncated to `K_TOP` | deproject + `fill` → argmax to one `uint32_t` |
| free-space carve: `applyCarveUpdate` staged per ray, `flushCarveFrame` | **no counterpart.** `space_carving_` is false, so its ray starts at `depth − sdf_trunc` |
| surface walk + Dirichlet/Beta deposit | `openvdb::math::DDA` over `[t0, t1]`, `alpha[label] += 1`, TSDF + weight |
| `clearTouchedSemDir` / `clearTouchedTsdf` | no counterpart |

**How the split is measured, and why not with per-ray clocks.** `bench.hpp`
brackets per frame on purpose — at ~30 ns a read, per-ray bracketing charges
seconds to whichever mapper walks more rays, which is the mapper under
suspicion. So the split reuses `s_walker` (already shipped, per-ray, scoped to
the ray walks plus `flushCarveFrame`) and adds one bracket on the other side:
`s_integrate`, around `vol.Integrate()`, two clock reads per frame. `prep` is
then `s_map − s_walker` and `s_map − s_integrate`, and both cover the caller's
per-pixel work. The per-ray reads of `s_walker` land *outside* its own bracket,
so **the SCovox `prep` column is an upper bound** by the 2.3–2.6 ms/frame the
six ungated `steady_clock::now()` sites in `scovox_map_split.hpp` cost (E-W26);
`walk` is the clean column. `load` agreeing to 6% across four arms is
the harness check that the two drivers have not drifted apart.

**The result.** The free-space carve is `143.36 − 47.53` = **95.83 ms/frame**,
66.8% of the shipped walk and **63.8% of the shipped frame** — the largest
single line item in the mapper, and the one with no SLIM-VDB counterpart. Take
it out of the pipe and the band-matched gap on the step that does have a
counterpart is **1.453x [1.397, 1.568], 3/3 reps** on `walk` and 1.471x on
`map`, with SCovox holding an eighth of the grid.

This is narrower than E-W21's band-matched 1.92x for a configuration reason, not
a disagreement: E-W21 ran `--tsdf-enabled 1` (its 016 arms were 63.62 against
33.32 = 1.910x), which keeps `trunc + h` in `useful_front` and disarms
`trim_tail`. Both are shipped settings; quote whichever matches the config under
discussion.

**Span is context, not a divisor.** Mean depth over 016's used pixels is
1.5154 m, so per ray SCovox walks ≈1.62 m shipped and ≈0.34 m carve-off (0.137 m
of that is `kWalkMarginVox`), against SLIM-VDB's `2·sdf_trunc` = 0.20 m and
0.08 m. Do **not** divide the walk column by these: E-W22 ablated
`kWalkMarginVox` at 38% of the ray for 8.2% of the time, which is exactly the
proof that cost is not uniform along a ray — margin voxels are travel to reach
the deposit window and write nothing.

**The table is a time table, and carve-off is not free.** Same dumps, scored:
intersection 0.60780 → 0.56651, union 0.38108 → 0.32327, occupancy IoU
0.55797 → 0.52039. Nothing here proposes moving the default; 016 is also the
scene that overstates carving ~4x, so those are upper bounds.

**Protocol note that supersedes §4.10's.** Arms ran in **palindromic order**
within each rep (`A B C D D C B A`) so no arm is permanently last — §4.10's
final caution was that the last arm of a forward bracket is the exposed one.
Each arm's rep value is the mean of its two passes. Absolute ms/frame drifted up
~12% across the run as host load rose 1.1 → 1.8, which is why every claim is a
paired ratio or a within-rep difference.

### 4.12 Tuning the carve-off pipeline: three byte-identical wins (measured 2026-09-07)

§4.11 leaves the carve-off arm at 52.28 ms/frame against SLIM-VDB's 35.61 at a
matched band. This section is what happened when that arm was profiled on its
own terms. **1.217x [1.178, 1.267], 3/3 reps, zero accuracy cost — every dump
byte-identical.**

**Why the shipped partition (§4.8) could not be reused.** It splits the
*shipped* frame: carve 55.8 %, band 17.2 %, TSDF 5.2 %, walker clocks 1.48 %.
Under `--w-free 0` the carve block does not exist and the walk is 5.04x
shorter, so every share there is a fraction of a denominator this arm does not
have. Re-profiled with `perf` on a native host build (the container has no
`perf`; the container build of the same source replays to the same dump md5,
which is what makes the host profile admissible), the carve-off frame is:

| block | share |
|---|---|
| `exact_body` — the per-voxel body | 28.1 % |
| the DDA loop itself | 23.3 % |
| Dirichlet deposit (`dirichletUpdate` + `applyBandSemantic` + saturation) | ~11 % |
| Bonxai accessor (`getOrAllocateDirOn`, `getLeafGrid`, root `find`) | 7.7 % |
| `seed_carve_off_walk`, once per ray | 4.5 % |
| **the walker's own `steady_clock` reads** | **4.0 %** |
| `applyCarveUpdate` — **a call that returns immediately here** | 2.6 % |
| `HitStage` staging map + flush + sort | 3.1 % |
| driver: `main` pixel loop + `TopkImage::fill` + PNG/zlib | ~7 % |

**Traversal is 51.4 % of the carve-off frame.** §4.5's conclusion — *stop
optimising traversal, attack per-surface-voxel model cost* — was derived on the
shipped frame, where the carve dominates, and does **not** transport to this
one.

#### The three changes

**(1) `far_thr` carried a `trunc` that no branch could reach.**
`scovox_map_split.hpp:459-502`. The radius outside which the walker takes a
fast path was built from the widest gate that *exists*,
`max(trunc + h, sem_band_)`. But `--tsdf-enabled 0` is the shipped replay
setting and `TsdfMap::sanitise` re-clamps a non-positive `sdf_trunc` back to
0.15 m, so `trunc` survives as 0.15 on a run where the `sdf <= trunc + h`
branch can never fire — inflating the radius from 3 voxels to 5, i.e. **4.6x
the volume** reserved for the expensive exact body. `trim_tail` (`:309`)
already fixed the identical phantom on the **back** end, and `useful_front`
inside `seed_carve_off_walk` already carried the `tsdf_writes ?` guard.
`far_thr` was the last member of that family still stale. Front and back halves
of one window must not disagree about its size.

**`far_skip` and `far_carve` need different radii — this is the load-bearing
detail.** They are different reductions:

| | what it does past the radius | precondition on the radius |
|---|---|---|
| `far_skip` | returns; claims the voxel writes nothing | the widest gate that can **fire on this ray** |
| `far_carve` | **carves** unconditionally | that, **and** `far_thr·res − res ≥ back_reach`, because only a front-of-surface voxel may be carved and the fast path has no `sdf > 0` test |

Tightening the shared value broke the second precondition (`3·0.05 − 0.05 =
0.10 < back_reach 0.15`) and moved the shipped carve-on dump. The tightening is
therefore conditional on `far_skip`; `far_carve` keeps the
widest-gate-that-exists form verbatim. The comment at `:463-481` states both
derivations so the shared name cannot hide them again.

**(2) `carve_off` short-circuits the semantic-carve branch.**
`exact_body` called `SemSplitMap::applyCarveUpdate` for every in-band,
in-front voxel; with `--w-free 0` its first statement is
`if (w_inc <= 0.f) return true;` (`sem_split_map.cpp:436`) — no state read, no
write, and `true`, so `carve_blocked` cannot latch. The call is out-of-line, so
declining to make it is worth more than the test that declines. Inert by that
same test.

**(3) `SCOVOX_WALKER_TIMERS`, in the new `walker_timers.hpp`.** Ten
`steady_clock::now()` reads across five brackets, **two per ray**,
unconditional. The header supplies `walkNow()` / `walkNs()`, which fold to `{}`
and a constant `0` at `=0`, so the brackets keep their shape at every call site
and the optimiser removes them; no grid is touched either way. **Default is 1**
because `scovox_node.cpp:1331-1332` publishes `tsdf_ms` / `sembeta_ms` straight
out of `tsdfTimeUs()` / `semdirTimeUs()`, and a 0-build makes those columns read
`0.00` — which is indistinguishable from "the walk was instant" to anything that
only prints the number. That is why the switch is on the build banner.

#### What it is worth

016, 1300 frames, four arms in palindromic order, 3 reps kept after a discarded
warm-up. ms/frame, mean of 3 reps × 2 passes:

| arm | load | prep | walk | **map** | **fps** |
|---|---|---|---|---|---|
| HEAD | 4.04 | 4.70 | 46.97 | 51.67 | 19.35 |
| + `far_thr` + carve elide | 4.00 | 4.59 | 42.27 | 46.86 | 21.34 |
| + timers off | 4.00 | — | — | **42.50** | **23.53** |
| SLIM-VDB, band 0.10 | 3.93 | 2.86 | 33.11 | 35.97 | 27.80 |

| paired ratio, geomean [min, max], n=3 | |
|---|---|
| `far_thr` + carve elide, on `s_map` | 1.104x [1.069, 1.156] |
| the same, on `s_walker` | 1.113x [1.079, 1.165] |
| timers off, further | 1.102x [1.096, 1.109] |
| **the whole change** | **1.217x [1.178, 1.267]** |
| **gap to SLIM-VDB 0.10: before → after** | **1.438x → 1.182x [1.178, 1.189]** |

`prep` and `walk` are `-` on the timers-off arm because `prep` is derived as
`s_map − s_walker` and `s_walker` is 0 by construction there; that arm is
compared on `s_map` alone.

Footprint is untouched and printed as a check: grid **4.43 MB**, peak mapping
RSS **~16 MB** on all three scovox arms, against SLIM-VDB's 34.58 MB and
47.3 MB. The carve-off arm now runs within **18 %** of SLIM-VDB while holding
an eighth of the grid bytes and a third of the peak mapping RSS.

#### Proof, and what "byte-identical" needed

Four 400-frame gates on 016, all green, re-run after a later header refactor:
`--w-free 0` → `8b1b82740062` and `--w-free 1.0` → `161ec60dd513`, on the
candidate with and without the timers, matching §4.11's gates B and C exactly.

Identity is the *absence* of a difference, so the gates were paired with
counters that must **move**:

| carve-off counter | HEAD | candidate |
|---|---|---|
| `total` traversed | 336 704 713 | 336 704 713 — unchanged |
| `fused_far` (skips) | 10 344 101 | 94 431 255 — **9.13x** |
| `fused_exact` (float bodies) | 326 360 612 | 242 273 458 — **−25.8 %** |

Same traversal, a quarter fewer expensive bodies. The carve-on counters are
bit-for-bit unchanged, which is the positive evidence that the shipped fast
path was not disarmed.

**mIoU is unchanged by construction** — the dumps are the same bytes. The
carve-off *configuration* still carries §4.11's price against the shipped
pipeline (016: intersection −0.0413, union −0.0578, occupancy −0.0376), and
nothing here proposes moving that default.

#### Diagnostics: `buildSwitches()` now names all ten switches

`version.cpp` reported seven of nine; this work added a tenth
(`SCOVOX_WALKER_TIMERS`) and closed the gap. `SCOVOX_WALK_MARGIN_VOX`'s
`#ifndef` block was hoisted out of a function body to file scope in
`scovox_map_split.hpp` so `version.cpp` can include it, and
`SCOVOX_SPARSE_BRANCH_COUNTERS` came free from an include already present. The
banner is a `.rodata` string literal, so no map byte can move, and the gates
were re-run to say so. The line now reads:

    scovox 0.1.0 K_TOP=2 BETA_U16=1 BETA_U16_SCALE=8 TRACK_QMAX=1 TRACK_NHIT=1
    DEPOSIT_TRACE=0 E0_COUNTERS=0 WALKER_TIMERS=1 SPARSE_BRANCH_COUNTERS=0
    WALK_MARGIN_VOX=2.7320508f NDEBUG=1

The drift audit is `grep -rn '^#ifndef SCOVOX_' src/scovox_core/include`
against that line. This is not hypothetical bookkeeping: §4.11's `fullwalk`
control (`-DSCOVOX_WALK_MARGIN_VOX=1e9f`) and §4.6's counters build both used
to print the same banner as the shipped default.

#### Levers seen and not taken

Measured only as profile shares, never A/B'd, all on the carve-off arm: the
`HitStage` staging map at 3.1 % (this is review item C3); `seed_carve_off_walk`
at 4.5 %, ~27.8 ns per ray of Eigen; the Dirichlet deposit block at ~11 %
(C2's per-ray hoist targets part of it); accessor overhead at 7.7 %.
`kWalkMarginVox` is **not** among them — it is 38 % of the ray and it stays,
because it is what buys byte-exactness (§4.5).
