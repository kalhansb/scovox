# Fine-resolution Beta–Dirichlet lattice — localized occupancy + semantics refinement

**Date:** 2026-08-02
**Status:** Design doc — implementation not yet started
**Companion to:** `fine_tsdf_band_dbh_2026_07_30.md` (the fine TSDF band).
That doc listed "fine-resolution occupancy or semantics" as a non-goal and
risk #4 called it "a separate (and much more expensive) design". This is that
design. The two share the refinement-region gate and should ship the wire
bump together.
**Depends on:** the `SemSplitMap` split substrate
([sem_split_map.hpp](src/scovox_core/include/scovox/sem_split_map.hpp)),
the `ScovoxMapSplit` fused walker
([scovox_map_split.hpp](src/scovox_core/include/scovox/scovox_map_split.hpp)),
the rev-6 block-run wire format
([binary_serializer.hpp](src/scovox_core/include/scovox/binary_serializer.hpp)),
and the refinement-cylinder gate from the fine-TSDF design.
**Non-goals:** whole-map refinement; more than two levels; refining the
transient (dynamic-class) grids; changing any coarse-grid behaviour.

## Motivation

The fine TSDF band answers "where exactly is the surface" inside a refinement
region, but two questions stay coarse:

1. **Fine occupancy.** Thin structure near a trunk — low branches, stems of
   neighbouring saplings, the trunk's own taper — is eroded or swallowed at
   10 cm: a 3 cm branch fills 3 % of a coarse voxel and loses the
   `w_occ`-vs-`w_free` race against the dense carve fan behind it. Inside a
   refinement region we want a Beta posterior at 1–2.5 cm so thin obstacles
   and the trunk cross-section itself survive as *occupancy*, not only as a
   TSDF zero-crossing (which consumers like clearance checks never read).
2. **Fine semantics.** The Dirichlet class boundary (tree ↔ ground at the
   trunk base, tree ↔ OTHER at foliage) is quantized to the coarse lattice.
   The DBH pipeline needs the local **base height** (`base + 1.3 m` defines
   breast height) — a fine tree/ground boundary tightens `base_z` from
   ±10 cm to ±2.5 cm, which is ±7.5 cm of slab placement on a tapering stem.
   Fine labels also let the fine TSDF mesh be labelled at its own resolution
   instead of inheriting the containing coarse voxel's class.

The reason this was deferred in the TSDF doc: **occupancy is full-ray**. The
Beta carve walks every voxel along every ray (the hot path), so refining it
globally multiplies the dominant per-scan cost by `2^k` and its memory by
volume, not surface. The entire design below is about bounding that carve to
the same refinement regions the TSDF band already gates on — where it turns
out to be nearly free (see Cost).

## Options considered

- **(A) Reuse the fine-TSDF two-lattice scheme — this design.** A second
  `SemSplitMap` instance at `resolution / 2^k`, written only inside
  refinement regions. All voxel math (priors, `sparse_add_class`, saturation,
  consensus merge, quantized wire records) is resolution-agnostic and reused
  byte-for-byte.
- **(B) Per-voxel refs: coarse voxels point at high-res sub-blocks.** Each
  refined coarse voxel would carry a pointer/index to a `2^k`³ fine block.
  Rejected:
  - `BetaVoxel`/`DirVoxel` are pool-allocated trivial PODs by static_assert
    ([beta_voxel.hpp:61-67](src/scovox_core/include/scovox/beta_voxel.hpp#L61));
    a ref member breaks zero-init pool semantics and adds 8 B to **every**
    coarse voxel map-wide to serve the <1 % that are ever refined.
  - It buys nothing over coordinate math: with a power-of-two ratio the
    coarse↔fine correspondence is already exact and deterministic
    (`posToCoord` through each grid) — there is no inter-level registration
    that a ref could make "tighter".
  - It is **not** drift-free (a hope worth killing explicitly): odometry
    drift corrupts `T_map_sensor` at integration time, i.e. the world
    coordinates of the points *before any data structure sees them*. A fine
    block hung off a coarse voxel is indexed by exactly the same drifting
    pose as an independent fine lattice — an orbit smears the trunk
    identically in both. Drift immunity requires re-estimating an anchor
    per scan, not storing a pointer; see extension E1.
- **(C) Coarse-voxel sub-occupancy bitmask** (`uint64_t` = 4×4×4). Same
  rejection as in the TSDF doc, plus: binary sub-occupancy cannot carry a
  Beta posterior (no free-evidence accumulation, no `p_occ` for consumers)
  and has no room for class mass at all.
- **(D) Natively multi-resolution backends.** Same rejection as the TSDF
  doc (Bonxai's O(1) cached accessor on the carve path is the thing we must
  not give up; two fixed levels don't need an octree).

## Design

### A second `SemSplitMap` at `res_fine`

`ScovoxMapSplit` grows an optional fine substrate:

```
TsdfMap        tsdf_        (coarse, unchanged)   ┐
SemSplitMap    semsplit_    (coarse, unchanged)   ├─ today
TsdfGrid       fine_tsdf_   (fine band, TSDF doc) ┤
SemSplitMap    fine_sem_    (this design)         ┘ null when disabled
```

- `res_fine = resolution / 2^k`, sharing `fine_ratio_log2` with the fine
  TSDF grid — **one** k for all fine grids, so all fine lattices are
  coord-identical (the same invariant the coarse grids rely on for
  cross-grid queries and `labelMesh`).
- The fine instance reuses `SemSplitMap` wholesale: Beta(1,1) occupancy
  prior, `(C − K_TOP)·α₀` Dirichlet prior, Stream A/B two-stream update,
  batched carve frames, per-grid touched sets. Same `num_classes`, same
  compile-time `K_TOP`. Its **transient grids stay unused**: the gate targets
  static trunks, and `is_dynamic` rays never route fine.
- Weights: the same `w_occ`/`w_free`/`kappa0` (or per-source `HitWeights`)
  apply. Evidence is an observation count and counts are
  resolution-independent; what changes is the *rate* (see Risks #1).

### Integration routing — clip the ray, don't walk it

The refinement-region set (cylinders `(cx, cy, r + margin, z_lo, z_hi)`
registered from `TreeDetection`, per the TSDF doc) becomes shared
infrastructure consulted by both fine paths. Per ray, in the fused walker:

1. **Clip** the segment `[origin, endpoint]` against the region set —
   ray-cylinder intersection, a few multiplies per registered region, done
   once per ray before the DDA. Regions are few (tens); a 2D hash over
   trunk centres prunes to the regions near the ray's XY footprint. Output:
   0–2 parameter intervals `[t0, t1]` (typically 0 — see Cost).
2. **Coarse walk unchanged.** The existing single DDA runs exactly as
   today; the clip result adds one flag per visited voxel at most.
3. **Fine carve** per clipped interval: a second, short DDA at `res_fine`
   over `[t0, t1]` only, staging `applyCarveUpdate` into the fine
   instance's carve frame. Pass-through rays (hit elsewhere) still carve
   fine free space inside the region — that is where most fine free
   evidence comes from and it is what kills ghost branches.
4. **Fine hit** iff the endpoint lies inside a region: fine Stream A
   (Beta) + gated Stream B (Dir) at the fine endpoint voxel — one
   `applyHitUpdate` on the fine instance, same gate the fine TSDF band
   uses, evaluated once and shared.
5. `beginCarveFrame`/`flushCarveFrame` wrap both instances; occupied-wins
   holds per lattice independently.

The hot path — the long coarse carve — is untouched; rays that miss every
region (the overwhelming majority, even mid-orbit) pay only the clip test.

### Consumers — coarse remains the authority, fine is opt-in

- **Default: nothing changes.** Planner, frontier detection, costmap, and
  the SSMI/information layer keep reading the coarse grids. Fine geometry
  is invisible unless a consumer asks.
- **Multi-resolution point queries** (new, on `ScovoxMapSplit`):
  `pOccMultires(pos)` / `dominantClassMultires(pos)` — return the fine
  voxel's posterior when a fine voxel exists at `pos` **and** its evidence
  `s_total()` clears a floor (an under-observed fine voxel must not veto a
  well-observed coarse one), else fall through to coarse. Deterministic,
  no interpolation.
- **Fine mesh labelling**: the per-tree fine TSDF mesh labels against
  `fine_sem_.dirGrid()` where allocated, falling back to the containing
  coarse voxel — upgrading the TSDF doc's coarse-lookup rule.
- **DBH support**: `base_z` from the fine tree/ground Dirichlet boundary in
  the region; fine Beta cross-section as a sanity check against the TSDF
  circle fit (the two are independent estimators of the same cylinder).
- **Optional coarse export**: a region's fine Beta can be down-projected
  as `max p_occ` over the `2^k`³ children into the coarse costmap layer if
  nav ever needs thin obstacles it currently misses. Conservative by
  construction (occupancy only ever increases). Off by default.

### Wire format & merge — additive, one bump with the fine TSDF

- Header gains `fine_ratio_log2: u8` (0 ⇒ no fine sections follow, which
  keeps the frame byte-identical to rev 6 semantics when the feature is
  off). After the existing three sections, three more:
  `fine_tsdf_count` (from the TSDF design), `fine_beta_count`,
  `fine_dir_count` — same block-run coord coding and u16 quantized payloads,
  coords in fine-lattice units. **VERSION 6 → 7 once**, covering both fine
  designs; mixed-revision peers fail loud, as with the 5→6 bump.
- Receiver (`dscovox`) reconstructs fine grids at
  `resolution / 2^fine_ratio_log2` and merges them with the *identical*
  per-grid conjugate rules (`mergeBeta`/`mergeDir`/`mergeTsdf` are
  resolution-blind). **No region synchronisation is required for
  consistency**: the gate is sender-side integration policy; receivers
  accept whatever fine voxels arrive. Robots with disjoint region sets
  simply contribute fine evidence where they chose to look — the merge is
  still conservative and mass-preserving.
- Bandwidth: fine deltas are region-bounded. A DBH region (r ≈ 0.8 m gated,
  0.6 m slab) fully carved at k=2 is ~77 k fine Beta voxels ≈ 310 KB
  quantized *once over the region's life*; per-frame touched sets are a few
  thousand voxels. Dir stays hit-sparse (~2 k shell voxels ≈ 20 KB). Small
  next to the coarse Beta stream.

## Cost

- **Runtime.** Per-ray: one region clip (O(active regions), ~4 mul each).
  Fine work only on region-crossing rays: chord ≤ 2·r_gate ≈ 1.6 m ⇒ ≤64
  fine steps at 2.5 cm. Geometry makes this cheap: orbiting at 4 m, the
  gate cylinder subtends ~23° of azimuth and the slab ~9° of elevation ⇒
  ~2 % of a VLP-16 scan's rays cross the region ⇒ **≲1 % added steps per
  scan** while actively orbiting, ~0 % otherwise. The expensive-sounding
  design is, gated, almost free.
- **Memory.** Volume-proportional but region-bounded: ~620 KB Beta +
  ~30 KB Dir per DBH region at k=2 (≈1 MB with leaf overhead); a 3.5 m
  full-stem region ≈ 5–6× that. Hundreds of concurrent DBH regions are
  fine; full-stem regions want the lifecycle below.
- **Region lifecycle** (needed for occupancy where the TSDF band was not,
  because Beta allocates the region's *interior*, not just its shell):
  `ACTIVE` (integrating) → `FROZEN` (measurement done — DBH reported, fit
  RMS converged; no more writes, still queryable/mergeable) → `EVICTED`
  (blocks released; keep the scalar outputs). Freezing is the default end
  state; eviction is a memory valve for long missions.

## Risks / limits

1. **Evidence-rate asymmetry.** The number of rays crossing a voxel scales
   with its cross-section: fine voxels accumulate `2^2k` (k=2: 16×) less
   evidence per second than coarse ones at the same range. Consequences:
   fine `p_occ` converges slower, and consumers must threshold on evidence
   (`s_total()`), not probability alone — a fresh fine voxel at Beta(1,1)
   is *unknown*, not free. The orbit behaviour concentrates exactly the
   rays the region needs, which is why gating on orbited trunks works at
   all. VLP-16 vertical ray spacing (~14 cm at 4 m) exceeds `res_fine`, so
   single-scan fine coverage is striped; coverage fills in from platform
   motion over the orbit.
2. **Two posteriors can disagree.** Coarse says free (branch lost the
   carve race), fine says occupied. This is the feature, not a bug — but
   policy must be explicit: fine wins only through the opt-in multires
   queries / the conservative coarse export; the coarse grid is never
   rewritten in place.
3. **Pose drift** — same ceiling as the fine TSDF band (risk #1 there), and
   per option (B) above, no storage scheme fixes it. Mitigation hooks are
   shared: robust fits, arc-coverage flags, and E1 below.
4. **Wire discipline.** One VERSION bump shared with the fine TSDF stream;
   `K_TOP` remains compile-time-locked across the fleet; a peer without the
   feature (fine_ratio_log2 = 0) interops cleanly.

## Extension E1 — region-anchored frames (the actual drift fix; deferred)

The base design keeps all fine lattices in the map frame (matching the fine
TSDF doc). If orbit-scale drift proves to be the accuracy ceiling the
validation data says it is, the region — not the voxel — is the right unit to
anchor: give each refinement region a pose `T_map_region`, integrate fine
data in region coordinates, and **re-fit the anchor each scan** by
registering that scan's in-region points against the trunk cylinder model
(a 2-DoF horizontal alignment is enough for DBH). Then drift accumulated in
`T_map_sensor` is absorbed into the per-scan anchor correction instead of
smearing the fine map; the fine lattice becomes drift-free *relative to its
tree*, which is the frame DBH is defined in anyway. Costs: per-scan fit
(needs enough in-region points — fall back to odometry when thin), a pose
per region on the wire, and merge semantics that must reconcile two robots'
anchors for the same tree (register region-to-region, or nominate the
detection owner's anchor). Deferred until the world-frame version's fit RMS
demonstrates the need; nothing in the base design blocks retrofitting it.

## Validation plan

1. **Thin-structure recall:** replay a forest bag with regions on a plot of
   known low branches; count branch voxels surviving at fine vs coarse
   (expect coarse to erode them, fine to keep a Beta-occupied core).
2. **`base_z` accuracy:** fine tree/ground boundary vs surveyed ground at
   each trunk; propagate to DBH slab placement error.
3. **Cross-estimator check:** fine-Beta cross-section area vs the fine-TSDF
   circle fit per tree; disagreement flags bad regions before either number
   is trusted.
4. **Cost audit:** per-scan integration time and wire bytes with 0 / 1 / 10
   active regions (expect ≲1 % / region while orbiting); memory over a
   full mission with and without the freeze/evict lifecycle.
5. **Merge parity:** two-robot replay where only one robot runs regions;
   assert the merged fine grids equal the region-runner's (receiver needs
   no gate), and that a rev-6 peer fails loud on rev-7 frames.
