# Fine-resolution TSDF band for trunk DBH — localized two-lattice refinement

**Date:** 2026-07-30 (updated 2026-08-02: TSDF-only decision, pose-error
section, per-scan anchor re-registration; 2026-08-03: scope revert — the
mapper only *generates* the fine map, DBH measurement moved out to
post-processing; raw-return full-density path added)
**Status:** Implemented — core (`refinement_regions.hpp`, `ScovoxMapSplit`
fine grid + anchor fit + `refineHit`), wire (codec rev 7), node plumbing
(`fine_*` params, `RefinementRegion` subscription, raw-return downsample
bypass), post-processing utility (`dbh_fit.hpp` — **not** called by the
mapping runtime), unit tests (`test_fine_tsdf.cpp`)
**Depends on:** TSDF band integration (`Params::sdf_trunc`), the ScovoxMapSplit
parallel-grid architecture, and `explo_planner`'s `TreeDetector`
**Wire format:** additive — codec rev 6 → 7: `fine_ratio_log2` header byte
(0 = no fine grid) + one fine-TSDF stream after the Dir stream; existing
streams unchanged
**Non-goals:** general multi-resolution queries / LOD; fine-resolution
occupancy or semantics (designed separately in
`fine_beta_dirichlet_2026_08_02.md` but **not implemented** — the 2026-08-02
decision is TSDF-only refinement, keeping one fine lattice, one wire stream,
and one fit; the Beta/Dirichlet doc stays as the recorded alternative);
**in-mapper measurement** — scovox_mapping's responsibility ends at
producing maps, so the node publishes no estimates: DBH extraction runs
downstream on the shared rev-7 fine stream or the saved map

## Motivation

We want to measure DBH (diameter at breast height, 1.3 m above local ground)
of individual trees directly from the SCovox map. DBH is a 1–2 cm-accuracy
measurement of a 10–50 cm cylinder, and two things make it impossible at the
current mapping resolution (10–15 cm outdoors):

1. **Truncation-band interference destroys thin objects.** With 10 cm voxels
   and `sdf_trunc_voxels: 3`
   ([lidar_mapping.yaml:52](src/scovox_mapping/config/lidar_mapping.yaml#L52))
   the truncation band is ±30 cm — wider than the trunk. Once the trunk is
   observed from opposite sides (which the exploitation node's circling
   behaviour guarantees), the far side's positive TSDF tail overlaps the near
   side's surface and the Curless–Levoy average biases or erodes the zero
   crossing. This is the classic thin-structure TSDF failure. Rule of thumb:
   **truncation distance ≲ trunk radius**. A 15 cm-DBH tree (7.5 cm radius)
   needs trunc ≤ ~7 cm, i.e. ~1–2.5 cm voxels at 3-voxel truncation.
2. **Sampling.** At 2.5 cm the breast-height circumference of a 30 cm trunk
   yields ~20–40 zero-crossing samples for a circle fit; at 10 cm it yields
   ~5–10, quantized.

Refining the *whole* map to 1–2.5 cm is not an option: the volumetric Beta
occupancy layer is walked voxel-by-voxel along every ray in the full-ray carve
(the hot path), and its memory scales with observed volume. The TSDF layer,
by contrast, only exists in a thin shell around surfaces — its cost scales
with surface area. The two layers only share a resolution today because they
share one lattice. This design unchains them.

## Options considered

- **(A) Lower the global resolution.** Zero code, but ~2× hot-path ray steps
  and ~4× surface memory per halving, paid everywhere for detail needed at a
  few trunks. Kept as the benchmarking baseline only.
- **(B) Two-lattice pyramid — this design.** Coarse grid unchanged; a second
  sparse `Bonxai::VoxelGrid<TsdfVoxel>` at `resolution / 2^k`, written only
  inside the truncation band, and only near registered trees.
- **(C) Sub-voxel bitmask payload** (`uint64_t` = 4×4×4 sub-occupancy per
  voxel). Cheap, but binary hit detail only — no fused TSDF at fine scale, so
  no sub-voxel surface fit. Rejected for DBH.
- **(D) Natively multi-resolution backends** (supereight2, wavemap,
  UFOMap/OctoMap). True adaptive refinement, but replaces Bonxai wholesale:
  O(tree-depth) pointer-chasing per voxel on the carve path where Bonxai's
  cached accessor is amortized O(1), plus a rewrite of every grid consumer.
  Only worth revisiting if we ever need many LOD levels rather than two.

Bonxai itself has no native multi-resolution — the inner/leaf hierarchy is
bitmask-only sparsity structure, not coarse data storage — so (B) builds the
second level as an independent grid, which the split-map architecture already
knows how to host ([sem_split_map.cpp](src/scovox_core/src/sem_split_map.cpp)
runs four parallel grids with per-grid cached accessors and per-grid
`drainTouched` streams today).

## Design

### Two lattices, power-of-two ratio

- `res_fine = resolution / 2^k`, `k = fine_ratio_log2` (new `Params` field;
  k=2 → 10 cm / 2.5 cm, k=3 → 10 cm / 1.25 cm). Power-of-two so level
  conversion is a shift; in practice always derive coarse coords from world
  position via the coarse grid's own `posToCoord` rather than shifting fine
  coords (right-shift of negative signed coords is arithmetic-shift on
  gcc/clang but going through the grid is unambiguous).
- The fine grid holds **TSDF only** — plain 8-byte `TsdfVoxel`
  ([tsdf_voxel.hpp:20-29](src/scovox_core/include/scovox/tsdf_voxel.hpp#L20-L29)),
  no Beta, no Dirichlet. Fine mesh vertices that need labels look them up
  from the containing coarse voxel, same as mesh labelling does today.
- The coarse TSDF grid stays exactly as-is (global mesh, existing consumers).
  The fine grid is purely additive.

### Integration routing

Per ray, today's walk is: carve free space coarse from origin to hit, then
write TSDF mass in the coarse band around the hit. With the fine band
enabled, per hit:

```
origin ────────────────────────────►│ hit │◄────►
        coarse carve (Beta, 10 cm)   coarse band (unchanged)
        unchanged, same cost         + fine band (±3 fine voxels)
                                       IFF hit passes the tree gate
```

The long segment — the hot path — is untouched. The fine band walk is
±`sdf_trunc_voxels` *fine* voxels around the hit, so its step count equals
today's coarse band walk (~6 steps); the band just shrinks in metres, which
is itself part of the win (thinner band → sharper surface, less rounding).
The fine truncation distance is `sdf_trunc_voxels * res_fine`, mirroring how
[scovox_node.cpp:230-231](src/scovox_mapping/src/scovox_node.cpp#L230)
derives the coarse one.

### Tree gate — refinement is spatially localized

The fine grid is sparse, so "high resolution only at trees" is an
integration-time routing policy, not a data-structure feature. A hit routes
to the fine band iff it falls inside a registered **refinement cylinder**:

- **Geometric gate (primary).** The map maintains a registry of cylinders
  `(cx, cy, r + margin, z_lo, z_hi)` sourced from `explo_planner`
  `TreeDetection` results (`tree_detector.hpp` already estimates per-trunk
  axis, radius, and height at 0.15 m resolution). For DBH only,
  `[z_lo, z_hi] = [base + 1.0, base + 1.6]`; extend to trunk height if stem
  taper/volume is wanted later. A 2D spatial hash over trunk centres makes
  the check O(1) per hit (`refinement_regions.hpp`). Registration API:
  `ScovoxMapSplit::addRefinementRegion(cylinder)`; the ROS interface is a
  `scovox_msgs/RefinementRegion` subscription (`fine_region_topic`,
  default `~/refinement_region`) carrying `(id, x, y, base_z, radius,
  remove)` — field-compatible with `TreeTarget`, so a one-line relay (or a
  later explo_planner publisher; explo_planner already depends on
  scovox_msgs) closes the detect → orbit → measure loop. The node derives
  `[z_lo, z_hi] = base_z + [fine_region_z_lo, fine_region_z_hi]`.
  Re-publishing an id updates its cylinder in place (slot-stable) — this
  is also how a downstream estimator refreshes a region's model (see the
  pose section's refresh loop).
- **Semantic gate (optional fallback).** Route to fine when the coarse
  voxel's dominant class is `veg_class` with sufficient confidence — zero
  extra state (the walk touches that voxel anyway), but it would refine
  canopy too, so it must be combined with a height slab if used.

This yields a **coarse-detect → orbit → fine-measure loop** that matches the
planner's existing structure: TreeDetector finds trunks in the coarse map,
detection registers the gate, the exploitation node circles the
under-informed trunk, and the fine TSDF accumulates exactly during the
orbit. The DBH fit's radius uncertainty (below) is a natural complement to
`info_deficit` as the per-tree "done" criterion.

### Full sensor density — the raw-return path

The node voxel-grid-downsamples each scan before integration
(`downsample_voxel_size`: 0.1 m in the geometric config, 0.5 m in the
raw-deskew one) — correct for the coarse map, but it would starve the fine
band to one return per downsample cell. So in-region raw (deskewed)
returns are additionally routed through `ScovoxMapSplit::refineHit`
(`fine_raw_returns`, default on): fine-band-only staging — one O(1) region
lookup, no coarse write — while the coarse map keeps its downsampled
medoid diet. The medoid itself still rides `integrateHit` (which stages
fine internally), so no return is fused twice, and both paths share the
per-scan anchor correction. Result: the fine lattice sees the **sensor's
native point density**, the only density there is.

What "maximum possible resolution" then means is set by the sensor, not
the lattice (`fine_ratio_log2` accepts up to k=8 = 0.4 mm voxels at a
10 cm base — mechanically fine, physically meaningless). For the Hesai
XT32 at a 3–4 m orbit range: horizontal beam spacing ≈ 1.1–1.3 cm
(0.18° @ 10 Hz), range σ ≈ 0.5–1 cm, vertical spacing ~6–7 cm per scan
(filled across the orbit by platform motion). **k=3 (1.25 cm) is the
sensor-matched ceiling**: at k=4 (6.25 mm) voxels fall below both the beam
spacing and the range noise, so per-voxel weights starve and adjacent
voxels carry correlated noise, 8× the memory for no new surface
information. The trunc rule holds at k=3 (`3 · 1.25 cm = 3.75 cm >
2σ_range`), and residual pose error after the anchor fit (~1 cm) is the
same order as the voxel — exactly the regime the anchor was built for.
k=2 (2.5 cm) remains the conservative default in `scovox_fine_band.yaml`.

### DBH measurement — post-processing, not a mapper responsibility

The TSDF *is* a distance field, so no meshing is needed. Take fine voxels in
the breast-height slab with `weight > 0`; for a candidate circle
`(cx, cy, r)`, a voxel at horizontal distance `d` from the axis should
satisfy `tsdf ≈ d − r`. Solve least-squares (robust loss for outliers) for
`(cx, cy, r)` over the fused values; `DBH = 2r`. Sub-voxel by construction —
the fusion has already averaged per-scan range noise (σ/√N per voxel), and
the fit averages around the circumference. With Hesai-class range noise
(1–2 cm) and multi-view accumulation from circling, 1–3 cm DBH error is the
target — mobile-laser-scanning forestry-inventory grade. Report `(DBH,
fit RMS, arc coverage)` per tree; arc coverage below ~180° should flag the
estimate as one-sided (projective bias does not cancel). Implemented as
`fitDbhCircle` (`dbh_fit.hpp`): robust IRLS over `(cx, cy, r)`, voxels
gated on weight and `|tsdf| < 0.5·trunc` — the near-surface half of the
band, because band-tail voxels carry the projective/curvature bias of
risk #3 (and clamped-at-trunc voxels carry a bound, not a distance), while
the near-surface voxels are exactly where a drift-smeared double wall
shows up, keeping the reported RMS a clean pose-quality flag.

The mapping node does **not** run this fit — generating the fine lattice is
where its job ends. The fit is a pure function over the fine grid, run by
whoever consumes the map: an offline tool over a saved/replayed map, a
separate analysis node subscribed to the rev-7 fine stream, or the dscovox
side after a consensus merge. `dbh_fit.hpp` lives in scovox_core only
because every such consumer already links this library to deserialize the
map (and the fine-band unit tests use the fit as their accuracy metric).
An estimator that wants the anchor to benefit from its refined radius
re-publishes the `RefinementRegion` with the fitted model — the refresh
loop of the pose section, closed outside the mapper.

### Serialization & meshing

- The delta serializer
  ([binary_serializer.hpp](src/scovox_core/include/scovox/binary_serializer.hpp))
  is coord+voxel and format-agnostic to resolution; codec rev 6 → 7 adds a
  `fine_ratio_log2: u8` header byte (0 = no fine grid; coords in the fine
  stream are fine-lattice indices at `resolution / 2^k`) and one fine-TSDF
  stream (flat 20 B records, same layout as the coarse TSDF stream) after
  the Dir stream. The fine stream ships iff `share_tsdf` is on and the fine
  grid is enabled; `mergeFrames` merges it Curless–Levoy per coord and
  rejects mismatched `fine_ratio_log2` (different fine lattices must not
  union-by-coord). The dscovox merger parses and ignores it, exactly as it
  does the coarse TSDF stream.
- [marching_cubes.hpp](src/scovox_core/include/scovox/marching_cubes.hpp) is
  already generic over `VoxelGrid<TsdfVoxel>` — pass the fine grid to get
  per-tree fine meshes for visualization/QA; the global mesh keeps coming
  from the coarse grid.

## Pose error — the real ceiling — and how this design absorbs it

TSDF fusion assumes every scan shares one consistent frame; the pose it
fuses with (`T_map_sensor`) drifts. At 10 cm voxels that drift is sub-voxel
and invisible; at 1–2.5 cm it is the dominant error source:

- **Translation drift:** good lidar-inertial odometry drifts ~0.2–1 % of
  distance traveled. An orbit at 4 m radius is ~25 m of driving → 5–25 cm
  of offset between the first and last scan of the orbit.
- **Rotation is worse:** error at the trunk ≈ translation error +
  range × attitude error. 1° of yaw at 4 m range displaces the trunk 7 cm.
- **Within-scan motion:** 1 m/s over a 100 ms sweep spreads points 10 cm if
  not deskewed. Deskew quality (IMU gyro noise — see the go1 `/hesai/imu`
  issue; use `/imu/data`) sets a floor.

Sensor range noise is zero-mean, so Curless–Levoy averaging helps (σ/√N).
Pose drift is a slowly varying **systematic** offset, so averaging smears
instead: the trunk fuses into a thickened or doubled wall, the zero
crossing shifts, and the circle fit returns an inflated radius with high
RMS. Fusion is **irreversible** — once samples are averaged in under wrong
poses, no later correction can un-mix them.

The saving insight: **DBH needs no global accuracy, only internal
consistency across one orbit (~30–60 s).** If the whole region sits 10 cm
wrong in the map frame, DBH is unaffected; only differential drift between
the scans that observe the trunk matters.

Mitigation ladder (all but the last are hygiene; the last is the fix):

1. **Bound the exposure window.** The gate already bounds integration
   spatially; the orbit bounds it temporally. Drift ∝ time and distance.
2. **Smoothest pose + deskew.** Feed integration locally-consistent
   odometry (never a stream that jumps from global corrections mid-orbit)
   and deskew with the good IMU.
3. **Pose-health gating.** Skip fine integration under angular-rate or
   covariance spikes (rotation error dominates). Cheap, coarse-grained.
4. **Per-scan anchor re-registration — implemented, see below.**
5. **Detect what remains:** fit RMS, arc coverage, and (future) a bimodal
   radial-residual histogram flag a smeared double wall. Report, re-orbit,
   or distrust — never average away.
6. Rejected: pose-graph SLAM with submap re-integration — the heavyweight
   answer to the same irreversibility, unnecessary for a 60 s orbit.

### Per-scan anchor re-registration

Each scan's in-region hits are **staged, not fused** (buffered inside the
`beginCarveFrame`/`flushCarveFrame` bracket the node already runs per
scan). At flush, per region: fit the 2-DoF horizontal shift `Δ` that best
aligns the staged endpoints with the region's canonical cylinder
`(cx, cy, r)` — robust IRLS on `‖p_xy + Δ − c‖ − r` with a Huber loss and
a small damping toward `Δ = 0` (the odometry prior, which also conditions
thin one-sided arcs) — then integrate every staged ray translated by `Δ`.
Every scan is drift-corrected at the door, so the fusion input is
consistent by construction; the scan buffer lives for one callback, the
map stays a single world-anchored fine lattice, and no point cloud is ever
persisted. Fit failure (fewer than `fine_anchor_min_points` in the region,
or `‖Δ‖ > fine_anchor_max_shift`) falls back to odometry (Δ = 0) for that
scan. The fine trunk therefore sits at the canonical (first-registered)
pose while the coarse map drifts around it — irrelevant for DBH, which is
region-internal. 2-DoF suffices: z drift moves the slab along a
near-cylindrical trunk (taper ~1 cm diameter / m → negligible), and
roll/pitch errors are already bounded by the IMU.

**Model-radius bias and the refresh loop.** With a one-sided arc, a wrong
canonical radius `r = r_true + δ` biases each scan's `Δ` outward along its
own view direction by up to `δ·g` (g < 1, shrinking as the arc widens), so
the fused surface forms at radius `r_true + δ·g` — coherent (no double
wall) but biased by the coarse TreeDetector's radius error. The DBH fit
then measures `r_true + δ·g`; when the downstream estimator re-publishes
the region with that fitted model (gated on the fit's RMS/arc quality —
the estimator's call, not the mapper's), the error contracts
geometrically: the fixed point of `r ← r_true + (r − r_true)·g` is
`r_true` for `g < 1`. The refresh moves the anchor *toward the fused
data*, so refreshes stay self-consistent. The mapper's part is only the
in-place model update on re-registration; the loop itself lives outside.

## Cost

- **Runtime:** hot path unchanged; per-hit adds one O(1) gate check, and for
  gated hits a band walk whose step count matches the existing coarse band.
  The raw-return path adds, per scan, one transform + hash probe per cached
  raw point (only while ≥1 region is registered) and band-only walks for the
  in-region subset — a trunk at 3–4 m subtends a few hundred returns per
  scan, each a ~6-step fine-band walk, no free-space carve.
- **Memory:** shell-proportional. A 30 cm trunk, 0.4 m breast slab, 1 cm
  voxels, 3-voxel band: π·0.3·0.4 m² × 0.06 m ≈ 23k voxels ≈ 180 KB payload
  (a few hundred KB with leaf-block overhead). Hundreds of trees are
  trivial; whole-map 1 cm would be hopeless. This asymmetry is the entire
  point of the design.

## Risks / limits

1. **Pose accuracy is the real ceiling.** See the dedicated section above:
   per-scan anchor re-registration absorbs differential drift; deskew, pose
   hygiene, and the RMS/arc flags cover the rest. The residual after
   anchor-corrected fusion is roughly sensor-noise level (~1 cm), inside
   the 1–3 cm DBH budget.
2. **Trunc-vs-radius rule still applies at the fine level.** For very thin
   stems (< ~2·`sdf_trunc_voxels`·`res_fine` diameter) the interference
   returns; k must be chosen per deployment (k=3 at 10 cm coarse covers
   ≥ ~8 cm DBH).
3. **Projective-TSDF bias** on convex surfaces pushes the zero crossing
   slightly outward for grazing rays. Circling gives near-normal incidence
   at the silhouette and the fit averages over views; acceptable at the
   1–3 cm target, but worth checking against ground truth before trusting
   sub-centimetre claims.
4. **Fine geometry is invisible to occupancy consumers** by design — the
   planner, frontier detection, and clearance checks still see the coarse
   Beta grid. If fine *occupancy* near obstacles is ever needed, that is a
   separate (and much more expensive) design.

## Validation plan

1. **Option A baseline:** one bag replayed at 2.5 cm global resolution to
   quantify what fine-band DBH must beat (and to confirm the hot-path cost
   that makes global refinement a non-starter).
2. **Offline two-pass replay** on the 2026-07-06 forest campaign bags:
   pass 1 builds the coarse map and runs TreeDetector; pass 2 re-integrates
   with the resulting cylinder gates active and fits DBH per tree.
3. **Ground truth:** field-measured DBH (tape/calliper) for the campaign
   plots if available; otherwise manual circle fits on the accumulated
   deskewed point cloud in the same slab as an upper-bound reference —
   the fine-TSDF fit should match it within noise while using fixed memory.
4. **Ablations:** k ∈ {2, 3}, `sdf_trunc_voxels` ∈ {2, 3}, geometric vs
   semantic+slab gate, with (DBH error, fit RMS, fine-voxel count, per-scan
   integration time) as the metric set.
