# Fine-resolution TSDF band for trunk DBH — localized two-lattice refinement

**Date:** 2026-07-30
**Status:** Design doc — implementation not yet started
**Depends on:** TSDF band integration (`Params::sdf_trunc`), the ScovoxMapSplit
parallel-grid architecture, and `explo_planner`'s `TreeDetector`
**Wire format:** additive — the fine grid serializes through the existing
coord+voxel delta path with a per-grid resolution tag; no change to existing
streams
**Non-goals:** general multi-resolution queries / LOD; fine-resolution
occupancy or semantics

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

- **Geometric gate (primary).** The node maintains a list of cylinders
  `(cx, cy, r + margin, z_lo, z_hi)` sourced from `explo_planner`
  `TreeDetection` results (`tree_detector.hpp` already estimates per-trunk
  axis, radius, and height at 0.15 m resolution). For DBH only,
  `[z_lo, z_hi] = [base + 1.0, base + 1.6]`; extend to trunk height if stem
  taper/volume is wanted later. A 2D hash over trunk centres makes the check
  O(1) per hit. Registration API: `Map::addRefinementRegion(cylinder)` +
  a ROS interface on the node (subscription or service).
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

### DBH measurement

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
estimate as one-sided (projective bias does not cancel).

### Serialization & meshing

- The delta serializer
  ([binary_serializer.hpp](src/scovox_core/include/scovox/binary_serializer.hpp))
  is coord+voxel and format-agnostic to resolution; the fine grid adds one
  more `TsdfDelta` stream tagged with its resolution. Additive format bump.
- [marching_cubes.hpp](src/scovox_core/include/scovox/marching_cubes.hpp) is
  already generic over `VoxelGrid<TsdfVoxel>` — pass the fine grid to get
  per-tree fine meshes for visualization/QA; the global mesh keeps coming
  from the coarse grid.

## Cost

- **Runtime:** hot path unchanged; per-hit adds one O(1) gate check, and for
  gated hits a band walk whose step count matches the existing coarse band.
- **Memory:** shell-proportional. A 30 cm trunk, 0.4 m breast slab, 1 cm
  voxels, 3-voxel band: π·0.3·0.4 m² × 0.06 m ≈ 23k voxels ≈ 180 KB payload
  (a few hundred KB with leaf-block overhead). Hundreds of trees are
  trivial; whole-map 1 cm would be hopeless. This asymmetry is the entire
  point of the design.

## Risks / limits

1. **Pose accuracy is the real ceiling.** At 1–2.5 cm voxels, odometry drift
   over an orbit smears the trunk into a double wall; no circle fit recovers
   that, and fit RMS will report it but not fix it. Deskew and pose quality
   (see the go1 `/hesai/imu` gyro issue) become the limiting factor before
   resolution does. Mitigation: robust loss + the arc-coverage/RMS flags;
   possibly gate fine integration on pose confidence.
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
