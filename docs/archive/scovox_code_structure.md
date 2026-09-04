# SCovox code structure — navigation and reuse guide

**Date:** 2026-08-26 (line anchors are approximate as of this date — refresh
them with the grep commands given inline before editing).
**Audience:** developers and code agents that need to locate, modify, or
extract parts of the mapping stack quickly, without reading 3000-line files
end to end.
**Companion docs:** [user_manual.md](user_manual.md) (running the system),
[design/](design/) (per-feature design docs),
[design/efficiency_audit_2026_08_26.md](design/efficiency_audit_2026_08_26.md)
(known hot spots — read before touching any hot path).

---

## 1. Orientation in 60 seconds

SCovox is a **sparse metric-semantic voxel mapping system** for ROS 2. Per
voxel it maintains three kinds of state, held in **three parallel sparse
grids** (this split is the central architectural fact — almost every file
below is explained by it):

| Grid | Voxel type | Size | Touched by | Holds |
|---|---|---|---|---|
| TSDF | `TsdfVoxel` | 8 B | band only (~2·trunc per ray) | Curless–Levoy distance + weight |
| Beta (occupancy) | `BetaVoxel` | 8 B | **full ray** (every carved voxel) | Beta(α_occ, α_free) posterior |
| Dir (semantics) | `DirVoxel` | 16 B | **hit only** | top-K sparse Dirichlet class counts |

All three are `Bonxai::VoxelGrid<T>` instances (vendored sparse hierarchical
grid, see §3.1). One fused ray walker (`ScovoxMapSplit::integrateHitFused`)
updates all three in a single DDA pass per ray.

Two ROS nodes:

- **`scovox_mapping_node`** ([scovox_node.cpp](../src/scovox_mapping/src/scovox_node.cpp))
  — the per-robot mapper. Sensors in (RGB-D pair or LiDAR cloud + optional
  soft-probability blobs), maps out (full-map message, LZ4 delta wire stream,
  point clouds, 2-D planning grid, mesh service).
- **`dscovox_mapping_node`** ([dscovox_node.cpp](../src/scovox_mapping/src/dscovox_node.cpp))
  — the multi-robot merger. Ingests each robot's delta wire stream into
  per-source grids and folds them into a fused map by Bayesian consensus.

The wire between them is `ScovoxMapBinary`: a custom binary codec
(rev-8, block-run coordinates, u8-companded evidence) + LZ4, carrying only
voxels whose values changed ("touched sets" + change gates).

**Dependency direction (never violated):**
`bonxai` → voxel types → substrates (`TsdfMap`, `SemSplitMap`) → walker
(`ScovoxMapSplit`) → ROS nodes. `scovox_core` has **zero ROS dependencies**
(Eigen + LZ4 only) — that is what makes it reusable outside ROS.

## 2. Workspace layout — what to read, what to ignore

```
scovox/                     ← the colcon workspace root (build from HERE)
├── src/
│   ├── scovox_core/        ← the reusable, ROS-free mapping library (§3)
│   ├── scovox_mapping/     ← the two ROS nodes + launch/config (§4)
│   ├── scovox_msgs/        ← message/service definitions (§5)
│   └── seg_pipeline/       ← Python RGB-D segmentation front-end (§6)
├── docs/                   ← this file, user manual, design docs
├── config/, docker/, scripts/   ← deployment glue, not mapping logic
├── experiments/            ← evaluation campaign (scorers, results, paper) — NOT runtime code
├── build*/ install*/ log/  ← colcon output. NEVER edit; see gotcha §9.8
└── README.md, compose.yaml
```

**Ignore for mapping work:** everything under `build*`, `install*`, `log`, and
`experiments/`. They contain copies of the source that are *not* the live code;
a grep that lands there wastes your edit.

## 3. `scovox_core` — the reusable library

Header-heavy; only three `.cpp` files. Builds one static/shared library
target `scovox_core` (links Eigen + LZ4). Everything is under
`include/scovox/` unless noted. Listed bottom-up by layer.

### 3.1 Foundation

- [third_party/bonxai/bonxai/](../src/scovox_core/include/third_party/bonxai/bonxai/)
  — vendored **Bonxai** sparse voxel grid (MPL-2.0): `bonxai.hpp` (the grid:
  root `unordered_map` → inner grid → pooled leaf blocks), `grid_coord.hpp`,
  `mask.hpp`, `grid_allocator.hpp`, `serialization.hpp`. Self-contained,
  header-only, no Eigen required internally. **Key API:** `VoxelGrid<T>`,
  `createAccessor()` / `createConstAccessor()` (cached, see gotcha §9.1),
  `posToCoord`/`coordToPos`, `forEachCell`, `activeCellsCount()`. New leaf
  blocks are **zero-initialised** — every voxel type uses "all-zero =
  unobserved" as its sentinel (gotcha §9.2).
- [version.hpp](../src/scovox_core/include/scovox/version.hpp) — version
  constants only.

### 3.2 Voxel types (plain structs, layout is load-bearing)

- [tsdf_voxel.hpp](../src/scovox_core/include/scovox/tsdf_voxel.hpp) —
  `TsdfVoxel {float distance; float weight}` (8 B). `weight == 0` =
  unobserved. Sign convention: positive in front of the surface
  (SLIM-VDB/VDBFusion-compatible).
- [beta_voxel.hpp](../src/scovox_core/include/scovox/beta_voxel.hpp) —
  `BetaVoxel {float a_occ; float a_free}` (8 B) + the symmetric Beta(1,1)
  prior constants `kBetaOccPrior`/`kBetaFreePrior`. The header's comment
  explains *why* occupancy is a separate grid — read it before proposing to
  re-unify anything.
- [dir_voxel.hpp](../src/scovox_core/include/scovox/dir_voxel.hpp) —
  `DirVoxel` (16 B): K_TOP sparse Dirichlet slots (class id + count) +
  OTHER/unknown mass, with Space-Saving eviction. Hit-only grid.
- [voxel.hpp](../src/scovox_core/include/scovox/voxel.hpp) — the **legacy
  unified** `Voxel` (Beta + Dirichlet + bookkeeping in one struct), plus two
  symbols that are still canonical for everyone: **`K_TOP` (compile-time,
  = 2)** and **`sparse_add()`** (the Space-Saving top-K update, strict `>`
  eviction). Layout `static_assert`s live here.
- [sembeta_voxel.hpp](../src/scovox_core/include/scovox/sembeta_voxel.hpp) —
  `SemBetaVoxel` (24 B, Beta + Dirichlet without TSDF): the intermediate
  combined type; used by mesh labelling and the merger's fused projection.

**Layout warning:** the wire codec and the tests `reinterpret_cast` /
`static_assert` these exact sizes. Changing any field is a **wire-format and
paper-claim change**, not a refactor (gotcha §9.3).

### 3.3 Pure math (no grid, no ROS — smallest reusable units)

- [semantics.hpp](../src/scovox_core/include/scovox/semantics.hpp) —
  `SemanticMode` enum + the Bayesian-soft Dirichlet update
  (`kappa0 · p_occ · quality` weighting; no hard occupancy gate).
- [uncertainty.hpp](../src/scovox_core/include/scovox/uncertainty.hpp) /
  [uncertainty.cpp](../src/scovox_core/src/uncertainty.cpp) — `digamma`,
  `variance`, `entropy`, `expectedInformationGain`, `semanticEntropy`,
  `betaKL`, for both `Voxel` and `SemBetaVoxel`. Note: EIG ≈ 3 digamma + 2
  log per call — expensive per-voxel (see efficiency audit item 3).
- [consensus_merge.hpp](../src/scovox_core/include/scovox/consensus_merge.hpp)
  — the multi-robot merge rules: Beta merge (`a_fused = a_A + a_B − prior`),
  the slot-reconciling Dirichlet merge (union → sum coinciding → re-truncate
  to K_TOP, remainder → OTHER), Curless–Levoy TSDF merge. Pure functions;
  the header comment is the authoritative spec of the math.

### 3.4 Integration substrates

- [ray_iterator.hpp](../src/scovox_core/include/scovox/ray_iterator.hpp)
  (40 lines) — integer Bresenham/DDA over grid coords. Visits ~max|Δ| voxels,
  stops one short of the endpoint. Shared by every integration path.
- [tsdf_map.hpp](../src/scovox_core/include/scovox/tsdf_map.hpp) /
  [tsdf_map.cpp](../src/scovox_core/src/tsdf_map.cpp) — `TsdfMap`: band-only
  TSDF integration (SLIM-VDB-equivalent), coarse + optional fine lattice,
  **touched-set** recording (`touched_.push_back` per band write;
  `drainTouchedTsdf()` = sort+unique+move-out vs `clearTouchedTsdf()` = drop;
  see gotcha §9.4).
- [sem_split_map.hpp](../src/scovox_core/include/scovox/sem_split_map.hpp) /
  [sem_split_map.cpp](../src/scovox_core/src/sem_split_map.cpp) —
  `SemSplitMap`: owns the Beta + Dir grids **plus transient grids for
  dynamic classes**, the batched free-space carve
  (`carve_stage_`/`carve_hits_` staging, `flushCarveFrame()` block-sorted by
  leaf), the semantic hit update with optional BKI/spread kernel
  (`applyHitUpdateKernel`), evidence saturation, and its own touched
  sets/drains. `Params` (w_occ, w_free, kappa0, num_classes,
  `batch_free_carve`, …) is the behavior switchboard.
- [refinement_regions.hpp](../src/scovox_core/include/scovox/refinement_regions.hpp)
  — registry of vertical-cylinder fine-TSDF regions (O(1) point-in-region
  via 2-D spatial hash — it runs per ray hit) + the IRLS anchor-shift fit
  that keeps a fine lattice drift-free relative to its tree.

### 3.5 The fused walker (the hot path)

- [scovox_map_split.hpp](../src/scovox_core/include/scovox/scovox_map_split.hpp)
  — `ScovoxMapSplit`: composes one `TsdfMap` + one `SemSplitMap` and
  implements **`integrateHitFused`**, the single-DDA-per-ray walker that
  updates TSDF band, Beta carve, and Dir hit in one pass. Contains the
  far-voxel skip (Chebyshev far test + guard ring; arms only when carving is
  batched OFF), the inlined bit-exact `coordToPos`, and deliberate per-ray
  timing brackets. **This file is verification-sensitive:** changes here are
  expected to reproduce byte-identical maps (see gotcha §9.5). Read its long
  comments before editing; they encode the invariants.

### 3.6 Wire codec

- [binary_serializer.hpp](../src/scovox_core/include/scovox/binary_serializer.hpp)
  — the rev-8 delta codec: per-stream records (Beta / Dir / TSDF / fine),
  block-run coordinate encoding (64 B bitmask or index list per leaf),
  u8 sqrt-companded evidence, u8 class ids, deterministic block sort,
  DoS-guarded deserialization. `serialize*` take touched-coord lists;
  `deserialize*` return decoded records for the merger.
- [lz4_codec.hpp](../src/scovox_core/include/scovox/lz4_codec.hpp) —
  `compressLZ4`/`decompressLZ4` with a 4-byte big-endian size header and a
  256 MB sanity cap. (Struct is named `ScovoxBinarySerializer` for historical
  call-site compatibility.)

### 3.7 Offline / analysis utilities

- [marching_cubes.hpp](../src/scovox_core/include/scovox/marching_cubes.hpp)
  — TSDF mesh extraction (`extractMesh`, `extractZeroCrossing`), PLY writer
  (ASCII), backs the `ExtractMesh` service and the TSDF cloud publisher.
- [mesh_labelling.hpp](../src/scovox_core/include/scovox/mesh_labelling.hpp)
  — `labelMesh` / `labelPointCloud`: attach semantics to extracted geometry
  by cross-grid lookup. Missing semantic voxel → sentinel class `0xFFFF`
  ("unknown") — consumers must filter it.
- [dbh_fit.hpp](../src/scovox_core/include/scovox/dbh_fit.hpp) —
  post-processing robust circle fit for trunk DBH on the fine TSDF lattice.
  **Nothing in the runtime calls it**; it lives here so map consumers can
  link one library.
- [map_interface.hpp](../src/scovox_core/include/scovox/map_interface.hpp) —
  legacy `Params` + one-stop re-export header (`voxel.hpp`, `uncertainty.hpp`,
  `semantics.hpp`, `ray_iterator.hpp`). Include this to get the unified-voxel
  toolkit in one line.

### 3.8 Tests and tools

`test/` — gtest suites, one per unit (`test_tsdf_map`, `test_sem_split_map`,
`test_scovox_map_split`, `test_binary_serializer`, `test_consensus_merge`,
`test_sparse_add`, `test_voxel_layouts`, `test_uncertainty`,
`test_mesh_labelling`, `test_fine_tsdf`). **Read the tests as the behavioral
spec** — they encode contracts (merge algebra, codec round-trips, layout
asserts) that the headers only summarise. `tools/split_memory_demo.cpp` is a
standalone memory-footprint demo.

## 4. `scovox_mapping` — the ROS layer

Builds library `scovoxmap` (the legacy unified `Map`) + two node executables
(`scovox_mapping_node`, `dscovox_mapping_node`), all linking `scovox_core`,
Eigen, LZ4, and the usual ROS 2 stack (tf2, message_filters, …).

### 4.1 [scovox_node.cpp](../src/scovox_mapping/src/scovox_node.cpp) — the mapper (~3100 lines)

Single class, single file. **Region map** (anchors approximate; refresh with
`grep -n "publishScovoxMap\|publishBinaryMap\|publishPlanningMap\|publishPointCloud\|publishTSDFPointCloud\|int main" src/scovox_mapping/src/scovox_node.cpp`):

| Lines ≈ | Region | Notes for editing |
|---|---|---|
| 40–230 | Constructor: TF buffer (600 s cache, L54), map construction, 1 Hz publish timer (163–178, holds one `shared_lock` across all publishers), `ExtractMesh` service (157), fine-region subscription (210) | Timer callbacks run on the **same thread** as sensor callbacks |
| 138–148 | `GateBeta`/`GateDir` change-gate structs | Wire-delta semantics — comparison values must stay exact |
| ~400–830 | `declare_parameter` block — **every runtime knob is declared here**, grouped by feature | To find any parameter's default: grep `declare_parameter` |
| 813–900 | Subscriptions + publishers: DiagnosticArray status, LiDAR cloud, IMU, CameraInfo, dataset-mode depth/seg image subs (857–862, `KeepLast(1000)` reliable), `ScovoxMap`/`ScovoxMapBinary`/cloud/planning publishers (879–898). Topic names come from parameters, not literals | Grep the parameter name (e.g. `input_pc_topic`) to trace a topic |
| ~1050–1350 | RGB-D path: ApproximateTime depth+seg sync, per-pixel back-projection, optional top-K soft-prob overlay via `TopkProvider` | |
| ~1450–1790 | LiDAR path: medoid downsample, deskew, per-ray `integrateHitFused` loop, `flushCarveFrame`, per-frame diag log + `/proc` RSS read (1766–1781) | The per-frame log tokens (`frame_ms`, `tsdf_ms`, …) are **parsed by experiment tooling** — do not rename/re-scope them |
| 1956–1990 | `publishScovoxMap` — full-map `ScovoxMap` message (subscriber- + `sm_dirty_`-gated) | Ships all active Beta cells incl. free space |
| 2056–2310 | `publishBinaryMap` — drains touched sets, applies change gates, rev-8 serialize + LZ4, chunked emit with byte budget + deferral FIFO | Zero-subscriber path currently drains-and-discards (audit item 2) |
| 2316–2500 | Gate helpers (`gateTauEff`, heartbeat walk), snapshot emit paths | |
| 2496–2601 | `publishPlanningMap` — 2-D occupancy/terrain grid for nav (called per scan, 1763) | Full-grid walk; audit item 4 |
| 2608–2745 | `publishPointCloud` — semantic cloud + per-voxel variance/EIG fields, reuses `pc_scratch_` | |
| 2749–2900 | `publishTSDFPointCloud` / fine cloud → `extractZeroCrossing` | |
| 3026–3060 | Gate-grid allocation (rolling-mode setup) | |
| 3104 | `main` — **`rclcpp::spin` on a SingleThreadedExecutor** | All callbacks serialize on one thread |

### 4.2 [dscovox_node.cpp](../src/scovox_mapping/src/dscovox_node.cpp) — the merger (~900 lines)

| Lines ≈ | Region | Notes |
|---|---|---|
| 60–190 | Includes, params, per-source grid bookkeeping | Includes `scovoxmap.hpp` only for the core re-exports |
| 194–230 | Publishers (fused cloud, fused `ScovoxMap` latched transient-local), one `ScovoxMapBinary` subscription per robot, `GetRegion`/`GetOccupancyGrid` services | |
| 256–263 | Diagnostics: `activeCellsCount` over all grids per tick (5 s-throttled log) | |
| ~380–560 | Binary ingest: LZ4 → deserialize → per-source grid write with `map_from_source` transform (per-binary INFO at 524–528) | The transform rides in the message — the merger needs **no TF tree** |
| ~560–608 | Delta-proportional **reset-then-refold consensus**: only cells whose source contribution changed are re-folded, via allocation-free scratch (`refold_beta_src_`/`refold_dir_src_`). The per-cell math lives in `dscovox_consensus.hpp` | The well-designed core — protect it |
| 609–688 | `publishPointCloud` — fused viz cloud (two full-grid walks; audit item 5) | |
| 694–787 | `regionOnGrid`/`fillRegion`/`publishFusedMap` — box queries + full fused-map message | |
| 794–902 | Service handlers, occupancy-grid projection, `main` | |

### 4.3 Headers ([include/scovox/](../src/scovox_mapping/include/scovox/))

- [node_utils.hpp](../src/scovox_mapping/include/scovox/node_utils.hpp) —
  shared node helpers: semantic color palette generation (10 fixed +
  golden-angle HSV), small formatting utilities. No node state.
- [scovoxmap.hpp](../src/scovox_mapping/include/scovox/scovoxmap.hpp) /
  [scovoxmap.cpp](../src/scovox_mapping/src/scovoxmap.cpp) — the **legacy
  unified `scovox::Map`** (single grid of `Voxel`, main + transient). The
  live mapper uses the split substrate instead; this class survives for the
  older tests and as the simplest self-contained example of the update math
  (`integrateRay` in ~500 lines). Good starting point for a minimal port.
- [topk_provider.hpp](../src/scovox_mapping/include/scovox/topk_provider.hpp)
  — file-backed loader/cache for per-frame soft-probability `.topk` blobs
  (binary layouts documented in the header). Only ROS dependency is the
  injected logger/clock.
- [dscovox_consensus.hpp](../src/scovox_mapping/include/scovox/dscovox_consensus.hpp)
  — **pure** receiver-side consensus helpers (at-prior tests, Beta+Dir →
  fused `SemBetaVoxel` projection, per-cell refold core). No ROS/Bonxai/Eigen.
  The node and its tests run the *same* code through this header.

### 4.4 Launch and config

| Launch file | Runs |
|---|---|
| `scovox_single_robot.launch.py` / `scovox_multi_robot.launch.py` | mapper(s), live sensors |
| `dscovox_single_robot.launch.py` / `dscovox_multi_robot.launch.py` | mapper(s) + merger, wire sharing on |
| `lidar_mapping.launch.py` | LiDAR profile |
| `scenenn_eval.launch.py`, `scenenet_eval.launch.py`, `scenenet_eval_fusion.launch.py`, `semantickitti_eval.launch.py` | dataset-replay evaluation profiles |

Config YAMLs in [config/](../src/scovox_mapping/config/): `default_params.yaml`,
`lidar_mapping.yaml`, `dscovox_params.yaml`, `scovox_bin_min.yaml` (minimal
wire profile). Parameter **defaults** live in the node's `declare_parameter`
block; YAMLs override them per profile — check both when tracing a value.

### 4.5 Tests

`test/`: `test_beta_update`, `test_dirichlet_update`, `test_tsdf_band`,
`test_semantic_audit`, `test_consensus`, `test_marching_cubes`,
`test_topk_provider` — these exercise the legacy `Map` and node-level helper
contracts; the split-substrate suites live in `scovox_core/test/`.

## 5. `scovox_msgs` — interface definitions

- **`ScovoxMapBinary.msg`** — the wire: `version`, `little_endian`,
  `map_from_source` (`geometry_msgs/Transform`, captured at publish so the
  merger needs no TF), `uint8[] data` (LZ4 blob). `header.frame_id` = source
  robot's integration frame — the merger keys per-source grids by it.
- **`ScovoxMap.msg`** — full-map snapshot: resolution, origin, thresholds,
  `ScovoxVoxel[]`.
- **`ScovoxVoxel.msg`** — position + `a_occ`/`a_free` + `a_unk` +
  `ScovoxSemanticEvidence[]` (class id + fractional count). Consumers derive
  p_occ and best-class themselves.
- **`RefinementRegion.msg`** — register/unregister a fine-TSDF trunk
  cylinder; field-compatible with the planner's TreeTarget.
- **Services:** `ExtractMesh` (min_weight + output path → PLY),
  `GetRegion` (AABB → `ScovoxMap` of that box), `GetOccupancyGrid`
  (z-range → `nav_msgs/OccupancyGrid`). All three served by the nodes as
  noted in §4.

## 6. `seg_pipeline` — segmentation front-end (Python)

[seg_node.py](../src/seg_pipeline/seg_pipeline/seg_node.py) — runs an outdoor
semantic segmentation model on the color stream and republishes seg + depth +
CameraInfo **all stamped with the depth frame's timestamp and a body
`frame_id`**, which is what makes the mapper's 0.05 s ApproximateTime sync
robust. `outdoor_palette.py` holds the class palette. Runs in Docker
(`compose.yaml`). Replaceable by anything that publishes the same three
topics with the same stamping contract.

## 7. Data flow (follow a measurement through the system)

**Single robot:**
sensor msg → node callback (§4.1) → pose from TF → per-ray
`ScovoxMapSplit::integrateHitFused` (TSDF band + Beta carve staging + Dir
hit) → `flushCarveFrame` (batched, leaf-sorted) → touched sets accumulate →
1 Hz timer → `publishBinaryMap`: drain touched → change gates → rev-8
serialize → LZ4 → `ScovoxMapBinary` out (+ `ScovoxMap`, clouds, planning
grid, each behind its own subscriber/dirty gate).

**Multi-robot:**
each robot's `ScovoxMapBinary` → merger ingest (§4.2): LZ4 → deserialize →
transform by `map_from_source` → per-source grid → delta-proportional refold
(`dscovox_consensus.hpp` math, `consensus_merge.hpp` algebra) → fused grids →
fused cloud / `ScovoxMap` / region services.

## 8. Reuse recipes — minimal file sets for other projects

`scovox_core` is deliberately ROS-free: everything below needs only a C++17
compiler, Eigen (for the walker/substrates), and liblz4 (wire only).
Bonxai is MPL-2.0 (file-level copyleft — keep its license header).

| You want | Take | Plus |
|---|---|---|
| A sparse voxel grid, nothing else | `include/third_party/bonxai/` | — (header-only, self-contained) |
| TSDF fusion (SLIM-VDB-style) | `tsdf_map.hpp/.cpp`, `tsdf_voxel.hpp`, `ray_iterator.hpp` | bonxai, Eigen |
| Bayesian occupancy + top-K semantics | `sem_split_map.hpp/.cpp`, `beta_voxel.hpp`, `dir_voxel.hpp`, `voxel.hpp` (for `K_TOP`/`sparse_add`), `semantics.hpp` | bonxai, Eigen |
| The full fused mapper (all three grids, one walker) | `scovox_map_split.hpp` | both rows above |
| The delta wire codec | `binary_serializer.hpp`, `lz4_codec.hpp` | voxel types, liblz4 |
| Multi-robot map merging (math only) | `consensus_merge.hpp`, `scovox_mapping/include/scovox/dscovox_consensus.hpp` | voxel types only — both are pure |
| Uncertainty measures (EIG, entropy, KL) | `uncertainty.hpp/.cpp` | `voxel.hpp`, `sembeta_voxel.hpp` |
| TSDF → mesh (+ labels) | `marching_cubes.hpp`, `mesh_labelling.hpp` | tsdf/sembeta voxels, bonxai, Eigen |
| Trunk DBH from a fine lattice | `dbh_fit.hpp`, `refinement_regions.hpp` | tsdf types, Eigen |
| Simplest possible reference implementation | `scovox_mapping`'s legacy `scovoxmap.hpp/.cpp` + `map_interface.hpp` | bonxai, Eigen — one class, ~500 lines, whole update math |
| A ROS node skeleton around any of the above | crib from `scovox_node.cpp` regions in §4.1 rather than copying the file | — |

## 9. Conventions and gotchas (read before editing)

1. **Accessor discipline.** Never call `grid.createAccessor()` per voxel —
   create once per loop/scan and reuse; the accessor caches the last leaf so
   same-leaf access is ~2 mask ops. `ConstAccessor` caches *null* leaves too:
   for read-modify-write use `setValue`/mutable accessor, or a stale null
   cache will hide the write.
2. **Zero = unobserved.** Bonxai zero-initialises new leaves; every voxel
   type treats all-zero as its unobserved sentinel (`weight == 0`,
   `a_occ == a_free == 0`, empty Dir slots). Never give a voxel type a
   default state that isn't all-zero.
3. **Voxel layouts are frozen by the wire.** 8/16/24 B sizes are
   `static_assert`ed and `reinterpret_cast` by the serializer and tests.
   A field change = codec revision bump + merger compatibility + published
   figures. Treat as an interface change with its own design doc.
4. **`drainTouched*` vs `clearTouched*`.** Drain = sort + unique + move-out
   (expensive, returns coords for the wire); clear = drop. If you consume a
   drain's result, you own emitting those voxels — the change gates assume
   every drained coord was considered.
5. **The hot walker is verification-gated.** Changes to
   `scovox_map_split.hpp` / the carve path are expected to reproduce
   **byte-identical** map digests across configs (this harness exists and has
   been used — see the design docs). Do not land hot-path changes without it.
6. **Timing tokens are an interface.** Per-frame log fields (`frame_ms`,
   `tsdf_ms`, per-ray brackets) are parsed by experiment tooling, and the
   bracket *scope* is deliberate — a past attempt to re-scope per-scan was
   reverted as a defect. Never rename, re-scope, or re-cadence them without
   checking `experiments/`.
7. **Parameters live in two places.** Defaults in the node's
   `declare_parameter` block; per-profile overrides in
   `config/*.yaml` + launch arguments. Grep both when tracing behavior.
8. **Build from the workspace root** (`scovox/`):
   `colcon build --packages-select scovox_core scovox_mapping`. Building from
   any other directory creates a stray `build/` and your binaries silently
   don't update. The install trees use **symlink-install for launch/config**
   (verified: `install/.../launch/*.py` are symlinks into `src/`) — editing a
   launch or YAML in `src/` immediately changes every install tree that links
   it, including pinned experiment installs (`install_*`). C++ changes, by
   contrast, reach nothing until the matching build tree is rebuilt.
9. **Multiple pinned build/install trees exist** (`build_*`/`install_*`) for
   past experiment configurations. Do not assume any of them matches git
   HEAD. The live development pair is `build/` + `install/`.
10. **Class-id conventions.** Slot/class 0 = unknown/unlabeled in the top-K
    blobs; `0xFFFF` = "no semantic voxel" sentinel from mesh/point labelling;
    `num_classes` is runtime (node default 14), `K_TOP` is compile-time (2).
11. **Hot-spot map.** Before optimising anything, read
    [design/efficiency_audit_2026_08_26.md](design/efficiency_audit_2026_08_26.md)
    — it ranks the known costs and lists the fixes already deemed
    out-of-bounds.
