# Mapping-stack efficiency audit — findings and suggested fixes

**Date:** 2026-08-26
**Status:** Findings log — review only. Nothing here has been implemented,
built, or benchmarked; every item is a suggestion with an estimated impact.
When one of these ships, move it to a dated "done" note or strike it here.
**Method:** read-only source review of the full mapping stack — the node,
the fused walker, both substrates, the Bonxai internals, the wire codec, and
the `dscovox` merger. No profiling was run for this audit. Where a percentage
is quoted it comes from the earlier measured campaigns and is labelled as such;
everything else is an estimate from reading the code.
**Scope:** `scovox_mapping` + `scovox_core` live code, and
[dscovox_node.cpp](../../src/scovox_mapping/src/dscovox_node.cpp). Excludes
experiments and offline tooling (marching cubes noted briefly since it backs a
live publisher).
**Non-goals:** re-scoping the per-ray timing brackets to per-scan (that exact
change was tried and reverted as a defect — it silently changed what `tsdf_ms`
measures under an unchanged log token); any change to `batch_free_carve`
semantics; anything that alters integration output, unless explicitly flagged
as behaviour-changing.

## What is already good (protect these)

The problems below are mostly *around* well-optimized cores, not in them:

- The rev-8 delta wire codec (block-run coords, u8 companded evidence, LZ4,
  byte-budget deferral) in
  [binary_serializer.hpp](../../src/scovox_core/include/scovox/binary_serializer.hpp)
  is tight — reserve-before-fill, DoS guards, deterministic block streams.
- The batched carve flush is block-sorted by leaf before touching the grid.
- Bonxai `Accessor`/`ConstAccessor` caching is used correctly on the hot paths.
- `pc_scratch_` is reused across point-cloud publishes.
- `publishBinaryMap` is gated by `sm_dirty_`, so an idle map costs nothing
  on the wire path.
- The far-voxel skip (when armed) is verified byte-identical, 1.69× measured.
- `dscovox`'s delta-proportional reset-then-refold uses allocation-free
  scratch buffers.

## Priority list — biggest expected win first

### 1. Free-space carving pays a full float body + a hash probe per walked voxel, and the far-skip cannot arm

**Status, part (a): IMPLEMENTED 2026-08-26 (Run 7a, edit-only — build/test/digests deferred to the batched pass).** New `carve_stage.hpp` (`CarveStage`): staging keyed by leaf block (`coord >> leaf_bits`) into a reused open-addressed index (Teschner-prime hash, load ≤ ½) of dense per-block weight arrays + occupancy bitmask, with a last-leaf cache for ray-coherent adds; flush sorts only the block keys (same ascending `(x>>lb,y>>lb,z>>lb)` order the retired per-voxel sort produced — root-map first-touch order and hence serialized bytes preserved; within-block order was never observable) and walks the bitmask. Replaces the `unordered_map<CoordT,float>` member and the per-scan sort in `flushCarveFrame`. Differential gate `test_carve_stage.cpp` (vs the retired staging verbatim: staged set, max weights, block sequence; lb=1 and 3, negative coords, index growth, frame reuse). Part (b) is Run 7b.

**Status, part (b): IMPLEMENTED 2026-08-26 (Run 7b, edit-only — build/test/digests deferred to the batched pass).** Far-voxel fast CARVE in `integrateHitFused`: the `batch_free_carve=true` counterpart of the far skip. Arms on `!space_carving && carveFrameOpen() && batch_free_carve && depth >= trunc` (mutually exclusive with the skip via the batch flag); beyond the SAME `far_thr` a voxel's exact body reduces to the carve branch alone (TSDF and band gates provably fail; no behind-surface voxel reaches `far_thr`; `depth >= trunc` keeps every walk voxel in front of the origin), so it is carved with integer-only Chebyshev tests — no `vc`/`dist`/`proj` floats — with the `carve_blocked` latch guard verbatim. Voxels within Chebyshev 2 of the EXACT origin voxel (recomputed via `posToCoord(origin)`, immune to the ±1 `k0` float-reconstruction wobble) fall through to the exact body, absorbing the `dist > carve_band` / `proj -> 0` degeneracies near the sensor. Kill-switch `SCOVOX_DISABLE_FAR_CARVE`, latched per instance like the skip's. Differential gate in `test_scovox_map_split.cpp` (`ScovoxMapSplitFarCarve`): byte-identity full-walk-vs-fast on the shipped batched config over both band arms plus short rays straddling `depth == trunc`, and a `space_carving=true` inertness config (the one observable arming conjunct). Post-review hardening: latch-canary accessors (`farSkipDisabled()`/`farCarveDisabled()`) asserted in both identity tests so a broken kill-switch latch cannot silently turn them into fast-vs-fast; and the numeric operating envelope the reduction rests on — |world coord|/res ≲ 8×10⁷, ray length/res ≲ 3×10⁷, an assumption shared with the far skip — documented at the arming site.

**Where:**
[sem_split_map.cpp#L235-L251](../../src/scovox_core/src/sem_split_map.cpp#L235-L251)
(staging),
[sem_split_map.hpp#L508-L509](../../src/scovox_core/include/scovox/sem_split_map.hpp#L508-L509)
(the `unordered_map`/`unordered_set` members),
[scovox_map_split.hpp#L258-L261](../../src/scovox_core/include/scovox/scovox_map_split.hpp#L258-L261)
(the far-skip arming condition).

**Problem:** the single largest frame-time item on dense RGB-D. The measured
campaigns put free-space handling at **~59 % of frame time on SceneNN** and
~7 % on KITTI. Two compounding causes:

- Every carved voxel on every ray does `carve_stage_.find(c)` plus an
  `emplace`/update — two hash probes on first touch, one on every later touch,
  millions of node-based hash operations per indoor scan. The flush then
  re-sorts the entire staging map into a vector each scan.
- The far-voxel skip only arms when `!batch_free_carve`, i.e. when carving is
  **off**. In the default carving-on config every interior ray voxel pays the
  full body: inlined `coordToPos`, two vector subtractions, `.norm()` (sqrt),
  a dot product, and the branch chain — even though a voxel far from the
  endpoint can only ever take the carve branch.

**Suggested fix (two independent parts):**

(a) Per-leaf staging: key by leaf coordinate (`coord >> leaf_bits`) into a
small open-addressed table whose slots are 512-entry float arrays + a bitmask,
reused across frames. Same-leaf ray steps become an array write instead of a
hash probe, and the flush no longer needs its per-scan sort — entries are
already grouped by leaf.

(b) Extend the far-skip idea to the carve-on path: beyond the Chebyshev far
distance from the hit, the carve test needs only the *sign* of the projection
(no norm/sqrt) — stage the carve directly. Keep two guard rings on the exact
body: the existing +1 ring at the hit end, and a ring at the **origin** end,
because when `trunc > depth` there is a thin near-origin shell where `dist`
exceeds the carve band. The order-dependent `carve_skip_occ_threshold` latch
is off by default; the fast path still walks in DDA order so the latch
semantics hold when armed, but that is the configuration to test hardest.

**Impact:** speed **high** (dense RGB-D; low–medium on LiDAR) · memory low
(removes the transient hash map).
**Risk:** **medium** — numerics-adjacent. The byte-identical digest
verification recipe used for the original far-skip is the acceptance gate.

### 2. Touched-set churn: duplicate pushes, per-scan sort — then discarded in the default config

**Status: IMPLEMENTED 2026-08-26 (Run 4, edit-only — build/test deferred to the batched pass).** All `drainTouched*` now return a `const&` into a member scratch (swap, sort+unique in place — capacity recycled on both sides); the eight drain-and-discard sites in `publishBinaryMap` are O(n) clears (`clearTouchedBeta`/`clearTouchedDir` added); bonus: the two persistent-mode no-`bin_pub_` blocks also cleared Fine, fixing a latent unbounded-growth leak when refinement regions run without a binary subscriber.

**Where:**
[tsdf_map.cpp#L181](../../src/scovox_core/src/tsdf_map.cpp#L181) (push per
band write),
[tsdf_map.cpp#L189-L205](../../src/scovox_core/src/tsdf_map.cpp#L189-L205)
(drain),
[sem_split_map.cpp#L696-L721](../../src/scovox_core/src/sem_split_map.cpp#L696-L721)
(same pattern ×3), and the key site
[scovox_node.cpp#L2067-L2070](../../src/scovox_mapping/src/scovox_node.cpp#L2067-L2070).

**Problem:** every TSDF band write pushes its coord onto `touched_` — ~7–13
pushes per ray, duplicates included — and each drain runs a full
`sort` + `unique` over millions of entries per scan. Three multipliers:

- **With no `scovox_bin` subscriber** (the standard single-robot eval config)
  `publishBinaryMap` still calls all four drains and throws the result away:
  `(void)split_map_->drainTouchedTsdf(); …Beta(); …Dir(); …Fine();` — a full
  O(N log N) sort per scan, computed to be discarded. The same
  drain-and-discard fires for individually disabled streams at
  [scovox_node.cpp#L2198](../../src/scovox_mapping/src/scovox_node.cpp#L2198),
  [#L2218](../../src/scovox_mapping/src/scovox_node.cpp#L2218), and
  [#L2305](../../src/scovox_mapping/src/scovox_node.cpp#L2305) — and
  `share_tsdf` / `share_dir` default false, so two of these run on every
  *emitting* tick as well.
- The drain returns `std::move(touched_)`, so the vector's multi-megabyte
  capacity is freed and re-grown from zero every scan.
- Duplicates inflate N by roughly the band width before the sort runs.

**Suggested fix:** (1) on non-emitting paths call the existing
`clearTouchedTsdf()` / `clearTouchedFine()` (already used on the LiDAR path at
[scovox_node.cpp#L1760-L1762](../../src/scovox_mapping/src/scovox_node.cpp#L1760-L1762))
instead of void-drains — `SemSplitMap` needs a one-line `clearTouchedDir()`
added; (2) keep capacity by swapping with a member scratch instead of moving
out; (3) optionally dedup at source with a "last leaf+offset" check per ray.

**Impact:** speed **high** per effort (the discard fix alone deletes a
per-scan sort in the default config) · memory **medium** (steady-state
allocator churn).
**Risk:** **low** — clear-instead-of-discard is behaviour-identical.

### 3. O(map) publishers on the single executor thread; cloud publishers have no dirty gate

**Status: IMPLEMENTED 2026-08-26 (Runs 4+6, edit-only — NOT yet built/tested).** Run 4: per-publisher dirty gates `pc_dirty_`/`tsdf_pc_dirty_` (set via a single `markMapDirty()` at the three mutation sites, consumed after the subscriber gate so an unsubscribed tick keeps the flag, and a subscriber-count RISE re-arms one publish so a late/re-subscriber on a quiescent map is not stranded — review finding, mirrors `publishBinaryMap`'s `prev_sub_count_`); `extractZeroCrossing` reserve (+ conditional shrink); `publish_uncertainty_fields` param (default off) that OMITS posterior_variance/eig from the cloud schema — omission, not zero-fill, so `pointcloud_to_npz.py`'s `if name in fields` copy degrades gracefully (checked: it is the only live consumer, and the uncertainty campaign reads offline snapshots). Run 6: the executor change — viz timer (`sm_timer_`) moved to its own MutuallyExclusive callback group; `main()` now runs a 2-thread `MultiThreadedExecutor`; ALL other callbacks stay in the default (also mutually exclusive) group, so the serialization that the node's lock-free plain members rely on (`loc_*` gate state, `imu_buf_`, `di_`, dataset caches, `frame_recv_`) is preserved group-locally — audited every viz-path member: map state is under the timer's `shared_lock(map_mtx_)` (integration already takes unique at its two sites), dirty flags are atomics, and `pc_scratch_`/prev-sub-counts are written only inside the viz group. Net effect: an O(map) viz walk overlaps a scan's pre-lock phase (decode, TF, snapshot assembly) instead of occupying the sensor callbacks' executor slot; headless eval runs are unaffected (viz bodies early-out on zero subscribers). Four stale "SingleThreadedExecutor serializes…" comments rewritten to name the default-group guarantee.

**Where:**
[scovox_node.cpp#L3104](../../src/scovox_mapping/src/scovox_node.cpp#L3104)
(single-threaded spin), timer at
[scovox_node.cpp#L163-L178](../../src/scovox_mapping/src/scovox_node.cpp#L163-L178),
[publishPointCloud #L2608](../../src/scovox_mapping/src/scovox_node.cpp#L2608),
[publishTSDFPointCloud #L2749](../../src/scovox_mapping/src/scovox_node.cpp#L2749)
→ [extractZeroCrossing, marching_cubes.hpp#L700-L764](../../src/scovox_core/include/scovox/marching_cubes.hpp#L700-L764).

**Problem:** everything shares one executor thread, so a publisher that walks
the whole map stalls integration for its full duration — and that duration
grows with map size, which is exactly the "slows down as the map grows"
failure mode. `publishScovoxMap` is at least `sm_dirty_`-gated; the two cloud
publishers are only *subscriber*-gated, so with RViz attached they re-walk the
entire map every tick even when nothing changed. The TSDF cloud additionally
probes 3 neighbours per active voxel and `push_back`s with no reserve. Each
cloud voxel also gets `posterior_variance` and `eig` filled at
[scovox_node.cpp#L2729](../../src/scovox_mapping/src/scovox_node.cpp#L2729):
`expectedInformationGain`
([uncertainty.cpp#L63-L85](../../src/scovox_core/src/uncertainty.cpp#L63-L85))
is 3 digamma evaluations (each a recurrence loop + log + polynomial,
[uncertainty.cpp#L13-L30](../../src/scovox_core/src/uncertainty.cpp#L13-L30))
plus 2 logs, and variance adds 3 more digammas — ~6 transcendental-heavy calls
per voxel per tick for fields only an uncertainty consumer reads.

**Suggested fix:** (1) a `viz_dirty_` flag set alongside `sm_dirty_`,
early-out when clean; (2) move visualization publishers to a separate callback
group under a `MultiThreadedExecutor` (or a snapshot thread) so they contend
on the shared lock instead of blocking sensor callbacks; (3) `reserve` from
`activeCellsCount()` in `extractZeroCrossing`; (4) parameter-gate the
eig/variance fill (default off).

**Impact:** speed **high** at large map scale · memory low.
**Risk:** low for gate/reserve; **medium** for the executor change.

### 4. Planning map: full-grid walk per scan for a fixed-size window, default ON

**Status: IMPLEMENTED 2026-08-26 (Run 4, edit-only).** New file-local `forEachCellInBox` mirrors Bonxai's `forEachCell` descent but rejects whole root blocks (and leaf blocks, and cells) outside an inclusive coord box; `publishPlanningMap` derives the box from the window (±1 voxel slack; terrain mode leaves z open with ±INT32_MAX/2 sentinels) and keeps every per-cell predicate unchanged, so output is identical. Terrain column map is now the persistent member `plan_cols_`. Launch check PASSED: all 16 eval/experiment launches set `publish_planning_map: False`; only the operational single/multi-robot launches enable it.

**Where:** called per scan at
[scovox_node.cpp#L1763](../../src/scovox_mapping/src/scovox_node.cpp#L1763),
body at
[scovox_node.cpp#L2496-L2601](../../src/scovox_mapping/src/scovox_node.cpp#L2496-L2601).

**Problem:** `publishPlanningMap` runs from every sensor callback
(subscriber-gated, but `publish_planning_map` defaults **true**). It walks
**every active Beta cell** via `forEachCell` even though the output is a
fixed-size local window around the robot. Terrain mode also builds a fresh
`unordered_map` of per-column z-vectors and sorts each column, and obstacle
inflation copies the whole occupancy grid and does an r² neighbourhood pass
per cell. O(map) + allocations, per scan, on the executor thread, for a
constant-size output.

**Suggested fix:** iterate only leaf blocks whose bounds intersect the window
(the root map's keys give this with integer compares per block) — cost becomes
O(window) regardless of map size. Keep the column scratch as a persistent
member. Consider a 2–5 Hz timer instead of per-scan. Also confirm eval
launches actually disable it, so it is not silently taxing benchmark runs.

**Impact:** speed **high** (per-scan O(map) → O(window)) · memory low.
**Risk:** **low** — iteration-domain change only, identical output.

### 5. dscovox: full-map nested-vector message per dirty tick + double full-grid walk for the cloud

**Status: IMPLEMENTED 2026-08-26 (Run 5, edit-only).** Viz cloud: `pc_dirty_` set on binary ingest, checked after a subscriber-rise re-arm (`pc_prev_subs_` atomic exchange before the zero-subscriber return, so detach→reattach counts as a rise on a KeepLast(1) non-latched topic); single collect pass into `static thread_local` scratch, then one sized fill — the count pass is gone. `regionOnGrid`: `reserve` bounded by min(`activeCellsCount()`, bbox volume) with per-axis extent guards, then a windowed walk via the shared `scovox::forEachCellInBox` (node_utils.hpp) using the clip's **own** `mn`/`mx` as the window, so the visited set equals the clip-accepted set for every request corner — responses byte-identical. `occupancyGridOnGrid`: z-band window (floor(z/res)±1 slack, ±INT32_MAX/2 sentinels for non-finite bounds); the metric per-cell predicate stays the sole membership authority. The delta topic stays parked (contract change).

**Where:**
[dscovox_node.cpp#L770-L787](../../src/scovox_mapping/src/dscovox_node.cpp#L770-L787)
(`publishFusedMap`),
[dscovox_node.cpp#L694-L752](../../src/scovox_mapping/src/dscovox_node.cpp#L694-L752)
(`regionOnGrid`/`fillRegion`),
[dscovox_node.cpp#L609-L688](../../src/scovox_mapping/src/dscovox_node.cpp#L609-L688)
(`publishPointCloud`).

**Problem:** ingest/refold is well designed; the outputs are the weak point.
When subscribed, `publishFusedMap` serializes the **entire fused map** into a
`ScovoxMap` message every dirty tick — `push_back` with no `reserve`, a heap
`semantic_evidence` vector per voxel, all under the timer's shared lock — and
the latched `transient_local` QoS keeps a full serialized copy alive in DDS
permanently. The viz cloud walks the whole fused grid **twice** (count pass,
fill pass) and is triggered from every received binary tail at up to 10 Hz.
`regionOnGrid` bbox-rejects *inside* the per-cell lambda, visiting every voxel
in the map to select a region.

**Suggested fix:** `reserve` from `activeCellsCount()`; single-pass cloud fill
into a persistent scratch (the main node's `pc_scratch_` pattern); dirty-gate
the cloud independently of binary arrival; skip non-intersecting leaf blocks
in `regionOnGrid` via root keys. Longer term: publish the fused map as deltas
(the rev-8 codec already exists) rather than a full nested-vector message.

**Impact:** speed **high** for multi-robot runs · memory **medium–high**
(latched copy + per-voxel vectors).
**Risk:** low for reserve/single-pass/block-skip; medium for a delta topic
(contract change for consumers).

### 6. `publishScovoxMap` ships every carved free-space voxel

**Status: IMPLEMENTED 2026-08-26 (Run 6, edit-only — NOT yet built/tested).** The consumer check came back decisive (fresh-agent enumeration of every `ScovoxMap` consumer): `explo_planner_node` in the sibling `hmr_explo_ws` workspace consumes free voxels as LOAD-BEARING input — frontier extraction requires `p_occ < 0.5` cells, coverage termination counts free-swept columns, and absent cells score as max-EIG unknown — and the coupling config lives inside this repo (`config/exploration_fused_bag.yaml`). So the filter is opt-in exactly as the ⚠ required: new param `scovox_occupied_only` (default **false** = ship everything, current behavior everywhere), which when enabled skips voxels with `p_occ() < min_occ_` — the same `occupancy_threshold` the message itself advertises, so a consumer that binarizes at the advertised threshold sees an identical occupied set. Safe to enable for `tree_detector_node` (already discards `p < occ_thresh`); `simple_nav_3d` uses the dscovox GetRegion service, not this topic; zero Python/bag/RViz consumers found.

**Where:**
[scovox_node.cpp#L1956-L1990](../../src/scovox_mapping/src/scovox_node.cpp#L1956-L1990).

**Problem:** the walk emits **all** active Beta cells with no occupancy/prior
filter. With full carving on (the default), free space vastly outnumbers
surface voxels — on outdoor LiDAR the message is dominated by empty air, each
entry with a heap `semantic_evidence` vector. Dirty + subscriber gates exist,
but a firing tick is O(carved volume), not O(surface).

**Suggested fix:** a parameter filtering to non-prior / occupied voxels (or a
compact side stream for free space if a consumer needs it).
**⚠ Check consumers first:** if anything downstream reconstructs free space
from this topic, the filter must be opt-in.

**Impact:** speed **medium** · memory **medium** (message + DDS buffers).
**Risk:** low–medium pending the consumer check.

### 7. Gate shadow grids: 40 B per emitted voxel, 16 B dead by default

**Status: IMPLEMENTED 2026-08-26 (Run 6, edit-only — NOT yet built/tested).** The `GateBeta`/`GateDir` wrapper structs are gone: `gate_beta_`/`gate_dir_` are now raw `VoxelGrid<BetaVoxel>` (8 B) / `VoxelGrid<DirVoxel>` (16 B), and the per-voxel `t_emit` doubles live in separate `VoxelGrid<double>` twins (`gate_beta_t_`/`gate_dir_t_`) allocated ONLY when `share_heartbeat_sec > 0` — so the default config sheds the full 16 B/voxel of dead stamp weight (40→24 B of shadow state). Stamp stays `double`, not float/u32: epoch seconds exceed float's exact-integer range, and the twin grid is heartbeat-armed-only so there is nothing to save by default. The twins are written in lockstep at EVERY wire emit (both snapshot and delta paths go through the same emit lambdas), which makes the twin's active set and Bonxai iteration order identical to the gate grid's — the heartbeat walk (extracted to `scovox::heartbeatReemit` in node_utils.hpp, shared Beta/Dir via accept/emit lambdas) walks the stamps twin and produces byte-identical wire output and order to the pre-split combined-struct walk. Gate COMPARISON values untouched (`betaChangedSinceEmit`/`dirChangedSinceEmit` unchanged — the wire-delta semantics invariant). Regression test added: `test/test_heartbeat.cpp` (registered in `SCOVOX_TEST_SOURCES`) covers stale re-emit + gate refresh + stamp advance, fresh skip, the ==period boundary, accept-veto and missing-live-voxel both leaving the stamp untouched (retry-next-tick semantics), the Dir binarize-style veto, and a split-vs-pre-split reference-equivalence walk over multi-root coords comparing emitted (coord, value) SEQUENCE and post-state.
[scovox_node.cpp#L3026-L3029](../../src/scovox_mapping/src/scovox_node.cpp#L3026-L3029),
structs at
[scovox_node.cpp#L138-L148](../../src/scovox_mapping/src/scovox_node.cpp#L138-L148).

**Problem:** change-gating keeps `GateBeta{BetaVoxel; double t_emit}` (16 B)
and `GateDir{DirVoxel; double t_emit}` (24 B) per emitted voxel — 40 B of
shadow state against 24 B of live payload (1.67×), scaling with map size. The
`t_emit` doubles are only read by the heartbeat walk, and `share_heartbeat_sec`
defaults to 0.0, so 16 B per voxel is dead weight in the default config.

**Suggested fix:** allocate `t_emit` only when the heartbeat is armed
(separate parallel grid, or a compile-time/variant split of the gate structs);
optionally store it as float / u32 seconds-since-start. The gate *comparison*
values must stay exact — they define wire-delta semantics.

**Impact:** memory **medium** (≈40 % of gate overhead) · speed none.
**Risk:** low–medium; needs a heartbeat-armed regression test.

### 8. Dataset mode: `KeepLast(1000)` reliable queues on both image topics

**Status: IMPLEMENTED 2026-08-26 (Run 6, edit-only — NOT yet built/tested) — parameterized, default UNCHANGED, and the ⚠ resolved AGAINST shrinking.** A fresh-agent replay-pacing audit proved the deep queue is the lossless-replay mechanism: every image replay node (`scenenet_replay_node` + subclasses, `replica_replay_node`, `gazebo_replay_node`) is open-loop — fixed-rate timer, no backpressure, only a one-shot startup discovery wait — and RELIABLE provides none either (a KEEP_LAST reader acks samples as it substitutes them out, so the writer never blocks; verified empirically: replay held 4.000 Hz while the mapper ran 30+ s behind). Measured peak queue occupancy on PASSING eval cells: **176 frames** (`scenenet_soft_e2/0_182`), 161 (`s1_ktop/K20`), 158 (K2) — the mapper integrates ~17% slower than the 4 Hz feed so the backlog grows monotonically all sequence. `KeepLast(200)` leaves only 12–24% margin and a slower host/finer res/heavier mechanism config would silently drop frames (invisible in artefacts: a DDS-dropped frame produces no `recv=` line at all); `KeepLast(100)` drops on cells that pass today. The real invariant is depth > longest replay sequence. Also: sizing to the 200-entry pairing caches is a category error — those hold only *unmatched* halves and empirically stay near-empty (300/300 matches, zero overflows, in a run with a 130-frame queue backlog). Fix as landed: `dataset_queue_depth` param (default 1000), with the mechanism + measurements documented at the QoS site. Shrink only together with replay-side pacing (the `ssmi_bridge_node` `max_in_flight` pattern). The ~2 GB RSS estimate revises to ~95 MB at the measured peak for SceneNet 320×240 (≈4× per-frame at SceneNN VGA) — if memory bounding is still wanted later, bound by *bytes*, not count.
[scovox_node.cpp#L857-L862](../../src/scovox_mapping/src/scovox_node.cpp#L857-L862).

**Problem:** dataset mode subscribes depth *and* segmentation with
`KeepLast(1000).reliable()`. When replay outruns integration, up to 1000
depth + 1000 seg images sit in DDS buffers — on the order of ~2 GB transient
RSS at VGA float depth (estimate), which also inflates any memory numbers
recorded during evals.

**Suggested fix:** `KeepLast(100–200)` (matching the bounded pairing caches),
or a parameter. **⚠** If the eval protocol relies on lossless replay with no
publisher-side pacing, shrinking this can drop frames — size it to the pairing
cache and confirm the bag player paces.

**Impact:** memory **medium–high** in dataset mode · speed none.
**Risk:** low–medium (eval-protocol dependent).

### 9. `std::function` weight callback per ray, indirect call per band voxel

**Status: IMPLEMENTED 2026-08-26 (Run 4, edit-only).** Plain-float `applyBandUpdate(c, sdf, w)` overload holds the original body; the fused walker passes `constexpr float 1.0f` (no `std::function` on the hot path). The `WeightFn` overload gates on the band FIRST and then delegates, so `weight_fn` is still evaluated only in-band with the same argument — bit-identical, and `integrateRay` call sites are untouched.

**Where:**
[scovox_map_split.hpp#L272](../../src/scovox_core/include/scovox/scovox_map_split.hpp#L272),
consumed at
[tsdf_map.cpp#L170](../../src/scovox_core/src/tsdf_map.cpp#L170).

**Problem:** `TsdfMap::constant(1.0f)` builds a `std::function` per ray, and
every band voxel (~7–13 per ray) pays a non-inlinable indirect call — in the
hottest loop in the system — to return a constant.

**Suggested fix:** template the band update on the functor type, or add a
plain-float overload for the constant-weight case (the only case the node
uses). ~Ten lines.

**Impact:** speed **low–medium** · memory none. **Risk:** **low**.

### 10. BKI/spread kernel: sqrt per offset, ~half outside support, recomputed per hit

**Status: IMPLEMENTED 2026-08-26 (Run 5, edit-only).** In-support offset+weight list precomputed once when the radii are configured (same `d = res·sqrt(dx²+dy²+dz²)` expression, same iteration order over the surviving offsets); the per-hit loop iterates the compact list with zero transcendentals. Default config (both radii 0) untouched.

**Where:**
[sem_split_map.cpp#L531-L572](../../src/scovox_core/src/sem_split_map.cpp#L531-L572).

**Problem:** with `kernel_radius`/`spread_radius` > 0, each hit loops the full
(2R+1)³ cube computing `d = res·sqrt(dx²+dy²+dz²)` **before** the
compact-support test — ~48 % of offsets are outside spherical support, so
nearly half the sqrts are discarded. The (dx,dy,dz) → weight mapping is
invariant per (resolution, length-scale) yet recomputed for every hit.
(The trig is *not* per-offset — only the sqrt is.)

**Suggested fix:** precompute, at construction, a compact list of in-support
offsets with weights; the per-hit loop becomes iterate-and-accumulate with
zero transcendentals.

**Impact:** speed **medium** when armed (RGB-D overlay profiles); **none in
the default config** (both radii default 0). **Risk:** **low**.

### 11. TF: 600 s buffer cache + blocking waits on the executor thread

**Status: IMPLEMENTED 2026-08-26 (Run 6, edit-only — NOT yet built/tested) — parameterized, defaults UNCHANGED, and the ⚠ resolved AGAINST shrinking the cache.** The replay-pacing audit (item 8) settles the cache question with a measurement: the mapper falls up to ~44 s of TF-stamp span behind the open-loop replay feed, so the exact-stamp lookup for frame N needs history far older than the newest broadcast — a 10–30 s cache would break every current eval cell. Fix as landed: `tf_cache_time_sec` param (default 600.0, declared in the member initializer — safe because the Node base is fully constructed before member init), shrink only for live runs with a real-time TF source. The blocking waits: onImages' two hardcoded 0.2 s exact-stamp waits now use a new `rgbd_tf_timeout_sec` param (default 0.2 — deliberately SEPARATE from `tf_lookup_timeout_sec`, which shipped configs raise to 1.0 for GLIM's LiDAR TF latency; reusing it would silently change the image path in `scovox_fused_lidar_rgbd.yaml`). Setting either knob to 0.0 gives exactly the suggested zero-timeout-drop behavior per-config; the DEFAULT keeps the wait because it is load-bearing live (GLIM broadcasts the pose after the scan arrives — a zero-timeout default would reintroduce the stale-pose smear the wait exists to prevent) and a frame-0 discovery race in replay could otherwise drop a frame and change eval output. Requeue-once was rejected: re-injection changes integration order, which is tie-break-visible. Bonus correction from the same audit, verified in-code: the zero-timeout comment in `publishBinaryMap` claimed the TF listener shares the executor thread — false (`tf_listener_(tf_buffer_)` is the `spin_thread=true` overload with a dedicated thread, which is also why 30+ s-lagging eval runs show zero TF failures); comment rewritten with the correct rationale (Time(0) lookup ⇒ already cached if it exists), conclusion unchanged. The two 0.05 s `Time(0)` fallback waits (deskew extrinsic, planning-map crop) only ever wait while the buffer is still empty at startup and were left alone.
[scovox_node.cpp#L54](../../src/scovox_mapping/src/scovox_node.cpp#L54);
0.2 s-timeout lookups in the scan callbacks; 0.05 s in the planning map.

**Problem:** the TF buffer keeps 600 s of history — 60× the default — which at
a 100 Hz TF rate retains ~60 k transforms per frame pair (memory, plus lookup
cost grows with depth). The scan callbacks block up to 200 ms on
`lookupTransform` on the single executor thread: one late TF stalls *all*
callbacks instead of deferring one scan.

**Suggested fix:** shrink the cache toward 10–30 s — **⚠ verify first that
dataset replay does not rely on the deep buffer** (bags with sparse or global
TF would). Replace blocking waits with zero-timeout `canTransform` +
requeue-once.

**Impact:** speed low (stall spikes) · memory **low–medium**.
**Risk:** low–medium (replay dependency).

### 12. Per-frame logging: /proc parse + 18-field INFO per scan; dscovox per-binary INFO + full diag counts

**Status: IMPLEMENTED 2026-08-26 (Run 5, edit-only).** `sampledVmRSSKB()` behind a new `rss_sample_every` parameter, **default 1 = parse /proc every frame** (current behaviour and the `rss_mb=` token cadence preserved exactly, per the ⚠ below); >1 returns the cached value between samples. dscovox: the diag `activeCellsCount()` walks now run only when the 5 s throttle will actually fire, via a manual replication of Humble's `RCLCPP_INFO_THROTTLE` (fire when `now >= last + 5 s`, `last` starts 0) so the emitted line and its cadence are byte-identical.

**Where:**
[scovox_node.cpp#L1766-L1781](../../src/scovox_mapping/src/scovox_node.cpp#L1766-L1781);
dscovox
[#L524-L528](../../src/scovox_mapping/src/dscovox_node.cpp#L524-L528) and
[#L256-L263](../../src/scovox_mapping/src/dscovox_node.cpp#L256-L263).

**Problem:** every frame opens and parses `/proc/self/status` (`getVmRSSKB`)
and formats an 18-field INFO line on the executor thread. dscovox logs per
received binary and computes `activeCellsCount()` over the fused grid *plus
every source grid* on **every** publish tick, feeding a log line that is
5 s-throttled anyway.

**Suggested fix:** sample RSS on the existing every-10th-frame cadence;
compute dscovox diag counts only when the throttled log will fire.
**⚠ Do not simply demote the log line:** the experiment parsers grep these
per-frame tokens (`frame_ms` etc.) — cadence must be a parameter with the
current default preserved, or bench tooling breaks.

**Impact:** speed **low–medium** (fixed per-frame cost, worst at LiDAR rates)
· memory none. **Risk:** low if parameterized as above.

### 13. Wire-path copies: LZ4 decompress round-trips an extra full-payload buffer

**Status: IMPLEMENTED 2026-08-26 (Run 5, edit-only), receive side only.** `decompressLZ4` now decompresses directly into `std::string(orig_size, '\0')` — the intermediate `std::vector<char>` copy is gone. **Correction to this item's send-side claim:** the send side already did `bin.data = std::move(comp)` — there was no serialized-string → vector → message copy chain to remove, so no send-side change was made.

**Where:**
[lz4_codec.hpp#L59-L64](../../src/scovox_core/include/scovox/lz4_codec.hpp#L59-L64);
plus the serialize → compress → message copy chain on the send side.

**Problem:** `decompressLZ4` decompresses into a `std::vector<char>` and then
constructs the returned `std::string` from it — a second full copy of every
received payload (matters most in dscovox, which ingests peer deltas
continuously). The send side goes serialized-string → LZ4 vector →
`msg.data` copy.

**Suggested fix:** decompress directly into `std::string(orig_size, '\0')`
(or return the vector and adjust the two call sites); compress straight into
the message's `data` field on the send side.

**Impact:** speed **low** (bandwidth-proportional, not map-proportional) ·
memory low. **Risk:** **low**.

### 14. Smaller items

**Status: PARTIALLY IMPLEMENTED 2026-08-26 (Run 5, edit-only).** Done: medoid scratch → members + `clear()`; `gateTauEff` → exact-double memo keyed on the evidence total (**correction: the item over-counted — there is exactly one `std::pow` per call, not two**; the memoised value is the same exact double, wire gate semantics untouched); marching-cubes anchor re-fetch removed in **both** `extractMesh` overloads (corner 0 and the semantic-label read now use the visited cell directly); the dscovox GetRegion/occupancy clips got the item-5 windowed walks. The `extractZeroCrossing` reserve (+ overshoot shrink) landed earlier, in Run 4's item-3 pass (TsdfVoxel overload — the one cited below). Not done: binary PLY (parked), the two per-ray `steady_clock::now()` compile-gate (Run 7 territory — touches the timing bracket), and the legacy Voxel-typed `extractZeroCrossing`'s reserve (dormant legacy path, left alone).

- **Medoid downsample scratch rebuilt per scan** (LiDAR path,
  ~[scovox_node.cpp#L1560-L1620](../../src/scovox_mapping/src/scovox_node.cpp#L1560-L1620)):
  per-scan `unordered_map` + vectors → members + `clear()`. Speed low–medium,
  risk low.
- **`gateTauEff`: two `std::pow` per gated voxel** per emit tick → precompute
  or cache per evidence bucket. Speed low, risk low.
- **Marching cubes:** `extractZeroCrossing` has no `reserve`; `extractMesh`
  re-fetches the anchor voxel it already holds
  ([marching_cubes.hpp#L700-L764](../../src/scovox_core/include/scovox/marching_cubes.hpp#L700-L764));
  the PLY writer is ASCII `ofstream<<` (binary PLY ≈ 5–10× faster to write).
  Offline-path, low.
- **Two `steady_clock::now()` per ray** in the fused walker (~2 ms/scan on
  dense scans): could be compile-gated. **Per the non-goals:** keep per-ray
  semantics — only gate, never re-scope to per-scan.
- **dscovox `GetRegion` / occupancy-grid service** clip inside the per-cell
  lambda
  ([dscovox_node.cpp#L694-L725](../../src/scovox_mapping/src/dscovox_node.cpp#L694-L725))
  — same leaf-block pre-skip fix as item 4.

### 15. Flagged, higher-risk: shrink `BetaVoxel` 8 B → 4 B

**Where:** the Beta grid — the largest structure in full-carve maps
(`BetaVoxel` is 2× float with evidence saturated at ~1000 by the node).

**Opportunity:** under the saturation cap both fields fit u16 fixed-point at a
quantization step finer than the wire codec already applies — halving the
biggest grid's memory.

**Why flagged:** touches numeric parity end-to-end (integration accumulates in
these fields), the wire path's `reinterpret_cast` / 8 B static_asserts, and
published per-voxel-byte claims. Only worth doing if memory pressure actually
demands it, and only with the digest harness adapted from byte-identical to a
tolerance check. Speed neutral-to-slightly-negative (pack/unpack) · memory
**high** · risk **high**.

## Do these first (best payoff per effort)

1. **Stop drain-and-discard**
   ([scovox_node.cpp#L2067-L2070](../../src/scovox_mapping/src/scovox_node.cpp#L2067-L2070)
   + the disabled-stream sites): void-drains → `clearTouched*()` (add the
   one-line `clearTouchedDir`), keep touched-vector capacity via swap.
   ~An hour; deletes a per-scan O(N log N) sort in the default config.
   (Item 2)
2. **Dirty-gate the cloud publishers + `reserve` in `extractZeroCrossing` +
   parameter-gate the eig/variance fill.** A few small edits; eliminates
   repeated O(map) walks and per-voxel digammas when nothing changed.
   (Item 3)
3. **Block-restrict `publishPlanningMap`** to leaf blocks intersecting the
   window, and confirm it is off in eval launches. Per-scan O(map) →
   O(window). (Item 4)
4. **Constant-weight fast path** replacing the per-ray `std::function`
   ([scovox_map_split.hpp#L272](../../src/scovox_core/include/scovox/scovox_map_split.hpp#L272)).
   Trivial, hot loop. (Item 9)
5. **Per-leaf carve staging, then the carve-on far path** (Item 1). The
   biggest single frame-time lever on dense RGB-D — schedule after the quick
   wins, with the byte-identical digest verification as the acceptance gate.

## Open verification notes

- Item 6: identify all consumers of the `ScovoxMap` topic before filtering
  free space out of it.
- Item 8 / Item 11: confirm what dataset replay actually needs (queue depth,
  TF history) before shrinking either.
- Item 12: enumerate which log tokens the experiment parsers consume before
  touching cadence.
- The ~59 % / ~7 % carve share and the 1.69× far-skip figures are from the
  earlier measured campaigns; all other impact ratings in this doc are
  estimates from code reading and should be confirmed with a profile before
  large refactors.
