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

**Where:** allocation at
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

**Where:**
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

**Where:** cache at
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
