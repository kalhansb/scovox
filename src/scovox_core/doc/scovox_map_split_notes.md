# scovox_map_split.hpp — design notes and history

The long comments of `include/scovox/scovox_map_split.hpp`, moved out of the code on 2026-09-23 so the source carries short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word.

Sections follow the order of the source file and are grouped by the function (or section) they sit in. Each gives the line of code the comment was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [Params — declarations](#params--declarations) — 2
- [resolution_](#resolution_) — 1
- [integrateHitFused](#integratehitfused) — 7
- [ScovoxMapSplit — declarations](#scovoxmapsplit--declarations) — 7

## Params — declarations

### split-tsdf-enabled

**The tsdf_enabled switch** — attached to `bool tsdf_enabled = true;` (line 61)

```text
When false, the fused walker skips every TSDF band write, so the TsdfMap
grid stays empty. For callers that disabled TSDF (sdf_trunc passed as 0)
but whose TsdfMap::Params still sanitises sdf_trunc back to a positive
default — the band integration is then pure dead work if the TsdfMap grid
is never read. Default true → TSDF integrates exactly as before. The
occupancy (Beta) and semantic (Dir) substrates are unaffected either way.
```

### fine-band-params

**Fine TSDF band parameters** — attached to `uint8_t fine_ratio_log2       = 0;` (line 70)

```text
docs/design/fine_tsdf_band_dbh_2026_07_30.md. 0 = off (default —
byte-identical behaviour, no fine grid allocated). k > 0 adds a second
sparse TSDF-only lattice at res_fine = resolution / 2^k, written only
inside registered refinement cylinders (addRefinementRegion), in a
±fine_sdf_trunc_voxels fine-voxel band around gated hits. Independent
of tsdf_enabled: the fine band can run with the coarse TSDF off
(e.g. the occupancy-only LiDAR config).
```

## resolution_

### split-band-needs-fused-walker

**Semantic band requires the fused walker** — attached to `if (sem_band_ > 0.f && !fused_walker_) {` (line 108)

```text
The band lives in the fused walker only (see integrateHitSplit). Asking
for both is a request that cannot be honoured, and honouring it silently
as "endpoint only" would hand back a null result that looks like a
measurement. Refuse loudly and zero the knob so `sem_band_` and the
params() the node prints agree with what actually runs.
```

## integrateHitFused

### fused-degenerate-ray-timing

**Timing of degenerate rays** — attached to `const auto t1 = clk::now();` (line 185)

```text
Degenerate ray (origin≈endpoint): no TSDF or semantic work happens.
Attribute the (near-zero) bracket to tsdf_ns_ — same accumulator the
main return below uses — so the fused walker reports ALL of its time in
one bucket. (Routing this to sem_ns_ would be doubly wrong: it does
no semantic work, and it splits the fused path's time across two
accumulators whose per-substrate split is meaningless on this path.)
```

### fused-band-exclusions

**When the semantic band is off** — attached to `const bool band_active = sem_band_ > 0.f && !is_dynamic && !geometry_off` (line 202)

```text
SLIM-VDB-style flat semantic band (Params::semantic_band_length). Decided
once per ray, not per voxel, so the branch inside the DDA is a bool test.
Excluded cases, all of them deliberate:
  is_dynamic     — a moving object must not paint its class onto the
                   static surfaces its beam passes through or stops on;
                   its endpoint already routes to the transient grids.
  geometry_off /
  kernel_radius  — an RGB-D overlay source owns the BKI ball path
                   (applyHitUpdateKernel). Running the band as well would
                   deposit that source's class twice per hit.
  no sem_probs   — a bare geometric return has no opinion to pool.
```

### fused-band-back-reach

**Extending the walk for the band** — attached to `const float back_reach = band_active ? std::max(trunc, sem_band_) : trunc;` (line 220)

```text
The band is symmetric about the surface, but the walk behind it normally
stops at `trunc`. Extend the far end when the band reaches deeper, else
the behind-surface half is silently clipped and the knob stops meaning
what it says. At band ≤ trunc — the mirror configuration, both 0.30 m on
KITTI — this is the old expression exactly and costs no extra steps.
Voxels gained beyond trunc have sdf ≤ −trunc, which applyBandUpdate drops
and the semCarve gate (sdf > 0) never sees, so TSDF/occupancy are unmoved.
```

### fused-hit-before-proj-check

**Hit update before the proj check** — attached to `if (c == k_hit) {` (line 252)

```text
(3) Hit (endpoint voxel) — semantic/occupancy update. Run this BEFORE
the proj≈0 early-return: when the endpoint lands exactly on a voxel
centre, v_point_voxel≈0 so proj≈0, and returning here would skip the
hit update entirely — leaving the surface voxel at prior and diverging
the fused walker from the non-fused SemSplitMap::integrateHit, which
applies the hit unconditionally. The TSDF band update below may still
skip on proj≈0 (its sign is ill-defined there), but semHit must not.
```

### fused-tsdf-band-gate

**TSDF band gate in the fused walker** — attached to `if (tsdf_enabled_ && !is_dynamic && !geometry_off && (tparams.space_carving || sdf <= trunc + h)) {` (line 267)

```text
(1) TSDF band update — gate + clamp + Curless–Levoy. The fused walker
always walks back to the origin (walk_back = max(depth, trunc)), so the
upper gate here must MATCH the non-fused TsdfMap::integrateRay band,
which depends on space_carving:
  - space_carving=false (Replica/KITTI default): the non-fused path
    walks only [hit−trunc, hit+trunc], so we keep the `sdf <= trunc + h`
    band gate; dropping it would write the whole front ray that the
    non-fused path never touches (and break the band invariant).
  - space_carving=true: the non-fused path walks [origin, hit+trunc] and
    applyBandUpdate clamps every in-front voxel (incl. sdf > trunc) to
    +trunc, so we drop the upper gate to integrate the full carve front.
applyBandUpdate owns the lower gate (`sdf <= -trunc` → drop) for both.
Dynamic rays write NO persistent TSDF: a moving object must not leave a
permanent surface. Its occupancy/semantics live in the transient grids
(semHit above, is_dynamic=true); the free-space carve below stays
persistent (the air the object passed through is genuinely free).
```

### fused-semantic-band-order

**Semantic band deposit and its order** — attached to `if (band_active && c != k_hit && sdf > -sem_band_ && sdf <= sem_band_) {` (line 287)

```text
(1b) SLIM-VDB-style flat semantic band. This is the exact window
SLIM-VDB's Integrate writes `alpha[label] += 1` over — `sdf > -trunc`
on a ray it truncates at depth ± trunc — evaluated on voxels this DDA
is already standing on, which is why it costs no traversal.

The endpoint is excluded: semHit above already deposited there, and
banding it too would give the surface voxel double weight relative to
its neighbours, inverting the smoothing this is meant to apply.

Ordered BEFORE the carve so the occupancy `applyBandSemantic` reads is
this voxel's pre-carve state on the immediate path. On the live batched
path the carve is staged until flushCarveFrame, so p_occ cannot move
mid-scan and the two orders coincide — but they must not diverge
between paths, so the order is pinned here rather than left to luck.
```

### fused-walker-timing

**Fused walker time accounting** — attached to `tsdf_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();` (line 327)

```text
Fused walker: TSDF band updates and semantic hit/carve are interleaved in
ONE per-voxel loop, so wall-clock cannot be cleanly attributed per
substrate without bracketing every applyBandUpdate vs semHit/semCarve with
a clock read — two steady_clock::now() calls per voxel would dominate and
distort the very cost being measured in this hot Bresenham loop. We
therefore report the COMBINED TSDF+semantic time under tsdf_ns_ and leave
sem_ns_ untouched on the fused path (it reads 0). For a true per-substrate
split, run the non-fused integrateHitSplit walker, which times the two
DDAs separately. See tsdfTimeUs()/semdirTimeUs() docs.
```

## ScovoxMapSplit — declarations

### split-walker-no-band

**Split walker lacks the semantic band** — attached to `void integrateHitSplit(const Eigen::Vector3f&    origin,` (line 339)

```text
Non-fused split walker (two DDAs). Kept for A/B parity testing.

NOTE: this path does NOT implement `semantic_band_length`. The band is
defined as "deposit on the voxels the walker is already standing on", and
the whole claim being tested is that this costs no extra traversal — which
is only true of the fused walker's single DDA. Reimplementing it here
would mean a third DDA and would measure something else. Callers get a
hard warning at construction rather than silent endpoint-only numbers;
see the `band + !fused_walker` check in ScovoxMapSplit's constructor.
```

### carve-frame-begin

**Opening a carve frame** — attached to `void beginCarveFrame() {` (line 388)

```text
Open a carve frame on the semantic substrate: every carve (fused walker or
non-fused integrateHit/Miss) is staged read-free until flushCarveFrame().
Wrap a whole scan's rays in beginCarveFrame()/flushCarveFrame(). See
SemSplitMap for the rationale (full-ray free-space, one write per voxel).
Also opens the fine-band scan frame: gated hits are staged per region
so flushCarveFrame can anchor-correct the whole scan before fusing.
```

### carve-frame-flush

**Flushing the carve frame** — attached to `std::size_t flushCarveFrame() {` (line 399)

```text
Write all staged carves for the scan (one Beta update per unique voxel,
block-ordered, occupied-wins). Timed into the same tsdf_ns_ bucket as the
fused walk, so tsdfTimeUs() reflects total carve cost (walk staging +
flush). Returns the number of voxels written (carve only — fine-band ray
count is reported via fineLastFrameRays()).
```

### fine-refine-hit

**Raw returns into the fine band** — attached to `void refineHit(const Eigen::Vector3f& origin, const Eigen::Vector3f& endpoint) {` (line 444)

```text
Feed one raw sensor return to the fine band ONLY — no coarse-map write.
This is the full-density path: nodes typically voxel-grid-downsample
each scan before `integrateHit`, which caps what the fine lattice can
see at one return per downsample cell. Routing every raw (deskewed)
return here instead gives refinement regions the sensor's native point
density while the coarse map keeps its downsampled diet. Same gate,
staging, and per-scan anchor treatment as dispatcher-staged hits;
out-of-region endpoints are a no-op after one O(1) hash lookup.
```

### split-tsdf-time-meaning

**What tsdfTimeUs measures** — attached to `std::int64_t tsdfTimeUs()   const noexcept { return tsdf_ns_ / 1000; }` (line 485)

```text
Accumulated TSDF time. NOTE: on the fused walker (fused_walker=true, the
default) this is the COMBINED TSDF+semantic integration time — the fused
loop interleaves both substrates and is not separable without per-voxel
clock overhead. Only the non-fused integrateHitSplit / integrateMiss paths
attribute TSDF and semantic time to separate accumulators.
```

### fine-stage-hit

**Staging fine-band hits** — attached to `void stageFineHit(const Eigen::Vector3f& origin,` (line 582)

```text
Gate + stage one hit for the fine band. Inside an open scan frame the
hit is buffered per region for the anchor-corrected flush; outside a
frame (direct integrateHit callers, e.g. unit tests without the carve
bracket) it fuses immediately, uncorrected — mirroring SemSplitMap's
immediate-vs-batched carve semantics.
```

### fine-flush-anchor

**Anchor-corrected fine-band flush** — attached to `std::size_t flushFineFrame() {` (line 599)

```text
Per-region anchor fit + band fusion of all staged hits. The whole
scan's in-region rays are translated by the fitted Δ (a rigid shift of
the sensor pose in the horizontal plane) so the fusion input aligns
with the region's canonical cylinder — drift is absorbed at the door,
before the irreversible Curless–Levoy average. Fit failure → Δ = 0.
```
