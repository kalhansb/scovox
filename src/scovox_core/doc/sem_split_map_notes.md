# sem_split_map.hpp — design notes and history

The long comments of `include/scovox/sem_split_map.hpp`, moved out of the code on 2026-09-23 so the source carries short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word.

Sections follow the order of the source file and are grouped by the function (or section) they sit in. Each gives the line of code the comment was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [HitWeights — declarations](#hitweights--declarations) — 2
- [SemSplitMap — declarations](#semsplitmap--declarations) — 1
- [Params — declarations](#params--declarations) — 6
- [SemSplitMap — declarations (part 2)](#semsplitmap--declarations-part-2) — 7

## HitWeights — declarations

### hitweights-per-ray-profile

**Per-ray source weight profile** — attached to `struct HitWeights {` (line 47)

```text
Per-ray source weight profile for multi-sensor fusion. A non-null
`HitWeights*` threaded into an integration call overrides the map's global
`params_` weights for THAT ray only — letting two sensors (e.g. LiDAR and
RGB-D) write the SAME SemSplitMap with their own inverse-sensor-model
weights (LiDAR w_occ=8/w_free=4.67 high-evidence ToF; RGB-D w_occ=0/w_free=0
semantics-only "pure LiDAR authority"). A NULL pointer — the default on every
signature below — binds to `params_`, so the whole fusion path is inert and
byte-identical to the single-sensor code until a caller opts in.

`geometry_off` is consumed by `ScovoxMapSplit` to suppress the TSDF band
write for this ray (noisy RGB-D depth must not drive the Curless–Levoy
surface); SemSplitMap itself never reads it. `dirichlet_min_p_occ` lets a
source carry its own Stream-B gate height. `evidence_saturation` and
`alpha_0` stay GLOBAL (per-grid caps / priors, not per-source evidence).
```

### hitweights-kernel-radius

**BKI kernel radius for RGB-D spread** — attached to `float kernel_radius = 0.0f;` (line 67)

```text
BKI kernel radius `l` (metres) for RGB-D→LiDAR semantic spread. When
`> 0`, a semantics-only source (e.g. RGB-D) does NOT commit its class at the
single endpoint voxel; instead it deposits kernel-weighted Dirichlet
evidence onto EVERY persistent-Beta (LiDAR-occupied, `p_occ >=
dirichlet_min_p_occ`) voxel within `l` of the endpoint — the S-BKI update
`α*ᵏ += k(d)·(κ₀·p_occ·q)` with the Melkumyan–Ramos compact-support kernel
(exactly 0 at `d >= l`). This decouples the label commit from exact
voxel coincidence (which the LiDAR downsample makes rare) while keeping pure
LiDAR authority: only LiDAR-occupied voxels ever receive a label. `0`
(the zero-init / null-prof default) = classic exact-voxel gate, byte-identical.
```

## SemSplitMap — declarations

### semsplit-params-occ-prior

**SemSplitMap params and occupancy prior** — attached to `struct Params {` (line 86)

```text
Knobs mirror `SemDirMap::Params` 1:1 so a launch file porting between the
two substrates needs no edits. The occupancy prior is the symmetric
Beta(1,1) constant (`kBetaOccPrior`), independent of `num_classes` /
`alpha_0` (which set only the semantic Dirichlet prior); see
docs/occupancy_prior.md.
```

## Params — declarations

### semsplit-dir-leaf-bits

**Separate leaf bits for the Dir grid** — attached to `uint8_t dir_leaf_bits              = 2;` (line 95)

```text
Leaf block bits for the **Dir (semantics) grid only**. The two grids do
NOT need the same block geometry: Beta is full-ray (dense — ~73% of an
8³ block is live on SceneNet), while Dir is hit-only and Stream-B gated
(~12% of the same block), so an 8³ Dir block allocates ~9× more voxel
slots than it fills. Coordinates are a pure function of `resolution`
(VoxelGrid::posToCoord), so a per-grid block size changes only the
container, never a voxel's identity or value — the Beta/Dir coord
identity that labelMesh / labelPointCloud / the publisher's cross-grid
joins rely on is preserved.

2 → 4×4×4 = 64 voxel slots per block. Measured on the 13 E6.6 SceneNet
snapshots this halves the allocated slots (fill 12% → 25%) and cuts the
Dir grid ~55%; 1 (2³) saves a further ~20% but makes per-leaf
bookkeeping ~48% of the grid and is slower to write.
Clamped into [1, leaf_bits] by sanitise(), so a leaf_bits=1 sparse-LiDAR
config is byte-identical to before.
```

### semsplit-evict-by-confidence

**Evicting by confidence** — attached to `bool    evict_by_confidence        = false;` (line 119)

```text
Use the incoming class's probability, rather than its accumulated
evidence, to decide whether it displaces the weakest occupied slot.
Requires a `-DSCOVOX_TRACK_QMAX=1` build (the per-slot confidence has
nowhere to live otherwise); ignored, with a warning from the node, on a
stock build. See `sparse_add_class`.
```

### semsplit-spread-radius

**Single-sensor semantic spread radius** — attached to `float   semantic_spread_radius     = 0.0f;` (line 126)

```text
Semantic spread radius `l` in metres for the SINGLE-SENSOR path. When
`> 0`, Stream A still commits occupancy at the endpoint voxel alone, but
Stream B's class evidence is spread over every already-occupied voxel
within `l` using the same Melkumyan–Ramos kernel as the RGB-D→LiDAR
fusion path (`HitWeights::kernel_radius`), which this does NOT disturb.
The two are mutually exclusive per hit: a source carrying its own
`kernel_radius` keeps the fusion semantics, including its endpoint
occupancy skip. `0` (default) = classic exact-voxel commit.
```

### semsplit-band-length

**Semantic band versus spread** — attached to `float   semantic_band_length       = 0.0f;` (line 136)

```text
SLIM-VDB-style flat semantic BAND half-width, in metres, for the
SINGLE-SENSOR fused-walker path. `0` (default) = classic exact-voxel
commit, byte-identical to before.

This is NOT `semantic_spread_radius` with a different kernel. The two
differ in the shape of the deposit set and therefore in cost:

  spread `l`  — a 3-D BALL around the endpoint. Scans (2⌊l/res⌋+1)³
                voxels per point (~113 at l=0.30, res=0.10), each an
                independent Beta lookup the walker never otherwise
                touches, weighted by the Melkumyan–Ramos kernel.
  band `b`    — the 1-D SEGMENT of the ray already being walked, kept
                to |sdf| ≤ b about the surface (~2b/res = 6 voxels at
                b=0.30, res=0.10), flat-weighted. This mirrors
                SLIM-VDB's `Integrate`, whose semantic write is three
                lines (`alpha[label] += 1`) inside the DDA iteration
                the TSDF update is already paying for.

The fused walker spans [hit − max(depth, sdf_trunc), hit + sdf_trunc],
a strict SUPERSET of SLIM-VDB's [depth − sdf_trunc, depth + sdf_trunc]
at equal trunc, so a band with `b ≤ sdf_trunc` adds ZERO traversal —
only the per-voxel Stream B deposit. `b > sdf_trunc` extends the walk
behind the surface (see `integrateHitFused`) and does cost extra steps.

Stream A (occupancy) is NOT written along the band: only the endpoint
asserts a surface. Band voxels take a Stream B deposit only, gated on
EXISTING persistent Beta occupancy (`dirichlet_min_p_occ`) exactly as
the BKI kernel is — SLIM-VDB has no occupancy model to gate on, so this
is a deliberate, documented deviation (see `applyBandSemantic`).

Mutually exclusive with `semantic_spread_radius` and with a source
profile's `kernel_radius`; `sanitise()` drops the band if either is set.
```

### semsplit-band-require-occ

**Band occupancy requirement** — attached to `bool    semantic_band_require_occ  = true;` (line 170)

```text
Whether the band deposit requires LiDAR to have already confirmed the
voxel occupied (`p_occ ≥ dirichlet_min_p_occ`). Only read when
`semantic_band_length > 0`.

`true` (default) — SCovox's invariant is preserved: a Dir voxel means a
surface. Beta is read, never allocated; the band lands only where a beam
has already stopped. Costs coverage relative to SLIM-VDB, because the
free-space half of the band and the interior of objects both go
unlabelled.

`false` — the FAITHFUL SLIM-VDB mirror. SLIM-VDB exports every TSDF
voxel with `weight ≥ min_weight` (kitti_pipeline.cpp), i.e. the entire
±sdf_trunc shell including the free-space side and the object interior,
each carrying its `alpha` histogram — it has no occupancy model and
applies no surface filter. Setting this false reproduces that: the Beta
read is skipped entirely and every band voxel takes a FLAT
`kappa0 · quality` deposit, the analogue of `alpha[label] += 1`.

This is the single knob on which "same rule as SLIM-VDB" turns, so it is
exposed rather than decided here. Note the two settings are not merely
stricter/looser: `false` also drops the `p_occ` weighting, because a
voxel with no Beta entry has no `p_occ` to weight by.
```

### semsplit-carve-wall-guard

**Wall guard on the immediate carve** — attached to `float   carve_skip_occ_threshold   = 0.0f;   ///< wall-blocking guard (immediate path; <=0 = off)` (line 194)

```text
Wall-blocking guard for the IMMEDIATE (unbatched) carve path only, and
`<= 0` disables it (the shipped default). We trust the most recent LiDAR
scan: if a beam physically reached its return, every voxel it traversed to
get there is free NOW — including one previously marked occupied (a moved
or stale obstacle), which is exactly the dynamic-clearing we want. The
batched carve path (beginCarveFrame/flushCarveFrame — the live pipeline)
never applies this guard: it stages free-space read-free and writes once
per scan. A positive value re-enables the guard only for direct
applyCarveUpdate callers (offline tools / ablations).
```

## SemSplitMap — declarations (part 2)

### semsplit-apply-carve

**Per-voxel carve, batched or immediate** — attached to `bool applyCarveUpdate(const CoordT& c, float quality,` (line 262)

```text
Per-voxel carve: Beta `a_free += w_free·quality`. Touches the Beta grid
only. Returns `false` only when the (optional, default-off) immediate-path
wall guard blocks the voxel (`carve_skip_occ_threshold > 0` and
`p_occ > threshold`); otherwise `true`.

Behaviour depends on whether a carve frame is open (beginCarveFrame):
  - frame OPEN  (batched, the live pipeline): the update is STAGED into a
    per-scan accumulator (max `w_free·quality` per voxel) with NO grid read
    and NO guard, then written once at flushCarveFrame. Always returns
    `true` (a scan never blocks itself — see class docs).
  - frame CLOSED (immediate): writes the Beta grid in place, applying the
    wall guard iff `carve_skip_occ_threshold > 0`.
```

### semsplit-batched-carve

**Batched per-scan carve** — attached to `void beginCarveFrame();` (line 280)

```text
Full-ray free-space carving re-traverses the dense near-origin ray fan, so
the same free voxel is crossed by thousands of rays per scan. Writing it
once per ray is both wasteful (thousands of Beta writes to one cell) and
radially biased (near-sensor voxels saturate purely from ray density). The
batched path fixes both: every carve is STAGED read-free during the walk,
then written ONCE per voxel per scan at flush (max evidence), block-ordered
for cache locality. This is the OctoMap computeUpdate model.

Contract: wrap a whole scan's integrateHit/Miss (or the fused walker) in
  beginCarveFrame(); ... integrate every ray ...; flushCarveFrame();
Occupied-wins: a voxel that is a HIT (surface) anywhere in the same scan is
never carved free, even if another ray grazes through it.
```

### semsplit-dynamic-hit

**Dynamic-aware per-voxel hit** — attached to `void applyHitUpdate(const CoordT&             c,` (line 314)

```text
Dynamic-aware per-voxel hit. When `is_dynamic` is true the endpoint's
occupancy (Stream A) + semantics (Stream B) are deposited into the parallel
TRANSIENT Beta/Dir grids — decayed each frame by `decayTransient`, overlaid
on the persistent map at publish — so moving objects (e.g. people) show
live but fade fast and never pollute the persistent map. The transient
path uses the IDENTICAL Stream A/B update; only the target grids differ and
it records no touched-set (transient is local-only, never drained to the
fusion wire). `is_dynamic == false` is byte-identical to the 3-arg form.
```

### semsplit-band-deposit

**Band deposit versus the hit update** — attached to `void applyBandSemantic(const CoordT&             c,` (line 328)

```text
Per-voxel SLIM-VDB-style BAND deposit: Stream B ONLY, at a voxel the fused
walker is already standing on (`Params::semantic_band_length`). Never call
this for the endpoint voxel — `applyHitUpdate` owns that one, and running
both would double-count it.

The three ways this differs from `applyHitUpdate`, all deliberate:

 1. NO Stream A. Occupancy is evidence that a beam *stopped* here, which
    is true only at the endpoint. Incrementing `a_occ` along the band would
    manufacture a ~2·b-thick occupied shell around every surface and fight
    the free-space carve running over the same voxels. SLIM-VDB has no
    occupancy grid, so it has nothing to mirror here.
 2. Beta is READ, never allocated. A band voxel that LiDAR has not already
    confirmed occupied (`p_occ ≥ dirichlet_min_p_occ`, or no Beta voxel at
    all) takes no deposit. SLIM-VDB writes `alpha` unconditionally; it can
    afford to because nothing downstream distinguishes a labelled free
    voxel from a labelled surface. For SCovox the gate is what keeps the
    Dir grid from growing a label shell through the free space in front of
    every surface. THIS IS THE ONE PLACE THE MIRROR IS NOT EXACT.
 3. FLAT weight — `kappa0 · p_occ · quality`, the same expression the
    endpoint uses, with no distance kernel. That is the point of contrast
    with `applyHitUpdateKernel`, which tapers by `k(d)`. SLIM-VDB's
    `alpha[label] += 1` is flat across its whole band, and at p_occ→1 this
    reduces to the same constant.

Deposits into the PERSISTENT Dir grid only. Dynamic endpoints do not reach
here (see `integrateHitFused`): a moving object must not smear its class
across the static surfaces behind it.
```

### semsplit-decay-transient

**Transient grid decay** — attached to `void decayTransient(float rate);` (line 361)

```text
Per-frame multiplicative decay of the transient grids toward their priors
(Beta `a → prior`, Dir `cnt → α₀`, `other → (C−K_TOP)·α₀`). Voxels whose
evidence decays back to within epsilon of prior are pruned (`setCellOff`)
so the transient grids stay bounded over a long run. `rate ∈ [0,1]`
(clamped): `rate == 1` is a no-op, `rate == 0` clears all transient
evidence. Call once per integrated frame.
```

### semsplit-carve-accumulator

**Per-scan carve accumulator** — attached to `std::unordered_map<CoordT, float> carve_stage_;` (line 502)

```text
Per-scan carve accumulator (batched path). `carve_stage_` maps each free
voxel to the MAX `w_free·quality` seen this scan (strongest free evidence,
one write); `carve_hits_` holds voxels observed as a surface this scan so
flush can honour occupied-wins. Both are clear()ed (capacity retained) at
beginCarveFrame, so steady-state framing allocates no new buckets.
```

### semsplit-kernel-spread

**BKI kernel semantic spread** — attached to `void       applyHitUpdateKernel(const CoordT&             c,` (line 540)

```text
BKI (S-BKI) semantic spread for a semantics-only source with
`prof->kernel_radius > 0` — the RGB-D→LiDAR fusion path. Deposits the
observed class (`sem_probs`) onto every LiDAR-occupied voxel within the
kernel radius of endpoint `c`, weighted by the compact-support kernel and
gated on the PERSISTENT Beta occupancy (LiDAR authority), regardless of
which Dir grid (`dacc`) is the deposit target. Touches only the Dir grid —
RGB-D deposits zero occupancy, so no Beta allocation/carve here.
Also serves the single-sensor `semantic_spread_radius` path, which passes
the map-global weights instead of a source profile's; the kernel itself is
identical either way, and in both cases occupancy is read, never written.
```
