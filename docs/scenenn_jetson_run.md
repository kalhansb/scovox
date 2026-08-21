# SCovox map from SceneNN (HKUST) — Jetson runbook

Builds a semantic SCovox map from a SceneNN scene on a Jetson (JetPack /
Ubuntu 22.04 / ROS 2 Humble), natively — no container.

SceneNN is real Asus Xtion capture with a hand-annotated mesh, so it exercises
the RGB-D path against genuine sensor noise and dropouts rather than SceneNet's
synthetic renders.

## What the dataset does and does not give you

| Need | Where it comes from |
|---|---|
| RGB-D frames | `main/oni/<id>.oni`, extracted with SceneNN's `playback` tool |
| Camera poses | `main/<id>/trajectory.log` — Redwood `.log`, 4x4 camera-to-world |
| Intrinsics | `main/intrinsic/asus.ini` — 640x480, fx=fy=544.473, cx=320, cy=240 |
| Semantics | **not shipped as 2D label PNGs** — derived from the annotated mesh |

The README's Google Drive folder (`0B-aa7y5Ox4eZWE8yMkRkNkU4Tk0`) is dead (404),
and with it the 2D annotation package. Everything else — including the `.oni`
files the Drive copy was needed for — is still served over plain HTTP from
`https://hkust-vgd.ust.hk/scenenn/`, and `check.txt` there is a full md5
manifest of the server, which is the quickest way to find what still exists.

Because the 2D labels are gone, per-pixel GT is recovered from
`contrib/nyu_class/<id>/<id>.ply`, whose vertices carry a `nyu_class`
attribute directly. Note its instance ids differ from the main annotation's
(walls are merged into one instance), so the nyu_class `.ply` and `.xml` are a
matched pair and must not be mixed with `main/<id>/<id>.xml`.

## Three things that are not what you would guess

**1. Depth is already Z-depth, and needs no flip.** OpenNI returns perspective
depth, so the ray-length→Z rescaling that `scenenet_replay_node.py` performs
must *not* be applied here (it would shrink depth by ~8% at the corners).
`PlaybackSync.cpp` looks like it flips depth and colour inconsistently — it
applies `FreeImage_FlipVertical` to colour only — but the saved PNGs come out
in matching orientation, so no correction is needed. Verified by measurement,
not by reading the source: `scenenn_check_relpose.py` scores the standard
optical convention `(x right, y down, z forward)` with no flip at **1.3 cm**
against 2–7 cm for every alternative.

**2. `trajectory.log` covers fewer frames than `playback` emits.** Scene 016
extracts 1364 frames (the tool discards depth frames to re-sync against colour)
but ships 1300 poses. Replay therefore stops at `min(frames, poses)`; frame `i`
pairs with pose `i`.

**3. The trajectory and the annotated mesh do not share a world frame.** The
poses reproduce inter-frame motion to ~1.3 cm, so replayed depth is
self-consistent and SCovox maps from it directly — but the same frames land
~30 cm off the mesh, and a single rigid fit only reaches 0.077 m / 36% inliers
because the trajectory drifts relative to the globally-optimised mesh.
Labelling therefore *tracks* each frame onto the mesh, seeded from the previous
frame. On 016 that holds **6.1 mm median with 99.8% inliers and 0/1300 weak
frames**. This matters only for labels — the map itself is built in the
trajectory frame and never needs the mesh.

## Run it

```bash
cd ~/jetbot-slam/scovox
source /opt/ros/humble/setup.bash && source install/setup.bash

# 0. build once (native; all deps ship with JetPack 22.04)
colcon build --packages-select scovox_msgs scovox_core scovox_mapping \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

# 1. fetch + extract a scene (016 is the smallest useful one: 229 MB oni)
experiments/scovox_eval/scripts/scenenn_fetch.sh 016

# 2. register the trajectory onto the annotated mesh (global fit + per-frame track)
python3 experiments/scovox_eval/scripts/scenenn_register_mesh.py \
    --scene ~/datasets/scenenn/016 --intrinsics ~/datasets/scenenn/asus.ini \
    --out ~/datasets/scenenn/016/mesh_align.json --per-frame

# 3. render per-pixel NYU-40 GT labels
python3 experiments/scovox_eval/scripts/scenenn_make_labels.py \
    --scene ~/datasets/scenenn/016 --intrinsics ~/datasets/scenenn/asus.ini \
    --align ~/datasets/scenenn/016/mesh_align.json --jobs 10

# 4. map
experiments/run_scenenn.sh 016

# 5. look at it (no open3d on the Jetson; this is a numpy splat)
python3 experiments/scovox_eval/scripts/scenenn_render_map.py \
    --npz experiments/results/scenenn/016/scovox.npz \
    --colormap ~/datasets/scenenn/nyu_color.xml \
    --out experiments/results/scenenn/016/map_semantic.png
```

Output lands in `experiments/results/scenenn/016/scovox.npz` with the node log
beside it. Fields per voxel: `points`, `occupancy_prob`, `semantic_class`,
`semantic_confidence`, `posterior_variance`, `eig`, `a_occ`/`a_free`/`a_unk`,
and the top-2 semantic counts.

## Result on 016 (reference numbers)

**1292 frames integrated** (replay indices 7..1298, no gaps — the node receives
1300 and reports `recv=1299`, but the first 7 are TF-gated and never integrate;
`recv` counts received, not integrated). **28 484 occupied voxels** at 5 cm over
a 5.20 x 3.00 x 4.70 m extent.

Cost, from the node's own per-frame counters in `scovox.log` (stride 1,
1292 samples, 508.7 s from first to last integrated frame):

| | mean | p50 | p95 | max |
|---|---|---|---|---|
| `frame_ms` | 393.1 | 396.7 | 439.2 | 475.2 |
| `integrate_ms` | 393.0 | 396.6 | 439.1 | 475.1 |
| `tsdf_ms` | 365.7 | 369.2 | 411.5 | 447.2 |
| `tf_ms` | 0.1 | 0.1 | 0.1 | 0.2 |
| `publish_ms` | 0.0 | 0.0 | 0.0 | 0.0 |

So **93% of frame time is TSDF integration**; TF and publish are free.

Note `frame_ms` does NOT include the per-frame INFO log or the `getVmRSSKB()`
read that feeds it — `scovox_node.cpp:1233` takes `t_end` before both. They sit
inside the frame callback, so they cost throughput without appearing here; see
"Does the logging distort the timings?" below for the measured size of that gap.

RSS climbs monotonically with map extent — 68 MB at the first frame, 317 MB at
25%, 613 MB at 50%, and **782.8 MB peak**, flat from ~75% on once the room is
fully swept. That is the whole-node figure including ROS and the input queue,
not the grid alone — and per the RSS caveat below it is not a map-size metric.

CPU comes from `jetson_telemetry.py`, which `run_scenenn.sh` now starts
alongside the node and writes to `telemetry.csv` (1 Hz, 508 active samples):

| | value |
|---|---|
| process CPU | **100.5% mean, 101.6 p95, 102.6 max** — i.e. exactly one core |
| cores busy | 1.00 of 12 (**8.4% of the machine**) |
| system-wide | 2.19 cores busy incl. replay; busiest cores 85% / 58% |
| RSS | 68 → 783 MB, 786 MB peak |
| VSZ peak | 1400 MB |
| host MemAvailable | 57.4 → 56.6 GB (−777 MB) |

**Integration is single-threaded** — the node itself never exceeds one core.
The machine is not saturated either, but "11 cores idle" would overstate it:
system-wide 2.16 of 12 cores are busy, because the Python replay node costs
~0.6 of a core (core4 57%) on top of the mapping node (core11 83%, migrating).
The replay node is harness, not system, so the headroom for threading the
integrator is real — just don't quote it as 11 free cores.

## Is it real time?

No, and not close. At stride 1 the node integrates at **2.54 Hz** against an
Asus Xtion that captured at 30 Hz — a **~12x** shortfall.

The run only completed because this is a replay: `run_scenenn.sh` drives frames
at 6 Hz, so replay finished in ~217 s while integration took 509 s, and the
dataset-mode input queue (`KeepLast(1000)`) absorbed the ~500-frame backlog. It
fits under the 1000-message limit, but only just. On a live sensor those frames
would be dropped, not buffered.

### TSDF is dead weight here — turn it off

`enable_tsdf:=false` (5th arg to `run_scenenn.sh`). This config never reads the
TsdfMap grid — `share_tsdf=0`, no `~/tsdf_pointcloud` publisher, no
`~/extract_mesh` — so the band writes inside the fused walker are pure cost.
A/B on 016 at stride 1:

| | TSDF on | TSDF off | |
|---|---|---|---|
| `frame_ms` mean | 393.1 | **334.4** | −15% |
| rate | 2.54 Hz | **2.99 Hz** | +18% |
| `tsdf_ms` mean | 365.7 | 307.4 | −16% |
| peak RSS | 783 MB | **429 MB** | **−45%** |
| process CPU | 100.5% | 100.3% | one core either way |

The two maps are **bit-identical across all 13 npz fields** (`points`,
`occupancy_prob`, `semantic_class`, `semantic_confidence`,
`posterior_variance`, `eig`, `a_occ`/`a_free`/`a_unk`, the four `sem_*`), so
that memory and time bought nothing. Leave it on only if you need the mesh or
the TSDF cloud.

Note `tsdf_ms` is a misleading counter name: it is the fused-walk plus
carve-flush bucket, so it still accounts for 92% of frame time with TSDF off —
what remains is free-space Beta carving, not TSDF.

### Widening the TSDF band to ±0.30 m

`sdf_trunc_voxels:=6` at 5 cm = 0.30 m truncation, TSDF on, stride 1, full
1300-frame scene, both runs integrating 1299/1300 frames:

| | ±0.15 m (default) | ±0.30 m | delta |
|---|---|---|---|
| mean frame_ms | 393.1 | 435.0 | **+10.7%** |
| tail-200 frame_ms | 350.6 | 392.5 | +12.0% |
| mean `tsdf_ms` | 365.7 | 407.7 | +42.0 |
| Hz | 2.54 | 2.30 | −0.24 |
| peak RSS | 782.8 MB | 1000.4 MB | **+27.8%** |
| proc CPU (mean / p95) | 97.4 / 101.6% | 97.6 / 101.9% | unchanged |
| published voxels | 28 484 | 28 495 | +11 |
| occupancy IoU | — | 0.99436 | |
| class agreement | — | 0.99982 | |

The whole +41.9 ms lands in `tsdf_ms`, which is what should happen: the band
gate is `sdf <= trunc + h`, so doubling `trunc` doubles the voxels per ray that
take an `applyBandUpdate`. Still one core — widening the band does not change
the single-threaded ceiling.

Two things worth knowing before setting this:

**It perturbs the occupancy map slightly, and not for a good reason.** The
fused walker sets `back_reach = trunc`, so `k_far` moves from 0.15 m to 0.30 m
past the surface. `RayIterator` interpolates between integer coords, so moving
the endpoint reshuffles which intermediate voxels the DDA lands on — 75 voxels
lost, 86 gained. Same mechanism as the Tier 0 `back_reach` finding. This is
traversal jitter, not better geometry.

**In this config nothing reads the TsdfMap grid** (`share_tsdf=0`, no
`~/tsdf_pointcloud` subscriber, no `~/extract_mesh` call), so the +42 ms/frame
and +218 MB buy nothing that appears in the npz. A wider band only pays off
through mesh extraction or the TSDF cloud. The RSS rise is consistent with a
±0.30 band allocating roughly twice the TSDF voxels, but the TSDF grid is not
published, so that is not verified by an allocated-voxel count — see the RSS
caveat below.

### Where the time actually goes

Measured with `scenenn_time_sweep.sh` — 400 frames of 016, one knob changed per
row, warm window (frames 50+), baseline is TSDF-off / stride 1:

| config | frame_ms | walk_ms | sem_ms | RSS | Hz | voxels | IoU vs base |
|---|---|---|---|---|---|---|---|
| base | 345.0 | 319.3 | 0.0 | 157 | 2.90 | 10709 | — |
| `enable_tsdf:=true` | 407.2 | 380.9 | 0.0 | 284 | 2.46 | 10709 | 1.0000 |
| `stride:=2` | 113.7 | 102.9 | 0.0 | 38 | 8.80 | 9976 | — |
| `stride:=4` | 55.2 | 48.8 | 0.0 | 36 | 18.11 | 9031 | — |
| `resolution:=0.10` | 207.0 | 182.0 | 0.0 | 42 | 4.83 | 2275 | — |
| `max_range:=2.0` | 305.7 | 281.4 | 0.0 | 87 | 3.27 | 9467 | 0.8740 |
| **`carve_band:=0.1`** | **129.9** | 101.6 | 0.0 | 43 | **7.70** | 11021 | 0.9704 |
| `fused_walker:=false` | 283.7 | 78.1 | 180.7 | 73 | 3.52 | 10734 | 0.9932 |

The stride rows fit `T(s) = 308.9/s^2 + 36.1 ms` to under 0.3 ms residual, so
**90% of frame time is strictly per-ray** and only 36 ms is fixed.

**The cost is `carve_band:=-1`, i.e. full-ray free-space carving.** With
`carve_band <= 0`, `ScovoxMapSplit::integrate` sets `carve_band = depth`, so
every ray runs `semCarve` on every voxel from the camera to the surface. On 016
that is:

Counted directly (temporary counter in `visit_one`, 48 frames, stride 1,
TSDF off — not estimated):

- **267 000 rays/frame**
- **31.8 voxels/ray** mean walk
- **8.50 M voxel visits/frame** — against **10 400 unique staged voxels**
- **818x redundancy**, at ~41 ns/visit

Where those 41 ns go is measured, not inferred. `batch_free_carve:=false`
returns from `applyCarveUpdate` before touching the staging container while
leaving the entire walk intact — same rays, same `visit_one` body, same
`semCarve` call — so the difference is the staging container and the flush,
exactly and only:

| | frame_ms | ns/visit | share |
|---|---|---|---|
| full (`base`) | 350.4 | 41 | 100% |
| walk only (`batch_free_carve:=false`) | 282.3 | 33 | **81%** |
| → staging + flush | 68.0 | 8 | **19%** |

**The container is not the bottleneck — the traversal is.** The staged
`std::unordered_map` costs ~8 ns/visit, not the 36 ns assumed earlier; the other
33 ns is `RayIterator`'s serial DDA plus `visit_one`'s per-voxel `coordToPos`
(double), two vector subtractions, `norm()` and `dot()`, and an out-of-line call
into `SemSplitMap::applyCarveUpdate` for every one of the 8.5 M visits. The
19% ceiling on any staging-container change is what makes Tier 1 below a dead
end before it is written.

The Replica and KITTI launches already default to `carve_band:=0.1`; the
SceneNN launch inherited `-1.0`. Setting `0.1` gives **2.9 -> 7.7 Hz** at
IoU 0.9704 / 99.9% class agreement. It is not free: less carving means fewer
voxels get pushed below the occupancy threshold, so the map gains 3% voxels.
That is an accuracy trade, unlike `enable_tsdf:=false`, which is bit-identical.

**The fused walker is the slower path here.** `fused_walker:=false` runs two
DDAs and is 30% faster with TSDF *on* (283.7 vs 407.2 ms) at IoU 0.9932. The
reason is in `visit_one` (`scovox_map_split.hpp:243-250`): it computes
`coordToPos`, two vector subtractions, a `norm()` (sqrt) and a `dot()` on
**every** voxel, unconditionally — before the `tsdf_enabled_` test on line 283.
So the full-ray carve pays TSDF-shaped math on all 8.5 M visits even when TSDF
is disabled.

Ranked, on this scene:

1. `carve_band:=0.1` — **-62%**, costs IoU 0.97
2. `stride:=2` — **-67%**, costs thin structure (and stride 4 drops 16% of voxels)
3. `enable_tsdf:=false` — **-15% and -45% RSS**, costs nothing (bit-identical)
4. `fused_walker:=false` — -30% with TSDF on, costs IoU 0.993
5. `max_range:=2.0` — only -11%, and IoU 0.874. Not worth it.

None of these touch the real ceiling: 818x redundant work on one core.

### Tier 0 was tried and reverted (negative result)

The obvious micro-optimisation — don't compute the signed distance when nothing
reads it — was implemented and measured on the full 1300-frame run, then backed
out. Two sub-changes, isolated:

| variant | frame_ms | RSS | Hz | voxels | IoU vs baseline |
|---|---|---|---|---|---|
| baseline | 334.4 | 429 | 2.99 | 28484 | — |
| defer the sqrt in `visit_one` | 323.8 | 345 | 3.09 | 28505 | 0.99898 |
| + `back_reach = 0` when both bands off | 305.5 | 211 | 3.27 | 28562 | 0.99315 |

Two reasons it was reverted:

**It does not preserve the output.** The carve gate `sdf > 0 && sdf <= band`
restated as `proj > 0 && dist2 <= band^2` is algebraically identical but not
identical in float, and the boundary `dist ≈ carve_band` is exactly where the
ray's *origin* voxel sits — so every ray evaluates the flipped comparison.
`back_reach = 0` additionally moves `k_far`, and `RayIterator` interpolates
between integer coords, so shortening the ray reshuffles which intermediate
voxels the DDA lands on. Bit-identical was the acceptance bar; neither variant
clears it.

**The payoff is ~3%, not the ~40% predicted.** Deferring the sqrt is worth
10.6 ms of 334. The estimate came from reading the 139 ms fused-vs-non-fused gap
as discarded sdf math — that reading is wrong, and the gap must come from the
non-fused path doing fewer map operations, not less arithmetic.

That is the useful finding: 10.6 ms out of 334 for removing a per-visit `sqrt`
says the walk is not one dominant instruction but a long serial dependency
chain — DDA state, then `coordToPos`, then the vector math — that no single
term shortens much. Only reducing the *count* of visits can move it.

### Tier 1 was tried and reverted (negative result)

Premise: staging is the hot path, so replace `carve_stage_`'s
`std::unordered_map<CoordT,float>` with a `Bonxai::VoxelGrid` of stamped cells
plus a caching `Accessor`. Consecutive voxels along a ray share an 8³ leaf, so
most visits should resolve on the cached leaf pointer with no hash lookup, and a
frame-id stamp makes `beginCarveFrame` O(1) instead of a clear over the staged
set. Implemented (stamped `{stamp, hit_stamp, w}` cells, separate accessors for
the carve and hit writers, `carve_hits_` folded into the same cell); all 156
`scovox_core` tests pass.

It is **slower**, consistently, on every config — 400 frames of 016, same
harness (`scenenn_ab.sh`):

| config | baseline | Tier 1 | delta |
|---|---|---|---|
| `base` (stride 1, TSDF off) | 350.4 ms | 365.4 ms | **+4.3%** |
| `enable_tsdf:=true` | 412.2 ms | 420.6 ms | +2.0% |
| `carve_band:=0.1` | 130.6 ms | 133.1 ms | +1.9% |

Staging + flush went from 68.0 ms to 83.1 ms. Two reasons, and the first one
alone was enough to predict the outcome:

**The premise was wrong by 4x.** Staging is 19% of frame time, not the
~90% the "36 ns/visit = one hash miss" reading implied. Even a *free* staging
container caps out at −19%. That number was available from a one-line
`batch_free_carve:=false` run and should have been measured before writing any
code — the attribution table above now exists because of this.

**A sparse-in-dense grid is the wrong container for this access pattern.** The
hash map stores 10.4 k live entries in ~500 KB. The stamped grid spreads the
same 10.4 k values across full 512-cell leaf blocks at 12 B/cell, so the
per-frame working set is several MB of mostly-cold cache lines, and crossing a
leaf boundary every ~5–8 voxels costs a shared-ptr chase the hash map does not
pay. The leaf-locality argument was real but smaller than the footprint
inflation it bought.

The patch is kept at `scratchpad/tier1_stamped_grid.patch` rather than in the
tree. Note the outputs were not shown equivalent: the two runs integrated 392 vs
387 frames, giving IoU 0.995, so frame count and the change are confounded.
Equivalence rests on the argument (flush writes `+=` to distinct voxels, so
order does not matter) and the test suite, not on that IoU.

### Peak RSS does not measure map content — do not compare builds with it

The RSS drop above looked like a memory win. It is not. Chasing it down on
400-frame runs (`rss_probe` results dir):

| | baseline | Tier 0 |
|---|---|---|
| allocated voxels (`occupancy_vis_threshold:=0.0`) | 41 236 | 41 090 – 41 174 |
| staged carves per frame (`carve_n`) | 11 153 | 11 095 |
| total staged over the run | 4 338 583 | 4 360 426 |
| **peak RSS** | **166 MB** | **91 MB** |

Same live data, same allocation counts, same staging work — **45% less RSS**.
Ruled out, each by measurement:

- **Map content.** Allocated voxel counts match to 0.2%. Note the default npz
  hides this: only `p_occ >= 0.5` is published, and **74% of allocated voxels
  are free-space carve results** (30 528 free vs 10 708 occupied), so an IoU
  check on the published cloud cannot see most of the map.
- **Publishing.** `publishPointCloud` early-returns on
  `get_subscription_count() == 0`, and nothing subscribes during a run;
  `scovox_publish_rate:=0.1` changed nothing (164.1 vs 156–169 MB).
- **glibc arenas.** `MALLOC_ARENA_MAX=1` + a 128 KB trim threshold bought 12%
  (178.3 → 156.8 MB), not 45%.
- **Run-to-run noise.** Identical repeats span 8% (156.3 / 155.6 / 168.6 at
  400 frames; 429 vs 394 at 1300). Real, but an order below the gap.

The residual is heap layout/fragmentation and would need a heap profiler to
attribute — none is installed on this Jetson (no valgrind, no heaptrack).

**Consequence:** every RSS figure in this document is a process-level
observation, not a map-size measurement. The `enable_tsdf:=false` RSS drop is
the one that is probably genuine content (TsdfMap allocates a second voxel grid
over the same extent), but it has not been verified by allocated-voxel count and
should not be quoted as one. To compare memory between builds, count allocated
voxels with `occupancy_vis_threshold:=0.0` — not RSS.

Both variants are preserved at `tier0_full.hpp` / `tier0_sqrtonly.hpp` in the
session scratchpad.

### The rest of the gap

Two independent levers, and the telemetry says which one to pull first:

- **Fewer visits.** 8.5 M visits produce ~12 k unique staged voxels in the same
  window (682x; the 818x quoted elsewhere pairs a visit count from early frames
  with a staged count from a later window — `carve_n` runs 9.0 k at replay 0-49
  and 13.8 k at replay 200-249, so the ratio must be taken within one window).
  Nothing that changes the cost *per* visit — container (Tier 1) or arithmetic
  (Tier 0) — can beat cutting that. The two knobs that already do this are
  `carve_band:=0.1` (measured 2.7x, IoU 0.97) and `stride` (measured 8.80 Hz at
  s=2, 18.11 Hz at s=4). A voxblox-style per-frame visited-set with ray abort
  on consecutive collisions attacks the same redundancy without the accuracy
  trade, and is the one untried idea with the right shape.
- **Threading.** ~80% of frame time is the ray walk on a single core,
  and the batched carve is a commutative max-reduce, so rays are independent
  until flush. This is where the remaining 10x is, and it costs no accuracy.

Determinism is NOT established by the artifacts here. `scovox.npz` and
`scovox_rss_rep.npz` differ (28 484 vs 28 426 voxels) — but they integrated
different frames (1292 from replay 7 vs 1286 from replay 13), so the comparison
is confounded and says nothing either way. What IS exactly true is the
`enable_tsdf` check: `scovox.npz` and `scovox_notsdf.npz` are bit-identical
across all 13 fields. To test determinism, two runs must integrate the same
replay indices; none of the stored pairs do.

**Update:** the replay node now holds playback until every image subscriber
has matched (+2 s grace for /tf), so runs DO integrate identical frame sets,
and determinism IS established on that footing: `fps_ab/det1.npz` vs
`det2.npz` — two identical arms, 393 frames each — are bit-identical across
all 13 fields.

### Far-voxel skip: carve-off no longer pays for the walk it doesn't use

When the free-space carve is disabled (`batch_free_carve:=false` on the live
batched path), every voxel the DDA visits in front of the TSDF band does only
dead float math: the TSDF gate fails on distance, the semantic band gate fails
on distance, and `applyCarveUpdate` returns before touching anything
(`sem_split_map.cpp`, frame-open + carve-off early return). The fused walker
now proves that per voxel with three integer subtractions — skip any voxel
whose Chebyshev coord distance to the hit voxel exceeds
`(max(trunc + h, sem_band) + h)/res + 1` — and early-outs `visit_one`. The
test is sufficient regardless of the DDA's path shape, so the walk visits the
same voxels with identical writes: bit-identical by construction, and
verified three independent ways (adversarial differential build over 20
configs — 1.05 M state lines byte-identical; a 5 M-ray fuzz asserting the
bound on every skipped voxel — 0 violations; live A/B on 016 with matched
frame intake — `fps_ab/det1.npz` vs `noskip.npz` bit-identical across all
13 fields, where `noskip` sets `SCOVOX_DISABLE_FAR_SKIP=1`).

Same binary, 393 frames, TSDF on, 0.5–3.5 m, carve off — **numbers below are the
repo build at `962a9c1`** (an earlier local build measured ~30 ms faster across
the board on the same configs; those figures were from a tree that no longer
exists and have been replaced here rather than kept):

| arm | mean ms | Hz |
|---|---|---|
| far skip active (`det1`) | 231.3 | 4.32 |
| `SCOVOX_DISABLE_FAR_SKIP=1` (`noskip`) | 352.2 | 2.84 |

−120.9 ms, +52%, and the two maps are bit-identical across all 13 npz fields.

**6 Hz is not reached on this build in any configuration measured.** The ceiling
is `sdf_trunc_voxels:=1` at **185.1 ms → 5.40 Hz**, and it buys that by narrowing
the TSDF band to ±1 voxel *on top of* carve-off. Default band is 231.3 ms
(4.32 Hz); the shipped config, carving on, is 418.4 ms (2.39 Hz).

Two counter-intuitive results worth keeping:

* **`enable_tsdf:=false` is slower than TSDF-on with a thin band.** At matched
  range: 191.5 ms TSDF-off vs 185.1 ms at `sdf_trunc_voxels:=1`. `scovox_node.cpp:318`
  sends `sdf_trunc = 0` when TSDF is off and `tsdf_map.cpp:23` rewrites `<= 0` to
  `0.15f`, so disabling TSDF leaves the far-skip window at its **widest** (±5
  voxels) while doing no TSDF work inside it. Turning the module off is currently
  strictly dominated.
* Band width, not TSDF writes, is what the skip window costs — `trunc1` and the
  default-band run produce **bit-identical** clouds, because the published npz
  carries occupancy and semantics only and is blind to the TSDF grid.

Carve-off abandons the free-space map (~74% of allocated voxels), so this
operating point is for surface/TSDF-only consumers. With carving ON the skip is
inert by construction and the full-quality config still needs threading.

`ScovoxMapSplitFusedWalker.FarSkipBitIdenticalToFullWalk` guards this: one
process builds a full-walk map (env kill-switch, latched per instance) and a
skipping map, feeds both identical rays including exact gate-boundary
geometry, and demands bit-equality of every grid and touched-list. Both
mutations that matter — shrinking the window by a voxel, and adding any side
effect to the carve no-op path — were verified to fail it.

| class | share |  | class | share |
|---|---|---|---|---|
| wall | 29.8% | | sofa | 4.9% |
| bed | 20.3% | | pillow | 3.8% |
| floor | 13.4% | | unknown | 3.5% |
| otherfurniture | 9.0% | | door | 2.9% |
| window | 5.6% | | otherprop | 2.8% |

Only 3.5% unknown, which is the expected residue: labels cover 86.9% of pixels
(the rest is invalid depth), and voxels seen only at grazing incidence never
accumulate a confident class.

Note `sembeta_ms` stays 0.0 in the node log on this path. That is **not** because
of `use_split`, and not evidence that semantics are missing — it is the fused
walker: `semdirTimeUs()` is "0 by design (the combined cost is reported under
`tsdfTimeUs()`); non-zero only via `integrateHitSplit` (non-fused)"
(`scovox_map_split.hpp:566-569`). So on a default run **`tsdf_ms` is the whole
walk, not TSDF cost** — do not quote it as TSDF. Run `fused_walker:=false` to
split the two (see the E7 stage table below), and check `semantic_class` in the
captured cloud to confirm semantics are present.

## E7 embedded feasibility — measured (AGX Orin, repo build `962a9c1`)

> ✅ **PROMOTED 2026-08-21 — SCOPED TO SCENENN.** On the user's instruction
> *"Set E7 done only with scenenn results"*, this section is claim-bearing and
> the E7 cell is **done**. Promoted copy: `experiments/RESULTS.md` **Part XI**;
> decision recorded in `experiments/PLAN.md` **rev 58**. The SceneNN scope is
> binding and "done" does not widen it: this is an **AGX Orin, not the Jetson
> Nano E7 was specified against** (read it as *"embedded ARM, upper end"*, never
> as a Nano estimate), and there is **no KITTI leg** — see the end of this
> section for why none can be produced from this box. Three caveats are promoted
> *with* the numbers and are not detachable: sustained Hz is not readable off
> this harness, neither resolution meets the sensor rate, and RSS overstates the
> map by ~an order of magnitude. This section is hand-written — the scripts emit
> logs and a report, not this prose — so the banner lives here rather than in a
> generator.

```bash
             ./experiments/scovox_eval/scripts/scenenn_e7_embedded.sh 400 016   # 5 cm
RES=0.03     ./experiments/scovox_eval/scripts/scenenn_e7_embedded.sh 400 016   # 3 cm
python3 experiments/scovox_eval/scripts/scenenn_e7_report.py \
        experiments/results/scenenn/016/e7 25.0        # or .../e7_r03
```

Orin Developer Kit, 12 cores @ 2.2 GHz, MAXN, `schedutil`, 62.8 GB, JetPack
5.15.148-tegra, native build. SceneNN 016, stride 1, oracle labels, 393 frames,
last 200 analysed. n=3 on each fast point.

| res | config | mean ms | p50 | p95 | p99 | **Hz** | CPU %/core | RSS MB | core grid MB | +TSDF MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 cm | fast (±1 band, carve off) | **185.3 ± 0.6** | 184.0 | 193.4 | 198.6 | **5.40** | 64.5 | 76.5 | 3.081 | 5.107 |
| 5 cm | shipped default | 421.6 | 415.4 | 462.4 | 469.5 | 2.37 | 100.9 | 316.4 | 3.088 | 5.117 |
| 3 cm | fast (±1 band, carve off) | **221.0 ± 0.3** | 219.3 | 232.6 | 235.4 | **4.52** | 77.0 | 78.0 | 3.711 | 5.779 |
| 3 cm | shipped default | 695.9 | 690.5 | 779.9 | 798.9 | 1.44 | 100.7 | 529.0 | 5.737 | 7.819 |

Run-to-run SD is **0.35%** (5 cm) and **0.12%** (3 cm) — tighter than the ~0.55%
noise floor seen elsewhere, so differences above ~1% are real.

**3 cm costs far less than 1/res would suggest — but only with the skip armed.**
With `sem_band = 0` (default, unplumbed) and `h = res/2`, the far-skip threshold
`(max(trunc+h, sem_band)+h)/res + 1` reduces to **`sdf_trunc_voxels + 2` voxels,
independent of resolution** — 3 voxels per ray at both 5 and 3 cm. So:

| 5 cm → 3 cm | predicted | measured |
|---|---|---|
| fast (skip armed → window-bounded) | ~flat + grid density | **+19.3%** |
| shipped (carve on → skip inert → ray-bounded) | +66.7% (= 5/3) | **+65.1%** |

The shipped arm tracks the ray-length law to within 1.6 points. The fast arm's
residual is voxel count, not path length: the TSDF bracket is **flat** across
resolutions (45.5 → 45.0 ms) while the semantic bracket grows 115.8 → 160.4 ms
against 3.5× more active voxels. Dropping to 5 cm to buy speed is therefore a
poor trade at the armed operating point.

**The frame cost is single-threaded CPU-bound work.** Integrated process
CPU-seconds ÷ Σ`frame_ms` = **0.99–1.01** on every run. Not blocked, not
parallel — one core.

**Do not read "sustained Hz" off this harness.** The Python replay (PIL decode
+ ~2 MB/frame of byte copies) caps near **3.4 Hz**, so at the fast point the
mapper is *starved*: 64.5% of one core, idle 35% of wall, end-to-end 3.43 Hz
against a 5.40 Hz capability. The control is clean — at the shipped default the
mapper (2.37 Hz) is slower than the loader, duty rises to **101%**, and
end-to-end equals per-frame (2.38 vs 2.37). Quote `1000/mean(frame_ms)`; a real
sustained-rate number needs a pre-decoded in-RAM loader.

**RTF against the 25 Hz sensor rate is 4.63 / 10.54 (5 cm fast / shipped) and
5.53 / 17.40 (3 cm).** Even at 10 Hz the best cell is RTF 1.85. Neither
resolution meets the sensor rate in any configuration measured, and coarsening
to 5 cm does not rescue it (it only buys 19%).

**RSS is not map size, and these pairs prove it:** at 5 cm the two configs are
**4.1×** apart in RSS (76.5 vs 316.4 MB) but **0.2%** apart in grid bytes (5.107
vs 5.117 MB); at 3 cm, **6.8×** (78.0 vs 529.0 MB) against **35%** (5.779 vs
7.819 MB). Carving touches far more voxels transiently and the allocator keeps
the arena.

**Block fill, not struct size, sets grid bytes — until fill saturates.** At 5 cm,
carving takes the Beta grid from 11 390 to 40 542 active voxels (3.6×) for
**+0.007 MB**, because leaf fill goes 0.158 → 0.415; allocated bytes/voxel fall
186 → 53 B. At 3 cm that free ride ends: 39 860 → 175 727 voxels (4.4×) costs
**+98%** (2.065 → 4.093 MB) as fill goes 0.223 → 0.574 and real leaves get
allocated. "Carving is nearly free in memory" is a sparsity artifact, not a
property of the encoding. The analytical 16 B/voxel is a struct fact and
predicts neither number.

Voxel counts track geometry: surface-only 11 390 → 39 860 (**3.50×**, vs 2.78×
for an ideal (5/3)² law), free-space 40 542 → 175 727 (**4.33×**, vs 4.63× for
(5/3)³).

**Stage split** (`fused_walker:=false`, n=1) — necessary because the fused
walker reports 0 for `sembeta_ms` and folds everything into `tsdf_ms`:

| bracket | 5 cm ms | of frame | 3 cm ms | of frame |
|---|---|---|---|---|
| semantic substrate (Beta+Dirichlet) | **115.8** | 63% | **160.4** | 70% |
| TSDF substrate | 45.5 | 25% | 45.0 | 20% |
| caller work outside both brackets | 21.9 | 12% | 22.7 | 10% |

Semantics cost 2.5–3.6× TSDF, and are the only bracket that grows with
resolution. Caveats: this is the ±1-band config, which minimises TSDF by
construction, and `sembeta_ms` is Beta+Dirichlet combined, not Dirichlet alone.

At 5 cm that arm came out **183.2 vs 185.3 ms fused** — ~1% *faster*, which
looked like a lead. It does not replicate: at 3 cm it is **228.1 vs 221.0 ms**,
3% *slower*. The sign flips, so there is no evidence the non-fused path beats
the fused walker; treat the 5 cm gap as noise.

`log_mem_usage:=true` costs **+0.5%** (5 cm) and **−0.3%** (3 cm) on the frame
mean — within noise — so timing and memory can be read from the same operating
point.

**KITTI cannot be run on this Jetson.** There is no SemanticKITTI on the box:
both roots the harness expects (`$HOME/Projects/HMR_Exploration_Experiment/…/
semantickitti/dataset`, `${WS}/data/semantickitti/dataset`) are absent and a
filesystem sweep finds no `.label`, `velodyne/`, or `predictions_topk/`. Staging
it needs (1) seq 06–10 scans+labels, >13 GB against **11 GB free** on the single
57.8 GB eMMC with no external mount; (2) PolarNet top-K predictions, which need
an inference stack this box does not have and so must be produced off-robot;
(3) a resolution decision — the plan's KITTI principal is 10 cm, so "both
resolutions" there most likely means 10 cm + 5 cm rather than the 5/3 cm pair
used for SceneNN.

Caveat: an unrelated RViz session (up since 2026-08-05, ~17% of one core) was
live for these runs, as it was for every earlier measurement here. On 12 cores it
does not contend for the mapper's core, but the box was not otherwise idle.

## Sanity checks

`scenenn_check_relpose.py` is the one to run first on a **new scene** — it
re-derives the optical convention and pose offset from the data, and needs no
mesh. If its best residual is not ~1 cm, replay is misconfigured and nothing
downstream is worth looking at.

`scenenn_check_geometry.py` compares depth against the mesh directly; expect it
to look bad before registration (that is finding 3, not a bug).

## Tuning notes

`stride` defaults to 2. A 5 cm voxel spans `0.05*544/z` px, so at the 1.5–3 m
range SceneNN actually observes, stride 1 puts 80–330 correlated pixels into
one voxel — that saturates the Beta posterior on what is effectively a single
surface observation and makes the uncertainty output meaningless (same argument
as `start_scovox.sh` on the robot). Stride 2 is 4x cheaper and still
oversamples; drop to 4 if frame time matters more than thin structures.

`max_range` defaults to 4.0 m, the Xtion's usable band, rather than SceneNet's
synthetic 10 m.
