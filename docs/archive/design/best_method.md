# The Winning Method

**Configuration:** `e5/k2_i0_evid`  
**Evaluation:** E9, cell `abl_inherit`

## 1. Build Configuration

Source: `/home/kalhan/Documents/projects/.build/e5/k2_i0_evid/CMakeCache.txt`

| Option | Value | Meaning |
| --- | --- | --- |
| Build type | `-O3 -DNDEBUG` | Release build |
| `SCOVOX_K_TOP` | `2` | Two class slots per voxel |
| `SCOVOX_EVICT_INHERIT` | `0` | An evicted slot resets rather than inheriting mass |
| `SCOVOX_TRACK_QMAX` | `1` | Track peak confidence; required by `--evict-by-confidence` |
| `SCOVOX_TRACK_NHIT` | `0` | Per-slot deposit counter. **Not set as of 2026-09-04** — no reader; `DirVoxel` 24 B -> 20 B. The E9 numbers below were produced by a build that still set it, and are unaffected: the field is write-only, so the dump is byte-identical either way ([storage_defaults_2026_09_04.md](storage_defaults_2026_09_04.md)). |
| `SCOVOX_E0_COUNTERS` | `1` | Enable counter instrumentation |

## 2. Run Configuration

Run from `results_e9/RUNS`:

```text
--resolution 0.05 --evict-by-confidence --sem-band 0.10
--w-occ 1.5 --w-free 1.0
--dump-below-gate-as-unknown --dump-label-gate 0.5
--e0-counters <json> --dump-slots <slots>
```

## 3. Load-Bearing Defaults

These options are not passed on the command line, but define the method:

| Option | Value | Meaning |
| --- | --- | --- |
| `fused_walker` | `1` | One DDA per ray, rather than two |
| `tsdf_enabled` | `0` | TSDF band is off; no mesh is produced |
| `far_fast_paths` | `1` | Sphere gate plus ring latch |
| `stride` | `2` | Depth subsampling, approximately 77k rays/frame |
| `min_depth / max_depth` | `0.4 / 4.0 m` | Usable Asus Xtion depth range |
| `num_classes` | `14` | Number of semantic classes |
| `alpha0` | `0.01` | Dirichlet prior per class |
| `min_p_occ` | `0.5` | Mapper deposit gate |
| `sem_mode` | `0` | `DIRICHLET`, not naive or majority vote |
| `inc_mode` | `0` | Soft labels; deposit the full probability vector |
| `batch_hits` | `1` | Hit batching enabled |
| `carve_no_return` | `true` | Rays with no return still carve |
| `kappa0 / range_decay` | `1.0 / 50.0` | Evidence parameters |
| `evid_sat / cls_evid_sat` | `0.0 / -1.0` | Both saturation caps are off |
| `ray_spread / spread_radius` | `0 / 0.0` | No BKI kernel or front-spread fallback |
| `band_require_occ` | `false` | Band deposits are not occupancy-gated |
| Fine SDF truncation | `3 voxels = 0.15 m` | SDF truncation distance |

## 4. Algorithm Per Ray

- **Traversal:** Exact DDA (Bresenham was deleted on 2026-09-02) over
	`[endpoint - max(depth, trunc) * u, endpoint + max(trunc, sem_band) * u]`.
- **Far-voxel fast path:** Six per-axis Chebyshev comparisons, followed by a
	Euclidean `dx² + dy² + dz² > far_thr_sq` check and an origin-ring latch.
	Passing voxels use an integer-only path and skip the floating-point body. The
	result is bit-identical to the body it skips.
- **Occupancy:** A dedicated `BetaVoxel{a_occ, a_free}` grid with prior
	`Beta(1, 1)`, so unobserved voxels have `p_occ = 0.5`. At the hit,
	`a_occ += 1.5`; along the carve, `a_free += 1.0`.
- **Semantics:** A separate sparse `DirVoxel` grid, allocated only where a
	class is observed. A semantic deposit is made at the hit voxel when
	`p_occ >= 0.5`, plus in every voxel within `0.10 m` Euclidean distance of
	the hit point. This is a ball, not a normal-oriented slab, and excludes the
	hit voxel itself for the band deposit.
- **Slot maintenance:** Two slots are maintained using argmax-by-`cnt`. A
	third class evicts the weaker slot only when its single-deposit confidence
	beats that slot's all-time peak: `q_fx > qmax[min_i]`. The victim's mass is
	added to `other`; the newcomer starts fresh at `α₀ + inc` with no
	Space-Saving inheritance.

## 5. Output Representation

Each voxel uses 24 bytes:

```text
other (f32), cnt[2] (f32), cls[2] (u16), qmax[2] (u16), nhit[2] (u16)
```

At readout, the label is the argmax over raw `cnt`. Confidence is:

```text
cnt[best] / s_class()
```

`s_class()` is the voxel's total class evidence. It read `other + Σcnt` until
2026-09-04; it is now the stored word itself, so the denominator is an O(1) read
and is exact rather than a three-float sum (dir_total_basis_2026_09_04.md).
The formula is unchanged.

Probability is never stored. If `p_occ < 0.5`, the export keeps the label but
restates it as unknown (`state 4`) rather than dropping it.

## 6. Scores

Mean over eight SceneNN scenes at their per-scene frame caps:

| Metric | Result |
| --- | ---: |
| Semantic mIoU, intersection | `0.5853` |
| Semantic mIoU, union | `0.2981` |
| Occupancy IoU | `0.4393` |
| Occupancy precision / recall | `0.5236 / 0.7272` |
| Phantom voxels | `16,005` |
| Throughput | `6.2-6.7 fps`, single core |

## Caveats

- `--dump-slots` is analysis instrumentation, not part of the method.
	`SCOVOX_TRACK_NHIT` was likewise instrumentation and **is no longer set in
	the shipped build** (2026-09-04). `nhit` is never read by a mapping
	decision; eviction reads `qmax`. Removing it cannot change any output
	value, which is why the scores above still stand unchanged.
- `tsdf_enabled = 0` means the winning numbers were produced with the TSDF band
	never written. This is the correct configuration for these scores because no
	mesh is needed, but it is a default rather than an explicit command-line
	choice. Most traversal/TSDF review findings are therefore inert against this
	pipeline.