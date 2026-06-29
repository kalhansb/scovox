# SCovox map accuracy vs GLIM full SLAM map

**Date:** 2026-06-29
**Bag:** `2026_06_19_18_19_06__kalhan-map-test-2_` (Ouster + IMU, ~500 s)
**Pipeline:** GLIM LiDAR-IMU SLAM → TF → SCovox occupancy, *live*, GLIM-deskewed feed.
**Reference:** GLIM dense factor-graph map (`glim_map.pcd`) — treated as geometric ground truth.

## Configuration under test

| param | value |
|---|---|
| input cloud | `/glim_ros/points` (GLIM-deskewed, frame `imu`) |
| `base_frame` / `integration_frame` | `imu` / `odom` |
| `resolution` | 0.10 m |
| `max_range` | 15.0 m |
| `occupancy_vis_threshold` | 0.6 |

Config: `scovox/config/glim/scovox_lidar_glim.yaml`.

## Method

1. Both clouds voxel-downsampled at 0.10 m.
2. SCovox (`odom`) rigidly ICP-aligned onto GLIM (`map`), outlier-gated at 1 m so the
   alignment is not biased by any sparse far voxels.
3. Bidirectional nearest-neighbour distances (scipy `cKDTree`):
   - **SCovox → GLIM** = precision/accuracy (is each SCovox voxel a real surface?)
   - **GLIM → SCovox** = recall/coverage (is each GLIM surface mapped?)
4. **Fair coverage:** recall recomputed over only the GLIM surfaces within `max_range`
   (15 m) of the trajectory — i.e. the region SCovox actually attempts to map.

Same-run pair. The live SCovox map was salvaged from the running node via
`glim_localisation/scripts/glim/salvage_capture_seq.py` (advancing `/clock` +
`SALVAGE_RELIABLE=1`). Comparison tool:
`glim_localisation/scripts/glim/compare_scovox_glim.py`.

## Inputs

| | raw points | @0.10 m voxels | bbox (m) |
|---|---|---|---|
| SCovox (`odom`) | 1,043,389 | 575,636 | 82 × 100 × 18 |
| GLIM (`map`) | 12,250,000 | 4,270,126 | 232 × 249 × 56 |

ICP `odom`→`map`: **|t| = 0.095 m, RMSE 0.072 m** → GLIM was essentially drift-free
on this bag, so `odom` ≈ `map` and the metrics below are alignment-robust.

## Results

### Accuracy / precision (SCovox → GLIM) — excellent

| metric | value |
|---|---|
| median voxel → surface error | **0.057 m** (~½ voxel) |
| mean / RMSE | 0.063 / 0.075 m |
| p95 / p99 / max | 0.118 / 0.191 / 4.17 m |
| precision @0.2 m / @0.3 m | **0.991** / 0.998 |
| spurious voxels (> 0.5 m from any GLIM surface) | **0.1 %** |

Nearly every SCovox voxel lies on a real GLIM surface.

### Coverage (GLIM → SCovox)

| denominator | recall @0.2 m | recall @0.5 m | median gap |
|---|---|---|---|
| full GLIM extent | 0.30 | 0.33 | 6.4 m |
| **within 15 m of trajectory (fair)** | **0.72** | **0.77** | **0.098 m** |

Only **41.9 %** of GLIM's full map lies within SCovox's 15 m range gate; the rest is
far/tall structure SCovox excludes **by design**. Restricted to the mappable
near-field, coverage is 72–77 % with ~10 cm median gap. The ~25 % shortfall is
occlusion / observation geometry plus the 0.6 occupancy-probability pruning.

### F-score (full-extent, for reference)

| τ (m) | precision | recall | F |
|---|---|---|---|
| 0.10 | 0.909 | 0.216 | 0.349 |
| 0.20 | 0.991 | 0.302 | 0.463 |
| 0.30 | 0.998 | 0.315 | 0.479 |
| 0.50 | 0.999 | 0.328 | 0.493 |

### Vertical extent / over-fill

| | z range (m) | z span | z (p1 … p99) |
|---|---|---|---|
| GLIM | −7.27 … 48.38 | 55.65 | −1.24 … 28.74 |
| SCovox | −1.50 … 16.30 | 17.80 | −1.10 … 13.20 |

0 % of SCovox voxels exceed GLIM's z-envelope. **The historical ~6–7 m vertical
over-fill smear (raw-`/ouster/points` feed) is absent** — the GLIM-deskewed feed is
the fix.

## Verdict

SCovox produces a **geometrically faithful, low-noise** local occupancy map
(~6 cm median accuracy, ~0 % spurious occupancy) that covers ~72–77 % of the
near-field (≤ 15 m) GLIM geometry. It is intentionally a tight, planning-scale
subset of GLIM's full-extent SLAM map — that is correct behaviour, not a defect.

## Reproduce

```bash
# from HMR_Explo/ws/src
./run_glim_experiment.sh map "" 0.5          # builds glim_map.pcd + scovox_map.npy
# then, in the glim container:
python3 /ws/scripts/glim/compare_scovox_glim.py \
    /ws/output/scovox_node.npy /ws/output/glim_map.pcd 0.10 /ws/output/path_glim.csv
```

Artifacts: `glim_localisation/output/{glim_map.pcd, scovox_node.npy, path_glim.csv,
scovox_vs_glim_eval.md}`.
