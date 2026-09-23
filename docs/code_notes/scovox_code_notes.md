# scovox — code comment notes

Long comments from files in the `scovox` repository, moved out of the code on 2026-09-23 so the sources carry short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word. Files with many moved comments have their own notes doc next to this one.

One section per source file, in file order. Each entry names the function (or section) the comment sat in, the line of code it was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [compose.yaml](#composeyaml) — 1
- [config/exploration_fused_bag.yaml](#configexploration_fused_bagyaml) — 3
- [config/scovox_fine_band.yaml](#configscovox_fine_bandyaml) — 1
- [config/scovox_fused_lidar_rgbd.yaml](#configscovox_fused_lidar_rgbdyaml) — 1
- [config/scovox_lidar_raw_deskew.yaml](#configscovox_lidar_raw_deskewyaml) — 2
- [config/scovox_robot_share.yaml](#configscovox_robot_shareyaml) — 2
- [docker/build_and_test.sh](#dockerbuild_and_testsh) — 1
- [experiments/kitti/build_gt.py](#experimentskittibuild_gtpy) — 5
- [scripts/launch_scovox.sh](#scriptslaunch_scovoxsh) — 1

## compose.yaml

### compose-gpu-passthrough

**NVIDIA GPU passthrough in compose** — in `Top level`, attached to `runtime: nvidia` (line 32)

```text
NVIDIA GPU passthrough so in-container RViz renders on the discrete GPU
(hardware GL), not Mesa/llvmpipe. Host default-runtime is already nvidia,
but request it explicitly (mirrors glim_localisation/compose.yaml). On this
PRIME/Optimus laptop the RViz launch must also export
__NV_PRIME_RENDER_OFFLOAD=1 + __GLX_VENDOR_LIBRARY_NAME=nvidia (see
run_fused_experiment.sh); DRIVER_CAPABILITIES=all pulls in the GLX libs.
GPU passthrough: `runtime: nvidia` + NVIDIA_VISIBLE_DEVICES/DRIVER_CAPABILITIES
(in environment below) is the portable mechanism. The `gpus: all` short syntax
needs Compose >= v2.30; this host runs v2.26, which rejects it as an unknown
property. runtime+env gives the same full-GPU access (verified: nvidia-smi -L
lists the Quadro P2000 inside the container).
```

## config/exploration_fused_bag.yaml

### fused-bag-terrain-mode

**Terrain-relative 3D mode, no planning map** — in `explo_planner — bag-driven fused-map experiment (waypoint visualization)`, attached to `explo_planner:` (line 16)

```text
TERRAIN (3D mode): terrain_relative_z: true makes the planner fully
terrain-relative on this hilly site (ground spans ~[-11, +22] m):
 - the map z-band (roi_min_z/roi_max_z) rides WITH the robot,
 - candidates snap to local ground + candidate_z_clearance (frontier
   candidates to the ground under their own centroid), so goals and
   markers carry a real 3D z,
 - NO 2D planning_map: use_planning_map: false runs the planner in
   its straight-line fallback — Euclidean candidate costs, no 2D
   reachability/free-cell filter. The live scovox projection covers only
   the walked corridor and a static GT prior came out ~90% unknown at
   planning resolution (unknown counts as occupied), so both starved the
   candidate filter. Obstacle rejection still happens against the 3D map
   (candidate_occ_thresh at generation + ground snapping).
```

### fused-bag-roi-z-band

**Robot-relative ROI z band sizing** — in `Terrain-relative 3D mode (see header)`, attached to `roi_min_z: -5.0` (line 120)

```text
With terrain_relative_z: true this z band is RELATIVE to the robot's
current altitude and rides with it (map-cache ingest clip, frontier
extraction and the FOV/EIG ray z-clip all re-band as the robot climbs).
+4 covers the 10 m FOV rays at +-22.5 deg pitch plus local slope; the
floor is ground_search_below_m plus the ~1 m re-band hysteresis, so the
candidate ground search can never poke below the ingested slab.
```

### fused-bag-fov-unverified

**FOV values unverified for the bag sensor** — in `Terrain-relative 3D mode (see header)`, attached to `fov_hfov: 1.047` (line 129)

```text
FOV evaluation for EIG scoring (matches the RGB-D-ish default geometry).

UNVERIFIED FOR THIS CAMPAIGN. shared_params.yaml has moved fov_vfov to
0.5236 (the sim husky's VLP-16, +-15 deg) because its runs are that
sensor. These bags are a different unit, and nobody has written down its
actual vertical FOV, beam count or usable range here -- so the values are
left alone rather than assumed to match. Before this config is used for
anything but replaying the existing bags, set fov_vfov / fov_v_rays /
fov_max_range from that unit's spec sheet. The horizontal figure is a
60 deg stand-in for a 360 deg sensor in both files; see the longer note in
shared_params.yaml for why that is not a one-line fix.
```

## config/scovox_fine_band.yaml

### fine-band-lattice-ratio

**Choosing the fine lattice ratio** — in `Fine lattice`, attached to `fine_ratio_log2: 2            # res_fine = resolution / 4 (0.10 -> 0.025)` (line 20)

```text
res_fine = resolution / 2^k. At base 0.10 m: k=2 -> 2.5 cm, k=3 ->
1.25 cm. k=3 is the sensor floor for the Hesai XT32 at 3-4 m orbit
range (~1.1 cm horizontal beam spacing, ~1 cm range sigma); beyond it
voxels go hit-starved and noise dominates. Pair k=3 with
fine_sdf_trunc_voxels: 3 (trunc 3.75 cm, still > 2x range sigma).
```

## config/scovox_fused_lidar_rgbd.yaml

### fused-lidar-authority-policy

**Pure LiDAR authority fusion policy** — in `SCovox — FUSED LiDAR-occupancy + RGB-D-semantics into ONE SemSplitMap`, attached to `/**:` (line 9)

```text
Fusion policy (pure LiDAR authority — behind the fuse_lidar_rgbd switch):
  * LiDAR owns Beta occupancy: lidar_w_occ=8, lidar_w_free=4, full-ray carve.
  * RGB-D is semantics-ONLY: rgbd_w_occ=0, rgbd_w_free=0, rgbd_geometry_off —
    it deposits ZERO occupancy/carve/TSDF; a bad RGB-D pixel can only misplace
    a decayable Dirichlet label, never move geometry.
  * Stream-B gate (rgbd_dirichlet_min_p_occ=0.55) reads the LiDAR-only p_occ,
    so a semantic label commits ONLY where LiDAR says occupied. >0.5 is
    mandatory: at 0.5 an RGB-D hit on a voxel LiDAR never touched (prior
    p_occ=0.5) would commit semantics on prior-only geometry.
```

## config/scovox_lidar_raw_deskew.yaml

### raw-deskew-why

**Why deskew the raw cloud natively** — in `SCovox — LiDAR occupancy mapping on the RAW Ouster cloud, deskewed in-node`, attached to `/**:` (line 9)

```text
WHY: feeding /glim_ros/points fixes the 7.2 m vertical smear (-> 0.40 m) but
costs recall because GLIM downsamples to ~10k pts/scan. Native deskew keeps the
full-res cloud -> thin surfaces AND full coverage. The smear is intra-scan SKEW
(per the feed-cmp in scovox-vertical-overfill.md); rotation dominates it, and
gyro integration reproduces GLIM's rotational deskew without GLIM's downsampling.
```

### raw-deskew-downsample-sweep

**Downsample voxel size sweep** — in `Uniform voxel-grid downsample (the smear lever)`, attached to `downsample_voxel_size: 0.5` (line 56)

```text
GLIM thins its map by voxel-downsampling each scan in preprocessing
(config_preprocess.json: downsample_resolution 1.0, ~10k pts/scan). The raw
full-res cloud over-samples the noisy surface so every scan fills the tails
of each column's z-distribution → thick smear. Collapsing points to one
centroid per voxel (sensor frame, per scan, after deskew) cuts that
tail-sampling. 0.0 = off. Sweep up from 0.01 to trade thinness vs recall.

SWEEP RESULT (rate 0.5, 120s, max_range 20) — shared-column z-cells / XY
footprint(cols) / scovox pts:
  0.01: 17cells 163k 3.39M | 0.1: 16 164k 3.18M | 0.2: 11 162k 2.48M
  0.5:  4cells 151k 1.05M | 1.0: 2cells 125k 0.40M
Knee at 0.5: 4x thinner (17->4) for only -7% footprint; past it the footprint
tax accelerates for little extra thinness. 0.5 matches the 0.40m thinness of
feeding GLIM's downsampled /glim_ros/points, natively, at 2.6x the points.

0.5 is also the in-code default (the sweep knee above); kept explicit
here as this config's sweep is the evidence for it. Refinement regions
are unaffected (raw returns bypass this via fine_raw_returns).
```

## config/scovox_robot_share.yaml

### share-bandwidth-measured

**Share bandwidth on map-test-2** — in `Low-bandwidth share controls (Tier 1)`, attached to `share_change_gate: true      # re-send a voxel only when it changed vs its` (line 44)

```text
Measured on the map-test-2 stream: legacy 32.8 Mbps -> gate 20.7 ->
gate + 2 Hz coalescing + z-band 4.9 Mbps per robot.
```

### share-roi-z-band-sync

**Shared ROI z band coupling** — in `Low-bandwidth share controls (Tier 1)`, attached to `share_roi_z_min: -0.5` (line 52)

```text
Shared-ROI z-band (map frame, metres); min >= max disables.
KEEP IN SYNC with the receive-side clip (dscovox_params.yaml
share_roi_z_min/max) and with the planner band (explo_planner
shared_params.yaml roi_min_z/roi_max_z): the shared band must be a
SUPERSET of the planner band.
```

## docker/build_and_test.sh

### build-resource-caps

**Container build resource caps** — in `Top level`, attached to `set -euo pipefail` (line 11)

```text
Resource bounds: the C++ build is template-heavy (Bonxai/Eigen), so an
unbounded colcon fans out one g++ per host core and can exhaust RAM and hard-
freeze the host. We cap the container's CPUs/RAM and serialise packages so at
most SCOVOX_BUILD_JOBS compilers run at once. --memory-swap == --memory means
no swap, so a runaway TU is OOM-killed inside the container (clean build
failure) instead of dragging the host into swap-thrash. Override via env:
```

## experiments/kitti/build_gt.py

### gt-accumulator-alternatives

**Rejected per-voxel accumulator designs** — in `main`, attached to `import tempfile, os` (line 109)

```text
Running accumulator: key(int64) -> [counts_per_class] as numpy array
Use a simple approach: collect all (key, label) from all scans,
then do ONE final sort + groupby. But stream to disk to limit RAM.

Actually: just use a Python dict[int, np.uint16[20]].
50M voxels × (40 bytes array + ~120 bytes dict overhead) ≈ 8 GB. Too much.

Compromise: use dict[int, int] mapping key -> packed(best_label << 16 | best_count).
This is a simple running max — NOT a true majority vote, but close enough
when one label dominates (which it does for most voxels at 10cm).

Better compromise: dict[int, int] mapping key -> (label << 16 | count) for the
top label only. Update: if new label == stored label, increment count. If new
label != stored label, decrement count; if count reaches 0, replace with new.
This is the Boyer-Moore majority vote algorithm — exact majority in one pass.
```

### gt-boyer-moore-rejected

**Boyer-Moore dict accumulator rejected** — in `main`, attached to `import tempfile, os` (line 125)

```text
Boyer-Moore streaming majority vote per voxel
Store: voxel_key -> (candidate_label, count)
Python int is 28 bytes, tuple is 56 bytes, dict entry ~100 bytes
50M entries × ~180 bytes = 9 GB. Still too much.
```

### gt-external-sort-option

**External sort option** — in `main`, attached to `import tempfile, os` (line 130)

```text
NUCLEAR OPTION: just write per-scan (key, label) arrays to disk,
then process with external sort in chunks.
```

### gt-sort-groupby-option

**Sort and groupby option** — in `main`, attached to `import tempfile, os` (line 133)

```text
SIMPLEST OPTION THAT WORKS: Process scans, build per-scan voxelized
arrays, concatenate keys+labels, sort, groupby. But do it with
memory-mapped temp files.
```

### gt-two-pass-pairs

**Two-pass voxel label pairs on disk** — in `main`, attached to `import tempfile, os` (line 137)

```text
Let's try: accumulate per-scan unique (key, label) pairs.
~50k unique voxels/scan × 4071 scans = ~200M pairs × 9 bytes = 1.8 GB on disk.
Then mmap, sort by key, groupby.
```

## scripts/launch_scovox.sh

### launch-scovox-localizer-split

**Localizer and SCovox responsibilities on hardware** — in `Top level`, attached to `set -e` (line 4)

```text
A localizer (e.g. GLIM LiDAR-IMU SLAM, running in a separate container) owns the
TF tree (map -> odom -> imu -> os_lidar) and provides the per-scan pose; SCovox
subscribes to the full-resolution /ouster/points + /imu/data over ROS 2 DDS and
builds the occupancy map online. It deskews each scan natively (gyro rotation)
and voxel-downsamples it (downsample_voxel_size) to suppress the vertical smear
without GLIM's recall-costing cloud downsample -- see config/ + the README.
```
