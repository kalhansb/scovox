# Distributed multi-robot mapping — LiDAR-only (geometric) runbook

Multiple robots build **one shared occupancy map** from LiDAR alone — no camera,
no segmentation, no semantics. Each robot maps its own surroundings and streams
compact map deltas to the others; every robot fuses everyone's stream into its
own copy of the global map. There is **no central server**.

This is the geometry-only sibling of [distributed_mapping.md](distributed_mapping.md)
(which also fuses RGB-D semantics). If you only have LiDAR, use this one.

```
                 robot 1                                robot 2
 ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
 │ hmr_localisation (NDT vs gt_map) │   │ hmr_localisation (NDT vs gt_map) │
 │   map → odom → base_link         │   │   map → odom → base_link         │
 │ scovox mapper  (LiDAR occupancy) │   │ scovox mapper  (LiDAR occupancy) │
 │   → /robot1/.../scovox_bin ──────┼──→                                   │
 │                             ←────┼───── /robot2/.../scovox_bin          │
 │ dscovox merger (fuses BOTH)      │   │ dscovox merger (fuses BOTH)      │
 │   = robot 1's GLOBAL map         │   │   = robot 2's GLOBAL map         │
 └──────────────────────────────────┘   └──────────────────────────────────┘
```

Each robot runs the same **three programs**:

1. **Localizer** — [hmr_localisation](https://github.com/kalhansb/hmr_localisation)
   NDT against a shared ground-truth map. This gives every robot the same global
   `map` frame, so no robot-to-robot pose estimation is needed.
2. **Mapper** — `scovox_mapping_node`. Builds a local occupancy voxel map from
   `/ouster/points` and publishes a small LZ4-compressed delta stream (`~/scovox_bin`).
3. **Merger** — `dscovox_mapping_node`. Subscribes to **every** robot's delta
   stream and fuses them. Its output is that robot's copy of the global map —
   the topic the exploration planner reads.

Everything runs inside the repos' Docker containers; nothing is installed on the host.

## Two rules that make fusion work

1. **Each robot must map in a UNIQUE frame** (`r1_map`, `r2_map`, …), bridged to
   `map` by an identity static TF. The delta stream is tagged with this frame, and
   the merger keys each robot's data by it. If two robots both map directly in
   `map`, their streams collapse into one and overwrite each other where the maps
   overlap.
2. **Start the merger before the mapper.** The delta stream is only sent when
   someone is subscribed — a mapper with no listener throws its deltas away. (A
   late-joining merger still catches up: the mapper re-sends a full snapshot when a
   new subscriber appears.)

## Prerequisites (each robot)

1. Clone and build the workspace:
   ```bash
   git clone --recursive https://github.com/kalhansb/hmr_explo.git
   cd hmr_explo/ws/src/hmr_localisation && docker compose build && docker compose up -d
   cd ../scovox && docker compose build && docker compose up -d
   ```
   Follow the one-time localizer build steps in the
   [hmr_localisation README](https://github.com/kalhansb/hmr_localisation#run)
   (the 0.5 m localization map `gt_map/gt_map_us050.pcd` is committed).
2. **Sensors**: an Ouster driver publishing `/ouster/points` with **RELIABLE QoS**
   (the localizer subscribes RELIABLE; a best-effort publisher never connects) and
   an IMU on `/imu/data` (used for per-scan motion deskew).
3. **Extrinsics**: the `base_link → os_lidar` / `imu` transforms are baked into
   hmr_localisation's `run_localization_live.sh` for the map-test robot —
   re-measure them for a different platform.
4. **Network**: all robots on one LAN with the same `ROS_DOMAIN_ID`. The compose
   files pin DDS to loopback, so any `docker compose exec` that must talk across
   machines needs `-e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`.

## Config files (identical on every robot)

- [scovox_lidar_geometric.yaml](../config/scovox_lidar_geometric.yaml) — the
  LiDAR-only mapper base config (Beta occupancy, full-ray free-space carve, in-node
  deskew; `fuse_lidar_rgbd: false`).
- [scovox_robot_share.yaml](../config/scovox_robot_share.yaml) — share overlay:
  flips the mapper to `mode: rolling` (creates the delta stream) and sets the
  low-bandwidth controls (change gate + 2 Hz coalescing + z-band).
- [dscovox_params.yaml](../src/scovox_mapping/config/dscovox_params.yaml) — merger
  config. `input_topics` lists the fleet's bin topics; **extend it on every robot
  as the fleet grows**.

## Two robots

Set up **each** robot exactly as in
[single_robot_lidar.md](single_robot_lidar.md) — localizer → identity static TF →
merger → mapper — changing only the robot index:

| per-robot value       | robot 1     | robot 2     |
|-----------------------|-------------|-------------|
| namespace (`__ns`)    | `/robot1`   | `/robot2`   |
| `integration_frame`   | `r1_map`    | `r2_map`    |
| static TF child frame | `r1_map`    | `r2_map`    |

Everything else comes from the committed config files above, unchanged. On each
robot, extend `input_topics` in `dscovox_params.yaml` so it lists **all** robots'
bin topics (`/robot1/scovox_node/scovox_bin` + `/robot2/scovox_node/scovox_bin`).

Robots do not need to start at the same time: a late merger gets a fresh snapshot,
and a late mapper just starts contributing when it comes up.

**Verify the exchange** (on *either* robot; `$E`/`$S` are the exec shorthands
defined in [single_robot_lidar.md](single_robot_lidar.md)):

```bash
$E bash -lc "$S && ros2 topic hz /robot1/scovox_node/scovox_bin /robot2/scovox_node/scovox_bin"
```

Both streams visible on both machines proves DDS discovery. Then each merger's
`dscovox_diag` line should reach `sources=2`, and the two robots' `fused_voxels`
counts should converge toward each other.

**Adding robot 3+**: add its bin topic to `input_topics` (on all robots), give it
`/robot3` + `r3_map`, done.

## Bandwidth

Occupancy-only, so **no semantic records travel on the wire** — this stream is a
strict subset of, and cheaper than, the fused pipeline (which measured ~4.9 Mbps
per robot). Wire cost per delta ≈ `28 + 20·N_beta` bytes. The knobs live in
[scovox_robot_share.yaml](../config/scovox_robot_share.yaml): `share_change_gate`
(re-send a voxel only when it changed), `share_rate_hz` (coalesce deltas), and the
z-band `share_roi_z_min/max`. Keep the z-band a **superset** of the planner band
(`roi_min_z`/`roi_max_z` in explo_planner's `exploration_params.yaml`) and in sync
with the merger's receive-side clip (`share_roi_z_min/max` in `dscovox_params.yaml`).

## Validating without robots

[run_mapshare_experiment.sh](https://github.com/kalhansb/hmr_explo/blob/main/ws/src/run_mapshare_experiment.sh)
in the hmr_explo workspace replays a recorded bag through this exact topology (N
mappers + N mergers, one sensor stream standing in for every robot) and proves
cross-robot fusion. Use the LiDAR-only arm:

```bash
cd ws/src
LIDAR_ONLY=1 ./run_mapshare_experiment.sh 120   # ends with FUSION VERIFY: PASS/FAIL
```

`LIDAR_ONLY=1` drops the camera/segmentation entirely, so it exercises the same
occupancy-only fusion this runbook produces. Each simulated robot shares only a
disjoint z-slice, so any voxel a robot's merger holds inside a *peer's* slice can
only have arrived over the wire.

## Troubleshooting

- **Merger stuck at `sources=0`** — the mapper started before the merger
  subscribed (deltas discarded). Restart the mapper, or check the bin topics are
  visible on the peer (`ros2 topic list`; if missing, it's DDS discovery — check
  `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, same `ROS_DOMAIN_ID`, same subnet).
- **Mapper logs "no exact-stamp TF … DROPPING scan"** — the localizer isn't
  publishing `map → odom` yet (or died). The mapper deliberately drops scans
  rather than integrate at a stale pose. If NDT sits at "Configuring end" without
  "Activating end", nudge it: `ros2 lifecycle set /lidar_localization activate`.
- **Localizer gets no scans** — the Ouster driver is publishing BEST_EFFORT;
  switch it to RELIABLE.
- **Both robots' maps flicker/overwrite each other** — they share one
  `integration_frame`; give each its own `rK_map` + identity static TF (rule 1).
- **Merger logs `No TF for 'rK_map'`** — that robot's identity static TF isn't
  running or isn't reaching this machine; the merger waits until it appears.
- **Merger logs `prior mismatch … dropping frame`** — a robot is running a
  different config. Every robot must use the identical
  [scovox_lidar_geometric.yaml](../config/scovox_lidar_geometric.yaml).
