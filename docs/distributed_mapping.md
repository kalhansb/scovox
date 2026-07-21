# Distributed multi-robot mapping — real-robot runbook

Every robot runs the same three pieces; there is **no central merger**:

```
                 robot 1                                robot 2
 ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
 │ hmr_localisation (NDT vs gt_map) │   │ hmr_localisation (NDT vs gt_map) │
 │   map → odom → base_link         │   │   map → odom → base_link         │
 │ scovox mapper  (mode: rolling)   │   │ scovox mapper  (mode: rolling)   │
 │   → /robot1/scovox_node/scovox_bin ──┼──→                               │
 │                               ←──┼───── /robot2/scovox_node/scovox_bin  │
 │ dscovox merger (fuses BOTH)      │   │ dscovox merger (fuses BOTH)      │
 │   = robot 1's GLOBAL map         │   │   = robot 2's GLOBAL map         │
 └──────────────────────────────────┘   └──────────────────────────────────┘
```

- **Localizer** ([hmr_localisation](https://github.com/kalhansb/hmr_localisation)):
  NDT against the shared ground-truth map gives every robot the same global `map`
  frame — no cross-robot pose estimation needed.
- **Mapper** (this repo, `scovox_mapping_node`, `mode: rolling`):
  builds the local voxel map and publishes an LZ4-compressed delta stream
  (`~/scovox_bin`). With the low-bandwidth share controls (change gate + 2 Hz
  coalescing + z-band) this measured **~4.9 Mbps per robot** on the map-test-2
  stream (down from 32.8 Mbps legacy).
- **Merger** (`dscovox_mapping_node`): subscribes to **all** robots' bin streams and
  fuses them; its output is that robot's copy of the global map — the latched
  `~/scovox` topic the exploration planner consumes.

Everything below runs inside the repos' Docker containers; nothing is installed on
the host. Verified end-to-end against the recorded bag on 2026-07-02
(`FUSION VERIFY: PASS`, see [Validating without robots](#validating-without-robots)).

## Prerequisites (each robot)

1. Clone the [hmr_explo](https://github.com/kalhansb/hmr_explo) workspace
   recursively and build the two containers:
   ```bash
   git clone --recursive https://github.com/kalhansb/hmr_explo.git
   cd hmr_explo/ws/src/hmr_localisation && docker compose build && docker compose up -d
   cd ../scovox && docker compose build && docker compose up -d
   ```
   Follow the one-time build steps in the
   [hmr_localisation README](https://github.com/kalhansb/hmr_localisation#run)
   (vcs import + patch + colcon build; the 0.5 m localization map
   `gt_map/gt_map_us050.pcd` is committed).
2. **Sensors**: Ouster driver publishing `/ouster/points` with **RELIABLE QoS**
   (the localizer subscribes RELIABLE; a best-effort publisher never matches) and an
   IMU on `/imu/data`.
3. **Extrinsics**: the static `base_link → {os_lidar, imu}` transforms baked into
   hmr_localisation's
   [run_localization_live.sh](https://github.com/kalhansb/hmr_localisation/blob/main/scripts/run_localization_live.sh)
   are the map-test robot's — re-measure for a different platform.
4. **Network**: all robots on one LAN, same `ROS_DOMAIN_ID`. Both compose files pin
   DDS discovery to loopback, so every `docker compose exec` below that must talk
   across machines needs `-e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`.

## The one rule that makes fusion work

The bin stream's `header.frame_id` is the mapper's `integration_frame`, and the
merger keys its per-source grids by that frame_id. **Each robot must therefore map
in a unique frame** (`r1_map`, `r2_map`, …) bridged to `map` by an identity static
TF. If two robots both integrate directly in `map`, their streams collapse into one
source grid inside every merger and overwrite each other wherever their maps
overlap. Each robot's **mapper** resolves that `map ← rK_map` bridge from its own
local `/tf_static` at publish time and stamps the resulting pose into every delta
(`ScovoxMapBinary.map_from_source`). Mergers read that pose straight from the message
and run **no TF listener at all**, so no robot's localization TF ever has to reach
another robot's merger; each merger pins the **first** pose it sees per source (the
static-bridge assumption — c-SLAM re-alignment is out of scope).

Also: **start the merger before the mapper.** The bin publish is subscriber-gated —
deltas are drained (discarded) while nobody listens, and the mapper only re-sends a
full snapshot when its subscriber count rises, so a merger that starts late still
converges, but a merger that never starts means nothing is kept.

## Single robot

One robot is just a fleet of one: the merger fuses its own stream and its `~/scovox`
output is the (trivially global) planner map. Three terminals / background jobs:

```bash
# ── 1. Localization tree (hmr_localisation container) ─────────────────────────
cd ws/src/hmr_localisation
docker compose exec -e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET ros \
  bash /ws/scripts/run_localization_live.sh
# wait for "active. TF tree: map -> odom -> base_link -> {os_lidar, imu}"

# ── 2. Mapping (scovox container; run each in its own exec) ───────────────────
cd ../scovox
E="docker compose exec -e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET scovox"
S="source /opt/ros/jazzy/setup.bash && source /scovox/install/setup.bash"

# identity bridge map -> r1_map (see the rule above)
$E bash -lc "$S && ros2 run tf2_ros static_transform_publisher \
  --frame-id map --child-frame-id r1_map"

# merger FIRST (input_topics in dscovox_params.yaml lists the fleet's bin topics;
# for a single robot trim it to just /robot1/... or leave it — a missing peer
# topic simply never delivers)
$E bash -lc "$S && ros2 run scovox_mapping dscovox_mapping_node --ros-args \
  -r __ns:=/robot1 \
  --params-file /scovox/src/scovox_mapping/config/dscovox_params.yaml"

# mapper: shared base config + share overlay + the one per-robot param
$E bash -lc "$S && ros2 run scovox_mapping scovox_mapping_node --ros-args \
  -r __ns:=/robot1 -r __node:=scovox_node \
  --params-file /scovox/config/scovox_fused_lidar_rgbd.yaml \
  --params-file /scovox/config/scovox_robot_share.yaml \
  -p integration_frame:=r1_map"
```

The RGB-D/semantic stream (`/scovox/depth/*`, `/scovox/segmentation/colored` from
the seg container) is optional — without it the mapper still builds full LiDAR
occupancy from `/ouster/points`; semantics just stay empty.

**Verify** (any scovox exec): the merger logs
`dscovox_diag: sources=1 src_voxels=… fused_voxels=…` about every 5 s, and
`ros2 topic hz /robot1/scovox_node/scovox_bin` shows the share cadence
(~2 Hz with the overlay defaults). The fused map for the planner is
`/robot1/dscovox_node/scovox`; the viewable cloud is `/robot1/dscovox_node/pointcloud`.

## Two robots

Run the **same three pieces on each robot**, changing only the robot index:

| per-robot value       | robot 1     | robot 2     |
|-----------------------|-------------|-------------|
| namespace (`__ns`)    | `/robot1`   | `/robot2`   |
| `integration_frame`   | `r1_map`    | `r2_map`    |
| static TF child frame | `r1_map`    | `r2_map`    |

Everything else is identical and comes from the committed param files:

- [scovox_fused_lidar_rgbd.yaml](../config/scovox_fused_lidar_rgbd.yaml) —
  the fused mapping base config (LiDAR owns occupancy, RGB-D semantics-only).
- [scovox_robot_share.yaml](../config/scovox_robot_share.yaml) —
  real-robot share overlay: `mode: rolling`, wall clock, change gate, 2 Hz
  coalescing, z-band [−0.5, 2.0].
- [dscovox_params.yaml](../src/scovox_mapping/config/dscovox_params.yaml) —
  merger config. `input_topics` already lists
  `/robot1/scovox_node/scovox_bin` + `/robot2/scovox_node/scovox_bin`; **extend this
  list on every robot when the fleet grows** (it is fleet-wide, identical on all
  robots — robot K's own topic included, its local stream is just another source).

Order on each robot: localizer → static TF → merger → mapper. Robots do not need to
start simultaneously; a late merger receives full snapshots (subscriber-count
transition), and a late mapper simply starts contributing when it comes up.

**Verify the exchange** (on *either* robot):

```bash
$E bash -lc "$S && ros2 topic hz /robot1/scovox_node/scovox_bin /robot2/scovox_node/scovox_bin"
```

Both streams visible on both machines proves DDS discovery. Then each merger's
`dscovox_diag` line must reach `sources=2`, and the two robots' `fused_voxels`
should converge toward each other (in the bag validation run they matched to 1.1 %).
Spot-check content that could only have crossed the wire with the region service —
query robot 1's merger for a region only robot 2 has visited:

```bash
$E bash -lc "$S && ros2 service call /robot1/dscovox_node/get_region \
  scovox_msgs/srv/GetRegion '{min_corner: {x: <x0>, y: <y0>, z: -0.5}, \
                              max_corner: {x: <x1>, y: <y1>, z: 2.0}}'"
```

**Adding robot 3+**: add its bin topic to `input_topics` in `dscovox_params.yaml`
(all robots), give it `/robot3` + `r3_map`, done.

## Bandwidth

Per-robot wire cost on the map-test-2 stream, 0.1 m voxels, 14 semantic classes,
top-2 evidence: legacy per-scan publish 32.8 Mbps → change gate 20.7 Mbps → gate +
2 Hz coalescing + z-band **4.9 Mbps**. The knobs live in
[scovox_robot_share.yaml](../config/scovox_robot_share.yaml)
(`share_change_gate`, `share_rate_hz`, `share_roi_z_min/max`). Keep the z-band a
**superset** of the planner band (`roi_min_z`/`roi_max_z` in explo_planner's
`exploration_params.yaml`) and in sync with the merger's receive-side clip — the
three files cross-reference each other. Full measurements:
[map_share_bandwidth_experiment.md](https://github.com/kalhansb/hmr_explo/blob/main/ws/src/map_share_bandwidth_experiment.md)
in the hmr_explo workspace.

## Validating without robots

[run_mapshare_experiment.sh](https://github.com/kalhansb/hmr_explo/blob/main/ws/src/run_mapshare_experiment.sh)
in the hmr_explo workspace replays the recorded bag through this exact topology
(N mappers + N mergers, one sensor stream standing in for every robot) and proves
cross-robot fusion: each simulated robot shares only a disjoint z-slice, so any
voxel a robot's merger holds inside a *peer's* slice can only have arrived over
the wire.

```bash
cd ws/src && ./run_mapshare_experiment.sh 120        # ends with FUSION VERIFY: PASS/FAIL
SPLIT_BAND=0 ./run_mapshare_experiment.sh 120        # bandwidth-measurement arm
```

Reference result (2026-07-02, 120 s, 2 robots): both mergers `sources=2`,
fused totals 1,034,882 / 1,023,277 voxels (0.989 symmetry), every cross-robot probe
slab populated, fleet wire 5.0 Mbps — `FUSION VERIFY: PASS`.

## Troubleshooting

- **Merger stuck at `sources=0`** — mapper started before the merger subscribed
  (deltas drained) and no snapshot re-trigger since: restart the mapper, or check
  the bin topics are visible across machines (`ros2 topic list` on the peer; if
  missing, DDS discovery — `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, same
  `ROS_DOMAIN_ID`, same subnet).
- **Mapper logs "no exact-stamp TF … DROPPING scan"** — the localizer is not
  publishing `map → odom` yet (or died). The mapper deliberately drops scans rather
  than integrate at a stale fallback pose. Check the NDT log: if it sits at
  "Configuring end" without "Activating end", the lifecycle handoff was missed —
  `ros2 lifecycle set /lidar_localization activate`.
- **Localizer gets no scans** — the Ouster driver is publishing BEST_EFFORT;
  switch it to RELIABLE.
- **Both robots' contributions flicker/overwrite** — both mappers share one
  `integration_frame`; give each robot its own frame + identity static TF
  (see [the rule](#the-one-rule-that-makes-fusion-work)).
- **Merger logs `No TF for 'rK_map'`** — that robot's identity static TF publisher
  isn't running (or isn't reaching this machine); the merger drops frames until the
  TF appears, then caches it.
- **Merger logs `prior mismatch … dropping frame`** — mappers configured with
  different `num_classes`/`dirichlet_prior`; all robots must share one semantic
  configuration (they do if everyone uses the committed param files).
