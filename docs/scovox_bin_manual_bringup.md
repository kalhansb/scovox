# Manual bring-up: finding why `scovox_bin` is missing

A step-by-step runbook for the symptom **"my mapper isn't publishing
`.../scovox_node/scovox_bin`"**. You bring the pipeline up **one node at a
time** and verify the delta stream at each stage, so the missing piece is
obvious instead of buried in a full launch.

- Just want a known-good reference to diff against? Jump to
  [§1 Fast path](#1-fast-path-run-the-known-good-reference).
- Driving it from a recorded bag? See
  [publish_scovox_bin_from_bag.md](publish_scovox_bin_from_bag.md).
- Real multi-robot fleet? See
  [distributed_mapping_lidar.md](distributed_mapping_lidar.md).

---

## The three gates (memorise these)

`~/scovox_bin` (full name `/<ns>/<node>/scovox_bin`) only appears **and carries
data** when all three hold. Almost every "no scovox_bin" report is one of them:

| # | Gate | Symptom when it fails | Fix |
|---|------|-----------------------|-----|
| 1 | Publisher exists **only in `mode: rolling`** ([scovox_node.cpp:610](../src/scovox_mapping/src/scovox_node.cpp#L610)). The shipped LiDAR base configs set `mode: persistent` ([scovox_lidar_geometric.yaml:62](../config/scovox_lidar_geometric.yaml#L62)). | topic **absent** from `ros2 topic list` | `-p mode:=rolling` (or the share overlay / `scovox_bin_min.yaml`) |
| 2 | Publish is **subscriber-gated** — with 0 subscribers the node drains and emits nothing ([scovox_node.cpp:1600-1608](../src/scovox_mapping/src/scovox_node.cpp#L1600-L1608)). | topic **present** but `topic hz` **silent** | start a subscriber (the merger, or `topic echo`) — a fresh sub triggers a full snapshot |
| 3 | Needs a **TF at the scan's exact stamp**; a miss drops the scan, so the map (and the stream) stays empty. `tf_require_exact:true` ([scovox_lidar_geometric.yaml:79](../config/scovox_lidar_geometric.yaml#L79)). | `DROPPING scan` logs; hz trickles/empty even with a sub | supply `map→odom→base_link→sensor` TF; do **not** relax exactness |

Also non-obvious but common:

- **Namespace / node name.** The topic is `~/scovox_bin` = `/<ns>/<node>/scovox_bin`.
  The merger's default `input_topics` is `/robot1/scovox_node/scovox_bin`
  ([dscovox_params.yaml:16](../src/scovox_mapping/config/dscovox_params.yaml#L16)).
  If you don't pass `-r __ns:=/robot1 -r __node:=scovox_node`, the mapper
  advertises `/scovox_node/scovox_bin` (root ns) and the merger never matches it.
- **RGB-D vs LiDAR path.** An **empty** `input_pointcloud_topic` puts the node in
  RGB-D mode — it never subscribes to your cloud, so the map stays empty (gate 3).

---

## 1. Fast path: run the known-good reference

All commands run **inside the scovox container**
(`docker compose -f scovox/compose.yaml exec scovox bash`), with the workspace
sourced (`source /opt/ros/jazzy/setup.bash; source /scovox/install/setup.bash`).

```bash
colcon build --packages-select scovox_mapping   # pick up the new launch/config
source install/setup.bash
ros2 launch scovox_mapping dscovox_single_robot.launch.py \
    robot:=robot1 cloud_topic:=/ouster/points base_frame:=os_lidar
```

This starts a rolling mapper **and** a merger (subscriber gate open) with the
topic pinned to `/robot1/scovox_node/scovox_bin`. In another shell, with your
sensor + TF flowing (bag playing, or a live robot):

```bash
ros2 topic list | grep scovox_bin              # gate 1: topic exists?
ros2 topic hz   /robot1/scovox_node/scovox_bin # gate 2+3: data flowing?
```

- **Flows here but not in your setup** → your config differs. 90% of the time
  it's `mode` (not rolling) or the namespace/node-name. Diff your params against
  [scovox_bin_min.yaml](../src/scovox_mapping/config/scovox_bin_min.yaml).
- **Silent even here** → the sensor stream or TF is missing. Do the manual walk
  below — it isolates which.

---

## 2. Manual walk — one node at a time

Bring the graph up bottom-up. Each step has a check that must pass before the
next. Use a separate container shell per long-running node (or `ros2 run ... &`).

### Step 0 — environment

```bash
source /opt/ros/jazzy/setup.bash
source /scovox/install/setup.bash
ros2 doctor --report | sed -n '/NETWORK/,/^$/p'   # sanity: RMW + domain
echo "domain=$ROS_DOMAIN_ID"                       # every node must share it
```

All nodes must share one `ROS_DOMAIN_ID` and reach each other (host networking
in these compose files handles that on one machine).

### Step 1 — is the sensor stream actually there?

```bash
ros2 topic hz /ouster/points        # <-- your cloud_topic; must tick
ros2 topic echo /ouster/points --field header.frame_id --once   # e.g. os_lidar
```

No cloud ⇒ nothing to map. Start your bag / driver first. Note the cloud's
`frame_id` — that (or a frame rigidly linked to it) is your `base_frame`.

### Step 2 — is the TF tree complete? (gate 3)

The mapper needs `integration_frame → base_frame` resolvable at each scan stamp.
For the distributed setup that chain is `map → odom → base_link → <sensor>`.

```bash
# One-shot: can TF resolve map -> your base_frame right now?
ros2 run tf2_ros tf2_echo map os_lidar

# Full picture: who publishes what leg
ros2 run tf2_tools view_frames && ls -l frames.pdf   # or: ros2 topic echo /tf_static --once
```

- `tf2_echo` **succeeds** → gate 3 will pass.
- `"Could not transform"` / `"frame does not exist"` → a leg is missing. In the
  distributed pipeline: `map→odom` and `odom→base_link` come from the localizer
  (NDT + EKF), `base_link→sensor` from the bag/robot `/tf_static`. Start the
  localizer (see [distributed_mapping_lidar.md](distributed_mapping_lidar.md))
  and re-check. **Do not** work around it by relaxing `tf_require_exact`.

### Step 3 — start the mapper in rolling mode → does the topic appear? (gate 1)

```bash
ros2 run scovox_mapping scovox_mapping_node --ros-args \
    -r __ns:=/robot1 -r __node:=scovox_node \
    --params-file /scovox/src/scovox_mapping/config/scovox_bin_min.yaml \
    -p input_pointcloud_topic:=/ouster/points \
    -p base_frame:=os_lidar \
    -p use_sim_time:=true            # drop for a live robot
```

Watch the startup line — it prints the mode:

```text
SCovox ready res=0.100 mode=rolling frame=map share_tsdf=0 ...
```

Then in another shell:

```bash
ros2 topic list | grep scovox_bin
# EXPECT: /robot1/scovox_node/scovox_bin
```

- **Topic absent / `mode=persistent`** → GATE 1. Your params file forced
  persistent, or you loaded a base config without the rolling overlay. Add
  `-p mode:=rolling`.
- **Topic is `/scovox_node/scovox_bin`** (no `/robot1`) → you dropped the
  `-r __ns:=/robot1` remap; the merger won't match it.

### Step 4 — open the subscriber gate → does data flow? (gate 2)

With the mapper from Step 3 still running:

```bash
# Subscribing is what triggers publication; a fresh sub gets a full snapshot.
ros2 topic hz /robot1/scovox_node/scovox_bin
```

- **`average rate: …` ticks** → gates 1+2+3 all pass. You're done; go to Step 5
  only if you need the merger too.
- **`no new messages`** → topic exists (gate 1 OK) but nothing is emitted:
  - Check the mapper log for `DROPPING scan` / `no exact-stamp TF` → **gate 3**
    (fix TF, Step 2). The map is empty, so even a snapshot is empty.
  - `deskew on but IMU/extrinsic not ready … integrating raw scan` is **harmless**
    ([scovox_node.cpp:1161](../src/scovox_mapping/src/scovox_node.cpp#L1161)) —
    the scan is still integrated; add `-p deskew_mode:=off` to silence.

### Step 5 — attach the merger (the real consumer)

```bash
ros2 run scovox_mapping dscovox_mapping_node --ros-args \
    -r __ns:=/robot1 -r __node:=dscovox_node \
    --params-file /scovox/src/scovox_mapping/config/dscovox_params.yaml \
    -p input_topics:="['/robot1/scovox_node/scovox_bin']" \
    -p use_sim_time:=true
```

Its log announces the receiver's compile-time `K_TOP` — **every sender must
match it** or frames fail to deserialize (a build-skew trap). Watch the merger's
own console (it runs `output=screen`): every 5 s it prints a throttled diag line
([dscovox_node.cpp:263](../src/scovox_mapping/src/dscovox_node.cpp#L263)) — this
is a **log line, not a topic**:

```text
dscovox_diag: sources=1 src_voxels=... fused_voxels=...
```

`sources=1` and `fused_voxels>0` means the stream arrived and fused. `sources=0`
means no bin frame reached it — go back to Step 4 (nothing on the wire) or check
the topic name matches its `input_topics`.

### Step 6 — second robot = fusion

Repeat Steps 3+5 as `robot2` with a **unique** `integration_frame:=r2_map`
(the merger keys sources by the bin's `header.frame_id`; two robots sharing a
frame collapse to one source). Publish the identity `map → r2_map` static TF,
and list both streams in each merger's `input_topics`. The full orchestration is
[../run_dist_lidar_experiment.sh](../../run_dist_lidar_experiment.sh) — read its
header for the exact per-robot args.

---

## Quick diagnostic tree

```text
scovox_bin not in `ros2 topic list`
        └─► GATE 1: mode != rolling → -p mode:=rolling
            (or wrong ns/node → -r __ns:=/robotK -r __node:=scovox_node)

scovox_bin present, `topic hz` silent
        ├─► no subscriber?      GATE 2: start merger / `topic echo`
        ├─► `DROPPING scan`?    GATE 3: fix map→odom→base_link→sensor TF (Step 2)
        └─► `input_pointcloud_topic` empty? RGB-D mode → set your cloud topic

`topic hz` ticks but merger sees sources=0 / fused_voxels=0
        ├─► integration_frame not unique per robot → give robotK r{K}_map
        ├─► missing identity map→rK_map static TF
        └─► K_TOP build-skew (sender vs receiver) → rebuild both from one tree
```

## See also

- [scovox_bin_min.yaml](../src/scovox_mapping/config/scovox_bin_min.yaml) — minimal params that guarantee the stream
- [dscovox_single_robot.launch.py](../src/scovox_mapping/launch/dscovox_single_robot.launch.py) — one-shot single-robot dscovox map (rolling mapper + merger)
- [publish_scovox_bin_from_bag.md](publish_scovox_bin_from_bag.md) — bag-driven variant
- [distributed_mapping_lidar.md](distributed_mapping_lidar.md) — full multi-robot runbook
