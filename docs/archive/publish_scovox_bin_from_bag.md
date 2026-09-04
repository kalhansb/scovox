# Publishing `scovox_bin` from a recorded bag

Replay a rosbag through `scovox_mapping_node` and get the LZ4 `ScovoxMapBinary`
delta stream (`~/scovox_bin`) on the wire — the same topic `dscovox_mapping_node`
fuses in the multi-robot pipeline ([distributed_mapping_lidar.md](distributed_mapping_lidar.md)),
but driven by a bag.

## Why it wasn't publishing — three gates

`~/scovox_bin` is guarded by three independent gates; a naive bag replay trips all
three.

1. **Publisher only exists in `mode: rolling`.** `bin_pub_` is created only when
   `mode == "rolling"` ([scovox_node.cpp:610](../src/scovox_mapping/src/scovox_node.cpp#L610)),
   but the LiDAR configs ship `mode: persistent`
   ([scovox_lidar_geometric.yaml:62](../config/scovox_lidar_geometric.yaml#L62)).
   → `-p mode:=rolling`.
2. **Publish is subscriber-gated.** `publishBinaryMap()` drains and returns while
   `get_subscription_count() == 0` ([scovox_node.cpp:1600](../src/scovox_mapping/src/scovox_node.cpp#L1600)).
   → start the subscriber (`topic hz`, or the merger) **before** playback. A
   fresh subscriber triggers a full snapshot, so late joiners still get the whole map.
3. **No TF → every scan dropped.** For each scan the node looks up the sensor pose
   in `integration_frame` at the exact stamp and drops the scan if it's missing
   ([scovox_node.cpp:1061](../src/scovox_mapping/src/scovox_node.cpp#L1061)). The
   config integrates in `map`, which **is not in these raw sensor bags** — so the
   map stays empty and there's nothing to emit. This is the real trap: supply the
   missing `map → odom → base_link` with a localizer (see Troubleshooting).

## Troubleshooting

- **No `…/scovox_bin` in `topic list`** — gate 1: add `-p mode:=rolling`.
- **Topic exists but `topic hz` is silent** — gate 2 (subscriber joined too late,
  or QoS mismatch: the bin publisher is `KeepLast(50).reliable()`) or gate 3
  (`DROPPING scan` → TF).
- **`no exact-stamp TF … DROPPING scan`** — gate 3: the localizer isn't
  publishing `map → odom → base_link`, or scovox can't see it. Do **not** lower
  `tf_require_exact` — the Time(0) fallback mis-places whole scans.
- **`deskew on but IMU/extrinsic not ready … integrating raw scan`** — harmless
  (the node integrates the raw cloud, doesn't drop it,
  [scovox_node.cpp:1161](../src/scovox_mapping/src/scovox_node.cpp#L1161)); add
  `-p deskew_mode:=off` to silence.
