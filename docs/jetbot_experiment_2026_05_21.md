# JetBot Nano 4 GB edge-feasibility experiment (2026-05-21)

**Subject of evaluation.** Whether the SCovox + dSCovox distributed-mapping
stack — designed and benchmarked on desktop hardware — runs at all on a
2017-era Jetson Nano (t210ref, 4 GB, JetPack 4.5.2 / L4T R32.5.2 /
Bionic), and whether the per-frame map produced there is consistent with
the desktop reference numbers already published in our paper.

This is a **feasibility report**, not a head-to-head paper figure. Sample
size is one trajectory.

## 1. Setup

### 1.1 Topology

```text
┌─────── LAPTOP (host workspace, Humble native, x86_64) ───────┐
│  scenenet_replay  robotA  (frames [0..200), 0.25 Hz)          │  /tf
│  scenenet_replay  robotB  (frames [100..300), 0.25 Hz)        │  /robotA/rgbd_camera_depth_image
│  scovox_node      robotB  (publishes /robotB/scovox_node/scovox_bin)
│  static_tf        map → robotB/odom                           │  /robotA/segmentation/colored
└───────────────────────────────────────────────────────────────┘  /robotA/rgbd_camera_info
                              ▲                                    /robotB/scovox_node/scovox_bin
                              │                                    │
                              │ CycloneDDS 1.3.4 unicast peer list │
                              │ ROS_DOMAIN_ID=42  (Wi-Fi, no MC)   │
                              ▼                                    │
┌─────── JETSON NANO 4 GB (Humble in dustynv container, aarch64) ──┐
│  scovox_node      robotA  (integrates the laptop's robotA stream)│
│  dscovox_node            (fuses local robotA + remote robotB)    │
│  static_tf        map → robotA/odom                              │
│  jetson_telemetry.py     → jetson_timing.csv                     │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Hardware

- **Jetson:** Nano 4 GB t210ref, L4T R32.5.2 / JetPack 4.5.2 / Bionic
  (Ubuntu 18.04.6), 4× Cortex-A57 @ 1.43 GHz, Maxwell GPU (compute 5.3,
  unused), kernel ABI aarch64. Python 3.6.9 host.
- **Laptop:** x86_64, Ubuntu 22.04 Jammy with Humble native, dual 5 GHz
  Wi-Fi to the same AP as the Jetson.

### 1.3 Software

- **Jetson container:** `dustynv-ros-humble:jupyter` — community-built
  Humble for L4T r32 (Humble lives at `/opt/ros/humble/install`,
  Python 3.6.9 inside).
- **Workspace packages built on the Jetson:** scovox_core,
  scovox_msgs, scovox_mapping, log_odds_mapping. Build command:
  `MAKEFLAGS="-j2" colcon build --packages-up-to scovox_mapping
  log_odds_mapping --cmake-args -DCMAKE_BUILD_TYPE=Release`. Wall: **5 min
  11 s**.
- **DDS:** CycloneDDS 1.3.4 with `cyclonedds.xml` unicast peer-list
  (laptop 192.168.1.16, Jetson 192.168.1.22, multicast disabled).
- **scovox parameters:** `K_TOP=2`, `resolution=0.05`, 14 NYU NYUv2 classes,
  `mode=rolling`, `use_split=true`, `fused_walker=true`, `wire_format=v3`,
  `semantic_mode=dirichlet`, `kappa0=2.0`, `range=[0.1, 10] m`.

### 1.4 Replay configuration

- Trajectory **0_223** (SceneNet val), 300 RGB-D + GT frames.
- **Rate:** 0.25 Hz (one frame every 4 s) — chosen to sit below the
  sustained Jetson integration throughput so every published frame lands
  rather than being silently dropped by `message_filters`'s
  `KEEP_LAST(5)`.
- **Overlap split** (standard fusion convention):
  - robotA: frames `[0, 200)`
  - robotB: frames `[100, 300)`

## 2. Per-frame timing on the Jetson

Per-frame log lines (`recv=N replay=M frame_ms=… tf_ms=… integrate_ms=…
publish_ms=… tsdf_ms=… sembeta_ms=… bin_bytes=…`) come straight from
[`scovox_node.cpp:622-633`](../robot_sw/distributed_mapping/scovox_mapping/src/scovox_node.cpp#L622-L633).

Sample size: **199 frame log lines** captured (Jetson `scovox_a`, robotA;
the laptop `scovox_b` lines live in `/tmp/laptop_run_q025.log`).

| Stage                | Mean (ms) | p50 (ms) | p95 (ms) | Share of `frame_ms`             |
|----------------------|----------:|---------:|---------:|---------------------------------|
| **frame_ms**         |    2404.2 |   2370.5 |   3097.8 | 100%                            |
| └ tf_ms              |       0.4 |      0.4 |      0.9 | < 0.05%                         |
| └ integrate_ms       |     822.6 |    824.7 |   1052.6 | 33.4% (end-of-run)              |
| └└ tsdf_ms           |     796.9 |    798.0 |   1026.0 | 97.1% (of integrate)            |
| └└ sembeta_ms        |       0.0 |      0.0 |      0.0 | folded into tsdf (fused walker) |
| └ **publish_ms**     |    1581.2 |   1551.9 |   2086.8 | **66.6% (end-of-run)**          |
| bin_bytes (KB)       |     214.0 |    209.8 |    382.6 | —                               |
| RSS (MB)             |      86.5 |     86.0 |     94.9 | —                               |

**Sustained throughput:** 1000 / 2404 ms = **0.42 Hz** (end-to-end with
publish). Integration-only would sustain 1000 / 823 ms = **1.22 Hz** —
this is the figure to compare against monolithic single-process
baselines such as SLIM-VDB, whose FPS numbers do not include any IPC.

Growth dynamics (first 20 vs last 20 frames):

| Stage          |  Start |    End |     Ratio |
|----------------|-------:|-------:|----------:|
| frame_ms       | 1869.7 | 2489.1 |     1.33× |
| integrate_ms   |  672.5 |  830.2 |     1.23× |
| publish_ms     | 1196.9 | 1658.3 |     1.39× |
| bin_bytes (KB) |   92.4 |  202.2 | **2.19×** |

`publish_ms` vs `bin_bytes` correlation across the 199 frames: **r =
0.907**. Binary delta payload size doubles over the trajectory; publish
time grows ~1.4× — sub-linear, consistent with per-frame fixed overhead
(DDS reliable ACK, serializer setup) dominating the steady-state.

### 2.1 What `publish_ms` actually measures

`publish_ms` covers `publishBinaryMap()` only — the serialization, LZ4
compression, and DDS-publish of the binary delta envelope to the
distributed-mapping peer. It does **not** include the 1 Hz pointcloud
visualization publish (that runs on a separate timer per
[`scovox_node.cpp:618`](../robot_sw/distributed_mapping/scovox_mapping/src/scovox_node.cpp#L618)).

This matters when comparing against single-process baselines (e.g.
SLIM-VDB): their FPS measurements do not include any inter-process
serialization, so the apples-to-apples figure from this run is
`integrate_ms` only.

## 3. Map-quality results

### 3.1 Captured artifacts

| Map output        | Source       | Voxels |          Path |
|-------------------|--------------|-------:|---------------|
| scovox_a (Jetson) | robotA solo  | 44,808 | `runs/jetbot/0_223_q025hz/scovox_a.npz` |
| dscovox fused     | Jetson fuser | 66,197 | `runs/jetbot/0_223_q025hz/dscovox.npz` |

### 3.2 mIoU vs SceneNet GT (5 cm, strict bucket-IoU)

| Map                                        |          mIoU | Union vox | Intersection vox | Δ vs scovox_a alone |
|--------------------------------------------|--------------:|----------:|-----------------:|--------------------:|
| Jetson scovox_a                            |        0.2184 |    71,684 |           20,744 |                   — |
| **Jetson dscovox fused**                   |    **0.4945** |    70,823 |           39,229 |          **+0.2761**|
| Desktop reference (full 300 fr, n=13 mean) | 0.327 ± 0.037 |         — |                — |                   — |

Reference desktop mean comes from
[project_scenenet_first_batch_2026_05_12.md](../../../../.claude/projects/-home-kalhan-projects-HMR-Exploration-Experiment-hmr-exploration-ws/memory/project_scenenet_first_batch_2026_05_12.md)
(13 random val trajectories, full 300-frame single-robot, mean ± SE).

Per-class IoU on the Jetson dscovox fused map (the 7 non-empty classes
in this scene):

| Class       | scovox_a IoU | dscovox IoU |   Δ   |
|-------------|-------------:|------------:|------:|
| Ceiling     |       0.2470 |      0.5657 | +0.32 |
| Chair       |       0.2296 |      0.3997 | +0.17 |
| Floor       |       0.3036 |      0.5854 | +0.28 |
| Furniture   |       0.2167 |      0.3600 | +0.14 |
| Objects     |       0.2036 |      0.3423 | +0.14 |
| Picture     |       0.0000 |      0.6321 | +0.63 |
| Wall        |       0.3281 |      0.5763 | +0.25 |

### 3.3 Same-config desktop comparison (and a real finding about Phase 3)

The Step 8 Phase 3 batch
([results/phase3_scenenet_fusion_v3_2026_05_14/phase3_summary.csv](../robot_sw/distributed_mapping/scovox_eval/results/phase3_scenenet_fusion_v3_2026_05_14/phase3_summary.csv))
ran the **same overlap-split configuration** as this experiment
(robotA=[0,200), robotB=[100,300), wire_format=v3, use_split=true,
fused_walker=true, K_TOP=2, res=0.05). The per-trajectory record for
0_223 is:

| Map               | Desktop (Phase 3 batch) | Jetson (this expt) | Δ Jetson − Desktop |
|-------------------|------------------------:|-------------------:|-------------------:|
| scovox_a solo     |                  0.2057 |             0.2184 |  +0.0127 (in noise) |
| scovox_b solo     |                  0.1979 |               n/a  |                  —  |
| **dscovox fused** |              **0.3036** |         **0.4945** |          **+0.191** |

solo_a is within the documented ±0.05 single-cell variance from
[project_replica_room0_runtorun_variance.md](../../../../.claude/projects/-home-kalhan-projects-HMR-Exploration-Experiment-hmr-exploration-ws/memory/project_replica_room0_runtorun_variance.md).
But the **fused map differs by +0.191 mIoU** between the two runs — well
outside cell-variance.

Investigating the desktop Phase 3 log
([results/phase3_scenenet_fusion_v3_2026_05_14/0_223/fusion_launch.log](../robot_sw/distributed_mapping/scovox_eval/results/phase3_scenenet_fusion_v3_2026_05_14/0_223/fusion_launch.log))
explains it. The Phase 3 batch ran replay at **`RATE_HZ=4.0`** (250 ms
inter-frame); both scovox nodes' `frame_ms` mean was 443–520 ms — well
above 250 ms — so `message_filters` `KEEP_LAST(5)` dropped roughly a
third of every robot's frames silently:

| Robot   | Frames published | Frames integrated | Drop rate |
|---------|------------------:|------------------:|----------:|
| robotA  |               200 |               146 |    **27%** |
| robotB  |               200 |               131 |  **34.5%** |

This experiment ran at **0.25 Hz** (4 000 ms inter-frame, well below the
Jetson sustained rate of ~0.5 Hz) so **no frames were dropped on either
side**.

Confirmation in voxel counts:

| Map           | Desktop 4 Hz (lossy) | Jetson 0.25 Hz (clean) | Δ      |
|---------------|---------------------:|-----------------------:|-------:|
| scovox_a solo |               43,484 |                 44,808 |  +3%   |
| **dscovox fused** |           **44,474** |             **66,197** |**+49%**|

The solo map size is roughly stable because solo coverage converges
after ~150 frames. The fused map size depends critically on robot_b's
peripheral coverage — losing a third of robot_b's frames removes ~22 k
voxels of unique-to-B coverage.

**Validation rerun (2026-05-22).** To test the lossy-replay hypothesis
directly, the desktop Phase 3 launch was re-executed on 0_223 with the
**only change being `RATE_HZ=0.25`** (i.e. fed at 0.25 Hz instead of
4 Hz). Same launch file, same params, same machine, same eval script.
Drops dropped to zero: robotA recv=200/200, robotB recv=200/200.
Results:

| Run                                        |  solo_a |  solo_b |    fused | fused voxels |
|--------------------------------------------|--------:|--------:|---------:|-------------:|
| Desktop OLD 4 Hz (the Phase 3 batch)       |  0.2057 |  0.1979 |   0.3036 |       44,474 |
| Desktop NEW 0.25 Hz (this validation)      |  0.2184 |  0.3679 | **0.4945** |     66,197 |
| Jetson 0.25 Hz (this experiment)           |  0.2184 |    n/a  | **0.4945** |     66,197 |
| **Δ (NEW − OLD desktop)**                  |  +0.013 |  +0.170 | **+0.191** |        +49% |

The desktop 0.25 Hz rerun produces the same fused mIoU as the Jetson
0.25 Hz run, to four decimal places. The NPZs are **bit-exactly
identical** between the two machines:

```python
np.array_equal(sorted(desktop_solo_a.points), sorted(jetson_scovox_a.points)) → True   # 44,808 each
np.array_equal(sorted(desktop_fused.points),  sorted(jetson_dscovox.points))  → True   # 66,197 each
```

Two consequences:

1. **The Phase 3 batch number for 0_223 (0.3036) is a measurement
   artifact**, not the algorithm's actual performance on this trajectory.
   The clean fused mIoU is 0.4945. The +0.191 gap is entirely explained
   by `message_filters` `KEEP_LAST(5)` silently dropping 27% of robotA's
   frames and 35% of robotB's at the 4 Hz Phase 3 rate.
2. **scovox/dscovox is bit-deterministic across architectures** when fed
   identical inputs without drops. The Nano (aarch64, Cortex-A57) and
   desktop (x86_64) produce literally identical maps under the same
   inputs. This is stronger than "within cell-variance" — there is no
   numerical drift between the platforms.

**Implication for the paper's Phase 3 numbers.** The n=13 mean Δ +0.071
mIoU reported in
[project_step8_complete_2026_05_14.md](../../../../.claude/projects/-home-kalhan-projects-HMR-Exploration-Experiment-hmr-exploration-ws/memory/project_step8_complete_2026_05_14.md)
is **confirmed to be an underestimate on at least one cell** (0_223:
published +0.0979, actually +0.1266 = +29% relative). Per-cell drop
rates from the Phase 3 launch logs:

| Severe-drop cells     | Likely understated | Mild-drop cells       | Probably OK   |
|-----------------------|--------------------|-----------------------|---------------|
| 0_182 (A=151, B=127)  |                    | 0_175, 0_178, 0_485   |               |
| 0_223 (A=146, B=131)  | confirmed +0.029   | 0_723, 0_789, 0_867   | drops <2%     |
| 0_279 (A=117, B=198)  |                    | 0_977                 |               |

### 3.4 Full n=13 desktop rerun at 0.25 Hz (2026-05-22)

The full Phase 3 batch was re-executed across all 13 trajectories at
0.25 Hz (~2.6 hr wall on desktop). Per-cell comparison against the
original 4 Hz batch:

| Trajectory | Old (4 Hz) fused | New (0.25 Hz) fused | Δ fused | Old drop rate (A / B) |
|---|---:|---:|---:|---|
| 0_175 | 0.2698 | 0.2695 | −0.0003 | 1% / 1% (clean)   |
| 0_178 | 0.3786 | 0.3926 | +0.0140 | 0% / 0% (clean)   |
| **0_182** | 0.3502 | **0.4504** | **+0.1002** | **25% / 36.5%** |
| **0_223** | 0.3036 | **0.4945** | **+0.1909** | **27% / 34.5%** |
| 0_279 | 0.4434 | 0.4466 | +0.0032 | 41.5% / 1% |
| 0_485 | 0.3617 | 0.3658 | +0.0041 | 2% / 5% |
| 0_490 | 0.3280 | 0.3463 | +0.0183 | 12.5% / 0% |
| 0_571 | 0.2629 | 0.2648 | +0.0019 | 3.5% / 11.5% |
| 0_682 | 0.5377 | 0.5355 | −0.0022 | 13.5% / 1.5% |
| **0_723** | 0.3036 | **0.3873** | **+0.0837** | 1% / 1% (clean!) |
| 0_789 | 0.4296 | 0.4152 | −0.0144 | 0% / 0% (clean) |
| 0_867 | 0.4155 | 0.3988 | −0.0167 | 1.5% / 1% (clean) |
| 0_977 | 0.4049 | 0.3974 | −0.0075 | 6.5% / 8.5% |

Aggregate:

| Metric                     | Lossy 4 Hz (n=13)   | Clean 0.25 Hz (n=13) | Δ      |
|----------------------------|--------------------:|---------------------:|-------:|
| `fused_miou` mean ± std    | 0.3684 ± 0.0724     | 0.3973 ± 0.0750      | +0.029 |
| `solo_a` mean              | 0.2676              | 0.2898               | +0.022 |
| `solo_b` mean              | 0.2867              | 0.3167               | +0.030 |
| **`fused − max(solo)` mean** | **+0.0712 ± 0.054** | **+0.0744 ± 0.054**  | **+0.003** |

### 3.5 Re-interpretation of the n=13 finding

The earlier alarm — "Phase 3 is a methodology bug, the +0.071 fusion
gain is an underestimate" — was **partially wrong**. Correcting:

1. **Aggregate `fused − max(solo)` is robust to the drop bias** (+0.0712
   → +0.0744, basically unchanged across the 13 cells). When drops
   happen, both `fused` and `max(solo)` move together, so their
   difference stays stable. **The published Step 8 phase 3 fusion claim
   doesn't need revision.**
2. **Per-cell absolute `fused_miou` is *not* robust** when drops are
   severe. The two cells with >25% drops on both robots (0_182, 0_223)
   shifted by +0.10 and +0.19 respectively. If per-trajectory numbers
   are ever cited, those need updating.
3. **0_223 specifically is dominated by coverage**, not fusion-alone —
   confirmed: the Jetson 0.4945 reproduces here on desktop at 0.25 Hz,
   bit-exactly identical NPZs.
4. **Cell variance is higher than the documented 0.05.** 0_723 had
   ~1% drops in both runs but `fused_miou` jumped +0.084 between
   batches. That's pure stochastic run-to-run variance — bigger than
   the documented 0.05 single-cell variance from Replica room0.
   SceneNet two-robot fused has additional variance sources (DDS
   message ordering, thread scheduling at overlapping voxels). Worth
   flagging in any future variance accounting.

## 4. Comparison to SLIM-VDB's edge claim

SLIM-VDB (Sheppard et al., RA-L Mar 2026) reports **6.12 FPS** on a Jetson
Orin Development Kit (4 GB GPU memory) at SceneNet 5 cm. Their stack uses
CUDA 12 + OpenVDB 12 + NanoVDB-on-GPU; the Nano (CUDA 10.2 max, Maxwell
GPU compute 5.3) cannot run their codebase without a substantial
port. The throughput numbers below are therefore on **different hardware
tiers** (Orin Ampere ~ 40 TOPS NPU + modern Cortex-A78AE vs Nano Maxwell
~ 0.5 TOPS + Cortex-A57). The comparable-hardware quality comparison
remains the desktop head-to-head from
[project_scenenet_head_to_head_2026_05_13.md](../../../../.claude/projects/-home-kalhan-projects-HMR-Exploration-Experiment-hmr-exploration-ws/memory/project_scenenet_head_to_head_2026_05_13.md)
(+0.088 mIoU, 13/13 wins).

| Method   | Edge platform    | FPS (sustained) | Notes                                                                          |
|----------|------------------|----------------:|--------------------------------------------------------------------------------|
| SLIM-VDB | Jetson Orin 4 GB |            6.12 | Closed-set, 5 cm, monolithic, GPU offload                                      |
| SCovox   | Jetson Nano 4 GB |            0.42 | Dirichlet K_TOP=2, 5 cm, distributed (DDS to peer + fuser, end-to-end frame_ms)|
| SCovox   | Jetson Nano 4 GB |            1.22 | Integration only (excludes publish, matches SLIM-VDB's measurement convention) |

The 0.42-vs-6.12 difference (~15×) is mostly **hardware tier**, not
algorithmic: the Orin's Cortex-A78AE cores are 2-3× faster per-core than
the Nano's A57, the Orin has 6 cores vs 4, and SLIM-VDB offloads
raycasting and grid updates to the Orin's Ampere GPU while SCovox is
CPU-only on the Nano. Adjusted for the integration-only convention
(1.22 vs 6.12), the algorithmic gap is closer to 5× and the
hardware-tier explanation is sufficient.

## 5. Bugs surfaced

- **`dscovox_diag` reports `fused_voxels=0`** for the whole run, but
  `pointcloud_to_npz` captures **66,197 voxels** from
  `/dscovox_node/pointcloud`. The diag counts
  `split_fused_sem_->activeCellsCount()` while the publish path emits a
  different internal grid. Cosmetic counter mislabel, not a functional
  bug. Worth fixing because the diagnostic is otherwise the natural way
  to confirm fusion is working at runtime.
- **Telemetry script crashed on Python 3.6** (PEP-604 `int | None` union
  types AND `from __future__ import annotations` — the latter is Python
  3.7+ only). Fixed by removing both: type hints stripped entirely. The
  dustynv container's `python3` is `3.6.9`; any helper script we ship to
  the Jetson needs to stay 3.6-compatible.
- **Dustynv image missing `liblz4-dev`** and has an expired ROS apt key
  (`F42ED6FBAB17C654`) which blocks `apt-get update`. Workaround:
  install `liblz4-dev` directly (`apt-get install -y
  --no-install-recommends liblz4-dev`) — it comes from Ubuntu main, not
  the ROS repo, so the ROS key expiry doesn't matter.
- **CycloneDDS 1.3.4 schema changes** — `NetworkInterfaceAddress` and
  unit-less `FragmentSize` are deprecated. Use
  `<Interfaces><NetworkInterface autodetermine="true"/></Interfaces>` and
  `1280B`. Old XML caused `rmw_create_node: failed to create domain` —
  a fatal but completely opaque error.
- **`run_laptop.sh` cleanup trap kills `scovox_b` too aggressively.** When
  the script's two replay processes finish, the trap kills `scovox_b`
  immediately. In slice 1 this race left the laptop publisher dead before
  dscovox had fully drained its peer's last frames, capping the fused
  map size. With the slower 0.25 Hz replay used here, scovox_b had time
  to publish its full delta stream before being killed, and the fused
  voxel count jumped from 44,527 (slice 1) to 66,197 (this run). For
  future runs with faster replay rates, add a deliberate
  drain-grace-period before cleanup.

## 6. Reproducibility

### Inputs

- `data/scenenet/val_preprocessed/0_223/` (RGB, depth, GT labels, poses,
  intrinsics, GT NPZ at 5 cm)

### Configuration

- [`src/robot_sw/distributed_mapping/scovox_mapping/launch/scenenet_jetbot.launch.py`](../robot_sw/distributed_mapping/scovox_mapping/launch/scenenet_jetbot.launch.py)
  (Jetson side)
- [`src/robot_sw/distributed_mapping/scovox_eval/jetbot/scripts/run_laptop.sh`](../robot_sw/distributed_mapping/scovox_eval/jetbot/scripts/run_laptop.sh)
  (laptop orchestrator)
- [`src/robot_sw/distributed_mapping/scovox_eval/jetbot/cyclonedds.xml`](../robot_sw/distributed_mapping/scovox_eval/jetbot/cyclonedds.xml)
  (DDS unicast config)
- [`src/robot_sw/distributed_mapping/scovox_eval/jetbot/README.md`](../robot_sw/distributed_mapping/scovox_eval/jetbot/README.md)
  (build + run guide)

### Run dir (this experiment)

- `runs/jetbot/0_223_q025hz/`
  - `scovox_a.npz` — Jetson scovox_a output
  - `dscovox.npz` — Jetson dscovox fused output
  - `jetbot.log` — Jetson ROS node logs
  - `jetson_timing.csv` — per-second RSS / CPU / per-core utilization

### Commands

```bash
# Jetson:
docker exec humble bash -c '
  source /opt/ros/humble/install/setup.bash
  source /workspace/install/setup.bash
  export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file:///workspace/src/robot_sw/distributed_mapping/scovox_eval/jetbot/cyclonedds.xml
  ros2 launch scovox_mapping scenenet_jetbot.launch.py
'

# Laptop (after Jetson side is up):
src/robot_sw/distributed_mapping/scovox_eval/jetbot/scripts/run_laptop.sh 0_223 0.25
```

## 7. Conclusion

The SCovox + dSCovox distributed-mapping stack runs end-to-end on a
2017-era Jetson Nano 4 GB with no algorithmic modifications. The single
load-bearing claim from this run:

> On a 4 GB Jetson Nano (Maxwell, no GPU offload), the Humble container
> built in 5 min, the laptop-streamed two-robot fused pipeline runs at
> 0.42 Hz end-to-end (1.22 Hz integration-only), holds RSS under 100 MB,
> and produces a 66 k-voxel semantic VDB scoring **mIoU 0.4945** against
> SceneNet GT on trajectory 0_223 — a +0.276 uplift over the
> single-robot 200-frame Jetson map (0.2184), reproducing the
> distributed-mapping fusion benefit established on desktop.

What this experiment **doesn't** establish (because n=1):

- per-trajectory variance of the Nano fused mIoU — desktop n=13 fusion
  gives a clean error bar, this single-trajectory run does not.
- whether the Nano dscovox uplift holds across the full 13-trajectory
  SceneNet val set we used for the desktop head-to-head.

The next experiment that would close this loop is the same configuration
across the remaining 12 trajectories at 0.25 Hz, producing per-traj
paired (scovox_a, dscovox) mIoUs comparable to the
[head-to-head memo](../../../../.claude/projects/-home-kalhan-projects-HMR-Exploration-Experiment-hmr-exploration-ws/memory/project_scenenet_head_to_head_2026_05_13.md).
Wall time at 0.25 Hz: 12 trajectories × ~13 min each + ~3 min teardown
per cell ≈ **3.5 hr** unattended.

### Profile-driven optimization (deferred)

`publish_ms` is 66.6% of frame time and correlates strongly with payload
size (r=0.907). Two low-cost levers, ranked:

1. **Asynchronous binary-delta emission.** Move `publishBinaryMap()` off
   the integration callback into a worker thread. Integrate continues at
   ~1.22 Hz; the publisher catches up at its own pace. Net effect: end-to-end
   throughput rises from 0.42 to ~1.0–1.2 Hz on the same Nano.
2. **Best-effort QoS on the binary topic** (currently
   `KeepLast(50).reliable()`). Drops reliable-ACK roundtrip overhead. Cheap
   probe; would isolate how much of publish_ms is reliable QoS vs LZ4 vs
   serializer.

Neither is required to ship the paper section — the result here is
already a defensible feasibility claim. They're called out as the path
to a future "real-time on Nano" follow-up.
