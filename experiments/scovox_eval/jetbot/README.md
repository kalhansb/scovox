# JetBot SceneNet experiment

Run scovox + dscovox on a JetBot Nano 4 GB while a laptop streams sensor
input and acts as the peer robot for distributed-mapping fusion.

## Topology

```text
┌─────────── LAPTOP (host workspace, Humble native) ──────────┐
│  scenenet_replay  robot_a  (frames [0..200))                 │   /tf
│  scenenet_replay  robot_b  (frames [100..300))               │   /robotA/rgbd_camera_depth_image
│  scovox_node      robot_b  (publishes /robotB/.../scovox_bin)│ ─ /robotA/segmentation/colored
│  static_tf        map → robotB/odom                          │   /robotA/rgbd_camera_info
└─────────────────────────────────────────────────────────────┘ ─ /robotB/scovox_node/scovox_bin
                              ▲                                    │
                              │                                    │
                              │ DDS over WiFi (Cyclone + peer list)│
                              │                                    ▼
┌──────────── JETSON NANO (Humble in dustynv container) ────────────┐
│  scovox_node     robot_a  (integrates laptop's robot_a stream)    │
│  dscovox_node             (fuses local robot_a + remote robot_b)  │
│  static_tf       map → robotA/odom                                │
│  jetson_telemetry.py      → /tmp/jetson_timing.csv                │
└───────────────────────────────────────────────────────────────────┘
```

## One-time setup

### 1. Get the source on the Jetson

From the laptop:

```bash
JET=<jetson-user>@<jetson-host>
WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws

ssh $JET 'mkdir -p ~/jetbot_ws/src/robot_sw/distributed_mapping'

rsync -avz --delete \
  $WS/src/robot_sw/distributed_mapping/{scovox_core,scovox_mapping,scovox_msgs,scovox_eval,log_odds_mapping,covox_mapping,peer_bridge,odom_to_tf_ros2} \
  $JET:~/jetbot_ws/src/robot_sw/distributed_mapping/
```

### 2. Build inside the Humble container (Jetson)

```bash
# On the Jetson:
docker run --rm -it \
    --network host \
    -v $HOME/jetbot_ws:/jetbot_ws \
    -w /jetbot_ws \
    dustynv/ros:humble-desktop-l4t-r32.x bash

# Inside the container:
apt-get update && apt-get install -y liblz4-dev
source /opt/ros/humble/setup.bash
MAKEFLAGS="-j2" colcon build \
    --packages-up-to scovox_mapping log_odds_mapping scovox_eval \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
```

`-j2` keeps RAM headroom on the 4 GB Nano. Expect 15–25 min.

### 3. Configure DDS (both machines)

Edit `cyclonedds.xml` — replace the two `<Peer address=".."/>` lines with
the laptop and Jetson Wi-Fi IPs. Find each with `ip -4 addr show`.

## Running one trajectory

Order matters: bring up the Jetson side **first** (it subscribes and waits
for sensor frames). Then start the laptop side.

### Terminal 1 — Jetson (inside the Humble container)

```bash
/jetbot_ws/src/robot_sw/distributed_mapping/scovox_eval/jetbot/scripts/run_jetbot.sh
```

Wraps `ros2 launch scovox_mapping scenenet_jetbot.launch.py` — brings up
scovox_a + dscovox + the static TF for robot_a, plus jetson_telemetry.py
in the background writing `/tmp/jetson_timing.csv`.

### Terminal 2 — Laptop

```bash
src/robot_sw/distributed_mapping/scovox_eval/jetbot/scripts/run_laptop.sh 0_223 10.0
# Or, for "every frame processed by Jetson" quality runs:
src/robot_sw/distributed_mapping/scovox_eval/jetbot/scripts/run_laptop.sh 0_223 2.0
```

This is a shell orchestrator (matches the existing
`scenenet_run_batch_iter6.sh` convention since `scovox_eval` isn't a
ROS package). It backgrounds: the static TF for robot_b, scovox_node B,
and two `python3 -m scovox_eval.scenenet_replay_node` instances (one per
robot, distinct namespaces + frame splits). It blocks until both replays
exit, then Ctrl-C the Jetson terminal.

## Capturing the result map for scoring

The Jetson dscovox publishes `/dscovox_node/pointcloud` once the consensus
publish timer fires. On a third terminal (laptop or Jetson — either side
of DDS works):

```bash
ros2 run scovox_eval pointcloud-to-npz --ros-args \
    -p topic:=/dscovox_node/pointcloud \
    -p output:=~/jetbot_runs/0_223/dscovox_fused.npz \
    -p wait_secs:=10.0
```

For per-robot maps (separate scovox grids before fusion):

```bash
ros2 run scovox_eval pointcloud-to-npz --ros-args \
    -p topic:=/robotA/scovox_node/pointcloud \
    -p output:=~/jetbot_runs/0_223/scovox_a.npz \
    -p wait_secs:=10.0
```

Then on the laptop, score against ground truth:

```bash
python3 src/robot_sw/distributed_mapping/scovox_eval/scripts/scenenet_compute_metrics.py \
    --batch_root ~/jetbot_runs \
    --gt_root data/scenenet/val_preprocessed \
    --variant dscovox_fused \
    --out_csv ~/jetbot_runs/summary.csv
```

## What to measure

Two runs per trajectory, same trajectory, different rate_hz:

| Run            | rate_hz | Purpose                                                          |
|----------------|---------|------------------------------------------------------------------|
| **Throughput** | 10.0    | Headline: sustained Hz, frame drops, RSS, per-core CPU% on Nano. |
| **Quality**    | 2.0     | Every frame processed → mIoU vs desktop reference.               |

Quality target: within ±0.05 mIoU of the corresponding desktop run from
the project memory's Step 8 fusion numbers — that's the documented
single-cell run-to-run variance for iter6 fused walker.

## Troubleshooting

- `ros2 topic list` on one side doesn't show the other's topics → DDS
  discovery is failing. Re-check `cyclonedds.xml` IPs, confirm both
  sides export `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and the same
  `ROS_DOMAIN_ID`, and ping each side from the other.
- Jetson scovox_node OOM-killed → drop `--packages-up-to` build to
  `MAKEFLAGS=-j1`, or close other processes. Free RAM should stay above
  500 MB during runs.
- `liblz4-dev` missing → `apt-get install liblz4-dev` inside the container.
- Replay starts but Jetson sees no depth → frame topic mismatch. The
  replay node publishes under `/<robot_name>/rgbd_camera_depth_image`;
  the scovox node subscribes to the same relative topic in its namespace.
  Confirm both sides use `robot_a:=robotA` (same name).
