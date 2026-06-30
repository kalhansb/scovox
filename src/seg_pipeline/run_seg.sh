#!/usr/bin/env bash
# Build (if needed), start the seg container, and launch the online seg node.
# The node consumes the bag's depth+color and publishes the colored seg +
# re-framed depth for SCovox. Run a bag (with --clock) + a localizer separately.
#
#   ./run_seg.sh              # use_sim_time:=true (bag playback)
#   ./run_seg.sh live         # use_sim_time:=false (live sensor)
set -euo pipefail
cd "$(dirname "$0")"

SIM=true
[ "${1:-}" = "live" ] && SIM=false

echo "[seg] building image (first run downloads torch/cu128 — can take a while)…"
docker compose up -d --build

echo "[seg] launching online seg node (use_sim_time:=${SIM})…"
exec docker compose exec seg bash -lc "
  source /opt/ros/jazzy/setup.bash
  cd /seg && python3 -m seg_pipeline.seg_node --ros-args -p use_sim_time:=${SIM}
"
