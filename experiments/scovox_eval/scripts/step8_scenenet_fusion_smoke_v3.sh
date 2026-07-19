#!/bin/bash
# Step 8 Phase 3 scaffolding smoke — SceneNet two-robot fusion on v3.
#
# Mirror of step8_fusion_smoke_v3.sh but on SceneNet 0_223. Validates:
#   - scenenet_eval_fusion.launch.py boots all 3 nodes
#   - dscovox v3 receiver pins SceneNet's (num_classes=14, alpha_0=0.01)
#   - fused.npz is non-empty (proves the v3 publish/receive loop is alive
#     on the SceneNet topic layout, not just Replica)
#
# Trajectory split (NEW_EXPERIMENT_PLAN.md Phase 3 convention): 50%
# overlap. Robot A=[0..200), Robot B=[100..300). Short for smoke; the
# full Phase 3 batch will use [0..200) / [100..300) too but on 13 trajs.

set -o pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RES_ROOT="${EVAL_PKG}/results/step8_scenenet_fusion_smoke_v3"
SCENE="0_223"
ROBOT_A="robotA"
ROBOT_B="robotB"
N_PER_ROBOT=200
OVERLAP=100        # B starts at OVERLAP, ends at OVERLAP + N
RATE_HZ=4.0
SCENENET_ROOT="${WS}/data/scenenet_val_layout"
mkdir -p "${RES_ROOT}/${SCENE}"
CELL="${RES_ROOT}/${SCENE}"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

T0=$(date +%s)

# Defensive sweep.
pgrep -af 'scovox_mapping_node|dscovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 2

# 1. Launch v3 fusion graph (3 nodes).
LAUNCH_LOG="${CELL}/fusion_launch.log"
echo "━━ Launching SceneNet v3 fusion graph (${SCENE}, A=[0,${N_PER_ROBOT}), B=[${OVERLAP},$((OVERLAP+N_PER_ROBOT)))) ━━"
ros2 launch scovox_mapping scenenet_eval_fusion.launch.py \
    robot_a:=${ROBOT_A} \
    robot_b:=${ROBOT_B} \
    resolution:=0.05 \
    use_split:=true \
    share_tsdf:=false \
    fused_walker:=true \
    wire_format:=v3 \
    num_classes:=14 \
    dirichlet_prior:=0.01 \
    > "${LAUNCH_LOG}" 2>&1 &
LPID=$!
sleep 6

# 2. Replay both robots in parallel — SceneNet replay node uses sequence
# as a STRING param (see scenenet-first-batch-2026-05-12 memo).
echo "━━ Replaying robot A frames [0, ${N_PER_ROBOT}) ━━"
python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${SCENENET_ROOT}" \
    -p sequence:="${SCENE}" \
    -p robot_name:=${ROBOT_A} \
    -p rate_hz:=${RATE_HZ} \
    -p start_frame:=0 \
    -p n_scans:=${N_PER_ROBOT} \
    -p use_gt_labels:=true > "${CELL}/replay_a.log" 2>&1 &
RA_PID=$!

echo "━━ Replaying robot B frames [${OVERLAP}, $((OVERLAP + N_PER_ROBOT))) ━━"
python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${SCENENET_ROOT}" \
    -p sequence:="${SCENE}" \
    -p robot_name:=${ROBOT_B} \
    -p rate_hz:=${RATE_HZ} \
    -p start_frame:=${OVERLAP} \
    -p n_scans:=${N_PER_ROBOT} \
    -p use_gt_labels:=true > "${CELL}/replay_b.log" 2>&1 &
RB_PID=$!

wait ${RA_PID} 2>/dev/null || true
wait ${RB_PID} 2>/dev/null || true
echo "  both replays done — settling 10 s"
sleep 10

# 3. Capture the three pointclouds.
for tag in solo_a solo_b fused; do
  case "${tag}" in
    solo_a) TOPIC="/${ROBOT_A}/scovox_node/pointcloud" ;;
    solo_b) TOPIC="/${ROBOT_B}/scovox_node/pointcloud" ;;
    fused)  TOPIC="/dscovox_node/pointcloud" ;;
  esac
  OUT="${CELL}/${tag}.npz"
  echo "━━ Capture ${tag} ← ${TOPIC} ━━"
  timeout 90 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:="${TOPIC}" \
      -p output:="${OUT}" 2>&1 | tail -1 || echo "  WARN: ${tag} capture timeout"
done

# 4. Cleanup.
kill ${LPID} 2>/dev/null || true
wait ${LPID} 2>/dev/null || true
pgrep -af 'scovox_mapping_node|dscovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1

T1=$(date +%s); WALL=$((T1 - T0))

# 5. Sharp assertions (silent-fallback guard).
PINNED=$(grep -c "v3 receive: pinned num_classes" "${LAUNCH_LOG}" 2>/dev/null || true); PINNED=${PINNED:-0}
BADVER=$(grep -c "Bad version" "${LAUNCH_LOG}" 2>/dev/null || true); BADVER=${BADVER:-0}
MISMATCH=$(grep -c "v3 prior mismatch" "${LAUNCH_LOG}" 2>/dev/null || true); MISMATCH=${MISMATCH:-0}

count_pts() {
  python3 - "$1" <<'PY'
import sys, numpy as np
p = sys.argv[1]
try:
    d = np.load(p); print(int(d['points'].shape[0]))
except Exception:
    print(0)
PY
}
NA=$(count_pts "${CELL}/solo_a.npz")
NB=$(count_pts "${CELL}/solo_b.npz")
NF=$(count_pts "${CELL}/fused.npz")

REPORT="${RES_ROOT}/SMOKE_REPORT.md"
{
  echo "# Step 8 SceneNet Fusion Smoke — wire_format=v3 (${SCENE})"
  echo "Generated: $(date -Iseconds)"
  echo "Wall-clock: ${WALL}s"
  echo
  echo "## Capture sizes"
  echo "| Tag | Points | Note |"
  echo "|---|---|---|"
  echo "| solo_a | ${NA} | robot A [0, ${N_PER_ROBOT}) |"
  echo "| solo_b | ${NB} | robot B [${OVERLAP}, $((OVERLAP + N_PER_ROBOT))) |"
  echo "| fused  | ${NF} | dscovox onBinaryMapV3 + mergeSemDir |"
  echo
  echo "## Receiver assertions"
  echo "- pinned-prior INFO: ${PINNED} (expect ≥ 1)"
  echo "- Bad version WARN:  ${BADVER} (expect 0)"
  echo "- prior mismatch:    ${MISMATCH} (expect 0)"
  echo
  if [[ "${PINNED}" -ge 1 && "${BADVER}" -eq 0 && "${MISMATCH}" -eq 0 && "${NF}" -gt 100 ]]; then
    echo "**PASS** — SceneNet v3 fusion loop alive."
  else
    echo "**FAIL** — see assertion counts."
  fi
} > "${REPORT}"

cat "${REPORT}"
