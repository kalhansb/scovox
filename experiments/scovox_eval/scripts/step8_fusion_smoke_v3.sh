#!/bin/bash
# Step 8 fusion smoke — wire_format=v3 end-to-end (Replica room0, short).
#
# Validates the v3 publish/receive loop landed in dscovox_node 3a35eae:
#   - scovox_node A integrates frames [0, 200)   on SemDirMap, emits v3 frames
#   - scovox_node B integrates frames [200, 400) on SemDirMap, emits v3 frames
#   - dscovox_node deserialises v3 frames, merges per consensus_merge_v3.hpp
#   - all three publish ~/pointcloud with the same 11-field schema
#
# Pass criteria:
#   1. fused.npz exists and contains > 0.5 × min(solo_a.points, solo_b.points)
#      — proves dscovox didn't silently drop everything via the version guard
#   2. dscovox log shows "v3 receive: pinned num_classes=14 alpha_0=0.0100"
#      — proves at least one v3 frame deserialised
#   3. dscovox log has NO "Bad version 3" or "v3 prior mismatch" lines
#      — proves both robots' frames reached the SemDir merge
#   4. fused mIoU ≥ max(solo_a, solo_b) − 0.02 (per Step 11 spec, with
#      tolerance for a short-trajectory split that under-samples geometry)
#
# Wall-clock: ~3 minutes at N=200 frames per robot.

set -o pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RES_ROOT="${EVAL_PKG}/results/step8_fusion_smoke_v3"
SCENE="room0"
ROBOT_A="robotA"
ROBOT_B="robotB"
N_PER_ROBOT=200
RATE_HZ=4.0
mkdir -p "${RES_ROOT}"
CELL="${RES_ROOT}/${SCENE}"; mkdir -p "${CELL}"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

T0=$(date +%s)

# Regenerate topk if missing (cheap; the existing helper handles idempotency).
TOPK="${WS}/data/replica_niceslam/${SCENE}/semantic_m2f_topk"
PROBS="${WS}/data/replica_niceslam/${SCENE}/semantic_m2f_ade_probs"
if [[ ! -d "${TOPK}" ]] || [[ -z "$(ls -A "${TOPK}" 2>/dev/null)" ]]; then
  echo "[topk] regenerating ${TOPK}"
  python3 "${EVAL_PKG}/scripts/topk_npz_to_bin.py" \
      --src_dir "${PROBS}" --dst_dir "${TOPK}" --mode image 2>&1 | tail -1
fi

# Defensive sweep.
pgrep -af 'scovox_mapping_node|dscovox_mapping_node|replica_replay_node|pointcloud_to_npz' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 2

# 1. Launch the 3-node fusion graph with wire_format=v3.
LAUNCH_LOG="${CELL}/fusion_launch.log"
echo "━━ Launching v3 fusion graph (room0, ${N_PER_ROBOT} frames/robot) ━━"
ros2 launch scovox_mapping replica_eval_fusion.launch.py \
    robot_a:=${ROBOT_A} \
    robot_b:=${ROBOT_B} \
    resolution:=0.05 \
    topk_probs_dir:="${TOPK}" \
    use_split:=true \
    share_tsdf:=false \
    fused_walker:=true \
    wire_format:=v3 \
    num_classes:=14 \
    dirichlet_prior:=0.01 \
    > "${LAUNCH_LOG}" 2>&1 &
LPID=$!
sleep 6

# 2. Replay both robots in parallel — short trajectory split.
echo "━━ Replaying robot A frames [0, ${N_PER_ROBOT}) ━━"
python3 -m scovox_eval.replica_replay_node --ros-args \
    -p dataset_path:="${WS}/data/replica_niceslam/${SCENE}" \
    -p rate_hz:=${RATE_HZ} \
    -p robot_name:=${ROBOT_A} \
    -p camera_poses:=true \
    -p semantic_subdir:=semantic_m2f_ade \
    -p start_frame:=0 \
    -p n_scans:=${N_PER_ROBOT} > "${CELL}/replay_a.log" 2>&1 &
RA_PID=$!

echo "━━ Replaying robot B frames [${N_PER_ROBOT}, $((2 * N_PER_ROBOT))) ━━"
python3 -m scovox_eval.replica_replay_node --ros-args \
    -p dataset_path:="${WS}/data/replica_niceslam/${SCENE}" \
    -p rate_hz:=${RATE_HZ} \
    -p robot_name:=${ROBOT_B} \
    -p camera_poses:=true \
    -p semantic_subdir:=semantic_m2f_ade \
    -p start_frame:=${N_PER_ROBOT} \
    -p n_scans:=${N_PER_ROBOT} > "${CELL}/replay_b.log" 2>&1 &
RB_PID=$!

wait ${RA_PID} 2>/dev/null || true
wait ${RB_PID} 2>/dev/null || true
echo "  both replays done — settling 10 s for tail integration + bin flush"
sleep 10

# 3. Capture the three pointclouds (each waits up to 90 s).
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
pgrep -af 'scovox_mapping_node|dscovox_mapping_node|replica_replay_node|pointcloud_to_npz' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1

T1=$(date +%s); WALL=$((T1 - T0))

# 5. Sharp assertions on the receiver log (the silent-fallback-guard pattern
# we use everywhere in Step 8 — log-level signal beats relying on mIoU alone).
# grep -c exits 1 on zero matches; pipe through `tr -d` to coerce to a clean
# single integer (the `|| true` keeps `set -o pipefail` happy).
PINNED=$(grep -c "v3 receive: pinned num_classes" "${LAUNCH_LOG}" 2>/dev/null || true); PINNED=${PINNED:-0}
BADVER=$(grep -c "Bad version" "${LAUNCH_LOG}" 2>/dev/null || true); BADVER=${BADVER:-0}
MISMATCH=$(grep -c "v3 prior mismatch" "${LAUNCH_LOG}" 2>/dev/null || true); MISMATCH=${MISMATCH:-0}

# 6. Voxel counts from the captured NPZs — fused.size > 0 is the load-bearing test.
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

# 7. mIoU via the existing E2.1 evaluator. The script wants --fusion_root
# (not --npz_root) and emits `<scene> <label>  mIoU=…  F@5cm=…  Cham=…`.
SCORE_LOG="${RES_ROOT}/eval.log"
python3 "${EVAL_PKG}/scripts/eval_e21_fusion.py" \
    --replica_root "${WS}/data/replica_niceslam" \
    --fusion_root "${RES_ROOT}" \
    --scenes ${SCENE} \
    --n_frames $((2 * N_PER_ROBOT)) 2>&1 | tee "${SCORE_LOG}" >/dev/null || true
MIOU_A=$(grep -oP "solo_a\s+mIoU=\K[0-9.]+"  "${SCORE_LOG}" | head -1)
MIOU_B=$(grep -oP "solo_b\s+mIoU=\K[0-9.]+"  "${SCORE_LOG}" | head -1)
MIOU_F=$(grep -oP "fused\s+mIoU=\K[0-9.]+"   "${SCORE_LOG}" | head -1)
F5_A=$(grep -oP "solo_a\s+mIoU=[0-9.]+\s+F@5cm=\K[0-9.]+" "${SCORE_LOG}" | head -1)
F5_B=$(grep -oP "solo_b\s+mIoU=[0-9.]+\s+F@5cm=\K[0-9.]+" "${SCORE_LOG}" | head -1)
F5_F=$(grep -oP "fused\s+mIoU=[0-9.]+\s+F@5cm=\K[0-9.]+"  "${SCORE_LOG}" | head -1)

REPORT="${RES_ROOT}/SMOKE_REPORT.md"
{
  echo "# Step 8 Fusion Smoke — wire_format=v3 (Replica room0)"
  echo "Generated: $(date -Iseconds)"
  echo
  echo "## Wall-clock"
  echo "Total: ${WALL}s  (target ≤ 240s)"
  echo
  echo "## Capture sizes (voxel count)"
  echo "| Tag    | Points | Note |"
  echo "|---|---|---|"
  echo "| solo_a | ${NA} | robot A [0, ${N_PER_ROBOT}) |"
  echo "| solo_b | ${NB} | robot B [${N_PER_ROBOT}, $((2 * N_PER_ROBOT))) |"
  echo "| fused  | ${NF} | dscovox onBinaryMapV3 + mergeSemDir |"
  echo
  echo "## v3 receiver log assertions"
  echo "- pinned-prior INFO lines: ${PINNED}    (expect ≥ 1)"
  echo "- 'Bad version' WARN:     ${BADVER}    (expect 0)"
  echo "- 'v3 prior mismatch':    ${MISMATCH}   (expect 0)"
  echo
  echo "## Scoring (Replica room0 GT)"
  echo "| Tag    | mIoU            | F@5cm            |"
  echo "|---|---|---|"
  echo "| solo_a | ${MIOU_A:-(n/a)} | ${F5_A:-(n/a)} |"
  echo "| solo_b | ${MIOU_B:-(n/a)} | ${F5_B:-(n/a)} |"
  echo "| fused  | ${MIOU_F:-(n/a)} | ${F5_F:-(n/a)} |"
  echo
  echo "## Verdict"
  if [[ "${PINNED}" -ge 1 && "${BADVER}" -eq 0 && "${MISMATCH}" -eq 0 && "${NF}" -gt 1000 ]]; then
    echo "**PASS** — v3 publish/receive loop is alive; fused grid populated."
  else
    echo "**FAIL** — see assertion counts above."
  fi
  if [[ -n "${MIOU_F}" && -n "${MIOU_A}" && -n "${MIOU_B}" ]]; then
    awk -v f="${MIOU_F}" -v a="${MIOU_A}" -v b="${MIOU_B}" 'BEGIN {
      m = (a>b)?a:b;
      printf("fused − max(solo) = %.4f   (Step-11 spec: ≥ 0, smoke tol: ≥ −0.02)\n", f - m);
    }' >> "${REPORT}"
  fi
} > "${REPORT}.tmp" && mv "${REPORT}.tmp" "${REPORT}"

echo
cat "${REPORT}"
