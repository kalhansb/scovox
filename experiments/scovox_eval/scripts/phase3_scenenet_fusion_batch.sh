#!/bin/bash
# NEW_EXPERIMENT_PLAN.md Phase 3 — 13-trajectory SceneNet fusion batch.
#
# Per-trajectory split: robot A=[0,200), robot B=[100,300) (50% overlap
# per the plan). wire_format=v3 end-to-end (validated by Step 8 fusion
# smoke commit a958172).
#
# Iterates the existing step8_scenenet_fusion_smoke_v3.sh logic per
# trajectory rather than calling it as a black box — gives finer control
# over result-dir layout (one cell per trajectory) and per-trajectory
# logging.
#
# Default trajectory list = the 13 from the SceneNet head-to-head batch
# (project-scenenet-first-batch-2026-05-12). Idempotent: cells with all
# 3 NPZs present are skipped.
#
# Env knobs: TRAJS, N_PER_ROBOT, OVERLAP, RATE_HZ

set -o pipefail
WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RES_ROOT="${EVAL_PKG}/results/phase3_scenenet_fusion_v3_$(date +%Y_%m_%d)"
SCENENET_ROOT="${WS}/data/scenenet_val_layout"
ROBOT_A="robotA"; ROBOT_B="robotB"
N_PER_ROBOT="${N_PER_ROBOT:-200}"
OVERLAP="${OVERLAP:-100}"
RATE_HZ="${RATE_HZ:-4.0}"

# 13 val trajs actually staged on disk under scenenet_val_layout/train/
# (2026-05-14 audit). The prior default referenced the head-to-head 13
# from [[project-scenenet-first-batch-2026-05-12]], but only 4 of those
# overlap with what's on disk. The 13 below are all immediately usable
# without re-downloading the 262 GB train shards.
TRAJS_DEFAULT=(0_175 0_178 0_182 0_223 0_279
               0_485 0_490 0_571 0_682 0_723
               0_789 0_867 0_977)
if [[ -n "${TRAJS:-}" ]]; then IFS=' ' read -ra TRAJ_LIST <<< "${TRAJS}"; else TRAJ_LIST=("${TRAJS_DEFAULT[@]}"); fi

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"
mkdir -p "${RES_ROOT}"

# Filter to trajectories that actually have data staged. Per the
# [[project-scenenet-rgbd-mirror]] memo, only a subset of the head-to-
# head 13 has data on disk — full train shards are 262 GB and won't fit.
# Missing trajs would just silently NaN out the scoring; skipping them
# up front keeps the matrix honest.
_PRESENT=()
for _t in "${TRAJ_LIST[@]}"; do
  if [[ -d "${SCENENET_ROOT}/train/${_t}" ]]; then
    _PRESENT+=("${_t}")
  else
    echo "[phase3] SKIP ${_t} — no data dir at ${SCENENET_ROOT}/train/${_t}"
  fi
done
TRAJ_LIST=("${_PRESENT[@]}")
if [[ "${#TRAJ_LIST[@]}" -eq 0 ]]; then
  echo "[phase3] FATAL: no trajectories with data on disk; aborting" >&2
  exit 1
fi
echo "[phase3] running ${#TRAJ_LIST[@]} trajs: ${TRAJ_LIST[*]}"

run_traj() {
  local traj="$1"
  local cell="${RES_ROOT}/${traj}"
  mkdir -p "${cell}"
  local solo_a="${cell}/solo_a.npz" solo_b="${cell}/solo_b.npz" fused="${cell}/fused.npz"
  if [[ -s "${solo_a}" && -s "${solo_b}" && -s "${fused}" ]]; then
    echo "[${traj}] skip — all 3 NPZs present"; return 0
  fi

  echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Phase 3 SceneNet fusion v3 — ${traj}"
  echo "  A=[0,${N_PER_ROBOT})  B=[${OVERLAP},$((OVERLAP+N_PER_ROBOT)))"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  pgrep -af 'scovox_mapping_node|dscovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2

  local launch_log="${cell}/fusion_launch.log"
  ros2 launch scovox_mapping scenenet_eval_fusion.launch.py \
      robot_a:=${ROBOT_A} robot_b:=${ROBOT_B} resolution:=0.05 \
      use_split:=true share_tsdf:=false fused_walker:=true \
      wire_format:=v3 num_classes:=14 dirichlet_prior:=0.01 \
      > "${launch_log}" 2>&1 &
  local LPID=$!; sleep 6

  python3 -m scovox_eval.scenenet_replay_node --ros-args \
      -p data_root:="${SCENENET_ROOT}" -p sequence:="${traj}" \
      -p robot_name:=${ROBOT_A} -p rate_hz:=${RATE_HZ} -p use_gt_labels:=true \
      -p start_frame:=0 -p n_scans:=${N_PER_ROBOT} > "${cell}/replay_a.log" 2>&1 &
  local RA_PID=$!
  python3 -m scovox_eval.scenenet_replay_node --ros-args \
      -p data_root:="${SCENENET_ROOT}" -p sequence:="${traj}" \
      -p robot_name:=${ROBOT_B} -p rate_hz:=${RATE_HZ} -p use_gt_labels:=true \
      -p start_frame:=${OVERLAP} -p n_scans:=${N_PER_ROBOT} > "${cell}/replay_b.log" 2>&1 &
  local RB_PID=$!
  wait ${RA_PID} 2>/dev/null || true
  wait ${RB_PID} 2>/dev/null || true
  sleep 10  # tail integration + bin flush

  for tag in solo_a solo_b fused; do
    case "${tag}" in
      solo_a) TOPIC="/${ROBOT_A}/scovox_node/pointcloud" ;;
      solo_b) TOPIC="/${ROBOT_B}/scovox_node/pointcloud" ;;
      fused)  TOPIC="/dscovox_node/pointcloud" ;;
    esac
    timeout 90 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:="${TOPIC}" -p output:="${cell}/${tag}.npz" 2>&1 | tail -1 \
      || echo "  WARN: ${tag} timeout"
  done

  kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  pgrep -af 'scovox_mapping_node|dscovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1

  # Receiver-log assertions per cell (silent-fallback guard).
  local pinned bad mismatch
  pinned=$(grep -c "v3 receive: pinned num_classes" "${launch_log}" 2>/dev/null || true);  pinned=${pinned:-0}
  bad=$(grep -c "Bad version" "${launch_log}" 2>/dev/null || true);                        bad=${bad:-0}
  mismatch=$(grep -c "v3 prior mismatch" "${launch_log}" 2>/dev/null || true);             mismatch=${mismatch:-0}
  echo "  [${traj}] pinned=${pinned} bad_version=${bad} mismatch=${mismatch}"
}

for traj in "${TRAJ_LIST[@]}"; do
  run_traj "${traj}"
done

# Score (mIoU, F@5cm, Chamfer) per trajectory + aggregate.
echo ""; echo "━━ Scoring ━━"
CSV="${RES_ROOT}/phase3_summary.csv"
echo "traj,solo_a_miou,solo_b_miou,fused_miou,fused_minus_max" > "${CSV}"
for traj in "${TRAJ_LIST[@]}"; do
  cell="${RES_ROOT}/${traj}"
  gt="${SCENENET_ROOT}/train/${traj}/gt_5cm.npz"
  [[ -s "${gt}" ]] || { echo "${traj},NaN,NaN,NaN,NaN" >> "${CSV}"; continue; }
  ma=$(python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" --pred_npz "${cell}/solo_a.npz" --gt_npz "${gt}" --resolution 0.05 2>&1 | grep -oP 'mIoU\s*[:=]\s*\K[0-9.]+' | head -1)
  mb=$(python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" --pred_npz "${cell}/solo_b.npz" --gt_npz "${gt}" --resolution 0.05 2>&1 | grep -oP 'mIoU\s*[:=]\s*\K[0-9.]+' | head -1)
  mf=$(python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" --pred_npz "${cell}/fused.npz"  --gt_npz "${gt}" --resolution 0.05 2>&1 | grep -oP 'mIoU\s*[:=]\s*\K[0-9.]+' | head -1)
  delta=$(awk -v a="${ma:-0}" -v b="${mb:-0}" -v f="${mf:-0}" 'BEGIN { m=(a>b)?a:b; printf "%.4f", f-m }')
  echo "${traj},${ma:-NaN},${mb:-NaN},${mf:-NaN},${delta}" >> "${CSV}"
done

echo ""; echo "Summary:"
column -s, -t < "${CSV}"
