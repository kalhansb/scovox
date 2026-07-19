#!/bin/bash
# Two-robot Dirichlet fusion on one SceneNet trajectory.
# Robot A = frames [0, N), Robot B = frames [OVERLAP, OVERLAP+N).
# Disjoint (no shared frames): N=150 OVERLAP=150 -> A=[0,150) B=[150,300).
# Captures solo_a, solo_b, fused maps and scores each vs GT.
#
# Usage: run_scenenet_fusion.sh <traj> <N_PER_ROBOT> <OVERLAP> <out_root>
set -o pipefail
TRAJ="${1:-0_223}"
N="${2:-150}"
OVERLAP="${3:-150}"
OUT_ROOT="${4:-/home/kalhan/Projects/scovox_ws/experiments/results/fusion_disjoint}"
RATE_HZ="${RATE_HZ:-4.0}"

WS=/home/kalhan/Projects/scovox_ws
EVAL_PKG="${WS}/experiments/scovox_eval"
SCENENET_ROOT="${WS}/data/scenenet_val_layout"
ROBOT_A=robotA; ROBOT_B=robotB
CELL="${OUT_ROOT}/${TRAJ}"; mkdir -p "${CELL}"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/scovox/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

pgrep -af 'scovox_mapping_node|dscovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2

echo "== [${TRAJ}] launch fusion: A=[0,${N}) B=[${OVERLAP},$((OVERLAP+N))) =="
ros2 launch scovox_mapping scenenet_eval_fusion.launch.py \
    robot_a:=${ROBOT_A} robot_b:=${ROBOT_B} resolution:=0.05 \
    use_split:=true share_tsdf:=false fused_walker:=true \
    num_classes:=14 dirichlet_prior:=0.01 semantic_mode:=dirichlet \
    > "${CELL}/fusion_launch.log" 2>&1 &
LPID=$!; sleep 6

echo "== replay both robots (disjoint frame ranges) =="
python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${SCENENET_ROOT}" -p sequence:="${TRAJ}" \
    -p robot_name:=${ROBOT_A} -p rate_hz:=${RATE_HZ} -p use_gt_labels:=true \
    -p start_frame:=0 -p n_scans:=${N} > "${CELL}/replay_a.log" 2>&1 &
RA=$!
python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${SCENENET_ROOT}" -p sequence:="${TRAJ}" \
    -p robot_name:=${ROBOT_B} -p rate_hz:=${RATE_HZ} -p use_gt_labels:=true \
    -p start_frame:=${OVERLAP} -p n_scans:=${N} > "${CELL}/replay_b.log" 2>&1 &
RB=$!
wait ${RA} 2>/dev/null || true
wait ${RB} 2>/dev/null || true
echo "  replays done; integration + fusion tail 12s"; sleep 12

for tag in solo_a solo_b fused; do
  case "${tag}" in
    solo_a) TOPIC="/${ROBOT_A}/scovox_node/pointcloud" ;;
    solo_b) TOPIC="/${ROBOT_B}/scovox_node/pointcloud" ;;
    fused)  TOPIC="/dscovox_node/pointcloud" ;;
  esac
  timeout 90 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:="${TOPIC}" -p output:="${CELL}/${tag}.npz" 2>&1 | tail -1 \
    || echo "  WARN ${tag} capture timeout"
done

kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
pgrep -af 'scovox_mapping_node|dscovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
  | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 2

echo "== score each map vs GT =="
GT="${SCENENET_ROOT}/train/${TRAJ}/gt_5cm.npz"
for tag in solo_a solo_b fused; do
  M=$(python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" \
        --pred_npz "${CELL}/${tag}.npz" --gt_npz "${GT}" --resolution 0.05 2>&1 \
      | grep -oP 'mIoU\s*[:=]\s*\K[0-9.]+' | head -1)
  echo "${tag} ${M}" | tee -a "${CELL}/scores.txt"
done
echo "DONE ${TRAJ}"
