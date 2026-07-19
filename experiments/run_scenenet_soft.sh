#!/bin/bash
# SceneNet SOFT-label run: feed Mask2Former per-pixel .topk probabilities to
# scovox instead of GT one-hot labels. Usage: run_scenenet_soft.sh <seq>
set -o pipefail
SEQ="${1:-0_223}"

WS=/home/kalhan/Projects/scovox_ws
EVAL_PKG="${WS}/experiments/scovox_eval"
SCENENET_ROOT="${WS}/data/scenenet_val_layout"
TOPK_DIR="${SCENENET_ROOT}/train/${SEQ}/predictions_topk"
RES_ROOT="${WS}/experiments/results/scenenet_soft/${SEQ}"
mkdir -p "${RES_ROOT}"
LOG="${RES_ROOT}/scovox.log"; OUT="${RES_ROOT}/scovox.npz"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/scovox/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

echo "== launch scovox (SceneNet eval, SOFT topk=${TOPK_DIR}) =="
ros2 launch scovox_mapping scenenet_eval.launch.py \
    robot_name:=phase0 resolution:=0.05 semantic_mode:=dirichlet \
    use_split:=true share_tsdf:=false fused_walker:=true \
    topk_probs_dir:="${TOPK_DIR}" > "${LOG}" 2>&1 &
LPID=$!; sleep 5

echo "== replay SceneNet ${SEQ} =="
python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${SCENENET_ROOT}" -p sequence:="${SEQ}" \
    -p rate_hz:=10.0 -p robot_name:=phase0 -p use_gt_labels:=true >> "${LOG}" 2>&1

WAITED=0
while true; do
  LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
  [ "${LAST:-0}" -ge 285 ] 2>/dev/null && { echo "  recv=${LAST}, settling"; sleep 5; break; }
  sleep 3; WAITED=$((WAITED+3))
  [ ${WAITED} -ge 300 ] && { echo "  WARN timeout @ recv=${LAST}"; break; }
done

echo "== capture -> npz =="
timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/phase0/scovox_node/pointcloud -p output:="${OUT}" || echo "  WARN capture timeout"
kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' 2>/dev/null || true

echo "== topk loader telemetry (must show loaded>=285) =="
grep -oP 'topk loader: loaded=\K[0-9]+' "${LOG}" | tail -1
grep -oP 'topk: first frame loaded.*' "${LOG}" | tail -1

echo "== score vs GT =="
python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" \
    --gt_npz "${SCENENET_ROOT}/train/${SEQ}/gt_5cm.npz" \
    --pred_npz "${OUT}" 2>&1 | tee "${RES_ROOT}/score.log"
echo "DONE ${SEQ}"
