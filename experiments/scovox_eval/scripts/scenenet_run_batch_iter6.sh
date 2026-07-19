#!/bin/bash
# SceneNet val-batch evaluator for SCovox iter6 (use_split + fused_walker).
#
# For each preprocessed trajectory under data/scenenet_val_layout/train/<seq>/:
#   1. Launch scovox_mapping_node with scenenet_eval.launch.py
#   2. Replay 300 frames via scenenet_replay_node at 10 Hz
#   3. Capture ~/pointcloud → NPZ
#   4. Build GT voxel grid via scenenet_build_gt.py
#   5. Score with scenenet_compute_metrics.py at the end
#
# Defaults: use_split:=true, share_tsdf:=false, fused_walker:=true,
#           semantic_mode:=dirichlet, resolution:=0.05, K_TOP=2 (compile-time).
#
# Usage:
#   ./scenenet_run_batch_iter6.sh                    # 13 default cells
#   ./scenenet_run_batch_iter6.sh 0_223 0_175 0_485  # specific cells

set -o pipefail
WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
DATA_ROOT="${WS}/data/scenenet_val_layout"        # symlink farm → val_preprocessed
RES_ROOT="${EVAL_PKG}/results/scenenet_val_iter6"
RESOLUTION="0.05"
ROBOT="atlas"
RATE_HZ="10.0"

mkdir -p "${RES_ROOT}"
RUN_LOG="${RES_ROOT}/runner.log"
echo "[scenenet] starting; $(date)" | tee "${RUN_LOG}"

# Determine which sequences to run
if [[ $# -gt 0 ]]; then
  SEQS=("$@")
else
  SEQS=()
  for d in "${DATA_ROOT}/train"/*/; do
    SEQS+=("$(basename "$d")")
  done
fi
echo "[scenenet] sequences: ${SEQS[*]}" | tee -a "${RUN_LOG}"

# Clean ROS env, source workspace.
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV PYTHONNOUSERSITE
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cleanup_procs() {
  pkill -9 -f scovox_mapping_node 2>/dev/null || true
  pkill -9 -f scenenet_replay_node 2>/dev/null || true
  pkill -9 -f pointcloud_to_npz 2>/dev/null || true
  sleep 2
}

run_one_cell() {
  local SEQ="$1"
  local CELL_DIR="${RES_ROOT}/${SEQ}"
  local NPZ="${CELL_DIR}/scovox_dirichlet.npz"
  local GT_NPZ="${WS}/data/scenenet/val_preprocessed/${SEQ}/gt_5cm.npz"
  local LOG="${CELL_DIR}/scovox_run.log"

  mkdir -p "${CELL_DIR}"

  if [[ -f "${NPZ}" ]]; then
    echo "[scenenet ${SEQ}] skip NPZ — already exists" | tee -a "${RUN_LOG}"
    return 0
  fi

  echo "" | tee -a "${RUN_LOG}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${RUN_LOG}"
  echo "  ${SEQ}  $(date)" | tee -a "${RUN_LOG}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${RUN_LOG}"
  cleanup_procs

  # 1. Launch scovox_mapping_node in background.
  ros2 launch scovox_mapping scenenet_eval.launch.py \
      robot_name:="${ROBOT}" \
      resolution:="${RESOLUTION}" \
      semantic_mode:=dirichlet \
      use_split:=true \
      share_tsdf:=false \
      fused_walker:=true \
      > "${LOG}" 2>&1 &
  local LPID=$!
  sleep 4

  # 2. Replay all 300 frames (blocks until done).
  python3 -m scovox_eval.scenenet_replay_node --ros-args \
      -p data_root:="${DATA_ROOT}" \
      -p sequence:="${SEQ}" \
      -p rate_hz:="${RATE_HZ}" \
      -p robot_name:="${ROBOT}" \
      -p use_gt_labels:=true \
      >> "${LOG}" 2>&1 || true

  # 3. Let the integrator drain.
  sleep 4

  # 4. Capture pointcloud to NPZ.
  timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:="/${ROBOT}/scovox_node/pointcloud" \
      -p output:="${NPZ}" \
      -p wait_secs:=5.0 \
      >> "${LOG}" 2>&1 || echo "  WARN pointcloud capture timed out" | tee -a "${RUN_LOG}"

  kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  cleanup_procs

  # 5. Build GT NPZ if missing.
  if [[ ! -f "${GT_NPZ}" ]]; then
    echo "[scenenet ${SEQ}] building GT" | tee -a "${RUN_LOG}"
    python3 "${EVAL_PKG}/scripts/scenenet_build_gt.py" \
        "${DATA_ROOT}" "${SEQ}" "${GT_NPZ}" \
        --resolution "${RESOLUTION}" >> "${LOG}" 2>&1
  fi

  if [[ -f "${NPZ}" ]]; then
    echo "[scenenet ${SEQ}] ✓ NPZ saved ($(stat -c%s "${NPZ}") B)" | tee -a "${RUN_LOG}"
  else
    echo "[scenenet ${SEQ}] ✗ NPZ MISSING — run failed" | tee -a "${RUN_LOG}"
  fi
}

# ───────────────────────────── main ─────────────────────────────
for SEQ in "${SEQS[@]}"; do
  run_one_cell "${SEQ}"
done

# ───────────────────────────── score ─────────────────────────────
echo "" | tee -a "${RUN_LOG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${RUN_LOG}"
echo "  Scoring all cells  $(date)" | tee -a "${RUN_LOG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${RUN_LOG}"

python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" \
    --batch_root "${RES_ROOT}" \
    --gt_root "${WS}/data/scenenet/val_preprocessed" \
    --variant scovox_dirichlet \
    --out_csv "${RES_ROOT}/summary.csv" 2>&1 | tee -a "${RUN_LOG}"

echo "[scenenet] all done $(date)" | tee -a "${RUN_LOG}"
