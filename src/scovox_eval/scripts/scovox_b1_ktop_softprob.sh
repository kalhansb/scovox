#!/bin/bash
# B1 (post-cleanup) — K_TOP sparse Dirichlet sweep on the SOFT-PROB pipeline
# (dense per-pixel/per-point distributions ingested via topk_probs_dir).
#
# Differs from scovox_b1_ktop.sh in that it points the launch at the *_topk
# directories produced by run_mask2former_replica.py + topk_npz_to_bin.py
# (Replica) and run_polarseg_inference.py (KITTI), so the Dirichlet update
# uses the full per-class distribution rather than a one-hot from argmax.
#
# Recompile-driven: edit K_TOP in scovox_core/include/scovox/voxel.hpp,
# rebuild Release, then run this script. Cell:
#   results/softprob_<anchor>/k_top_<K>/scovox.npz
#
# Usage:
#   scovox_b1_ktop_softprob.sh <K>          # both anchors
#   scovox_b1_ktop_softprob.sh <K> replica
#   scovox_b1_ktop_softprob.sh <K> kitti
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <K> [replica|kitti]" >&2; exit 2
fi
K="$1"
WHICH="${2:-both}"

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
VOXEL_HPP="${WS}/src/robot_sw/distributed_mapping/scovox_core/include/scovox/voxel.hpp"
REPLICA_ROOT="${WS}/data/replica_niceslam"
KITTI_ROOT="${WS}/data/semantickitti/dataset"

# Sanity: source has the requested K
SRC_K=$(grep -oP 'constexpr int K_TOP = \K[0-9]+' "${VOXEL_HPP}")
if [[ "${SRC_K}" != "${K}" ]]; then
  echo "ERROR: voxel.hpp has K_TOP=${SRC_K} but script invoked with K=${K}." >&2
  echo "       edit ${VOXEL_HPP} and rebuild Release before running." >&2
  exit 3
fi

# Sanity: install/ has been rebuilt at this K (the sizeof check is in the
# scovox_node startup banner, but we cannot easily probe binaries; trust the
# user pipeline edit + rebuild + source flow).
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

run_replica() {
  local CELL_DIR="${EVAL_PKG}/results/softprob_replica_room0_m2f/k_top_${K}"
  local OUT="${CELL_DIR}/scovox.npz"
  local LOG="${CELL_DIR}/scovox_run.log"
  local TOPK_DIR="${REPLICA_ROOT}/room0/semantic_m2f_topk"
  mkdir -p "${CELL_DIR}"
  if [[ -f "${OUT}" ]]; then echo "[replica softprob k_top_${K}] skip"; return; fi
  if [[ ! -d "${TOPK_DIR}" ]]; then
    echo "ERROR: replica topk dir missing: ${TOPK_DIR}" >&2; exit 4
  fi
  echo "━━ replica room0 m2f softprob, K_TOP=${K} ━━"
  ros2 launch scovox_mapping replica_eval.launch.py \
      robot_name:=ablation \
      resolution:=0.05 \
      semantic_occ_gate:=0.0 \
      range_decay_length:=-1.0 \
      w_occ:=2.0 \
      w_free:=1.0 \
      carve_skip_occ_threshold:=0.7 \
      kappa0:=2.0 \
      evidence_saturation:=1000.0 \
      semantic_min_confidence:=0.1 \
      topk_probs_dir:="${TOPK_DIR}" \
      > "${LOG}" 2>&1 &
  local LPID=$!
  sleep 4
  python3 -m scovox_eval.replica_replay_node --ros-args \
      -p dataset_path:="${REPLICA_ROOT}/room0" \
      -p rate_hz:=2.0 \
      -p robot_name:=ablation \
      -p camera_poses:=true \
      -p semantic_subdir:=semantic_m2f_ade \
      -p n_scans:=2000
  local MIN_RECV=1980 WAITED=0 LAST=0
  while true; do
    LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 30; break; fi
    sleep 5; WAITED=$((WAITED+5))
    [ ${WAITED} -ge 1800 ] && { sleep 30; break; }
  done
  timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/ablation/scovox_node/pointcloud \
      -p output:="${OUT}" || true
  kill ${LPID} 2>/dev/null || true
  wait ${LPID} 2>/dev/null || true
  pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
  pkill -9 -f 'replica_replay_node' 2>/dev/null || true
  pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
  sleep 3
}

run_kitti() {
  local CELL_DIR="${EVAL_PKG}/results/softprob_kitti_seq08_polarseg/k_top_${K}"
  local OUT="${CELL_DIR}/scovox.npz"
  local LOG="${CELL_DIR}/scovox_run.log"
  local TOPK_DIR="${KITTI_ROOT}/sequences/08/predictions_topk"
  mkdir -p "${CELL_DIR}"
  if [[ -f "${OUT}" ]]; then echo "[kitti softprob k_top_${K}] skip"; return; fi
  if [[ ! -d "${TOPK_DIR}" ]]; then
    echo "ERROR: kitti topk dir missing: ${TOPK_DIR}" >&2; exit 4
  fi
  echo "━━ kitti seq08 polarseg softprob, K_TOP=${K} ━━"
  ros2 launch scovox_mapping semantickitti_eval.launch.py \
      robot_name:=ablation \
      resolution:=0.10 \
      semantic_mode:=dirichlet \
      semantic_occ_gate:=0.0 \
      range_decay_length:=50.0 \
      w_occ:=6.0 \
      w_free:=1.0 \
      carve_skip_occ_threshold:=0.4 \
      kappa0:=2.0 \
      evidence_saturation:=1000.0 \
      semantic_min_confidence:=0.1 \
      topk_probs_dir:="${TOPK_DIR}" \
      > "${LOG}" 2>&1 &
  local LPID=$!
  sleep 4
  python3 -m scovox_eval.semantickitti_replay_node --ros-args \
      -p dataset_path:="${KITTI_ROOT}" \
      -p sequence:=8 \
      -p rate_hz:=0.5 \
      -p robot_name:=ablation \
      -p max_range:=30.0 \
      -p min_range:=1.0 \
      -p labels_subdir:=predictions \
      -p n_scans:=100 \
      -p soft_prob_passthrough:=true
  local MIN_RECV=99 WAITED=0 LAST=0
  while true; do
    LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
    if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 10; break; fi
    sleep 3; WAITED=$((WAITED+3))
    [ ${WAITED} -ge 600 ] && { echo "  WARN: timeout @ recv=${LAST}"; break; }
  done
  timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/ablation/scovox_node/pointcloud \
      -p output:="${OUT}" || echo "  WARN: capture timeout"
  kill ${LPID} 2>/dev/null || true
  wait ${LPID} 2>/dev/null || true
  pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
  pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
  pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
  sleep 2
}

case "${WHICH}" in
  replica) run_replica ;;
  kitti)   run_kitti ;;
  both)    run_replica; run_kitti ;;
  *) echo "unknown anchor: ${WHICH}" >&2; exit 2 ;;
esac

echo "=== B1 softprob K_TOP=${K} (${WHICH}) done ==="
