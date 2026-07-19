#!/bin/bash
# Phase 1 KITTI K_TOP sweep — PolarSeg SOFT input (predictions_topk/).
#
# The original phase1_ktop_sweep.sh accidentally ran KITTI with HARD labels
# because it set `soft_prob_passthrough:=true` on the replay node but did
# NOT set `topk_probs_dir:=...` on scovox_node — both are required for
# soft-prob ingestion per [[project-softprob-pipeline-2026-05-04]]. The
# datasets table says "PolarSeg soft" but the executed numbers were hard.
#
# This script reruns the 30 KITTI cells (5 seqs × 6 K) with proper soft
# ingestion, writing into a parallel results subtree so the hard cells
# stay intact for reference.
#
# Output: results/phase1_ktop_sweep_2026_05_14/K{N}/kitti_{seq}_soft/scovox.npz
# (parallel naming: same K{N} dir but `_soft` suffix on the anchor name)
#
# K_TOP is compile-time → sed-edits voxel.hpp + rebuilds per K. EXIT trap
# restores K=2.

set -o pipefail
WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
VOXEL_HPP="${WS}/src/robot_sw/distributed_mapping/scovox_core/include/scovox/voxel.hpp"
RES_ROOT="${EVAL_PKG}/results/phase1_ktop_sweep_2026_05_14"
KITTI_ROOT="${WS}/data/semantickitti/dataset"

KS=(1 2 3 4 6 19)
KITTI_SEQS=(06 07 08 09 10)
N_KITTI=100
ORIG_K_TOP=2

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
# scovox_eval is a pip-editable package whose install was previously broken
# (egg-link pointed at /home/kalhan/Projects/... — capital P — but actual ws
# is /home/kalhan/projects/...). Bypass pip by injecting the source dir
# into PYTHONPATH directly so `python3 -m scovox_eval.*` resolves.
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH}"
source /opt/ros/humble/setup.bash
mkdir -p "${RES_ROOT}"

cleanup_on_exit() {
  echo "━━ Restoring K_TOP=${ORIG_K_TOP} in voxel.hpp ━━"
  sed -i -E "s|^(constexpr int K_TOP = )[0-9]+;|\1${ORIG_K_TOP};|" "${VOXEL_HPP}"
  ( cd "${WS}" && colcon build --packages-select scovox_core scovox_mapping \
      --cmake-args -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 ) || true
}
trap cleanup_on_exit EXIT

set_k_top() {
  local k="$1"
  echo "━━ Patching K_TOP = ${k} in voxel.hpp ━━"
  sed -i -E "s|^(constexpr int K_TOP = )[0-9]+;|\1${k};|" "${VOXEL_HPP}"
  grep -E "^constexpr int K_TOP" "${VOXEL_HPP}" | head -1
}

rebuild() {
  ( cd "${WS}" && colcon build --packages-select scovox_core scovox_mapping \
      --cmake-args -DCMAKE_BUILD_TYPE=Release ) || return 1
  source "${WS}/install/setup.bash"
}

run_kitti_soft_cell() {
  local k="$1" seq="$2"
  local cell="${RES_ROOT}/K${k}/kitti_${seq}_soft"
  mkdir -p "${cell}"
  local out="${cell}/scovox.npz"
  local log="${cell}/scovox.log"
  if [[ -s "${out}" ]]; then echo "[K${k} kitti_${seq}_soft] skip"; return 0; fi
  echo "━━ K=${k} KITTI seq${seq} (SOFT, predictions_topk/) ━━"
  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2

  local topk_dir="${KITTI_ROOT}/sequences/${seq}/predictions_topk"
  ros2 launch scovox_mapping semantickitti_eval.launch.py \
      robot_name:=ktop resolution:=0.10 \
      semantic_mode:=dirichlet semantic_occ_gate:=0.0 \
      range_decay_length:=50.0 w_occ:=6.0 w_free:=1.0 kappa0:=2.0 \
      carve_skip_occ_threshold:=0.4 evidence_saturation:=1000.0 \
      use_split:=true share_tsdf:=false fused_walker:=true \
      num_classes:=20 dirichlet_prior:=0.01 \
      topk_probs_dir:="${topk_dir}" > "${log}" 2>&1 &
  local LPID=$!; sleep 4
  python3 -m scovox_eval.semantickitti_replay_node --ros-args \
      -p dataset_path:="${KITTI_ROOT}" -p sequence:=${seq#0} -p rate_hz:=0.5 \
      -p robot_name:=ktop -p max_range:=30.0 -p min_range:=1.0 \
      -p labels_subdir:=predictions -p n_scans:=${N_KITTI} \
      -p soft_prob_passthrough:=true >> "${log}" 2>&1
  local waited=0
  while true; do
    local last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
    [ "${last:-0}" -ge $((N_KITTI - 5)) ] 2>/dev/null && { sleep 5; break; }
    sleep 3; waited=$((waited+3)); [ ${waited} -ge 600 ] && break
  done
  timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/ktop/scovox_node/pointcloud -p output:="${out}" 2>&1 | tail -1
  kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1
}

echo "Phase 1 KITTI SOFT-prob sweep: Ks=[${KS[*]}] seqs=[${KITTI_SEQS[*]}] (n=${N_KITTI})"
for k in "${KS[@]}"; do
  set_k_top "${k}"
  rebuild || { echo "BUILD FAIL at K=${k}"; exit 1; }
  for seq in "${KITTI_SEQS[@]}"; do
    run_kitti_soft_cell "${k}" "${seq}"
  done
done
echo ""; echo "KITTI SOFT sweep done. NPZs under: ${RES_ROOT} (K*/kitti_*_soft/)"
