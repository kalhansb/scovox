#!/bin/bash
# NEW_EXPERIMENT_PLAN.md Phase 1 — K_TOP Pareto sweep on the post-SemDir
# substrate. K_TOP is compile-time (scovox_core/include/scovox/voxel.hpp:11
# `constexpr int K_TOP = …`), so this script edits that header in place,
# rebuilds, and batches all scenes/seqs at that K before moving on.
#
# Default K sweep: {1, 2, 3, 4, 6, 19}. 19 covers KITTI K_full and is a
# safe over-estimate for SceneNet (K_full=13) — the extra slots cost
# bytes per voxel but don't change accuracy.
#
# Env knobs (all overridable for smoke runs):
#   K_VALUES="2"           — single K, no rebuild loop  (smoke default)
#   SCENES_KITTI="08"      — KITTI sequence list
#   SCENES_SCENENET="0_223" — SceneNet trajectory list
#   N_KITTI=100  N_SCENENET=300
#
# Idempotency: per-(K, anchor) NPZs go in results/phase1_ktop_<DATE>/K<K>/<anchor>/.
# Cells with a non-empty NPZ are skipped.
#
# Restores K_TOP=2 at end (the production default).

set -o pipefail
WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
VOXEL_HPP="${WS}/src/robot_sw/distributed_mapping/scovox_core/include/scovox/voxel.hpp"
RES_ROOT="${EVAL_PKG}/results/phase1_ktop_sweep_$(date +%Y_%m_%d)"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
SCENENET_ROOT="${WS}/data/scenenet_val_layout"

# Defaults match NEW_EXPERIMENT_PLAN; smoke overrides via K_VALUES="2"
# (single K = no compile-loop wall-clock).
DEFAULT_KS=(1 2 3 4 6 19)
if [[ -n "${K_VALUES:-}" ]]; then IFS=' ' read -ra KS <<< "${K_VALUES}"; else KS=("${DEFAULT_KS[@]}"); fi
if [[ -n "${SCENES_KITTI:-}"   ]]; then IFS=' ' read -ra KITTI_SEQS <<<    "${SCENES_KITTI}";    else KITTI_SEQS=(08); fi
if [[ -n "${SCENES_SCENENET:-}" ]]; then IFS=' ' read -ra SCENENET_TRAJS <<< "${SCENES_SCENENET}"; else SCENENET_TRAJS=(0_223); fi
N_KITTI="${N_KITTI:-100}"
N_SCENENET="${N_SCENENET:-300}"
ORIG_K_TOP=2   # production default — restored on exit

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
mkdir -p "${RES_ROOT}"

cleanup_on_exit() {
  echo "━━ Restoring K_TOP=${ORIG_K_TOP} in voxel.hpp ━━"
  sed -i -E "s|^(constexpr int K_TOP = )[0-9]+;|\1${ORIG_K_TOP};|" "${VOXEL_HPP}"
  # Reset workspace to a known good state.
  ( cd "${WS}" && colcon build --packages-select scovox_core scovox_mapping \
      --cmake-args -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 ) || true
}
trap cleanup_on_exit EXIT

set_k_top() {
  local k="$1"
  echo "━━ Patching K_TOP = ${k} in ${VOXEL_HPP##*/} ━━"
  sed -i -E "s|^(constexpr int K_TOP = )[0-9]+;|\1${k};|" "${VOXEL_HPP}"
  grep -E "^constexpr int K_TOP" "${VOXEL_HPP}" | head -1
}

rebuild() {
  ( cd "${WS}" && colcon build --packages-select scovox_core scovox_mapping \
      --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3 )
  source "${WS}/install/setup.bash"
  export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"
}

run_kitti_cell() {
  local k="$1" seq="$2"
  local cell="${RES_ROOT}/K${k}/kitti_${seq}"
  mkdir -p "${cell}"
  local out="${cell}/scovox.npz"
  local log="${cell}/scovox.log"
  if [[ -s "${out}" ]]; then echo "[K${k} kitti_${seq}] skip"; return 0; fi
  echo "━━ K=${k} KITTI seq${seq} ━━"
  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2
  ros2 launch scovox_mapping semantickitti_eval.launch.py \
      robot_name:=ktop resolution:=0.10 \
      semantic_mode:=dirichlet semantic_occ_gate:=0.0 \
      range_decay_length:=50.0 w_occ:=6.0 w_free:=1.0 kappa0:=2.0 \
      carve_skip_occ_threshold:=0.4 evidence_saturation:=1000.0 \
      use_split:=true share_tsdf:=false fused_walker:=true \
      num_classes:=20 dirichlet_prior:=0.01 > "${log}" 2>&1 &
  local LPID=$!; sleep 4
  python3 -m scovox_eval.semantickitti_replay_node --ros-args \
      -p dataset_path:="${KITTI_ROOT}" -p sequence:=${seq#0} -p rate_hz:=0.5 \
      -p robot_name:=ktop -p max_range:=30.0 -p min_range:=1.0 \
      -p labels_subdir:=predictions -p n_scans:=${N_KITTI} \
      -p soft_prob_passthrough:=true >> "${log}" 2>&1
  # Wait briefly for tail integration to flush.
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

run_scenenet_cell() {
  local k="$1" traj="$2"
  local cell="${RES_ROOT}/K${k}/scenenet_${traj}"
  mkdir -p "${cell}"
  local out="${cell}/scovox.npz"
  local log="${cell}/scovox.log"
  if [[ -s "${out}" ]]; then echo "[K${k} scenenet_${traj}] skip"; return 0; fi
  echo "━━ K=${k} SceneNet ${traj} ━━"
  pgrep -af 'scovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2
  ros2 launch scovox_mapping scenenet_eval.launch.py \
      robot_name:=ktop resolution:=0.05 \
      semantic_mode:=dirichlet \
      use_split:=true share_tsdf:=false fused_walker:=true > "${log}" 2>&1 &
  local LPID=$!; sleep 4
  python3 -m scovox_eval.scenenet_replay_node --ros-args \
      -p data_root:="${SCENENET_ROOT}" -p sequence:="${traj}" \
      -p robot_name:=ktop -p rate_hz:=4.0 -p use_gt_labels:=true \
      -p start_frame:=0 -p n_scans:=${N_SCENENET} >> "${log}" 2>&1
  sleep 5
  timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/ktop/scovox_node/pointcloud -p output:="${out}" 2>&1 | tail -1
  kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  pgrep -af 'scovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1
}

echo "Phase 1 K_TOP sweep — Ks=[${KS[*]}] KITTI=[${KITTI_SEQS[*]}] SceneNet=[${SCENENET_TRAJS[*]}]"
for k in "${KS[@]}"; do
  set_k_top "${k}"
  rebuild
  for seq  in "${KITTI_SEQS[@]}";    do run_kitti_cell    "${k}" "${seq}";  done
  for traj in "${SCENENET_TRAJS[@]}"; do run_scenenet_cell "${k}" "${traj}"; done
done

# Restore voxel.hpp + rebuild is handled by the EXIT trap.
echo ""; echo "K_TOP sweep done. NPZs under: ${RES_ROOT}"
