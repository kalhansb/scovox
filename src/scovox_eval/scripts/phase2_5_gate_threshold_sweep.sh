#!/bin/bash
# NEW_EXPERIMENT_PLAN.md Phase 2.5 — gate threshold sweep.
#
# Varies the PUBLISH-time occupancy gate (occupancy_vis_threshold) and
# scores mIoU at each threshold. Isolates the contribution of the
# occupancy gate itself, separate from K_TOP (Phase 1) and the update
# strategy (Phase 2).
#
# Note on naming: NEW_EXPERIMENT_PLAN.md Phase 2.5 spec calls this knob
# `dirichlet_min_p_occ`, but in the code that name is the INTEGRATION-
# time semantic-commit gate (semdir_map.cpp:applyHitUpdate). The publish-
# time / labelling-envelope gate — which is what Phase 2.5 actually
# ablates per the spec text ("the publish-time threshold for `is this
# voxel occupied?`") — is `occupancy_vis_threshold` / `min_occ_`
# (scovox_node.cpp:publishPointCloud). This script varies the latter.
#
# Design: integrate each anchor ONCE with occupancy_vis_threshold=0.0
# (publishes every observed voxel), then post-filter the NPZ at
# thresholds in {0.0, 0.3, 0.5, 0.7} and score each filtered slice.
# 2 anchors × 4 thresholds = 8 cells; integration cost ~5 min
# total, filter+score is sub-second per cell.
#
# Env knobs:
#   ANCHORS=…       override anchor list (default "kitti_08 scenenet_0_223")
#   THRESHOLDS=…    override threshold list (default "0.0 0.3 0.5 0.7")
#   N_KITTI=…       frames/scene for KITTI    (default 100)
#   N_SCENENET=…    frames/scene for SceneNet (default 300)

set -o pipefail
WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RES_ROOT="${EVAL_PKG}/results/phase2_5_gate_sweep_$(date +%Y_%m_%d)"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
SCENENET_ROOT="${WS}/data/scenenet_val_layout"
SKITTI_YAML="${WS}/src/sem_seg_pipeline/polarseg/semantic-kitti.yaml"

if [[ -n "${ANCHORS:-}"    ]]; then IFS=' ' read -ra ANCHOR_LIST  <<< "${ANCHORS}";    else ANCHOR_LIST=(kitti_08 scenenet_0_223); fi
if [[ -n "${THRESHOLDS:-}" ]]; then IFS=' ' read -ra THRESH_LIST  <<< "${THRESHOLDS}"; else THRESH_LIST=(0.0 0.3 0.5 0.7); fi
N_KITTI="${N_KITTI:-100}"
N_SCENENET="${N_SCENENET:-300}"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RES_ROOT}"

# --- Step 1: integrate each anchor ONCE with vis_threshold=0.0 ---
integrate_kitti() {
  local seq="$1"
  local out="${RES_ROOT}/kitti_${seq}_full/scovox.npz"
  local log="${RES_ROOT}/kitti_${seq}_full/scovox.log"
  mkdir -p "$(dirname "${out}")"
  [[ -s "${out}" ]] && { echo "[kitti_${seq}] skip — unthresholded NPZ exists"; return 0; }
  echo "━━ Integrating kitti_${seq} @ occupancy_vis_threshold=0.0 ━━"
  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2
  ros2 launch scovox_mapping semantickitti_eval.launch.py \
      robot_name:=gate25 resolution:=0.10 \
      semantic_mode:=dirichlet semantic_occ_gate:=0.0 \
      range_decay_length:=50.0 w_occ:=6.0 w_free:=1.0 kappa0:=2.0 \
      carve_skip_occ_threshold:=0.4 evidence_saturation:=1000.0 \
      use_split:=true share_tsdf:=false fused_walker:=true \
      num_classes:=20 dirichlet_prior:=0.01 \
      occupancy_vis_threshold:=0.0 \
      > "${log}" 2>&1 &
  local LPID=$!; sleep 4
  python3 -m scovox_eval.semantickitti_replay_node --ros-args \
      -p dataset_path:="${KITTI_ROOT}" -p sequence:=${seq#0} -p rate_hz:=0.5 \
      -p robot_name:=gate25 -p max_range:=30.0 -p min_range:=1.0 \
      -p labels_subdir:=predictions -p n_scans:=${N_KITTI} \
      -p soft_prob_passthrough:=true >> "${log}" 2>&1
  local waited=0
  while true; do
    local last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
    [ "${last:-0}" -ge $((N_KITTI - 5)) ] 2>/dev/null && { sleep 5; break; }
    sleep 3; waited=$((waited+3)); [ ${waited} -ge 600 ] && break
  done
  timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/gate25/scovox_node/pointcloud -p output:="${out}" 2>&1 | tail -1
  kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1
}

integrate_scenenet() {
  local traj="$1"
  local out="${RES_ROOT}/scenenet_${traj}_full/scovox.npz"
  local log="${RES_ROOT}/scenenet_${traj}_full/scovox.log"
  mkdir -p "$(dirname "${out}")"
  [[ -s "${out}" ]] && { echo "[scenenet_${traj}] skip — unthresholded NPZ exists"; return 0; }
  echo "━━ Integrating scenenet_${traj} @ occupancy_vis_threshold=0.0 ━━"
  pgrep -af 'scovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2
  ros2 launch scovox_mapping scenenet_eval.launch.py \
      robot_name:=gate25 resolution:=0.05 semantic_mode:=dirichlet \
      use_split:=true share_tsdf:=false fused_walker:=true \
      occupancy_vis_threshold:=0.0 > "${log}" 2>&1 &
  local LPID=$!; sleep 4
  python3 -m scovox_eval.scenenet_replay_node --ros-args \
      -p data_root:="${SCENENET_ROOT}" -p sequence:="${traj}" \
      -p robot_name:=gate25 -p rate_hz:=4.0 -p use_gt_labels:=true \
      -p start_frame:=0 -p n_scans:=${N_SCENENET} >> "${log}" 2>&1
  sleep 5
  timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/gate25/scovox_node/pointcloud -p output:="${out}" 2>&1 | tail -1
  kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  pgrep -af 'scovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1
}

for anchor in "${ANCHOR_LIST[@]}"; do
  case "${anchor}" in
    kitti_*)    integrate_kitti    "${anchor#kitti_}"   ;;
    scenenet_*) integrate_scenenet "${anchor#scenenet_}" ;;
    *) echo "WARN unknown anchor format: ${anchor}" ;;
  esac
done

# --- Step 2: post-filter each unthresholded NPZ at every threshold, score ---
echo ""; echo "━━ Scoring sweep ━━"
CSV="${RES_ROOT}/phase2_5_summary.csv"
echo "anchor,threshold,n_published,miou" > "${CSV}"

for anchor in "${ANCHOR_LIST[@]}"; do
  src="${RES_ROOT}/${anchor}_full/scovox.npz"
  [[ -s "${src}" ]] || { echo "  ${anchor}: SKIP — no unthresholded NPZ"; continue; }
  for t in "${THRESH_LIST[@]}"; do
    filt="${RES_ROOT}/${anchor}_t${t//./_}/scovox.npz"
    mkdir -p "$(dirname "${filt}")"
    # Filter the NPZ to keep only voxels with occupancy_prob >= t.
    python3 - "${src}" "${filt}" "${t}" <<'PY'
import sys, numpy as np
src, dst, t = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = np.load(src)
mask = d['occupancy_prob'] >= t
out = {k: d[k][mask] for k in d.keys()}
np.savez_compressed(dst, **out)
print(f"  filtered {int(mask.sum())}/{len(mask)} voxels @ t>={t}")
PY
    n_pub=$(python3 -c "import numpy as np; print(int(np.load('${filt}')['points'].shape[0]))")
    # Score per anchor type.
    if [[ "${anchor}" == kitti_* ]]; then
      seq="${anchor#kitti_}"
      stage="${RES_ROOT}/_score_${anchor}_t${t//./_}/${seq}"; mkdir -p "${stage}"
      cp "${filt}" "${stage}/scovox.npz"
      miou=$(python3 "${EVAL_PKG}/scripts/eval_scovox_kitti_miou.py" \
          --kitti_root "${KITTI_ROOT}" --npz_root "${RES_ROOT}/_score_${anchor}_t${t//./_}" \
          --variant scovox --sequences "${seq}" --n_scans ${N_KITTI} \
          --semantic_kitti_yaml "${SKITTI_YAML}" \
          --replay_to_yaml_lut 2>&1 | grep -oP 'mIoU=\K[0-9.]+' | head -1)
    else
      traj="${anchor#scenenet_}"
      miou=$(python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" \
          --pred_npz "${filt}" \
          --gt_npz "${SCENENET_ROOT}/train/${traj}/gt_5cm.npz" \
          --resolution 0.05 2>&1 | grep -oP 'mIoU\s*[:=]\s*\K[0-9.]+' | head -1)
    fi
    echo "${anchor},${t},${n_pub},${miou:-NaN}" >> "${CSV}"
    echo "  ${anchor} t=${t}: n_pub=${n_pub} mIoU=${miou:-NaN}"
  done
done

echo ""; echo "Summary at ${CSV}:"
column -s, -t < "${CSV}"
