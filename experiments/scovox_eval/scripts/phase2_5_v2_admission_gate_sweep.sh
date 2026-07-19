#!/bin/bash
# NEW_EXPERIMENT_PLAN.md Phase 2.5-v2 — integration-time admission gate sweep.
#
# Companion to the publish-time Phase 2.5 (which returned a null result
# because low-p_occ voxels carry class=Unknown which the scorer ignores).
# This sweeps the integration-time gate `dirichlet_min_p_occ` instead,
# which actually controls which voxels get a real class commit.
#
# At t=0.0 every hit commits a class regardless of occupancy posterior
# (SLIM-VDB-like envelope). Predicts mIoU drops monotonically as threshold
# decreases, with the largest drop between 0.3 and 0.0.
#
# Anchors: KITTI seq08 (LiDAR + PolarSeg soft) and SceneNet 0_223 (RGB-D
# GT labels). 4 thresholds × 2 anchors = 8 cells.
#
# Wall-clock budget: ~50 min (each cell needs a fresh integration, unlike
# Phase 2.5 which reused the same SemDirMap state through different
# publish thresholds).

set -o pipefail
WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RES_ROOT="${EVAL_PKG}/results/phase2_5_v2_admission_gate_$(date +%Y_%m_%d)"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
SCENENET_ROOT="${WS}/data/scenenet_val_layout"

THRESHOLDS=(0.0 0.3 0.5 0.7)
N_KITTI=100
N_SCENENET=300

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"
mkdir -p "${RES_ROOT}"

run_kitti() {
  local thresh="$1"
  local cell="${RES_ROOT}/kitti_08_t${thresh//./_}"
  mkdir -p "${cell}"
  local out="${cell}/scovox.npz"
  local log="${cell}/scovox.log"
  if [[ -s "${out}" ]]; then echo "[kitti_08 t=${thresh}] skip — NPZ present"; return 0; fi

  echo ""; echo "━━ KITTI seq08 dirichlet_min_p_occ=${thresh} ━━"
  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2

  ros2 launch scovox_mapping semantickitti_eval.launch.py \
      robot_name:=ablation resolution:=0.10 semantic_mode:=dirichlet \
      use_split:=true share_tsdf:=false fused_walker:=true \
      num_classes:=20 dirichlet_prior:=0.01 dirichlet_min_p_occ:="${thresh}" \
      semantic_occ_gate:=0.0 range_decay_length:=50.0 \
      w_occ:=6.0 w_free:=1.0 kappa0:=2.0 \
      carve_skip_occ_threshold:=0.4 evidence_saturation:=1000.0 \
      > "${log}" 2>&1 &
  local LPID=$!; sleep 4
  python3 -m scovox_eval.semantickitti_replay_node --ros-args \
      -p dataset_path:="${KITTI_ROOT}" -p sequence:=8 \
      -p rate_hz:=0.5 -p robot_name:=ablation \
      -p max_range:=30.0 -p min_range:=1.0 \
      -p labels_subdir:=predictions \
      -p n_scans:=${N_KITTI} \
      -p soft_prob_passthrough:=false >> "${log}" 2>&1
  # Wait for tail integration to flush.
  local waited=0
  while true; do
    local last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
    [ "${last:-0}" -ge $((N_KITTI - 5)) ] 2>/dev/null && { sleep 5; break; }
    sleep 3; waited=$((waited+3)); [ ${waited} -ge 600 ] && break
  done
  timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/ablation/scovox_node/pointcloud \
      -p output:="${out}" 2>&1 | tail -1
  kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1
}

run_scenenet() {
  local thresh="$1"
  local cell="${RES_ROOT}/scenenet_0_223_t${thresh//./_}"
  mkdir -p "${cell}"
  local out="${cell}/scovox.npz"
  local log="${cell}/scovox.log"
  if [[ -s "${out}" ]]; then echo "[scenenet_0_223 t=${thresh}] skip — NPZ present"; return 0; fi

  echo ""; echo "━━ SceneNet 0_223 dirichlet_min_p_occ=${thresh} ━━"
  pgrep -af 'scovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true; sleep 2

  ros2 launch scovox_mapping scenenet_eval.launch.py \
      robot_name:=ablation resolution:=0.05 semantic_mode:=dirichlet \
      use_split:=true share_tsdf:=false fused_walker:=true \
      dirichlet_min_p_occ:="${thresh}" \
      > "${log}" 2>&1 &
  local LPID=$!; sleep 4
  python3 -m scovox_eval.scenenet_replay_node --ros-args \
      -p data_root:="${SCENENET_ROOT}" -p sequence:=0_223 \
      -p robot_name:=ablation -p rate_hz:=4.0 \
      -p use_gt_labels:=true \
      -p start_frame:=0 -p n_scans:=${N_SCENENET} >> "${log}" 2>&1
  sleep 5
  timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/ablation/scovox_node/pointcloud \
      -p output:="${out}" 2>&1 | tail -1
  kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  pgrep -af 'scovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1
}

echo "Phase 2.5-v2 integration-time gate sweep: thresholds=[${THRESHOLDS[*]}]"
for t in "${THRESHOLDS[@]}"; do run_kitti "${t}"; done
for t in "${THRESHOLDS[@]}"; do run_scenenet "${t}"; done

echo ""; echo "━━ Scoring ━━"
CSV="${RES_ROOT}/phase2_5_v2_summary.csv"
echo "anchor,threshold,n_voxels,n_unknown,n_labelled,miou" > "${CSV}"

count_classified() {
  # Count voxels with semantic_class != 0 (Unknown). The gate's whole
  # point is to control how many voxels get a real class commit.
  local npz="$1"
  python3 -c "
import numpy as np
d = np.load('${npz}')
sc = d['semantic_class']
n_total = len(sc)
n_unk = int((sc == 0).sum())
print(f'{n_total},{n_unk},{n_total - n_unk}')
"
}

for t in "${THRESHOLDS[@]}"; do
  cell="${RES_ROOT}/kitti_08_t${t//./_}"
  npz="${cell}/scovox.npz"
  if [[ -s "${npz}" ]]; then
    stage="${RES_ROOT}/_score_kitti_08_t${t//./_}/08"; mkdir -p "${stage}"
    cp "${npz}" "${stage}/scovox.npz"
    miou=$(python3 "${EVAL_PKG}/scripts/eval_scovox_kitti_miou.py" \
        --kitti_root "${KITTI_ROOT}" \
        --npz_root "${RES_ROOT}/_score_kitti_08_t${t//./_}" \
        --variant scovox --sequences 08 --n_scans ${N_KITTI} \
        --semantic_kitti_yaml "${WS}/src/sem_seg_pipeline/polarseg/semantic-kitti.yaml" \
        --replay_to_yaml_lut 2>&1 | grep -oP 'mIoU=\K[0-9.]+' | head -1)
    counts=$(count_classified "${npz}")
    echo "kitti_08,${t},${counts},${miou:-NaN}" | tee -a "${CSV}"
  else
    echo "kitti_08,${t},NaN,NaN,NaN,NaN" >> "${CSV}"
  fi
done

for t in "${THRESHOLDS[@]}"; do
  cell="${RES_ROOT}/scenenet_0_223_t${t//./_}"
  npz="${cell}/scovox.npz"
  if [[ -s "${npz}" ]]; then
    miou=$(python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" \
        --pred_npz "${npz}" \
        --gt_npz "${SCENENET_ROOT}/train/0_223/gt_5cm.npz" \
        --resolution 0.05 2>&1 | grep -oP 'mIoU\s*[:=]\s*\K[0-9.]+' | head -1)
    counts=$(count_classified "${npz}")
    echo "scenenet_0_223,${t},${counts},${miou:-NaN}" | tee -a "${CSV}"
  else
    echo "scenenet_0_223,${t},NaN,NaN,NaN,NaN" >> "${CSV}"
  fi
done

echo ""; echo "Summary:"
column -s, -t < "${CSV}"
