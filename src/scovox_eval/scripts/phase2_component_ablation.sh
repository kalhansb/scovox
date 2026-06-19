#!/bin/bash
# NEW_EXPERIMENT_PLAN.md Phase 2 — component ablation on the post-SemDir
# substrate. **Expanded from the plan's 4 anchors to 6** by adding two
# more KITTI sequences (seq06, seq10) alongside the canonical seq08 —
# same coverage convention as Phase 4 h2h, which uses 5 KITTI seqs.
# With 3 modes that's 6 × 3 = 18 cells; run on use_split=true to
# exercise the SemDirMap path.
#
# Anchors:
#   - KITTI seq06 (added; suburban, fast traffic — different distribution)
#   - KITTI seq08 (plan's canonical anchor)
#   - KITTI seq10 (added; highway/long-straight, less semantic clutter)
#   - SceneNet 0_789  (high SCovox mIoU = 0.385)
#   - SceneNet 0_723  (low  SCovox mIoU = 0.269)
#   - SceneNet 0_485  (mid  SCovox mIoU = 0.303)
# Modes: dirichlet (D), majority_vote (MV), naive (NP)
#
# ⚠️ SceneNet uses GT one-hot labels → soft-prob reduces to hard. The
# "history matters" finding (D vs NP, D vs MV) still tests cleanly on
# both datasets. The "soft-prob recovers calibration signal" finding
# is KITTI-only (PolarSeg .topk).
#
# Env knobs:
#   SCENES=…       — override the anchor list (smoke: "kitti_08")
#   MODES=…        — override the mode list  (smoke: "dirichlet")
#   N_KITTI=…      — frames/scene for KITTI    (default 100)
#   N_SCENENET=…   — frames/scene for SceneNet (default 300)

set -o pipefail
WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RES_ROOT="${EVAL_PKG}/results/phase2_component_ablation_$(date +%Y_%m_%d)"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
SCENENET_ROOT="${WS}/data/scenenet_val_layout"

# Defaults match the plan; smoke runs override.
ANCHORS_DEFAULT=(kitti_06 kitti_08 kitti_10 scenenet_0_789 scenenet_0_723 scenenet_0_485)
MODES_DEFAULT=(dirichlet majority_vote naive)
if [[ -n "${SCENES:-}" ]]; then IFS=' ' read -ra ANCHORS <<< "${SCENES}"; else ANCHORS=("${ANCHORS_DEFAULT[@]}"); fi
if [[ -n "${MODES:-}"  ]]; then IFS=' ' read -ra MODE_LIST <<< "${MODES}";  else MODE_LIST=("${MODES_DEFAULT[@]}"); fi
N_KITTI="${N_KITTI:-100}"
N_SCENENET="${N_SCENENET:-300}"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RES_ROOT}"

# Mode-tag → semantic_mode launch arg.
mode_tag()  { case "$1" in dirichlet) echo "D";; majority_vote) echo "MV";; naive) echo "NP";; *) echo "$1";; esac; }

run_cell() {
  local anchor="$1" mode="$2"
  local tag="${anchor}_$(mode_tag "$mode")"
  local cell="${RES_ROOT}/${tag}"
  mkdir -p "${cell}"
  local out="${cell}/scovox.npz"
  local log="${cell}/scovox.log"
  if [[ -f "${out}" ]]; then
    echo "[${tag}] skip — NPZ already present"
    return 0
  fi

  echo ""; echo "━━ ${tag}  (semantic_mode=${mode}) ━━"
  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  sleep 2

  if [[ "${anchor}" == kitti_* ]]; then
    local seq="${anchor#kitti_}"          # "08"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=ablation \
        resolution:=0.10 \
        semantic_mode:="${mode}" \
        semantic_occ_gate:=0.0 \
        range_decay_length:=50.0 \
        w_occ:=6.0 w_free:=1.0 kappa0:=2.0 \
        carve_skip_occ_threshold:=0.4 \
        evidence_saturation:=1000.0 \
        use_split:=true share_tsdf:=false fused_walker:=true \
        num_classes:=20 dirichlet_prior:=0.01 \
        > "${log}" 2>&1 &
    local LPID=$!; sleep 4
    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="${KITTI_ROOT}" -p sequence:=${seq#0} \
        -p rate_hz:=0.5 -p robot_name:=ablation \
        -p max_range:=30.0 -p min_range:=1.0 \
        -p labels_subdir:=predictions \
        -p n_scans:=${N_KITTI} \
        -p soft_prob_passthrough:=false >> "${log}" 2>&1
    # Wait briefly for tail integration to flush.
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
  else
    # SceneNet anchor (e.g. scenenet_0_789).
    local seq="${anchor#scenenet_}"      # "0_789"
    ros2 launch scovox_mapping scenenet_eval.launch.py \
        robot_name:=ablation \
        resolution:=0.05 \
        semantic_mode:="${mode}" \
        use_split:=true share_tsdf:=false fused_walker:=true \
        > "${log}" 2>&1 &
    local LPID=$!; sleep 4
    python3 -m scovox_eval.scenenet_replay_node --ros-args \
        -p data_root:="${SCENENET_ROOT}" \
        -p sequence:="${seq}" \
        -p robot_name:=ablation -p rate_hz:=4.0 \
        -p use_gt_labels:=true \
        -p start_frame:=0 -p n_scans:=${N_SCENENET} >> "${log}" 2>&1
    sleep 5
    timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${out}" 2>&1 | tail -1
    kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
  fi

  pgrep -af 'scovox_mapping_node|semantickitti_replay_node|scenenet_replay_node|pointcloud_to_npz' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true; sleep 2; ros2 daemon start 2>/dev/null || true; sleep 1
}

echo "Phase 2 component ablation — anchors=[${ANCHORS[*]}] modes=[${MODE_LIST[*]}]"
for anchor in "${ANCHORS[@]}"; do
  for mode in "${MODE_LIST[@]}"; do
    run_cell "${anchor}" "${mode}"
  done
done

echo ""; echo "━━ Scoring ━━"
# Scoring is per-anchor (KITTI scorer for kitti_*, SceneNet scorer for scenenet_*).
# Emits a flat CSV: anchor,mode,miou.
CSV="${RES_ROOT}/phase2_summary.csv"
echo "anchor,mode,miou" > "${CSV}"
for anchor in "${ANCHORS[@]}"; do
  for mode in "${MODE_LIST[@]}"; do
    tag="${anchor}_$(mode_tag "$mode")"
    npz="${RES_ROOT}/${tag}/scovox.npz"
    [ -f "${npz}" ] || { echo "${anchor},${mode},NaN" >> "${CSV}"; continue; }
    miou=""
    if [[ "${anchor}" == kitti_* ]]; then
      seq="${anchor#kitti_}"
      # Stage NPZ into the seq subdir layout the scorer expects.
      stage="${RES_ROOT}/_score_${tag}/${seq}"; mkdir -p "${stage}"
      cp "${npz}" "${stage}/scovox.npz"
      # `--replay_to_yaml_lut` remaps PolarSeg's REPLAY class layout (lane-
      # marking at id 15, veg/trunk/terrain/pole +1 vs yaml learning_map)
      # back to yaml space before bucket-IoU. Without it 4 classes score 0
      # and KITTI mIoU drops by ~0.10 absolute. See kitti-miou-replay-bug
      # 2026-05-11 memo + Phase 0 smoke for the canonical invocation.
      miou=$(python3 "${EVAL_PKG}/scripts/eval_scovox_kitti_miou.py" \
          --kitti_root "${KITTI_ROOT}" \
          --npz_root "${RES_ROOT}/_score_${tag}" \
          --variant scovox --sequences "${seq}" --n_scans ${N_KITTI} \
          --semantic_kitti_yaml "${WS}/src/sem_seg_pipeline/polarseg/semantic-kitti.yaml" \
          --replay_to_yaml_lut \
          2>&1 | grep -oP 'mIoU=\K[0-9.]+' | head -1)
    else
      seq="${anchor#scenenet_}"
      miou=$(python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" \
          --pred_npz "${npz}" \
          --gt_npz "${SCENENET_ROOT}/train/${seq}/gt_5cm.npz" \
          --resolution 0.05 2>&1 | grep -oP 'mIoU\s*[:=]\s*\K[0-9.]+' | head -1)
    fi
    echo "${anchor},${mode},${miou:-NaN}" >> "${CSV}"
  done
done

echo ""; echo "Summary written to ${CSV}:"
cat "${CSV}"
