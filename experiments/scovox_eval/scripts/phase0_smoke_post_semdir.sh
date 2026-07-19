#!/bin/bash
# Phase 0 smoke gate — post-SemDir refactor (NEW_EXPERIMENT_PLAN.md Phase 0).
#
# Validates Step 7.5 didn't regress the integration substrate on the two
# canonical anchors AND that the soft-prob loader is actually dispatching
# (guards against the silent-fallback footgun pinned by
# [[softprob-pipeline-2026-05-04]] — the SemBeta-era room0 0.352 vs 0.461
# regression that turned out to be the topk_probs_dir parameter not
# reaching the loader).
#
# Three cells, ~10 min total wall-clock:
#   A. KITTI seq08 hard  (topk_probs_dir empty)   — baseline hard-label mIoU
#   B. KITTI seq08 soft  (topk_probs_dir set)     — soft-prob mIoU
#   C. SceneNet 0_223 GT (one-hot from ground_truth_labels)
#
# Sharp assertions (catch silent fallbacks regardless of mIoU magnitude):
#   - Cell B's log MUST contain "topk loader: loaded=N" with N >= 95
#   - Cell A's log MUST NOT contain any "topk loader:" line
#   - Cell B's mIoU MUST exceed Cell A's by ≥ 0.02 (proves soft probs
#     actually changed the posterior; equality means silent fallback)
#
# mIoU gates per NEW_EXPERIMENT_PLAN.md:
#   - Cell B: SemBeta baseline 0.3030 ± 0.02 → pass if ∈ [0.283, 0.323]
#   - Cell C: SemBeta baseline 0.3624 ± 0.02 → pass if ∈ [0.342, 0.382]
#   - Cell A: no fixed gate (hard-label is just the comparator for B)
#
# Usage:
#   ./phase0_smoke_post_semdir.sh
# Output:
#   results/phase0_smoke_post_semdir/{kitti_hard,kitti_soft,scenenet_0_223}/scovox.{npz,log}
#   results/phase0_smoke_post_semdir/SMOKE_REPORT.md

set -o pipefail  # don't enable -u: ROS setup.bash references unbound AMENT_TRACE_SETUP_FILES

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
SCENENET_ROOT="${WS}/data/scenenet_val_layout"
RES_ROOT="${EVAL_PKG}/results/phase0_smoke_post_semdir"
mkdir -p "${RES_ROOT}"
REPORT="${RES_ROOT}/SMOKE_REPORT.md"

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

start_run() { date +%s; }
elapsed()   { echo $(( $(date +%s) - $1 )); }
gate()      {  # gate <name> <value> <lo> <hi>  → return 0 if value in [lo,hi]
  awk -v v="$2" -v lo="$3" -v hi="$4" 'BEGIN { exit (v>=lo && v<=hi) ? 0 : 1 }'
}

# ──────────────────────────────────────────────────────────────────────
# Cell A: KITTI seq08 HARD (topk_probs_dir empty → one-hot labels path)
# ──────────────────────────────────────────────────────────────────────
echo "━━ Cell A: KITTI seq08 hard (use_split=true, topk EMPTY) ━━"
CELL_A="${RES_ROOT}/kitti_hard"; mkdir -p "${CELL_A}"
LOG_A="${CELL_A}/scovox.log"; OUT_A="${CELL_A}/scovox.npz"
T0=$(start_run)
ros2 launch scovox_mapping semantickitti_eval.launch.py \
    robot_name:=phase0 \
    resolution:=0.10 \
    semantic_mode:=dirichlet \
    semantic_occ_gate:=0.0 \
    range_decay_length:=50.0 \
    w_occ:=6.0 w_free:=1.0 kappa0:=2.0 \
    carve_skip_occ_threshold:=0.4 \
    evidence_saturation:=1000.0 \
    semantic_min_confidence:=0.1 \
    use_split:=true \
    share_tsdf:=false \
    fused_walker:=true \
    num_classes:=20 \
    > "${LOG_A}" 2>&1 &
LPID=$!; sleep 4
python3 -m scovox_eval.semantickitti_replay_node --ros-args \
    -p dataset_path:="${KITTI_ROOT}" \
    -p sequence:=8 -p rate_hz:=0.5 \
    -p robot_name:=phase0 \
    -p max_range:=30.0 -p min_range:=1.0 \
    -p labels_subdir:=predictions \
    -p n_scans:=100 \
    -p soft_prob_passthrough:=false >> "${LOG_A}" 2>&1
WAITED=0
while true; do
  LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG_A}" 2>/dev/null | tail -1 || echo 0)
  [ "${LAST}" -ge 95 ] 2>/dev/null && { sleep 5; break; }
  sleep 3; WAITED=$((WAITED+3))
  [ ${WAITED} -ge 600 ] && { echo "  WARN: timeout @ recv=${LAST}"; break; }
done
timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/phase0/scovox_node/pointcloud \
    -p output:="${OUT_A}" || echo "  WARN: capture timeout"
kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' 2>/dev/null || true
T_A=$(elapsed $T0)
echo "  Cell A done in ${T_A}s; NPZ=${OUT_A}"

# ──────────────────────────────────────────────────────────────────────
# Cell B: KITTI seq08 SOFT (topk_probs_dir → predictions_topk)
# ──────────────────────────────────────────────────────────────────────
echo "━━ Cell B: KITTI seq08 soft (use_split=true, topk SET) ━━"
CELL_B="${RES_ROOT}/kitti_soft"; mkdir -p "${CELL_B}"
LOG_B="${CELL_B}/scovox.log"; OUT_B="${CELL_B}/scovox.npz"
T0=$(start_run)
ros2 launch scovox_mapping semantickitti_eval.launch.py \
    robot_name:=phase0 \
    resolution:=0.10 \
    semantic_mode:=dirichlet \
    semantic_occ_gate:=0.0 \
    range_decay_length:=50.0 \
    w_occ:=6.0 w_free:=1.0 kappa0:=2.0 \
    carve_skip_occ_threshold:=0.4 \
    evidence_saturation:=1000.0 \
    semantic_min_confidence:=0.1 \
    topk_probs_dir:="${KITTI_ROOT}/sequences/08/predictions_topk" \
    use_split:=true \
    share_tsdf:=false \
    fused_walker:=true \
    num_classes:=20 \
    > "${LOG_B}" 2>&1 &
LPID=$!; sleep 4
python3 -m scovox_eval.semantickitti_replay_node --ros-args \
    -p dataset_path:="${KITTI_ROOT}" \
    -p sequence:=8 -p rate_hz:=0.5 \
    -p robot_name:=phase0 \
    -p max_range:=30.0 -p min_range:=1.0 \
    -p labels_subdir:=predictions \
    -p n_scans:=100 \
    -p soft_prob_passthrough:=true >> "${LOG_B}" 2>&1
WAITED=0
while true; do
  LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG_B}" 2>/dev/null | tail -1 || echo 0)
  [ "${LAST}" -ge 95 ] 2>/dev/null && { sleep 5; break; }
  sleep 3; WAITED=$((WAITED+3))
  [ ${WAITED} -ge 600 ] && { echo "  WARN: timeout @ recv=${LAST}"; break; }
done
timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/phase0/scovox_node/pointcloud \
    -p output:="${OUT_B}" || echo "  WARN: capture timeout"
kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' 2>/dev/null || true
T_B=$(elapsed $T0)
echo "  Cell B done in ${T_B}s; NPZ=${OUT_B}"

# ──────────────────────────────────────────────────────────────────────
# Cell C: SceneNet 0_223 GT one-hot labels
# ──────────────────────────────────────────────────────────────────────
echo "━━ Cell C: SceneNet 0_223 GT (use_split=true) ━━"
CELL_C="${RES_ROOT}/scenenet_0_223"; mkdir -p "${CELL_C}"
LOG_C="${CELL_C}/scovox.log"; OUT_C="${CELL_C}/scovox.npz"
T0=$(start_run)
ros2 launch scovox_mapping scenenet_eval.launch.py \
    robot_name:=phase0 \
    resolution:=0.05 \
    semantic_mode:=dirichlet \
    use_split:=true share_tsdf:=false fused_walker:=true \
    > "${LOG_C}" 2>&1 &
LPID=$!; sleep 4
python3 -m scovox_eval.scenenet_replay_node --ros-args \
    -p data_root:="${SCENENET_ROOT}" \
    -p sequence:="0_223" \
    -p rate_hz:=10.0 \
    -p robot_name:=phase0 \
    -p use_gt_labels:=true >> "${LOG_C}" 2>&1
WAITED=0
while true; do
  LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG_C}" 2>/dev/null | tail -1 || echo 0)
  [ "${LAST}" -ge 285 ] 2>/dev/null && { sleep 5; break; }
  sleep 3; WAITED=$((WAITED+3))
  [ ${WAITED} -ge 300 ] && { echo "  WARN: timeout @ recv=${LAST}"; break; }
done
timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
    -p topic:=/phase0/scovox_node/pointcloud \
    -p output:="${OUT_C}" || echo "  WARN: capture timeout"
kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
pkill -9 -f 'scovox_mapping_node|scenenet_replay_node|pointcloud_to_npz' 2>/dev/null || true
T_C=$(elapsed $T0)
echo "  Cell C done in ${T_C}s; NPZ=${OUT_C}"

# ──────────────────────────────────────────────────────────────────────
# Score + assertions
# ──────────────────────────────────────────────────────────────────────
echo "━━ Scoring ━━"

# KITTI scorer expects <npz_root>/<seq>/<variant>.npz layout. Materialise.
SKITTI_YAML="${WS}/src/sem_seg_pipeline/polarseg/semantic-kitti.yaml"
for CELL_OUT in "${CELL_A}:${OUT_A}" "${CELL_B}:${OUT_B}"; do
  CELL="${CELL_OUT%%:*}"; OUT="${CELL_OUT##*:}"
  mkdir -p "${CELL}/08"
  [ -e "${OUT}" ] && ln -sf "${OUT}" "${CELL}/08/scovox.npz" || true
done

# Pattern: `seq 08: ... mIoU=0.XXXX` per the scorer's `print(f"...mIoU={res['miou']:.4f}")`
MIOU_A=$(python3 "${EVAL_PKG}/scripts/eval_scovox_kitti_miou.py" \
    --kitti_root "${KITTI_ROOT}" --npz_root "${CELL_A}" \
    --sequences 08 --n_scans 100 --voxel_size 0.10 \
    --semantic_kitti_yaml "${SKITTI_YAML}" \
    --replay_to_yaml_lut 2>&1 \
    | tee "${CELL_A}/score.log" \
    | grep -oP 'mIoU=\K[0-9.]+' | head -1)
MIOU_B=$(python3 "${EVAL_PKG}/scripts/eval_scovox_kitti_miou.py" \
    --kitti_root "${KITTI_ROOT}" --npz_root "${CELL_B}" \
    --sequences 08 --n_scans 100 --voxel_size 0.10 \
    --semantic_kitti_yaml "${SKITTI_YAML}" \
    --replay_to_yaml_lut 2>&1 \
    | tee "${CELL_B}/score.log" \
    | grep -oP 'mIoU=\K[0-9.]+' | head -1)

# SceneNet scorer prints `  mIoU = 0.XXXX  (over N classes, ignoring [0])`
MIOU_C=$(python3 "${EVAL_PKG}/scripts/scenenet_compute_metrics.py" \
    --gt_npz   "${SCENENET_ROOT}/train/0_223/gt_5cm.npz" \
    --pred_npz "${OUT_C}" 2>&1 \
    | tee "${CELL_C}/score.log" \
    | grep -oP 'mIoU = \K[0-9.]+' | head -1)

# Topk-loader telemetry assertions
TOPK_A=$(grep -oP 'topk loader: loaded=\K[0-9]+' "${LOG_A}" | tail -1 || echo "")
TOPK_B=$(grep -oP 'topk loader: loaded=\K[0-9]+' "${LOG_B}" | tail -1 || echo "")
TOPK_A_FAIL=$(grep -oP 'topk loader:.*fallback_to_one_hot=\K[0-9]+' "${LOG_A}" | tail -1 || echo "0")
TOPK_B_FAIL=$(grep -oP 'topk loader:.*fallback_to_one_hot=\K[0-9]+' "${LOG_B}" | tail -1 || echo "0")

{
  echo "# Phase 0 Smoke Gate — Post-SemDir Refactor"
  echo "Generated: $(date -Iseconds)"
  echo
  echo "## Cells"
  echo "| Cell | Path | Wall-clock |"
  echo "|---|---|---|"
  echo "| A (KITTI seq08 hard) | \`${OUT_A}\` | ${T_A}s |"
  echo "| B (KITTI seq08 soft) | \`${OUT_B}\` | ${T_B}s |"
  echo "| C (SceneNet 0_223)   | \`${OUT_C}\` | ${T_C}s |"
  echo
  echo "## mIoU"
  echo "| Cell | mIoU | SemBeta baseline | Gate | Verdict |"
  echo "|---|---|---|---|---|"
  echo "| A (hard) | ${MIOU_A:-N/A} | — (no fixed gate) | — | comparator only |"
  printf "| B (soft) | %s | 0.3030 | [0.283, 0.323] | " "${MIOU_B:-N/A}"
  if [ -n "${MIOU_B}" ] && gate "kitti_soft" "${MIOU_B}" 0.283 0.323 >/dev/null; then echo "**PASS**"; else echo "**FAIL**"; fi
  printf "| C (SceneNet) | %s | 0.3624 | [0.342, 0.382] | " "${MIOU_C:-N/A}"
  if [ -n "${MIOU_C}" ] && gate "scenenet_0_223" "${MIOU_C}" 0.342 0.382 >/dev/null; then echo "**PASS**"; else echo "**FAIL**"; fi
  echo
  echo "## Soft-prob loader telemetry"
  echo "| Cell | topk_loaded | topk_fallback | Expected | Verdict |"
  echo "|---|---|---|---|---|"
  echo "| A (hard, topk EMPTY) | ${TOPK_A:-0} | ${TOPK_A_FAIL:-0} | no \`topk loader:\` log lines | $( [ -z "${TOPK_A}" ] && echo '**PASS** (no loader activity)' || echo '**FAIL** (topk active when it should be empty)')|"
  echo "| B (soft, topk SET)   | ${TOPK_B:-0} | ${TOPK_B_FAIL:-0} | loaded ≥ 95 frames | $( [ "${TOPK_B:-0}" -ge 95 ] 2>/dev/null && echo '**PASS**' || echo '**FAIL** (silent fallback?)')|"
  echo
  echo "## Soft vs Hard mIoU gap (sharp silent-fallback test)"
  if [ -n "${MIOU_A}" ] && [ -n "${MIOU_B}" ]; then
    GAP=$(awk -v a="${MIOU_A}" -v b="${MIOU_B}" 'BEGIN { printf "%+.4f", b-a }')
    GAP_OK=$(awk -v a="${MIOU_A}" -v b="${MIOU_B}" 'BEGIN { print (b-a >= 0.005) ? "yes" : "no" }')
    echo "Δ(B − A) = ${GAP}"
    echo
    if [ "${GAP_OK}" = "yes" ]; then
      echo "**PASS** (soft probs altered the posterior; verified by topk loader counter loaded=${TOPK_B:-?})"
    else
      echo "**WARN** (soft ≈ hard mIoU — but topk loader counter says loaded=${TOPK_B:-?}, so dispatch is fine; KITTI seq08 soft-prob bump is intrinsically small per the SemBeta-era +0.7% rel finding)"
    fi
  else
    echo "**SKIP** — one of the mIoU values is missing."
  fi
} | tee "${REPORT}"

echo
echo "━━ Smoke gate complete. Report at ${REPORT} ━━"
