#!/bin/bash
# E1.3 head-to-head — split-grid SCovox vs SLIM-VDB.
#
# Per Step-12 cell sheet:
#   8 Replica scenes (M2F soft, K=2)
#   5 KITTI sequences (PolarSeg soft, K=2)
#   use_split:=true, share_tsdf:=false
#
# Per cell capture:
#   results/e13_split_2026_05_08/{replica,kitti}/<scene>/
#     scovox.npz, scovox_run.log, summary.txt
#
# After all 13 cells: scoring + summary CSV with mIoU / FPS / TsdfMap MB /
# SemBetaMap MB. Compares TsdfMap bytes against SLIM-VDB bytes per the
# E1.3 spec ("expect within 5%").
#
# Idempotent: cells whose scovox.npz exists are skipped.
# Sequential by user preference (feedback_sequential_execution.md).
set -o pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
DATE_TAG="${DATE_TAG_OVERRIDE:-$(date +%Y_%m_%d)}"
RES_ROOT="${EVAL_PKG}/results/e13_split_${DATE_TAG}"

REPLICA_SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
KITTI_SEQS=(06 07 08 09 10)

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RES_ROOT}/replica" "${RES_ROOT}/kitti"

# ---- Helpers ----
extract_metrics() {
  # Pull frame_ms mean, [memSplit] tsdf/sembeta MB from a scovox_run.log.
  # Prints: "<frame_ms_mean>,<tsdf_mb>,<sembeta_mb>,<n_frames>"
  local log="$1"
  local frame_mean tsdf_mb sembeta_mb n_frames
  frame_mean=$(grep -oP 'frame_ms=\K[0-9]+\.[0-9]+' "${log}" 2>/dev/null \
               | awk '{s+=$1; n++} END{if(n>0) printf "%.1f", s/n; else print ""}')
  n_frames=$(grep -c 'frame_ms=' "${log}" 2>/dev/null || echo 0)
  # Last [memSplit] line is the steady-state grid size.
  tsdf_mb=$(grep -oP 'tsdf_grid_mb=\K[0-9]+\.[0-9]+' "${log}" 2>/dev/null | tail -1)
  sembeta_mb=$(grep -oP 'sembeta_grid_mb=\K[0-9]+\.[0-9]+' "${log}" 2>/dev/null | tail -1)
  echo "${frame_mean:-NA},${tsdf_mb:-NA},${sembeta_mb:-NA},${n_frames}"
}

# ---- Replica cells ----
for scene in "${REPLICA_SCENES[@]}"; do
  cell_dir="${RES_ROOT}/replica/${scene}"
  npz="${cell_dir}/scovox.npz"
  mkdir -p "${cell_dir}"
  if [[ -f "${npz}" ]]; then
    echo "[replica ${scene}] skip — npz exists"
    continue
  fi
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  E1.3 replica ${scene}  (use_split=true, K=2 soft)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  topk="${WS}/data/replica_niceslam/${scene}/semantic_m2f_topk"
  probs="${WS}/data/replica_niceslam/${scene}/semantic_m2f_ade_probs"
  if [[ ! -d "${topk}" ]] || [[ -z "$(ls -A "${topk}" 2>/dev/null)" ]]; then
    echo "  [topk] regenerating from ${probs}"
    python3 "${EVAL_PKG}/scripts/topk_npz_to_bin.py" \
        --src_dir "${probs}" --dst_dir "${topk}" --mode image 2>&1 | tail -2
  fi

  SCENE_OVERRIDE="${scene}" \
  USE_SPLIT_OVERRIDE=true \
  SHARE_TSDF_OVERRIDE=false \
  N_SCANS_OVERRIDE=2000 \
  RESULTS_ROOT_OVERRIDE="${cell_dir}/_run" \
    bash "${EVAL_PKG}/scripts/scovox_softprob_replica_room0.sh" \
       > "${cell_dir}/wrapper.log" 2>&1 || echo "  [warn] wrapper exited non-zero"

  # The soft-prob wrapper stores under <RR>/baseline_softprob; lift NPZ + log up.
  if [[ -f "${cell_dir}/_run/baseline_softprob/scovox.npz" ]]; then
    mv "${cell_dir}/_run/baseline_softprob/scovox.npz" "${npz}"
    mv "${cell_dir}/_run/baseline_softprob/scovox_run.log" "${cell_dir}/scovox_run.log"
  fi
  rm -rf "${cell_dir}/_run"

  if [[ -f "${npz}" ]]; then
    metrics=$(extract_metrics "${cell_dir}/scovox_run.log")
    echo "  [done] ${scene}: $(du -h "${npz}" | cut -f1)  metrics=${metrics}"
    echo "${metrics}" > "${cell_dir}/perf.csv"
  else
    echo "  [warn] ${scene}: no npz produced"
  fi

  # Free disk for next scene (Replica topk dirs are ~29 GB each — without
  # rotation the second cell ENOSPCs on regen). The original "spare room0"
  # exclusion was wrong; topk is always re-derivable from ade_probs/.
  if [[ -d "${topk}" ]]; then
    rm -rf "${topk}"
  fi
done

# ---- KITTI cells ----
for seq in "${KITTI_SEQS[@]}"; do
  cell_dir="${RES_ROOT}/kitti/${seq}"
  npz="${cell_dir}/scovox.npz"
  mkdir -p "${cell_dir}"
  if [[ -f "${npz}" ]]; then
    echo "[kitti ${seq}] skip — npz exists"
    continue
  fi
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  E1.3 kitti seq${seq}  (use_split=true, K=2 soft)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  SEQ_OVERRIDE="${seq}" \
  USE_SPLIT_OVERRIDE=true \
  SHARE_TSDF_OVERRIDE=false \
  N_SCANS_OVERRIDE=100 \
  RESULTS_ROOT_OVERRIDE="${cell_dir}/_run" \
    bash "${EVAL_PKG}/scripts/scovox_softprob_kitti_seq08.sh" \
       > "${cell_dir}/wrapper.log" 2>&1 || echo "  [warn] wrapper exited non-zero"

  if [[ -f "${cell_dir}/_run/baseline_softprob/scovox.npz" ]]; then
    mv "${cell_dir}/_run/baseline_softprob/scovox.npz" "${npz}"
    mv "${cell_dir}/_run/baseline_softprob/scovox_run.log" "${cell_dir}/scovox_run.log"
  fi
  rm -rf "${cell_dir}/_run"

  if [[ -f "${npz}" ]]; then
    metrics=$(extract_metrics "${cell_dir}/scovox_run.log")
    echo "  [done] seq${seq}: $(du -h "${npz}" | cut -f1)  metrics=${metrics}"
    echo "${metrics}" > "${cell_dir}/perf.csv"
  else
    echo "  [warn] seq${seq}: no npz produced"
  fi
done

# ---- Score Replica cells ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  E1.3 scoring"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SUMMARY_CSV="${RES_ROOT}/summary.csv"
echo "dataset,scene,n_frames,frame_ms_mean,fps,tsdf_mb,sembeta_mb,mIoU" > "${SUMMARY_CSV}"

for scene in "${REPLICA_SCENES[@]}"; do
  cell_dir="${RES_ROOT}/replica/${scene}"
  npz="${cell_dir}/scovox.npz"
  if [[ ! -f "${npz}" ]]; then echo "replica,${scene},NA,NA,NA,NA,NA,NA" >> "${SUMMARY_CSV}"; continue; fi
  miou=$(python3 "${EVAL_PKG}/scripts/eval_scovox_replica.py" \
              --replica_root "${WS}/data/replica_niceslam" \
              --npz_root "${RES_ROOT}/replica" \
              --scenes "${scene}" \
              --do_miou 2>&1 | tee "${cell_dir}/eval.log" \
              | grep -oP 'mIoU.*?\K[0-9]+\.[0-9]+' | tail -1 || echo "")
  m=$(cat "${cell_dir}/perf.csv" 2>/dev/null)
  IFS=, read -r fms tsdf sb nf <<< "${m}"
  fps=$(python3 -c "print(f'{1000.0/${fms}:.2f}' if '${fms}' not in ('NA','') else 'NA')" 2>/dev/null || echo NA)
  echo "replica,${scene},${nf},${fms},${fps},${tsdf},${sb},${miou:-NA}" >> "${SUMMARY_CSV}"
done

KITTI_YAML="${WS}/src/sem_seg_pipeline/polarseg/semantic-kitti.yaml"
for seq in "${KITTI_SEQS[@]}"; do
  cell_dir="${RES_ROOT}/kitti/${seq}"
  npz="${cell_dir}/scovox.npz"
  if [[ ! -f "${npz}" ]]; then echo "kitti,${seq},NA,NA,NA,NA,NA,NA" >> "${SUMMARY_CSV}"; continue; fi
  # eval_scovox_kitti_miou expects npz_root/<seq>/scovox.npz; our cell_dir already matches.
  npz_root="${RES_ROOT}/kitti"
  ln -sf "${cell_dir}/scovox.npz" "${cell_dir}/scovox.npz" 2>/dev/null || true
  miou=$(python3 "${EVAL_PKG}/scripts/eval_scovox_kitti_miou.py" \
              --kitti_root "${WS}/data/semantickitti/dataset" \
              --npz_root "${npz_root}" \
              --semantic_kitti_yaml "${KITTI_YAML}" \
              --variant scovox \
              --sequences "${seq}" 2>&1 | grep -oP 'mIoU=\K[0-9]+\.[0-9]+' | tail -1 || echo "")
  m=$(cat "${cell_dir}/perf.csv" 2>/dev/null)
  IFS=, read -r fms tsdf sb nf <<< "${m}"
  fps=$(python3 -c "print(f'{1000.0/${fms}:.2f}' if '${fms}' not in ('NA','') else 'NA')" 2>/dev/null || echo NA)
  echo "kitti,${seq},${nf},${fms},${fps},${tsdf},${sb},${miou:-NA}" >> "${SUMMARY_CSV}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  E1.3 summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
column -ts, < "${SUMMARY_CSV}"
echo ""
echo "Full CSV: ${SUMMARY_CSV}"
