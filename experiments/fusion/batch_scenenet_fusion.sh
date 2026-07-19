#!/bin/bash
# Two-robot DISJOINT fusion batch on all 13 SceneNet trajs.
# A=[0,150), B=[150,300) — no shared frames. Collect solo_a/solo_b/fused mIoU.
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
FE="${WS}/experiments/fusion"
OUT="${WS}/experiments/results/fusion_disjoint"
SUMM="${OUT}/summary.csv"
TRAJS="0_175 0_178 0_182 0_223 0_279 0_485 0_490 0_571 0_682 0_723 0_789 0_867 0_977"
mkdir -p "${OUT}"
echo "traj,solo_a,solo_b,fused,fused_minus_max" > "${SUMM}"

for t in ${TRAJS}; do
  echo "################ FUSION ${t} ################"
  if [ ! -s "${OUT}/${t}/fused.npz" ]; then
    bash "${FE}/run_scenenet_fusion.sh" "${t}" 150 150 "${OUT}" > "${OUT}/${t}.log" 2>&1
  fi
  A=$(grep -oP 'solo_a \K[0-9.]+' "${OUT}/${t}/scores.txt" 2>/dev/null | tail -1)
  B=$(grep -oP 'solo_b \K[0-9.]+' "${OUT}/${t}/scores.txt" 2>/dev/null | tail -1)
  F=$(grep -oP 'fused \K[0-9.]+' "${OUT}/${t}/scores.txt" 2>/dev/null | tail -1)
  D=$(awk -v a="${A:-0}" -v b="${B:-0}" -v f="${F:-0}" 'BEGIN{m=(a>b)?a:b; printf "%.4f", f-m}')
  echo "${t},${A},${B},${F},${D}" >> "${SUMM}"
  echo "  ${t}: solo_a=${A} solo_b=${B} fused=${F} (fused-max=${D})"
done
echo "== FUSION BATCH DONE =="; cat "${SUMM}"
