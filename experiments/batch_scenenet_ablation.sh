#!/bin/bash
# Semantic-mode ablation on SceneNet SOFT input (Mask2Former topk): run
# majority_vote + naive for all 13 trajs and score. dirichlet already lives in
# results/scenenet_soft/. Sequential.
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
EXP="${WS}/experiments"
TRAJS="0_175 0_178 0_182 0_223 0_279 0_485 0_490 0_571 0_682 0_723 0_789 0_867 0_977"
MODES="majority_vote naive"

for mode in ${MODES}; do
  OUT="${EXP}/results/scenenet_soft_${mode}"
  SUMM="${OUT}/summary.csv"
  mkdir -p "${OUT}"
  echo "seq,soft_miou" > "${SUMM}"
  echo "################ SceneNet SOFT mode=${mode} ################"
  for t in ${TRAJS}; do
    if [ ! -f "${OUT}/${t}/scovox.npz" ]; then
      bash "${EXP}/run_scenenet_soft_mode.sh" "${t}" "${mode}" "${OUT}" > /dev/null 2>&1
    fi
    M=$(grep -oP 'mIoU = \K[0-9.]+' "${OUT}/${t}/score.log" 2>/dev/null | head -1)
    echo "${t},${M}" >> "${SUMM}"
    echo "  ${mode} ${t} -> mIoU=${M}"
  done
  echo "-- ${mode} summary --"; cat "${SUMM}"
done
echo "SCENENET ABLATION DONE"
