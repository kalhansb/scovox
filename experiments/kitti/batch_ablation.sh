#!/bin/bash
# Semantic-mode ablation on KITTI PolarSeg input: run majority_vote + naive for
# seqs 06-10 and score each. dirichlet already lives in results/kitti_polarseg/.
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
KE="${WS}/experiments/kitti"
SEQS="06 07 08 09 10"
MODES="majority_vote naive"

for mode in ${MODES}; do
  OUT="${WS}/experiments/results/kitti_polarseg_${mode}"
  echo "################ MODE=${mode} ################"
  for s in ${SEQS}; do
    mkdir -p "${OUT}/${s}"
    if [ -f "${OUT}/${s}/scovox.npz" ]; then echo "==== ${mode} seq ${s} already done, skip ===="; continue; fi
    echo "==== PolarSeg/${mode} seq ${s} ===="
    bash "${KE}/run_kitti.sh" "${s}" predictions "atlas_ps_${mode:0:2}" \
         "${OUT}/${s}/scovox.npz" "${OUT}/${s}/run.log" 0.10 "${mode}" 2>&1 \
         | grep -vE 'Waiting for TF|frames gated' | tail -6
  done
  echo "==== scoring ${mode} ===="
  export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
  python3 "${KE}/score_kitti.py" "${OUT}" --tag "${mode}"
done
echo "KITTI ABLATION DONE"
