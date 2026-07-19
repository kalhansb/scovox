#!/bin/bash
# GT-input scovox for SemanticKITTI seqs 06-10 (labels/ as semantic input),
# then score vs voxelized GT. Sequential (ROS namespace reuse).
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
KE="${WS}/experiments/kitti"
OUT="${WS}/experiments/results/kitti_gt"
SEQS="06 07 08 09 10"

for s in ${SEQS}; do
  echo "======== GT seq ${s} ========"
  bash "${KE}/run_kitti.sh" "${s}" labels "atlas_gt" \
       "${OUT}/${s}/scovox.npz" "${OUT}/${s}/run.log" 2>&1 \
       | grep -vE 'Waiting for TF|frames gated' | tail -8
done

echo "======== SCORING GT ========"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
python3 "${KE}/score_kitti.py" "${OUT}" --tag gt
echo "GT BATCH DONE"
