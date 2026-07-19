#!/bin/bash
# Soft-label scovox for all 13 SceneNet trajs (topk from swin-small inference).
# Requires predictions_topk/ already generated. Sequential.
WS=/home/kalhan/Projects/scovox_ws
TRAJS="0_175 0_178 0_182 0_223 0_279 0_485 0_490 0_571 0_682 0_723 0_789 0_867 0_977"
SUMM="${WS}/experiments/results/scenenet_soft/summary.csv"
mkdir -p "${WS}/experiments/results/scenenet_soft"
echo "seq,soft_miou" > "${SUMM}"
for t in $TRAJS; do
  echo "=== SOFT $t ==="
  bash "${WS}/experiments/run_scenenet_soft.sh" "$t" > /dev/null 2>&1
  M=$(grep -oP 'mIoU = \K[0-9.]+' "${WS}/experiments/results/scenenet_soft/${t}/score.log" 2>/dev/null | head -1)
  echo "$t,$M" >> "${SUMM}"
  echo "  $t -> soft mIoU=$M"
done
echo "SOFT BATCH DONE"; cat "${SUMM}"
