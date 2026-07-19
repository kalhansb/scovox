#!/bin/bash
# GT-input scovox for all 13 SceneNet trajs. Sequential (ROS namespace reuse).
WS=/home/kalhan/Projects/scovox_ws
TRAJS="0_175 0_178 0_182 0_223 0_279 0_485 0_490 0_571 0_682 0_723 0_789 0_867 0_977"
SUMM="${WS}/experiments/results/scenenet_gt/summary.csv"
mkdir -p "${WS}/experiments/results/scenenet_gt"
echo "seq,gt_miou" > "${SUMM}"
for t in $TRAJS; do
  echo "=== GT $t ==="
  M=$(bash "${WS}/experiments/run_scenenet_gt.sh" "$t" 2>&1 | tail -1)
  echo "$t,$M" >> "${SUMM}"
  echo "  $t -> mIoU=$M"
done
echo "GT BATCH DONE"; cat "${SUMM}"
