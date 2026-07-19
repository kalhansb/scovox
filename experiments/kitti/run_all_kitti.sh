#!/bin/bash
# End-to-end KITTI: PolarSeg inference (GPU) -> GT scovox batch -> PolarSeg
# scovox batch -> GT-vs-PolarSeg table. Stages are SEQUENTIAL on purpose: the
# scovox replay is timing-sensitive and concurrent CPU load drops frames.
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
KE="${WS}/experiments/kitti"

echo "############ [$(date +%T)] 1/4 PolarSeg inference (GPU) ############"
bash "${KE}/run_polarseg_inference.sh" 06,07,08,09,10 cuda:0

echo "############ [$(date +%T)] 2/4 GT-input scovox batch ############"
bash "${KE}/batch_gt.sh"

echo "############ [$(date +%T)] 3/4 PolarSeg-input scovox batch ############"
bash "${KE}/batch_polarseg.sh"

echo "############ [$(date +%T)] 4/4 GT-vs-PolarSeg table ############"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
python3 "${KE}/build_kitti_table.py"
echo "############ [$(date +%T)] KITTI PIPELINE DONE ############"
