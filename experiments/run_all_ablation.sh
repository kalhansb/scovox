#!/bin/bash
# Full semantic-mode ablation: dirichlet (done) vs majority_vote vs naive, on
# the predicted-label inputs where fusion matters (KITTI PolarSeg + SceneNet
# soft). Sequential (timing-sensitive ROS replay). Builds both 3-way tables.
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
EXP="${WS}/experiments"

echo "############ [$(date +%T)] KITTI PolarSeg ablation (mv + naive) ############"
bash "${EXP}/kitti/batch_ablation.sh"

echo "############ [$(date +%T)] SceneNet soft ablation (mv + naive) ############"
bash "${EXP}/batch_scenenet_ablation.sh"

echo "############ [$(date +%T)] build ablation tables ############"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
python3 "${EXP}/kitti/build_ablation_table.py"
echo ""
python3 "${EXP}/build_scenenet_ablation_table.py"
echo "############ [$(date +%T)] ABLATION PIPELINE DONE ############"
