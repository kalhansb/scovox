#!/bin/bash
# Autonomous remaining pipeline: wait for GT batch -> swin-small inference (13)
# -> soft scovox (13) -> soft-vs-GT table. Stages are sequential to avoid CPU
# contention (GT/soft scovox runs are timing-sensitive).
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
EXP="${WS}/experiments"
export PATH=/home/kalhan/miniconda3/bin:$PATH
export HF_HOME="${WS}/data/hf_cache"
export OMP_NUM_THREADS=$(nproc)
MODEL=facebook/mask2former-swin-small-ade-semantic
TRAJS="0_175,0_178,0_182,0_223,0_279,0_485,0_490,0_571,0_682,0_723,0_789,0_867,0_977"

echo "[$(date +%T)] waiting for GT batch to finish ..."
while pgrep -f batch_scenenet_gt >/dev/null; do sleep 15; done
echo "[$(date +%T)] GT batch done. GT summary:"; cat "${EXP}/results/scenenet_gt/summary.csv"

echo "[$(date +%T)] === swin-small soft inference (13 trajs, CPU) ==="
python "${EXP}/soft_scenenet/run_mask2former_scenenet.py" \
    --data_root "${WS}/data/scenenet_val_layout" \
    --seqs "${TRAJS}" --device cpu --model "${MODEL}" \
    > "${EXP}/soft_scenenet/infer_all.log" 2>&1
echo "[$(date +%T)] inference done. topk per traj:"
for t in ${TRAJS//,/ }; do
  echo "  $t: $(ls ${WS}/data/scenenet_val_layout/train/$t/predictions_topk/ 2>/dev/null | wc -l)"
done

echo "[$(date +%T)] === soft scovox (13 trajs) ==="
bash "${EXP}/batch_scenenet_soft.sh" > "${EXP}/results/scenenet_soft/batch.log" 2>&1

echo "[$(date +%T)] === build soft-vs-GT table ==="
python "${EXP}/build_soft_vs_gt_table.py"
echo "[$(date +%T)] FULL SOFT PIPELINE DONE"
