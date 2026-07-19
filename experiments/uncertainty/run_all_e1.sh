#!/bin/bash
# E1 full pipeline over SemanticKITTI seqs 06-10:
#   1) deterministic replay + split-map binary snapshot capture
#   2) raycast occupancy reference (offline, AFTER replay -> no CPU contention)
#   3) calibration/decomposition scoring
# Runs sequences one at a time so the CPU-heavy raycaster never competes with
# ROS integration (which drops frames under contention).
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
U="${WS}/experiments/uncertainty"
RES=0.10
SEQS="${1:-06 07 08 09 10}"

for SEQ in ${SEQS}; do
  OUT="${WS}/experiments/results/kitti_e1/${SEQ}"
  mkdir -p "${OUT}"
  echo "################ E1 seq ${SEQ} ################"
  if [ ! -s "${OUT}/map_bin.npz" ]; then
    bash "${U}/run_kitti_e1.sh" "${SEQ}" atlas_e1 "${OUT}" "${RES}" > "${WS}/experiments/results/kitti_e1/${SEQ}_driver.log" 2>&1
  fi
  if [ ! -s "${OUT}/occ_ref.npz" ]; then
    echo "  building occupancy reference (raycast) ..."
    python3 "${U}/build_occ_reference.py" --sequence "$((10#${SEQ}))" \
      --resolution "${RES}" -o "${OUT}/occ_ref.npz" > "${OUT}/ref.log" 2>&1
  fi
  echo "  scoring ..."
  python3 "${U}/score_e1.py" --map "${OUT}/map_bin.npz" --ref "${OUT}/occ_ref.npz" \
    --tag "seq ${SEQ}" -o "${OUT}/e1_score.md" > "${OUT}/score.log" 2>&1
  grep -E 'matched in map|^\| all ' "${OUT}/e1_score.md" | sed 's/^/    /'
done
echo "== E1 BATCH DONE =="
python3 "${U}/build_e1_table.py"
