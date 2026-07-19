#!/bin/bash
# E1 full pipeline over the 13 SceneNet val trajectories:
#   1) deterministic RGB-D replay + split-map binary snapshot capture
#   2) raycast occupancy reference (offline depth-camera DDA, AFTER replay)
#   3) calibration/decomposition scoring (same score_e1.py as KITTI)
# One trajectory at a time so the CPU-heavy raycaster never competes with ROS
# integration (which can drop frames under contention).
set -o pipefail
WS=/home/kalhan/Projects/scovox_ws
U="${WS}/experiments/uncertainty"
RES=0.05
SEQS="${1:-0_175 0_178 0_182 0_223 0_279 0_485 0_490 0_571 0_682 0_723 0_789 0_867 0_977}"

for SEQ in ${SEQS}; do
  OUT="${WS}/experiments/results/scenenet_e1/${SEQ}"
  mkdir -p "${OUT}"
  echo "################ E1 scenenet ${SEQ} ################"
  if [ ! -s "${OUT}/map_bin.npz" ]; then
    bash "${U}/run_scenenet_e1.sh" "${SEQ}" atlas_e1 "${OUT}" "${RES}" \
      > "${WS}/experiments/results/scenenet_e1/${SEQ}_driver.log" 2>&1
  fi
  if [ ! -s "${OUT}/occ_ref.npz" ]; then
    echo "  building occupancy reference (depth raycast) ..."
    python3 "${U}/build_occ_reference_scenenet.py" --sequence "${SEQ}" \
      --resolution "${RES}" -o "${OUT}/occ_ref.npz" > "${OUT}/ref.log" 2>&1
  fi
  echo "  scoring ..."
  python3 "${U}/score_e1.py" --map "${OUT}/map_bin.npz" --ref "${OUT}/occ_ref.npz" \
    --tag "scenenet ${SEQ}" -o "${OUT}/e1_score.md" > "${OUT}/score.log" 2>&1
  grep -E 'matched in map|^\| all ' "${OUT}/e1_score.md" | sed 's/^/    /'
done
echo "== E1 SCENENET BATCH DONE =="
python3 "${U}/build_e1_table_scenenet.py"
