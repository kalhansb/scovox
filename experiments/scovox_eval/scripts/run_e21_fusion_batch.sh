#!/bin/bash
# E2.1 fusion smoke — split-grid trajectory-split SCovox + dscovox fusion.
#
# Per-scene topology (replica_eval_fusion.launch.py + use_split=true):
#   /robotA/scovox_node  ← replay frames [0, 1000)
#   /robotB/scovox_node  ← replay frames [1000, 2000)
#   /dscovox_node        ← merges both via wire-format v2 (37 B/voxel SemBeta)
#
# Per cell capture:
#   results/e21_fusion_2026_05_08/<scene>/{solo_a,solo_b,fused}.npz
#
# Score: eval_e21_fusion.py — voxel-mIoU + Chamfer + F@5cm vs replica GT.
# Verdict per spec: fused beats max(solo_a, solo_b) on mIoU, F@5cm, Chamfer.
#
# Idempotent: scenes with all 3 npz present are skipped.
# Sequential execution per user preference.
set -o pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
DATE_TAG="${DATE_TAG_OVERRIDE:-$(date +%Y_%m_%d)}"
# Step 8 — wire format selector. Default v2 keeps every legacy invocation
# of this script byte-identical to its previous behaviour; new runs override
# with `WIRE_FORMAT=v3 bash run_e21_fusion_batch.sh` (or per-call).
# v3 paths the result dir suffix so v2 / v3 batches don't collide on disk.
WIRE_FORMAT="${WIRE_FORMAT:-v2}"
NUM_CLASSES="${NUM_CLASSES:-14}"           # Replica → NYU13
DIRICHLET_PRIOR="${DIRICHLET_PRIOR:-0.01}"
RES_TAG_SUFFIX=""
if [[ "${WIRE_FORMAT}" == "v3" ]]; then RES_TAG_SUFFIX="_v3"; fi
RES_ROOT="${EVAL_PKG}/results/e21_fusion_${DATE_TAG}${RES_TAG_SUFFIX}"

# Scenes default to the full 8-scene Replica batch. Override via env for
# smoke runs (`SCENES="room1" N_PER_ROBOT=200 bash run_e21_fusion_batch.sh`).
if [[ -n "${SCENES:-}" ]]; then
  IFS=' ' read -ra REPLICA_SCENES <<< "${SCENES}"
else
  REPLICA_SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
fi

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RES_ROOT}"

ROBOT_A="robotA"
ROBOT_B="robotB"
N_PER_ROBOT="${N_PER_ROBOT:-1000}"   # 2000 / 2 for trajectory split (env-overridable for smoke runs)

run_fusion_scene() {
  local scene="$1"
  local cell_dir="${RES_ROOT}/${scene}"
  mkdir -p "${cell_dir}"
  local solo_a="${cell_dir}/solo_a.npz"
  local solo_b="${cell_dir}/solo_b.npz"
  local fused="${cell_dir}/fused.npz"
  if [[ -f "${solo_a}" && -f "${solo_b}" && -f "${fused}" ]]; then
    echo "[fusion ${scene}] skip — all 3 npz exist"
    return 0
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  E2.1 fusion ${scene}  (use_split=true, K=2 soft, 2 robots)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  df -h "${WS}" | tail -1

  local topk="${WS}/data/replica_niceslam/${scene}/semantic_m2f_topk"
  local probs="${WS}/data/replica_niceslam/${scene}/semantic_m2f_ade_probs"
  if [[ ! -d "${topk}" ]] || [[ -z "$(ls -A "${topk}" 2>/dev/null)" ]]; then
    echo "  [topk] regenerating ${topk}"
    python3 "${EVAL_PKG}/scripts/topk_npz_to_bin.py" \
        --src_dir "${probs}" --dst_dir "${topk}" --mode image 2>&1 | tail -2
  fi

  # Cleanup any prior survivor procs.
  pkill -9 -f 'scovox_mapping_node|dscovox_node|replica_replay_node|pointcloud_to_npz' 2>/dev/null || true
  sleep 2

  # 1. Launch fusion (3 nodes: scovoxA, scovoxB, dscovox).
  local launch_log="${cell_dir}/fusion_launch.log"
  ros2 launch scovox_mapping replica_eval_fusion.launch.py \
      robot_a:=${ROBOT_A} \
      robot_b:=${ROBOT_B} \
      resolution:=0.05 \
      topk_probs_dir:="${topk}" \
      use_split:=true \
      share_tsdf:=false \
      fused_walker:=true \
      wire_format:=${WIRE_FORMAT} \
      num_classes:=${NUM_CLASSES} \
      dirichlet_prior:=${DIRICHLET_PRIOR} \
      > "${launch_log}" 2>&1 &
  local LPID=$!
  sleep 6

  # 2. Replay both robots in parallel: A=[0..1000), B=[1000..2000).
  echo "  [replay] robot A frames [0, ${N_PER_ROBOT})"
  python3 -m scovox_eval.replica_replay_node --ros-args \
      -p dataset_path:="${WS}/data/replica_niceslam/${scene}" \
      -p rate_hz:=2.0 \
      -p robot_name:=${ROBOT_A} \
      -p camera_poses:=true \
      -p semantic_subdir:=semantic_m2f_ade \
      -p start_frame:=0 \
      -p n_scans:=${N_PER_ROBOT} > "${cell_dir}/replay_a.log" 2>&1 &
  local RA_PID=$!

  echo "  [replay] robot B frames [${N_PER_ROBOT}, $((2 * N_PER_ROBOT)))"
  python3 -m scovox_eval.replica_replay_node --ros-args \
      -p dataset_path:="${WS}/data/replica_niceslam/${scene}" \
      -p rate_hz:=2.0 \
      -p robot_name:=${ROBOT_B} \
      -p camera_poses:=true \
      -p semantic_subdir:=semantic_m2f_ade \
      -p start_frame:=${N_PER_ROBOT} \
      -p n_scans:=${N_PER_ROBOT} > "${cell_dir}/replay_b.log" 2>&1 &
  local RB_PID=$!

  # 3. Wait for both replays to finish (each ~500s wallclock).
  wait ${RA_PID} 2>/dev/null || true
  wait ${RB_PID} 2>/dev/null || true
  echo "  [replay] both robots done — settling 30s"
  sleep 30

  # 4. Capture the three pointclouds.
  echo "  [capture] /${ROBOT_A}/scovox_node/pointcloud → solo_a.npz"
  timeout 180 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/${ROBOT_A}/scovox_node/pointcloud \
      -p output:="${solo_a}" 2>&1 | tail -2 || echo "  [warn] solo_a timeout"

  echo "  [capture] /${ROBOT_B}/scovox_node/pointcloud → solo_b.npz"
  timeout 180 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/${ROBOT_B}/scovox_node/pointcloud \
      -p output:="${solo_b}" 2>&1 | tail -2 || echo "  [warn] solo_b timeout"

  echo "  [capture] /dscovox_node/pointcloud → fused.npz"
  timeout 180 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/dscovox_node/pointcloud \
      -p output:="${fused}" 2>&1 | tail -2 || echo "  [warn] fused timeout"

  # 5. Cleanup.
  kill ${LPID} 2>/dev/null || true
  wait ${LPID} 2>/dev/null || true
  pkill -9 -f 'scovox_mapping_node|dscovox_node|replica_replay_node|pointcloud_to_npz' 2>/dev/null || true
  ros2 daemon stop 2>/dev/null || true
  sleep 4
  ros2 daemon start 2>/dev/null || true
  sleep 2

  # 6. Free disk.
  if [[ -d "${topk}" ]]; then rm -rf "${topk}"; fi

  for f in "${solo_a}" "${solo_b}" "${fused}"; do
    if [[ -f "${f}" ]]; then echo "  [done] $(du -h "${f}" | cut -f1)  $(basename "${f}")"
    else echo "  [warn] $(basename "${f}") not produced"; fi
  done
}

# ---- Run all scenes ----
for scene in "${REPLICA_SCENES[@]}"; do
  run_fusion_scene "${scene}"
done

# ---- Score ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  E2.1 scoring"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SCORE_LOG="${RES_ROOT}/eval_e21_fusion.log"
python3 "${EVAL_PKG}/scripts/eval_e21_fusion.py" \
    --replica_root "${WS}/data/replica_niceslam" \
    --npz_root "${RES_ROOT}" \
    --scenes "${REPLICA_SCENES[@]}" \
    --do_miou --do_chamfer 2>&1 | tee "${SCORE_LOG}"

echo ""
echo "Full results: ${RES_ROOT}"
