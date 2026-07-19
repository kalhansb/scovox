#!/bin/bash
# Final run: SCovox hard + SCovox soft + SLIM-VDB across ALL Replica scenes
# (room0..2, office0..4) and ALL KITTI sequences (06–10), strictly serial
# to avoid ROS namespace collisions and GPU contention.
#
# Phase 1:  KITTI SCovox (hard + soft) for all 5 seqs            ~35 min
# Phase 2:  Replica SCovox per-scene loop (M2F → topk → soft →   ~50 min × 8 = 6.7h
#           hard → cleanup)
# Phase 3:  KITTI SLIM-VDB all seqs (existing docker runner)     ~25 min
# Phase 4:  Replica SLIM-VDB all scenes (existing docker runner) ~4 h
#
# Total: ~11 hours. Idempotent — each cell has a `[[ -f scovox.npz ]]`
# guard so partial runs resume safely. Logs to:
#   /tmp/softprob_logs/final_run_phase{1..4}.log
#
# Disk: peak ~35 GB during a Replica scene (.topk + NPZ). Free ≥ 35 GB.
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
REPLICA_ROOT="${WS}/data/replica_niceslam"
KITTI_RESULTS="${EVAL_PKG}/results/final_kitti_2026_05_04"
REPLICA_RESULTS="${EVAL_PKG}/results/final_replica_2026_05_04"
COLLAPSE_JSON="${EVAL_PKG}/scripts/collapse_ade150_to_scovox18.json"
LOG_ROOT=/tmp/softprob_logs
mkdir -p "${LOG_ROOT}"

KITTI_SEQS=(06 07 08 09 10)
SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
KITTI_RES=0.10
KITTI_NSCANS=100
KITTI_HZ=0.5
REPLICA_RES=0.05
LABEL_DIR=semantic_m2f_ade

# ROS env
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

# ── helpers ───────────────────────────────────────────────────────────

ensure_no_ros_running() {
    pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
    pkill -9 -f 'replica_replay_node' 2>/dev/null || true
    pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
    pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
    sleep 2
}

run_kitti_cell() {
    local seq=$1; local mode=$2
    local seq_int=$((10#${seq}))
    local cell_dir="${KITTI_RESULTS}/${mode}/${seq}"
    local out="${cell_dir}/scovox.npz"
    local log="${cell_dir}/scovox_run.log"
    mkdir -p "${cell_dir}"
    [[ -f "${out}" ]] && { echo "[KITTI ${mode}/${seq}] skip"; return; }
    ensure_no_ros_running

    local topk_arg=""
    local passthrough="false"
    if [[ "${mode}" == "soft" ]]; then
        topk_arg="topk_probs_dir:=${KITTI_ROOT}/sequences/${seq}/predictions_topk"
        passthrough="true"
    fi

    echo "━━ KITTI ${mode}/${seq} ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=ablation \
        resolution:=${KITTI_RES} \
        semantic_mode:=dirichlet \
        semantic_occ_gate:=0.0 \
        range_decay_length:=50.0 \
        w_occ:=6.0 w_free:=1.0 \
        carve_skip_occ_threshold:=0.4 \
        kappa0:=2.0 \
        evidence_saturation:=1000.0 \
        semantic_min_confidence:=0.1 \
        ${topk_arg} \
        > "${log}" 2>&1 &
    local LPID=$!
    sleep 4

    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="${KITTI_ROOT}" \
        -p sequence:=${seq_int} \
        -p rate_hz:=${KITTI_HZ} \
        -p robot_name:=ablation \
        -p max_range:=30.0 -p min_range:=1.0 \
        -p labels_subdir:=predictions \
        -p n_scans:=${KITTI_NSCANS} \
        -p soft_prob_passthrough:=${passthrough}

    local MIN_RECV=$((KITTI_NSCANS * 99 / 100)); local WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 10; break; fi
        sleep 3; WAITED=$((WAITED+3))
        [ ${WAITED} -ge 600 ] && { echo "  WARN: timeout @ recv=${LAST}"; break; }
    done

    timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${out}" || echo "  WARN: capture timeout"

    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    ensure_no_ros_running
}

run_replica_cell() {
    local scene=$1; local mode=$2
    local cell_dir="${REPLICA_RESULTS}/${mode}/${scene}"
    local out="${cell_dir}/scovox.npz"
    local log="${cell_dir}/scovox_run.log"
    mkdir -p "${cell_dir}"
    [[ -f "${out}" ]] && { echo "[Replica ${mode}/${scene}] skip"; return; }
    ensure_no_ros_running

    local topk_arg=""
    if [[ "${mode}" == "soft" ]]; then
        topk_arg="topk_probs_dir:=${REPLICA_ROOT}/${scene}/semantic_m2f_topk"
    fi

    echo "━━ Replica ${mode}/${scene} ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        robot_name:=ablation \
        resolution:=${REPLICA_RES} \
        semantic_occ_gate:=0.0 \
        range_decay_length:=-1.0 \
        w_occ:=2.0 w_free:=1.0 \
        carve_skip_occ_threshold:=0.7 \
        kappa0:=2.0 \
        evidence_saturation:=1000.0 \
        semantic_min_confidence:=0.1 \
        ${topk_arg} \
        > "${log}" 2>&1 &
    local LPID=$!
    sleep 4

    python3 -m scovox_eval.replica_replay_node --ros-args \
        -p dataset_path:="${REPLICA_ROOT}/${scene}" \
        -p rate_hz:=2.0 \
        -p robot_name:=ablation \
        -p camera_poses:=true \
        -p semantic_subdir:=${LABEL_DIR} \
        -p n_scans:=2000

    local MIN_RECV=1980; local WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 30; break; fi
        sleep 5; WAITED=$((WAITED+5))
        [ ${WAITED} -ge 1800 ] && { sleep 30; break; }
    done

    timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${out}" || true

    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    ensure_no_ros_running
}

m2f_dense() {
    local scene=$1
    local probs_dir="${REPLICA_ROOT}/${scene}/semantic_m2f_ade_probs_collapsed18"
    local n=$(ls "${probs_dir}" 2>/dev/null | wc -l)
    if [[ "${n}" -ge 2000 ]]; then
        echo "[m2f/${scene}] already have ${n} dense NPZ — skip"
        return
    fi
    rm -rf "${probs_dir}"
    echo "━━ M2F dense / ${scene} ━━"
    (
        source /home/kalhan/miniconda3/etc/profile.d/conda.sh
        unset VIRTUAL_ENV
        conda activate sam_clip
        export PYTHONNOUSERSITE=1
        cd "${WS}"
        python src/sem_seg_pipeline/run_mask2former_replica.py \
            --replica_root data/replica_niceslam \
            --scenes "${scene}" \
            --save_topk_probs 5 \
            --probs_subdir semantic_m2f_ade_probs_collapsed18 \
            --collapse_map_json "${COLLAPSE_JSON}" \
            --collapse_n_classes 19 \
            --overwrite
    )
}

convert_topk() {
    local scene=$1
    local src="${REPLICA_ROOT}/${scene}/semantic_m2f_ade_probs_collapsed18"
    local dst="${REPLICA_ROOT}/${scene}/semantic_m2f_topk"
    local n=$(ls "${dst}" 2>/dev/null | wc -l)
    if [[ "${n}" -ge 2000 ]]; then
        echo "[topk/${scene}] already converted — skip"
        return
    fi
    rm -rf "${dst}"
    python3 "${EVAL_PKG}/scripts/topk_npz_to_bin.py" \
        --src_dir "${src}" --dst_dir "${dst}" --mode image > /dev/null
    echo "[topk/${scene}] $(ls "${dst}" | wc -l) files"
}

cleanup_scene() {
    local scene=$1
    rm -rf "${REPLICA_ROOT}/${scene}/semantic_m2f_topk"
    rm -rf "${REPLICA_ROOT}/${scene}/semantic_m2f_ade_probs_collapsed18"
    df -h /home/kalhan/projects/ | tail -1
}

# ── Phase 1: KITTI SCovox ─────────────────────────────────────────────
echo "════════════════ PHASE 1: KITTI SCovox ════════════════"
for seq in "${KITTI_SEQS[@]}"; do
    run_kitti_cell "${seq}" hard
    run_kitti_cell "${seq}" soft
done

# ── Phase 2: Replica SCovox per-scene loop ────────────────────────────
echo "════════════════ PHASE 2: Replica SCovox per-scene loop ════════════════"
for scene in "${SCENES[@]}"; do
    echo "── scene=${scene} ──"
    df -h /home/kalhan/projects/ | tail -1
    # If both cells already done, skip the M2F+topk pipeline entirely.
    hard_done=0; soft_done=0
    [[ -f "${REPLICA_RESULTS}/hard/${scene}/scovox.npz" ]] && hard_done=1
    [[ -f "${REPLICA_RESULTS}/soft/${scene}/scovox.npz" ]] && soft_done=1
    if [[ "${hard_done}" -eq 1 && "${soft_done}" -eq 1 ]]; then
        echo "[${scene}] both cells already done — skip M2F + topk + sweeps"
        cleanup_scene "${scene}"
        continue
    fi
    if [[ "${soft_done}" -eq 0 ]]; then
        m2f_dense   "${scene}"
        convert_topk "${scene}"
    fi
    run_replica_cell "${scene}" soft
    run_replica_cell "${scene}" hard
    cleanup_scene "${scene}"
done

# ── Phase 3: KITTI SLIM-VDB ───────────────────────────────────────────
echo "════════════════ PHASE 3: KITTI SLIM-VDB ════════════════"
bash "${WS}/third_party_sw/slim_vdb/scripts/run_slimvdb_kitti_all.sh"

# ── Phase 4: Replica SLIM-VDB ─────────────────────────────────────────
echo "════════════════ PHASE 4: Replica SLIM-VDB ════════════════"
bash "${WS}/third_party_sw/slim_vdb/scripts/run_slimvdb_replica_m2f.sh"

echo "════════════════ FINAL RUN COMPLETE ════════════════"
