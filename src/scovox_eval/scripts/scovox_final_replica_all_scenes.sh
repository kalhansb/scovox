#!/bin/bash
# Final-run driver: SCovox hard-label + SCovox soft-prob across all 8
# Replica scenes (room0..2, office0..4). Per-scene loop:
#   1. Run M2F dense inference (collapse_n_classes=19) → NPZ
#   2. Convert NPZ → .topk (~29 GB peak)
#   3. Run SCovox soft sweep
#   4. Run SCovox hard sweep (no .topk needed)
#   5. Delete this scene's .topk + NPZ to free disk for next scene
#
# Disk budget: ~30 GB peak/scene. Free disk should be ≥ 35 GB at start.
# Total compute: ~50 min/scene × 8 = ~6.5 hours.
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
REPLICA_ROOT="${WS}/data/replica_niceslam"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RESULTS_ROOT="${EVAL_PKG}/results/final_replica_2026_05_04"
COLLAPSE_JSON="${EVAL_PKG}/scripts/collapse_ade150_to_scovox18.json"

SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
RES=0.05
LABEL_DIR=semantic_m2f_ade

# Wait for any other scovox_mapping_node to exit (e.g. concurrent KITTI
# sweep) so we don't share the ROS namespace `ablation`. Polls every 30 s.
echo "[init] waiting for any other scovox_mapping_node to finish..."
while pgrep -f 'scovox_mapping_node' > /dev/null 2>&1; do sleep 30; done
echo "[init] no other scovox_mapping_node running — proceeding"

# ROS env (for sweeps)
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

run_sweep_cell() {
    local scene=$1; local mode=$2
    local cell_dir="${RESULTS_ROOT}/${mode}/${scene}"
    local out="${cell_dir}/scovox.npz"
    local log="${cell_dir}/scovox_run.log"
    mkdir -p "${cell_dir}"
    [[ -f "${out}" ]] && { echo "[${mode}/${scene}] skip — exists"; return; }

    local topk_arg=""
    if [[ "${mode}" == "soft" ]]; then
        topk_arg="topk_probs_dir:=${REPLICA_ROOT}/${scene}/semantic_m2f_topk"
    fi

    echo "━━ ${mode}/${scene} ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        robot_name:=ablation \
        resolution:=${RES} \
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
    pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
    pkill -9 -f 'replica_replay_node' 2>/dev/null || true
    pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
    sleep 3
}

run_m2f_dense() {
    local scene=$1
    local probs_dir="${REPLICA_ROOT}/${scene}/semantic_m2f_ade_probs_collapsed18"
    local n=$(ls "${probs_dir}" 2>/dev/null | wc -l)
    if [[ "${n}" -ge 2000 ]]; then
        echo "[m2f/${scene}] already have ${n} dense NPZ — skip inference"
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

for scene in "${SCENES[@]}"; do
    echo "════════════════════════════════════════════════════════════════"
    echo "==>  scene = ${scene}"
    echo "════════════════════════════════════════════════════════════════"
    df -h /home/kalhan/projects/ | tail -1

    run_m2f_dense "${scene}"
    convert_topk "${scene}"
    run_sweep_cell "${scene}" soft
    run_sweep_cell "${scene}" hard
    cleanup_scene "${scene}"
done

echo "=== Final Replica sweep done. Results under ${RESULTS_ROOT}/{hard,soft}/<scene>/ ==="
