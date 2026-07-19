#!/bin/bash
# Final-run driver: SCovox hard-label + SCovox soft-prob across KITTI
# sequences 06, 07, 08, 09, 10. Writes per-cell scovox.npz under
# results/final_kitti/{hard,soft}/<seq>/. Same parameters as the
# post-NG baseline used in scovox_ablation_kitti_seq08_polarseg.sh.
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
KITTI_ROOT="${WS}/data/semantickitti/dataset"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
RESULTS_ROOT="${EVAL_PKG}/results/final_kitti_2026_05_04"

SEQS=(06 07 08 09 10)
RES=0.10
N_SCANS=100
REPLAY_HZ=0.5

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

run_cell() {
    local seq=$1; local mode=$2  # mode: hard | soft
    local seq_int=$((10#${seq}))
    local cell_dir="${RESULTS_ROOT}/${mode}/${seq}"
    local out="${cell_dir}/scovox.npz"
    local log="${cell_dir}/scovox_run.log"
    mkdir -p "${cell_dir}"
    [[ -f "${out}" ]] && { echo "[${mode}/${seq}] skip — exists"; return; }

    local topk_arg=""
    local passthrough="false"
    if [[ "${mode}" == "soft" ]]; then
        topk_arg="topk_probs_dir:=${KITTI_ROOT}/sequences/${seq}/predictions_topk"
        passthrough="true"
    fi

    echo "━━ ${mode}/${seq} ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=ablation \
        resolution:=${RES} \
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
        -p rate_hz:=${REPLAY_HZ} \
        -p robot_name:=ablation \
        -p max_range:=30.0 -p min_range:=1.0 \
        -p labels_subdir:=predictions \
        -p n_scans:=${N_SCANS} \
        -p soft_prob_passthrough:=${passthrough}

    local MIN_RECV=$((N_SCANS * 99 / 100)); local WAITED=0
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
    pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
    pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
    pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
    sleep 2
}

for seq in "${SEQS[@]}"; do
    run_cell "${seq}" hard
    run_cell "${seq}" soft
done

echo "=== Final KITTI sweep done. Results under ${RESULTS_ROOT}/{hard,soft}/<seq>/ ==="
