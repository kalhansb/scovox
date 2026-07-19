#!/bin/bash
# SCovox-NG (semantic_occ_gate=0) on SemanticKITTI seqs 06-10, first 100
# frames each, PolarSeg predictions, 10 cm. Mirrors the baseline run in
# scovox_semantickitti_polarseg.sh but with the gate disabled. Outputs to
# results/semantickitti_polarseg_ng_10cm/<seq>/scovox.npz.
set -eo pipefail

RESOLUTION="${1:-0.10}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
KITTI_ROOT="${WS}/data/semantickitti/dataset"
RESULTS_ROOT="${WS}/src/robot_sw/distributed_mapping/scovox_eval/results/semantickitti_polarseg_ng_${RES_CM}cm"
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
SEQUENCES=(06 07 08 09 10)
N_SCANS=100
REPLAY_HZ=0.5

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RESULTS_ROOT}"
echo "━━ SCovox-NG × SemKITTI PolarSeg, ${RES_CM}cm, seqs ${SEQUENCES[*]} ━━"

for seq in "${SEQUENCES[@]}"; do
    SEQ_RES="${RESULTS_ROOT}/${seq}"
    mkdir -p "${SEQ_RES}"
    OUT_NPZ="${SEQ_RES}/scovox.npz"
    LOG="${SEQ_RES}/scovox_run.log"
    if [[ -f "${OUT_NPZ}" ]]; then echo "[${seq}] skip"; continue; fi
    echo "━━ seq ${seq} ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=ablation \
        resolution:=${RESOLUTION} \
        semantic_mode:=dirichlet \
        semantic_occ_gate:=0.0 \
        range_decay_length:=50.0 \
        > "${LOG}" 2>&1 &
    LPID=$!
    sleep 4
    SEQ_INT=$((10#$seq))
    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="${KITTI_ROOT}" \
        -p sequence:=${SEQ_INT} \
        -p rate_hz:=${REPLAY_HZ} \
        -p robot_name:=ablation \
        -p max_range:=30.0 \
        -p min_range:=1.0 \
        -p labels_subdir:=predictions \
        -p n_scans:=${N_SCANS}

    MIN_RECV=$((N_SCANS * 99 / 100)); WAITED=0
    while true; do
        LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 10; break; fi
        sleep 3; WAITED=$((WAITED+3))
        [ ${WAITED} -ge 600 ] && { echo "  WARN: timeout @ recv=${LAST}"; break; }
    done

    timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/ablation/scovox_node/pointcloud \
        -p output:="${OUT_NPZ}" || echo "  WARN: capture timeout"

    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node' 2>/dev/null || true
    pkill -9 -f 'semantickitti_replay_node' 2>/dev/null || true
    pkill -9 -f 'pointcloud_to_npz' 2>/dev/null || true
    sleep 2
done

echo "=== done. NG NPZs in ${RESULTS_ROOT}/<seq>/scovox.npz ==="
