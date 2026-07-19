#!/bin/bash
# E6.1 — SCovox SLIM-VDB-matched-config head-to-head, KITTI side.
#
# Per sequence (06,07,08,09,10):
#   1. Run scovox_node with matched config (band_only=true, range_decay=-1,
#      min/max_range=5/30, voxel=0.10, sdf_trunc=0.30, K_TOP=2,
#      min_tsdf_weight_publish=10.0, soft PolarSeg topk).
#   2. Replay 100 scans at rate_hz=0.5.
#   3. Capture ~/pointcloud → scovox.npz.
#
# Per cell:  results/matched_config_2026_05_08/kitti/<seq>/scovox.npz
#
# Idempotent: seqs whose scovox.npz exists are skipped. predictions_topk
# already exists per seq (no rotation needed).
set -o pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
RES_ROOT="${EVAL_PKG}/results/matched_config_2026_05_08/kitti"

SEQS=(06 07 08 09 10)

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RES_ROOT}"

run_kitti_seq() {
    local seq="$1"
    local cell_dir="${RES_ROOT}/${seq}"
    local npz="${cell_dir}/scovox.npz"
    local log="${cell_dir}/scovox_run.log"
    local topk_dir="${KITTI_ROOT}/sequences/${seq}/predictions_topk"
    mkdir -p "${cell_dir}"

    if [[ -f "${npz}" ]]; then
        echo "[kitti seq${seq}] skip — npz already exists"
        return 0
    fi
    if [[ ! -d "${topk_dir}" ]] || [[ -z "$(ls -A "${topk_dir}" 2>/dev/null)" ]]; then
        echo "[kitti seq${seq}] WARN: missing predictions_topk at ${topk_dir} — abort"
        return 1
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  kitti seq${seq} (matched config: voxel=0.10, trunc=0.30)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    pkill -9 -f 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' 2>/dev/null || true
    sleep 2

    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=atlas \
        resolution:=0.10 \
        sdf_trunc_voxels:=3 \
        band_only_integration:=true \
        tsdf_space_carving:=false \
        carve_band:=-1.0 \
        range_decay_length:=-1.0 \
        min_range:=5.0 \
        max_range:=30.0 \
        min_tsdf_weight_publish:=10.0 \
        semantic_mode:=dirichlet \
        topk_probs_dir:="${topk_dir}" \
        > "${log}" 2>&1 &
    local LPID=$!
    sleep 5

    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="${KITTI_ROOT}" \
        -p sequence:="$(echo $seq | sed 's/^0//')" \
        -p rate_hz:=0.5 \
        -p robot_name:=atlas \
        -p max_range:=30.0 \
        -p min_range:=5.0 \
        -p labels_subdir:=predictions \
        -p n_scans:=100 > /dev/null 2>&1

    local MIN_RECV=99 WAITED=0
    while true; do
        local last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
        if [ "${last}" -ge "${MIN_RECV}" ] 2>/dev/null; then
            echo "  [scovox] recv=${last}/100 — settling 15s"
            sleep 15
            break
        fi
        sleep 3; WAITED=$((WAITED+3))
        if [ ${WAITED} -ge 600 ]; then
            echo "  [scovox] WARN: timeout at recv=${last}"
            sleep 15
            break
        fi
    done

    timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/atlas/scovox_node/pointcloud \
        -p output:="${npz}" 2>&1 | tail -2 || echo "  [capture] WARN: timeout"

    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' 2>/dev/null || true
    sleep 3

    if [[ -f "${npz}" ]]; then
        echo "  [done] $(du -h "${npz}" | cut -f1) ${npz}"
    else
        echo "  [done] WARN: no npz produced"
    fi
}

for seq in "${SEQS[@]}"; do
    run_kitti_seq "${seq}"
done
echo ""
echo "All KITTI matched-config runs complete."
ls -la "${RES_ROOT}"/*/scovox.npz 2>&1 | head -10
