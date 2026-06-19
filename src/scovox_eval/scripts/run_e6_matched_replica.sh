#!/bin/bash
# E6.1 — SCovox SLIM-VDB-matched-config head-to-head, Replica side.
#
# Per scene:
#   1. (re)generate semantic_m2f_topk/ from semantic_m2f_ade_probs/
#   2. Run scovox_node with matched config (band_only=true, range_decay=-1,
#      min/max_range=0.2/8.0, voxel=0.05, sdf_trunc=0.15, K_TOP=2,
#      min_tsdf_weight_publish=5.0, soft m2f topk).
#   3. Replay 2000 frames at rate_hz=2.0.
#   4. Capture ~/pointcloud → scovox.npz.
#   5. Delete the topk dir to free disk for the next scene.
#
# Per cell saved under
#   results/matched_config_2026_05_08/replica/<scene>/
# in scovox.npz, scovox_run.log.
#
# Idempotent: scenes whose scovox.npz exists are skipped.
set -o pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
REPLICA_ROOT="${WS}/data/replica_niceslam"
RES_ROOT="${EVAL_PKG}/results/matched_config_2026_05_08/replica"

SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${RES_ROOT}"

run_replica_scene() {
    local scene="$1"
    local cell_dir="${RES_ROOT}/${scene}"
    local npz="${cell_dir}/scovox.npz"
    local log="${cell_dir}/scovox_run.log"
    local topk_dir="${REPLICA_ROOT}/${scene}/semantic_m2f_topk"
    local probs_dir="${REPLICA_ROOT}/${scene}/semantic_m2f_ade_probs"
    mkdir -p "${cell_dir}"

    if [[ -f "${npz}" ]]; then
        echo "[replica ${scene}] skip — npz already exists"
        return 0
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  replica ${scene} (matched config: band_only, trunc=0.15)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    df -h "${WS}" | tail -1

    # 1. Generate topk if not already there.
    if [[ ! -d "${topk_dir}" ]] || [[ -z "$(ls -A "${topk_dir}" 2>/dev/null)" ]]; then
        echo "  [topk] generating ${topk_dir} from ${probs_dir}"
        python3 "${EVAL_PKG}/scripts/topk_npz_to_bin.py" \
            --src_dir "${probs_dir}" \
            --dst_dir "${topk_dir}" \
            --mode image 2>&1 | tail -3
    else
        echo "  [topk] already present at ${topk_dir} ($(ls "${topk_dir}" | wc -l) files)"
    fi

    # 2. Launch scovox_node.
    echo "  [scovox] launching matched-config node"
    pkill -9 -f 'scovox_mapping_node|replica_replay_node|pointcloud_to_npz' 2>/dev/null || true
    sleep 2

    ros2 launch scovox_mapping replica_eval.launch.py \
        robot_name:=atlas \
        resolution:=0.05 \
        sdf_trunc_voxels:=3 \
        band_only_integration:=true \
        tsdf_space_carving:=false \
        carve_band:=-1.0 \
        range_decay_length:=-1.0 \
        min_range:=0.2 \
        max_range:=8.0 \
        min_tsdf_weight_publish:=5.0 \
        semantic_mode:=dirichlet \
        topk_probs_dir:="${topk_dir}" \
        > "${log}" 2>&1 &
    local LPID=$!
    sleep 5

    # 3. Replay 2000 frames at 2.0 Hz.
    echo "  [replay] streaming 2000 frames"
    python3 -m scovox_eval.replica_replay_node --ros-args \
        -p dataset_path:="${REPLICA_ROOT}/${scene}" \
        -p rate_hz:=2.0 \
        -p robot_name:=atlas \
        -p camera_poses:=true \
        -p semantic_subdir:=semantic_m2f_ade \
        -p n_scans:=2000 > /dev/null 2>&1

    # 4. Wait for scovox to finish ingesting.
    local MIN_RECV=1980 WAITED=0
    while true; do
        local last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
        if [ "${last}" -ge "${MIN_RECV}" ] 2>/dev/null; then
            echo "  [scovox] recv=${last}/2000 — settling 30s"
            sleep 30
            break
        fi
        sleep 5; WAITED=$((WAITED+5))
        if [ ${WAITED} -ge 2400 ]; then
            echo "  [scovox] WARN: timeout at recv=${last}"
            sleep 30
            break
        fi
    done

    # 5. Capture ~/pointcloud.
    echo "  [capture] ~/pointcloud → ${npz}"
    timeout 180 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/atlas/scovox_node/pointcloud \
        -p output:="${npz}" 2>&1 | tail -2 || echo "  [capture] WARN: timeout"

    # 6. Cleanup.
    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|replica_replay_node|pointcloud_to_npz' 2>/dev/null || true
    ros2 daemon stop 2>/dev/null || true
    sleep 4
    ros2 daemon start 2>/dev/null || true
    sleep 2

    # 7. Delete topk to free disk for next scene (rotation).
    if [[ -d "${topk_dir}" ]]; then
        echo "  [topk] deleting ${topk_dir} (rotation)"
        rm -rf "${topk_dir}"
    fi

    if [[ -f "${npz}" ]]; then
        echo "  [done] $(du -h "${npz}" | cut -f1) ${npz}"
    else
        echo "  [done] WARN: no npz produced"
    fi
}

for scene in "${SCENES[@]}"; do
    run_replica_scene "${scene}"
done
echo ""
echo "All Replica matched-config runs complete."
ls -la "${RES_ROOT}"/*/scovox.npz 2>&1 | head -10
