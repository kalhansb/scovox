#!/bin/bash
# TSDF surface extraction smoke test — exercises the new
# extractZeroCrossing-based publishTSDFPointCloud and the ExtractMesh service
# added on 2026-04-29. Runs:
#   - SemanticKITTI seq 08 (polarseg predictions, 100 frames, 10 cm)
#   - Replica room0 (m2f ADE labels, 2000 frames, 5 cm)
# and captures, alongside the usual occupancy npz:
#   - ~/tsdf_pointcloud → scovox_tsdf.npz (sub-voxel zero-crossings + semantic_class)
#   - ~/extract_mesh service → scovox_mesh.ply (marching cubes)
#
# Outputs land in results/tsdf_surface_eval/<scene>/. Re-run safe (skip-if-exists).
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
REPLICA_ROOT="${WS}/data/replica_niceslam"
OUT_ROOT="${EVAL_PKG}/results/tsdf_surface_eval"
KITTI_RES="${OUT_ROOT}/kitti_seq08"
REPL_RES="${OUT_ROOT}/replica_room0"

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${KITTI_RES}" "${REPL_RES}"

capture_topic() {
    # capture_topic <topic> <output_npz>
    local TOPIC="$1" OUT="$2"
    if [[ -f "${OUT}" ]]; then return 0; fi
    timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:="${TOPIC}" \
        -p output:="${OUT}" 2>&1 || echo "  WARN: capture timeout ${TOPIC}"
}

call_extract_mesh() {
    # call_extract_mesh <node_ns> <out_ply> <min_weight>
    local NS="$1" OUT="$2" MINW="$3"
    if [[ -f "${OUT}" ]]; then return 0; fi
    echo "  extract_mesh: ${NS}/extract_mesh → ${OUT}"
    timeout 120 ros2 service call "${NS}/extract_mesh" \
        scovox_msgs/srv/ExtractMesh \
        "{min_weight: ${MINW}, output_path: '${OUT}'}" 2>&1 | tail -10 \
        || echo "  WARN: extract_mesh failed"
}

run_kitti() {
    local LOG="${KITTI_RES}/scovox_run.log"
    local OCC_NPZ="${KITTI_RES}/scovox_occupancy.npz"
    local TSDF_NPZ="${KITTI_RES}/scovox_tsdf.npz"
    local PLY="${KITTI_RES}/scovox_mesh.ply"
    if [[ -f "${OCC_NPZ}" && -f "${TSDF_NPZ}" && -f "${PLY}" ]]; then
        echo "[kitti] all artefacts present, skipping run"
        return
    fi
    echo "━━ KITTI seq08 polarseg, TSDF surface smoke ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=ablation \
        resolution:=0.10 \
        semantic_mode:=dirichlet \
        semantic_occ_gate:=0.0 \
        range_decay_length:=50.0 \
        w_occ:=6.0 \
        w_free:=1.0 \
        carve_skip_occ_threshold:=0.4 \
        kappa0:=2.0 \
        evidence_saturation:=1000.0 \
        semantic_min_confidence:=0.1 \
        sdf_trunc_voxels:=3 \
        tsdf_space_carving:=false \
        carve_band:=-1.0 \
        > "${LOG}" 2>&1 &
    local LPID=$!
    sleep 4
    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="${KITTI_ROOT}" \
        -p sequence:=8 \
        -p rate_hz:=0.5 \
        -p robot_name:=ablation \
        -p max_range:=30.0 \
        -p min_range:=1.0 \
        -p labels_subdir:=predictions \
        -p n_scans:=100
    local MIN_RECV=99 WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then
            echo "  recv=${LAST}/100, grace 10s"; sleep 10; break
        fi
        sleep 3; WAITED=$((WAITED+3))
        [ ${WAITED} -ge 600 ] && { echo "  WARN: timed out at recv=${LAST}"; sleep 10; break; }
    done
    capture_topic /ablation/scovox_node/pointcloud      "${OCC_NPZ}"
    capture_topic /ablation/scovox_node/tsdf_pointcloud "${TSDF_NPZ}"
    call_extract_mesh /ablation/scovox_node "${PLY}" 1.0
    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' 2>/dev/null || true
    sleep 3
}

run_replica() {
    local LOG="${REPL_RES}/scovox_run.log"
    local OCC_NPZ="${REPL_RES}/scovox_occupancy.npz"
    local TSDF_NPZ="${REPL_RES}/scovox_tsdf.npz"
    local PLY="${REPL_RES}/scovox_mesh.ply"
    if [[ -f "${OCC_NPZ}" && -f "${TSDF_NPZ}" && -f "${PLY}" ]]; then
        echo "[replica] all artefacts present, skipping run"
        return
    fi
    echo "━━ Replica room0 m2f, TSDF surface smoke ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        robot_name:=ablation \
        resolution:=0.05 \
        semantic_occ_gate:=0.0 \
        range_decay_length:=-1.0 \
        w_occ:=2.0 \
        w_free:=1.0 \
        carve_skip_occ_threshold:=0.7 \
        kappa0:=2.0 \
        evidence_saturation:=1000.0 \
        semantic_min_confidence:=0.1 \
        sdf_trunc_voxels:=3 \
        tsdf_space_carving:=false \
        carve_band:=-1.0 \
        > "${LOG}" 2>&1 &
    local LPID=$!
    sleep 4
    python3 -m scovox_eval.replica_replay_node --ros-args \
        -p dataset_path:="${REPLICA_ROOT}/room0" \
        -p rate_hz:=2.0 \
        -p robot_name:=ablation \
        -p camera_poses:=true \
        -p semantic_subdir:=semantic_m2f_ade \
        -p n_scans:=2000
    local MIN_RECV=1980 WAITED=0
    while true; do
        local LAST=$(grep -oP 'recv=\K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo 0)
        if [ "${LAST}" -ge "${MIN_RECV}" ] 2>/dev/null; then
            echo "  recv=${LAST}/2000, grace 30s"; sleep 30; break
        fi
        sleep 5; WAITED=$((WAITED+5))
        [ ${WAITED} -ge 1800 ] && { echo "  WARN: timed out at recv=${LAST}"; sleep 30; break; }
    done
    capture_topic /ablation/scovox_node/pointcloud      "${OCC_NPZ}"
    capture_topic /ablation/scovox_node/tsdf_pointcloud "${TSDF_NPZ}"
    call_extract_mesh /ablation/scovox_node "${PLY}" 1.0
    kill ${LPID} 2>/dev/null || true
    wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|replica_replay_node|pointcloud_to_npz' 2>/dev/null || true
    ros2 daemon stop 2>/dev/null || true; sleep 4; ros2 daemon start 2>/dev/null || true
    sleep 2
}

case "${1:-both}" in
    kitti)   run_kitti ;;
    replica) run_replica ;;
    both|"") run_kitti; run_replica ;;
    *) echo "usage: $0 [kitti|replica|both]"; exit 1 ;;
esac
echo "done. outputs: ${KITTI_RES}, ${REPL_RES}"
