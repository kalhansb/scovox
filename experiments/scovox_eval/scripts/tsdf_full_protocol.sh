#!/bin/bash
# Full standard-protocol re-runs with TSDF integration enabled, mirroring the
# published evaluation:
#   - SemanticKITTI seq 06-10 × 100 scans × 10 cm × PolarSeg labels
#   - Replica all 8 scenes × 2000 frames × 5 cm × Mask2Former ADE labels
#
# Outputs go into `_tsdf/` sibling directories so the pre-TSDF results stay
# untouched; the SLIM-VDB outputs in third_party_sw/slim_vdb/outputs/ are the
# primary comparison reference.
#
# Usage: $0 [kitti|replica|both]
set -eo pipefail

WS=$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws
EVAL_PKG="${WS}/src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT="${WS}/data/semantickitti/dataset"
REPLICA_ROOT="${WS}/data/replica_niceslam"
KITTI_RES_ROOT="${EVAL_PKG}/results/semantickitti_polarseg_10cm_tsdf"
REPL_RES_ROOT="${EVAL_PKG}/results/replica_m2f_ade_5cm_tsdf"

KITTI_SEQS=(06 07 08 09 10)
REPL_SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

mkdir -p "${KITTI_RES_ROOT}" "${REPL_RES_ROOT}"

run_kitti_seq() {
    local seq="$1"
    local seq_int=$((10#$seq))
    local cell_dir="${KITTI_RES_ROOT}/${seq}"
    local out="${cell_dir}/scovox.npz"
    local log="${cell_dir}/scovox_run.log"
    mkdir -p "${cell_dir}"
    if [[ -f "${out}" ]]; then echo "[kitti ${seq}] skip"; return; fi
    echo "━━ kitti seq ${seq} ━━"
    ros2 launch scovox_mapping semantickitti_eval.launch.py \
        robot_name:=atlas \
        resolution:=0.10 \
        semantic_mode:=dirichlet \
        sdf_trunc_voxels:=3 \
        tsdf_space_carving:=false \
        > "${log}" 2>&1 &
    local LPID=$!
    sleep 4
    python3 -m scovox_eval.semantickitti_replay_node --ros-args \
        -p dataset_path:="${KITTI_ROOT}" \
        -p sequence:=${seq_int} \
        -p rate_hz:=0.5 \
        -p robot_name:=atlas \
        -p max_range:=30.0 \
        -p min_range:=1.0 \
        -p labels_subdir:=predictions \
        -p n_scans:=100
    local MIN_RECV=99 WAITED=0
    while true; do
        local last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
        if [ "${last}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 10; break; fi
        sleep 3; WAITED=$((WAITED+3))
        [ ${WAITED} -ge 600 ] && { sleep 10; break; }
    done
    timeout 60 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/atlas/scovox_node/pointcloud \
        -p output:="${out}" || echo "  WARN: capture timeout"
    kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|semantickitti_replay_node|pointcloud_to_npz' 2>/dev/null || true
    sleep 3
}

run_replica_scene() {
    local scene="$1"
    local cell_dir="${REPL_RES_ROOT}/${scene}"
    local out="${cell_dir}/scovox.npz"
    local log="${cell_dir}/scovox_run.log"
    mkdir -p "${cell_dir}"
    if [[ -f "${out}" ]]; then echo "[replica ${scene}] skip"; return; fi
    echo "━━ replica ${scene} ━━"
    ros2 launch scovox_mapping replica_eval.launch.py \
        robot_name:=atlas \
        resolution:=0.05 \
        sdf_trunc_voxels:=3 \
        tsdf_space_carving:=false \
        > "${log}" 2>&1 &
    local LPID=$!
    sleep 4
    python3 -m scovox_eval.replica_replay_node --ros-args \
        -p dataset_path:="${REPLICA_ROOT}/${scene}" \
        -p rate_hz:=2.0 \
        -p robot_name:=atlas \
        -p camera_poses:=true \
        -p semantic_subdir:=semantic_m2f_ade \
        -p n_scans:=2000
    local MIN_RECV=1980 WAITED=0
    while true; do
        local last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1 || echo 0)
        if [ "${last}" -ge "${MIN_RECV}" ] 2>/dev/null; then sleep 30; break; fi
        sleep 5; WAITED=$((WAITED+5))
        [ ${WAITED} -ge 1800 ] && { sleep 30; break; }
    done
    timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
        -p topic:=/atlas/scovox_node/pointcloud \
        -p output:="${out}" || echo "  WARN: capture timeout"
    kill ${LPID} 2>/dev/null || true; wait ${LPID} 2>/dev/null || true
    pkill -9 -f 'scovox_mapping_node|replica_replay_node|pointcloud_to_npz' 2>/dev/null || true
    ros2 daemon stop 2>/dev/null || true; sleep 4; ros2 daemon start 2>/dev/null || true
    sleep 2
}

case "${1:-both}" in
    kitti) for s in "${KITTI_SEQS[@]}"; do run_kitti_seq "$s"; done ;;
    replica) for sc in "${REPL_SCENES[@]}"; do run_replica_scene "$sc"; done ;;
    both|"")
        for s in "${KITTI_SEQS[@]}"; do run_kitti_seq "$s"; done
        for sc in "${REPL_SCENES[@]}"; do run_replica_scene "$sc"; done ;;
    *) echo "usage: $0 [kitti|replica|both]"; exit 1 ;;
esac
echo "done."
