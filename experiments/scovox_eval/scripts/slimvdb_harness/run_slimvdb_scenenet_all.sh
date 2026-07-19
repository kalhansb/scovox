#!/bin/bash
# SLIM-VDB scenenet_pipeline on the same 13 SceneNet val trajectories that
# the SCovox iter6 batch ran on (see ../../../src/robot_sw/distributed_mapping/
# scovox_eval/results/scenenet_val_iter6/summary.csv).
#
# Paired build: NCLASSES=14 (13 NYU13 classes + 0 Unknown), LANGUAGE=CLOSED.
# Built by: scripts/slimvdb_docker_build.sh 14 scenenet CLOSED
#
# Layout expected:
#   data/scenenet_val_layout/
#     weights/libtorch_nyuv2.pt           ← model file (always loaded, even with
#                                            realtime_segmentation: False)
#     train/0_<traj>/
#       depth/NNNN.png                    ← mm uint16
#       prediction/NNNN.png               ← uint8 class id 0..13 (symlink to
#                                            ground_truth_labels/ for GT-input
#                                            head-to-head)
#       poses.txt                         ← 300×12 row-major camera-to-world
#       intrinsics.txt                    ← fx/fy/cx/cy from fov=45°×60° @ 320×240
#
# Outputs per cell:
#   outputs/scenenet_val/0_<traj>/
#     scenenet_0_<traj>_300_scans.vdb         (TSDF grid)
#     scenenet_0_<traj>_300_semantics.vdb     (Dirichlet count grid)
#     voxels.bin                              (vdb_to_voxels dump for scoring)
#     timing.log
#     telemetry.csv
#
# Usage:
#   ./run_slimvdb_scenenet_all.sh                       # all 13 cells
#   ./run_slimvdb_scenenet_all.sh 0_223 0_485           # specific cells

set -euo pipefail

# Resolve workspace root: WS env override → walk up from script location → fallback.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${WS:-}" ]]; then
    _cand="${_script_dir}"
    while [[ "${_cand}" != "/" && ! -d "${_cand}/third_party_sw/slim_vdb" ]]; do
        _cand="$(dirname "${_cand}")"
    done
    if [[ -d "${_cand}/third_party_sw/slim_vdb" ]]; then
        WS="${_cand}"
    else
        WS=/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws
    fi
fi
DATA_ROOT="${WS}/data/scenenet_val_layout"
OUT_ROOT="${WS}/third_party_sw/slim_vdb/outputs/scenenet_val"
N_SCANS=300
CFG=config/scenenet.yaml

IMG=slimvdb_docker
OVDB_LIB=/workspace/third_party_sw/slim_vdb/build/install/lib
TORCH_LIB=/workspace/third_party_sw/slim_vdb/downloads/libtorch/lib
BIN=/workspace/third_party_sw/slim_vdb/build-scenenet/bin/scenenet_pipeline
VDB2VOX=/workspace/third_party_sw/slim_vdb/build-scenenet/bin/vdb_to_voxels
POLLER="${WS}/third_party_sw/slim_vdb/scripts/telemetry_poller.py"

mkdir -p "${OUT_ROOT}"

# Pick sequences.
if [[ $# -gt 0 ]]; then
    SEQS=("$@")
else
    SEQS=()
    for d in "${DATA_ROOT}/train"/*/; do
        SEQS+=("$(basename "$d")")
    done
fi
echo "[slimvdb-scenenet] running ${#SEQS[@]} sequences: ${SEQS[*]}"

for seq in "${SEQS[@]}"; do
    OUT_SEQ="${OUT_ROOT}/${seq}"
    mkdir -p "${OUT_SEQ}"
    TIMING_LOG="${OUT_SEQ}/timing.log"
    TELEMETRY_CSV="${OUT_SEQ}/telemetry.csv"
    SEM_VDB="${OUT_SEQ}/scenenet_${seq}_${N_SCANS}_semantics.vdb"
    VOX_BIN="${OUT_SEQ}/voxels.bin"

    if [[ -f "${VOX_BIN}" ]]; then
        echo "[${seq}] voxels.bin exists; skipping"
        continue
    fi

    echo ""
    echo "============================================================"
    echo " SLIM-VDB SceneNet ${seq}  n_scans=${N_SCANS}  $(date)"
    echo "============================================================"

    python3 "${POLLER}" --period 1.0 --out "${TELEMETRY_CSV}" &
    POLL_PID=$!

    # The pipeline binary's CWD must contain config/scenenet.yaml; we cd into
    # examples/cpp/ (where config/ lives in the source tree) and run from there.
    # We bind-mount the host workspace path twice — once at /workspace (legacy
    # convention shared with all other runners) and once at the host path
    # itself, so the host-absolute symlinks under data/scenenet_val_layout/...
    # and data/scenenet/val_preprocessed/.../depth/*.png resolve correctly
    # inside the container without rewriting ~10k symlinks to be relative.
    # scenenet_pipeline's Render() step uses Open3D/OpenCV which want a GUI
    # backend (GTK) — even for offscreen rendering. Forward the host X
    # display so cvInitSystem can find a usable connection.
    docker run --rm --gpus all \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -e DISPLAY="${DISPLAY:-:1}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "${WS}":/workspace \
        -v "${WS}":"${WS}" \
        -w /workspace/third_party_sw/slim_vdb/slim-vdb/examples/cpp \
        -e LD_LIBRARY_PATH="${OVDB_LIB}:${TORCH_LIB}" \
        "${IMG}" bash -c "cp ${BIN} . && ./scenenet_pipeline /workspace/data/scenenet_val_layout /workspace/third_party_sw/slim_vdb/outputs/scenenet_val/${seq} --sequence ${seq} --config ${CFG} --n_scans ${N_SCANS}" \
        2>&1 | tee "${TIMING_LOG}"

    kill "${POLL_PID}" 2>/dev/null || true
    wait "${POLL_PID}" 2>/dev/null || true

    # Convert semantics .vdb → voxels.bin (x,y,z,argmax) for our Python scorer.
    if [[ -f "${SEM_VDB}" ]]; then
        echo "[${seq}] dumping voxels.bin"
        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -e HOME=/tmp \
            -v "${WS}":/workspace \
            -v "${WS}":"${WS}" \
            -e LD_LIBRARY_PATH="${OVDB_LIB}" \
            "${IMG}" "${VDB2VOX}" \
                "/workspace/third_party_sw/slim_vdb/outputs/scenenet_val/${seq}/scenenet_${seq}_${N_SCANS}_semantics.vdb" \
                "/workspace/third_party_sw/slim_vdb/outputs/scenenet_val/${seq}/voxels.bin" \
            2>&1 | tee -a "${TIMING_LOG}"
    else
        echo "[${seq}] WARN: semantics.vdb missing — pipeline likely failed"
    fi
done

echo ""
echo "[slimvdb-scenenet] all done $(date)"
echo "Outputs in ${OUT_ROOT}/<seq>/"
echo "Next: python3 ${WS}/src/robot_sw/distributed_mapping/scovox_eval/scripts/scenenet_score_slimvdb.py \\"
echo "        --voxels_root ${OUT_ROOT} --gt_root ${WS}/data/scenenet/val_preprocessed \\"
echo "        --out_csv ${OUT_ROOT}/summary.csv"
