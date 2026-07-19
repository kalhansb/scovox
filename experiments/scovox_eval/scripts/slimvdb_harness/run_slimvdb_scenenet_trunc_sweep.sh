#!/bin/bash
# SLIM-VDB SceneNet truncation-distance sweep.
#
# Tests the hypothesis: SCovox's mIoU advantage over SLIM-VDB comes mostly from
# SLIM-VDB writing semantic labels into the ±sdf_trunc TSDF shell around every
# depth measurement, which leaks FPs into the immediate free-space envelope.
#
# We re-run SLIM-VDB at sdf_trunc ∈ {0.05, 0.075, 0.15} and score with the
# same strict bucket-IoU. Narrower trunc → fewer free-space FPs → mIoU ↑.
#
# Usage:
#   ./run_slimvdb_scenenet_trunc_sweep.sh 0.05      # one value
#   ./run_slimvdb_scenenet_trunc_sweep.sh 0.05 0.075 0.15   # full sweep

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
N_SCANS=300

IMG=slimvdb_docker
OVDB_LIB=/workspace/third_party_sw/slim_vdb/build/install/lib
TORCH_LIB=/workspace/third_party_sw/slim_vdb/downloads/libtorch/lib
BIN=/workspace/third_party_sw/slim_vdb/build-scenenet/bin/scenenet_pipeline
VDB2VOX=/workspace/third_party_sw/slim_vdb/build-scenenet/bin/vdb_to_voxels

# Configs live in examples/cpp/config/ (binary CWDs there).
CFG_DIR="${WS}/third_party_sw/slim_vdb/slim-vdb/examples/cpp/config"

if [[ $# -eq 0 ]]; then
    TRUNCS=(0.05 0.075 0.15)
else
    TRUNCS=("$@")
fi

# Sequences: same 13 cells.
SEQS=()
for d in "${DATA_ROOT}/train"/*/; do
    SEQS+=("$(basename "$d")")
done

for trunc in "${TRUNCS[@]}"; do
    TAG=$(printf "trunc%03d" $(python3 -c "print(int(round(${trunc}*1000)))"))
    CFG_NAME="scenenet_${TAG}.yaml"
    CFG_PATH="${CFG_DIR}/${CFG_NAME}"
    OUT_ROOT="${WS}/third_party_sw/slim_vdb/outputs/scenenet_val_${TAG}"
    mkdir -p "${OUT_ROOT}"

    # Generate config from the baseline scenenet.yaml.
    sed -E "s|^sdf_trunc: .*|sdf_trunc: ${trunc}|" "${CFG_DIR}/scenenet.yaml" > "${CFG_PATH}"
    grep "sdf_trunc" "${CFG_PATH}"

    echo ""
    echo "=========================================================="
    echo " trunc=${trunc} (tag ${TAG})  ${#SEQS[@]} cells  $(date)"
    echo "=========================================================="

    for seq in "${SEQS[@]}"; do
        OUT_SEQ="${OUT_ROOT}/${seq}"
        mkdir -p "${OUT_SEQ}"
        VOX_BIN="${OUT_SEQ}/voxels.bin"
        SEM_VDB="${OUT_SEQ}/scenenet_${seq}_${N_SCANS}_semantics.vdb"
        TIMING_LOG="${OUT_SEQ}/timing.log"

        if [[ -f "${VOX_BIN}" ]]; then
            echo "  [${seq}] voxels.bin exists, skip"
            continue
        fi
        echo "  [${seq}] $(date +%H:%M:%S)"

        docker run --rm --gpus all \
            --user "$(id -u):$(id -g)" \
            -e HOME=/tmp \
            -e DISPLAY="${DISPLAY:-:1}" \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "${WS}":/workspace \
            -v "${WS}":"${WS}" \
            -w /workspace/third_party_sw/slim_vdb/slim-vdb/examples/cpp \
            -e LD_LIBRARY_PATH="${OVDB_LIB}:${TORCH_LIB}" \
            "${IMG}" bash -c "cp ${BIN} . && ./scenenet_pipeline /workspace/data/scenenet_val_layout /workspace/third_party_sw/slim_vdb/outputs/scenenet_val_${TAG}/${seq} --sequence ${seq} --config config/${CFG_NAME} --n_scans ${N_SCANS}" \
            > "${TIMING_LOG}" 2>&1

        if [[ -f "${SEM_VDB}" ]]; then
            docker run --rm \
                --user "$(id -u):$(id -g)" \
                -e HOME=/tmp \
                -v "${WS}":/workspace \
                -v "${WS}":"${WS}" \
                -e LD_LIBRARY_PATH="${OVDB_LIB}" \
                "${IMG}" "${VDB2VOX}" \
                    "/workspace/third_party_sw/slim_vdb/outputs/scenenet_val_${TAG}/${seq}/scenenet_${seq}_${N_SCANS}_semantics.vdb" \
                    "/workspace/third_party_sw/slim_vdb/outputs/scenenet_val_${TAG}/${seq}/voxels.bin" \
                >> "${TIMING_LOG}" 2>&1
        else
            echo "    WARN: semantics.vdb missing"
        fi
    done

    # Score the sweep.
    python3 "${WS}/src/robot_sw/distributed_mapping/scovox_eval/scripts/scenenet_score_slimvdb.py" \
        --voxels_root "${OUT_ROOT}" \
        --gt_root "${WS}/data/scenenet/val_preprocessed" \
        --out_csv "${OUT_ROOT}/summary.csv"
done

echo ""
echo "[trunc-sweep] all done $(date)"
