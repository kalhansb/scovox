#!/bin/bash
# Runs SLIM-VDB kitti_pipeline on sequences 06, 07, 08, 09, 10 — first 100
# frames each (matches the SLIM-VDB paper protocol), using PolarSeg predicted
# labels. Emits per-frame "TIMING ..." lines + GPU/system RAM polling CSV
# for downstream stats (median/P95 FPS, peak memory).
set -euo pipefail

SEQUENCES=(06 07 08 09 10)
N_SCANS=100
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
OUT_ROOT=${WS}/third_party_sw/slim_vdb/outputs/kitti
mkdir -p "${OUT_ROOT}"

IMG=slimvdb_docker
OVDB_LIB=/workspace/third_party_sw/slim_vdb/build/install/lib
TORCH_LIB=/workspace/third_party_sw/slim_vdb/downloads/libtorch/lib
BIN=/workspace/third_party_sw/slim_vdb/build-kitti/bin/kitti_pipeline
POLLER="${WS}/third_party_sw/slim_vdb/scripts/telemetry_poller.py"

for seq in "${SEQUENCES[@]}"; do
    OUT_SEQ="${OUT_ROOT}/${seq}"
    mkdir -p "${OUT_SEQ}"
    TIMING_LOG="${OUT_SEQ}/timing.log"
    TELEMETRY_CSV="${OUT_SEQ}/telemetry.csv"
    echo ""
    echo "============================================================"
    echo " SLIM-VDB on KITTI seq ${seq}  (n_scans=${N_SCANS})"
    echo "============================================================"

    # Start memory poller in background, 0.5 s period.
    python3 "${POLLER}" --period 0.5 --out "${TELEMETRY_CSV}" &
    POLL_PID=$!

    # Run docker; tee stdout into the timing log.
    docker run --rm --gpus all \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -e DISPLAY="${DISPLAY:-:1}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "${WS}":/workspace \
        -w /workspace/third_party_sw/slim_vdb/slim-vdb/examples/cpp \
        -e LD_LIBRARY_PATH="${OVDB_LIB}:${TORCH_LIB}" \
        "${IMG}" bash -c "cp ${BIN} . && ./kitti_pipeline /workspace/data/semantickitti/dataset /workspace/third_party_sw/slim_vdb/outputs/kitti/${seq} --sequence ${seq} --config config/kitti.yaml --n_scans ${N_SCANS}" \
        2>&1 | tee "${TIMING_LOG}"

    kill "${POLL_PID}" 2>/dev/null || true
    wait "${POLL_PID}" 2>/dev/null || true

    # Peak GPU + system memory from the poller CSV
    python3 - <<PYEOF
import csv
with open("${TELEMETRY_CSV}") as f:
    rows = list(csv.DictReader(f))
if rows:
    gpu = max(float(r["gpu_mem_mb"]) for r in rows)
    sysm = max(float(r["sys_used_mb"]) for r in rows)
    dgpu = max(float(r["gpu_delta_mb"]) for r in rows)
    dsys = max(float(r["sys_delta_mb"]) for r in rows)
    print(f"[seq ${seq}] peak GPU={gpu:.0f} MB (Δ{dgpu:.0f}) | peak SYS={sysm:.0f} MB (Δ{dsys:.0f}) | samples={len(rows)}")
PYEOF
done

echo ""
echo "All sequences done. Outputs + telemetry in ${OUT_ROOT}/<seq>/"
