#!/bin/bash
# KITTI sdf_trunc sweep for SLIM-VDB, mirroring run_slimvdb_kitti_all.sh.
# Each run reuses the per-sequence kitti.yaml but with sdf_trunc overridden
# via kitti_trunc{010,015,020}.yaml configs. The KITTI default of sdf_trunc=0.30
# stays as `kitti.yaml` and is NOT re-run here (existing output already on disk).
#
# Usage:
#   bash run_slimvdb_kitti_trunc_sweep.sh           # seq 08 only (quick read)
#   SEQS="06 07 08 09 10" bash run_slimvdb_kitti_trunc_sweep.sh
#   WS=/path/to/hmr_exploration_ws bash run_slimvdb_kitti_trunc_sweep.sh
set -euo pipefail

SEQS=${SEQS:-08}
N_SCANS=100
TRUNCS=(010 015 020)
# Workspace root resolves three ways (in priority order):
#   1) WS env override
#   2) walk up from script location until a dir with third_party_sw/slim_vdb exists
#   3) hardcoded fallback (author's machine)
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
echo "WS=${WS}"
OUT_ROOT="${WS}/third_party_sw/slim_vdb/outputs"

IMG=slimvdb_docker
OVDB_LIB=/workspace/third_party_sw/slim_vdb/build/install/lib
TORCH_LIB=/workspace/third_party_sw/slim_vdb/downloads/libtorch/lib
BIN=/workspace/third_party_sw/slim_vdb/build-kitti/bin/kitti_pipeline
POLLER="${WS}/third_party_sw/slim_vdb/scripts/telemetry_poller.py"

for trunc in "${TRUNCS[@]}"; do
    BUCKET="${OUT_ROOT}/kitti_trunc${trunc}"
    mkdir -p "${BUCKET}"
    for seq in ${SEQS}; do
        OUT_SEQ="${BUCKET}/${seq}"
        mkdir -p "${OUT_SEQ}"
        TIMING_LOG="${OUT_SEQ}/timing.log"
        TELEMETRY_CSV="${OUT_SEQ}/telemetry.csv"
        echo ""
        echo "============================================================"
        echo " SLIM-VDB on KITTI seq ${seq}  trunc=0.${trunc}  (n_scans=${N_SCANS})"
        echo "============================================================"

        python3 "${POLLER}" --period 0.5 --out "${TELEMETRY_CSV}" &
        POLL_PID=$!

        docker run --rm --gpus all \
            --user "$(id -u):$(id -g)" \
            -e HOME=/tmp \
            -e DISPLAY="${DISPLAY:-:1}" \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "${WS}":/workspace \
            -w /workspace/third_party_sw/slim_vdb/slim-vdb/examples/cpp \
            -e LD_LIBRARY_PATH="${OVDB_LIB}:${TORCH_LIB}" \
            "${IMG}" bash -c "cp ${BIN} . && ./kitti_pipeline /workspace/data/semantickitti/dataset /workspace/third_party_sw/slim_vdb/outputs/kitti_trunc${trunc}/${seq} --sequence ${seq} --config config/kitti_trunc${trunc}.yaml --n_scans ${N_SCANS}" \
            2>&1 | tee "${TIMING_LOG}"

        kill "${POLL_PID}" 2>/dev/null || true
        wait "${POLL_PID}" 2>/dev/null || true

        python3 - <<PYEOF
import csv
with open("${TELEMETRY_CSV}") as f:
    rows = list(csv.DictReader(f))
if rows:
    gpu = max(float(r["gpu_mem_mb"]) for r in rows)
    sysm = max(float(r["sys_used_mb"]) for r in rows)
    print(f"[seq ${seq} trunc=0.${trunc}] peak GPU={gpu:.0f} MB | peak SYS={sysm:.0f} MB | samples={len(rows)}")
PYEOF
    done
done

echo ""
echo "Sweep done. Outputs in ${OUT_ROOT}/kitti_trunc{010,015,020}/<seq>/"
