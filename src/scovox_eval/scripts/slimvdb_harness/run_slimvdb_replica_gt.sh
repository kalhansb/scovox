#!/bin/bash
# SLIM-VDB replica_pipeline on all 8 Replica scenes (2000 frames each) using
# Replica GT semantics from data/replica_niceslam/<scene>/semantic_gt_fixed/.
# Paired build: NCLASSES=102 (101 Replica classes + 0 void).
# Emits per-frame "TIMING ..." lines + GPU/system RAM polling CSV per scene.
set -euo pipefail

SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
N_SCANS=2000
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
OUT_ROOT=${WS}/third_party_sw/slim_vdb/outputs/replica_gt
mkdir -p "${OUT_ROOT}"

IMG=slimvdb_docker
OVDB_LIB=/workspace/third_party_sw/slim_vdb/build/install/lib
TORCH_LIB=/workspace/third_party_sw/slim_vdb/downloads/libtorch/lib
BIN=/workspace/third_party_sw/slim_vdb/build-replica/bin/replica_pipeline
CFG=config/replica_gt.yaml
POLLER="${WS}/third_party_sw/slim_vdb/scripts/telemetry_poller.py"

for scene in "${SCENES[@]}"; do
    OUT_SCENE="${OUT_ROOT}/${scene}"
    mkdir -p "${OUT_SCENE}"
    TIMING_LOG="${OUT_SCENE}/timing.log"
    TELEMETRY_CSV="${OUT_SCENE}/telemetry.csv"
    echo ""
    echo "============================================================"
    echo " SLIM-VDB GT on Replica scene ${scene}  (n_scans=${N_SCANS})"
    echo "============================================================"

    python3 "${POLLER}" --period 1.0 --out "${TELEMETRY_CSV}" &
    POLL_PID=$!

    docker run --rm --gpus all \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -e DISPLAY="${DISPLAY:-:1}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "${WS}":/workspace \
        -w /workspace/third_party_sw/slim_vdb/slim-vdb/examples/cpp \
        -e LD_LIBRARY_PATH="${OVDB_LIB}:${TORCH_LIB}" \
        "${IMG}" bash -c "cp ${BIN} . && ./replica_pipeline /workspace/data/replica_niceslam ${OUT_ROOT}/${scene} --sequence ${scene} --config ${CFG} --n_scans ${N_SCANS}" \
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
    dgpu = max(float(r["gpu_delta_mb"]) for r in rows)
    dsys = max(float(r["sys_delta_mb"]) for r in rows)
    print(f"[scene ${scene}] peak GPU={gpu:.0f} MB (Δ{dgpu:.0f}) | peak SYS={sysm:.0f} MB (Δ{dsys:.0f}) | samples={len(rows)}")
PYEOF
done

echo ""
echo "All scenes done. Outputs + telemetry in ${OUT_ROOT}/<scene>/"
