#!/usr/bin/env bash
# Host-side wrapper: launch slimvdb_docker and run the in-container build script
# for a given NCLASSES variant.
#
# Usage:
#   scripts/slimvdb_docker_build.sh <NCLASSES> <VARIANT> [LANGUAGE]
#     NCLASSES  = number of closed-set classes (e.g. 20 for KITTI, 102 for Replica)
#     VARIANT   = short name for this build (e.g. kitti, replica) — picks build-<VARIANT>/
#     LANGUAGE  = CLOSED (default) | OPEN
#
# The container gets the whole workspace mounted at /workspace (matches host path
# structure) so built binaries are usable both inside the container and from the
# host via cached build dirs.
#
# Examples:
#   scripts/slimvdb_docker_build.sh 20  kitti
#   scripts/slimvdb_docker_build.sh 102 replica

set -euo pipefail

NCLASSES="${1:?usage: $0 <NCLASSES> <VARIANT> [LANGUAGE]}"
VARIANT="${2:?usage: $0 <NCLASSES> <VARIANT> [LANGUAGE]}"
LANGUAGE="${3:-CLOSED}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE="${SLIMVDB_DOCKER_IMAGE:-slimvdb_docker}"
CONTAINER_NAME="slimvdb_build_${VARIANT}_$$"

echo "[host] workspace   : ${WS_ROOT}"
echo "[host] image       : ${IMAGE}"
echo "[host] container   : ${CONTAINER_NAME}"
echo "[host] variant     : ${VARIANT}  (NCLASSES=${NCLASSES} LANGUAGE=${LANGUAGE})"

# Log directory on the host.
LOG_DIR="${WS_ROOT}/third_party_sw/slim_vdb/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/build_${VARIANT}_$(date +%Y%m%d_%H%M%S).log"

# Run the build inside the container.
# --user: match host UID/GID so file ownership stays sane outside the container.
# --rm: container is disposable; all persistent state lives in mounted build dirs.
# --gpus all: NanoVDB + libtorch need CUDA available at link/build time.
set -x
docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "${WS_ROOT}:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    bash /workspace/third_party_sw/slim_vdb/scripts/slimvdb_build_inside.sh \
         "${NCLASSES}" "${VARIANT}" "${LANGUAGE}" 2>&1 | tee "${LOG_FILE}"
set +x

echo ""
echo "[host] log saved to: ${LOG_FILE}"
