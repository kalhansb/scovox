#!/usr/bin/env bash
# Build the standalone scovox image and run a full colcon build + test inside a
# throwaway container. Independent of the hmr_localisation image/container.
#
#   ./docker/build_and_test.sh            # build image, then build + test workspace
#   ./docker/build_and_test.sh --shell    # build image, then drop into a dev shell
#
# The workspace source is bind-mounted, so build/install/log land in the repo
# (git-ignored) and edits on the host are picked up without rebuilding the image.
#
# Resource bounds: the C++ build is template-heavy (Bonxai/Eigen), so an
# unbounded colcon fans out one g++ per host core and can exhaust RAM and hard-
# freeze the host. We cap the container's CPUs/RAM and serialise packages so at
# most SCOVOX_BUILD_JOBS compilers run at once. --memory-swap == --memory means
# no swap, so a runaway TU is OOM-killed inside the container (clean build
# failure) instead of dragging the host into swap-thrash. Override via env:
#   SCOVOX_BUILD_CPUS=6  SCOVOX_BUILD_MEM=12g  SCOVOX_BUILD_JOBS=6
set -euo pipefail

IMAGE="${SCOVOX_IMAGE:-scovox:jazzy}"
BUILD_CPUS="${SCOVOX_BUILD_CPUS:-6}"
BUILD_MEM="${SCOVOX_BUILD_MEM:-12g}"
BUILD_JOBS="${SCOVOX_BUILD_JOBS:-6}"

# Resolve the repo root (the dir containing src/) regardless of where this runs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Shared container resource caps (see header). memory-swap pinned to memory.
LIMITS=(--cpus="${BUILD_CPUS}" --memory="${BUILD_MEM}" --memory-swap="${BUILD_MEM}"
        -e MAKEFLAGS="-j${BUILD_JOBS}")

echo ">>> Building image ${IMAGE}"
docker build -t "${IMAGE}" -f "${REPO_ROOT}/docker/Dockerfile" "${REPO_ROOT}"

if [[ "${1:-}" == "--shell" ]]; then
    exec docker run --rm -it "${LIMITS[@]}" -v "${REPO_ROOT}":/scovox "${IMAGE}"
fi

echo ">>> Building + testing scovox workspace in ${IMAGE} (cpus=${BUILD_CPUS} mem=${BUILD_MEM} -j${BUILD_JOBS})"
docker run --rm "${LIMITS[@]}" -v "${REPO_ROOT}":/scovox "${IMAGE}" bash -lc '
    set -e
    colcon build --packages-select scovox_msgs scovox_core scovox_mapping \
        --parallel-workers 1 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release
    colcon test --packages-select scovox_core scovox_mapping --parallel-workers 1
    colcon test-result --all
'
