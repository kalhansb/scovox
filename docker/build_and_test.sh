#!/usr/bin/env bash
# Build the standalone scovox image and run a full colcon build + test inside a
# throwaway container. Independent of the hmr_localisation image/container.
#
#   ./docker/build_and_test.sh            # build image, then build + test workspace
#   ./docker/build_and_test.sh --shell    # build image, then drop into a dev shell
#
# The workspace source is bind-mounted, so build/install/log land in the repo
# (git-ignored) and edits on the host are picked up without rebuilding the image.
set -euo pipefail

IMAGE="${SCOVOX_IMAGE:-scovox:jazzy}"

# Resolve the repo root (the dir containing src/) regardless of where this runs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo ">>> Building image ${IMAGE}"
docker build -t "${IMAGE}" -f "${REPO_ROOT}/docker/Dockerfile" "${REPO_ROOT}"

if [[ "${1:-}" == "--shell" ]]; then
    exec docker run --rm -it -v "${REPO_ROOT}":/scovox "${IMAGE}"
fi

echo ">>> Building + testing scovox workspace in ${IMAGE}"
docker run --rm -v "${REPO_ROOT}":/scovox "${IMAGE}" bash -lc '
    set -e
    colcon build --packages-select scovox_msgs scovox_core scovox_mapping \
        --cmake-args -DCMAKE_BUILD_TYPE=Release
    colcon test --packages-select scovox_core scovox_mapping
    colcon test-result --all
'
