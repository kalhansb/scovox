#!/usr/bin/env bash
# Runs INSIDE the slimvdb_docker container.
#
# Builds (once) the umfieldrobotics OpenVDB fork into a private prefix under
# third_party_sw/slim_vdb/build/install, then builds SLIM-VDB + examples for
# one NCLASSES variant into a variant-specific build/install tree.
#
# Usage (inside container):
#   /workspace/third_party_sw/slim_vdb/scripts/slimvdb_build_inside.sh <NCLASSES> <VARIANT_NAME>
#     e.g. slimvdb_build_inside.sh 20 kitti
#          slimvdb_build_inside.sh 102 replica
#
# Idempotent: re-running only rebuilds what changed.

set -euo pipefail

NCLASSES="${1:?missing NCLASSES}"
VARIANT="${2:?missing VARIANT name (e.g. kitti, replica)}"
LANGUAGE="${3:-CLOSED}"                       # CLOSED or OPEN
JOBS="${JOBS:-$(nproc)}"

# Layout inside the container — all paths are host-mounted.
SLIM_ROOT="/workspace/third_party_sw/slim_vdb"
OVDB_SRC="${SLIM_ROOT}/openvdb"
OVDB_BUILD="${SLIM_ROOT}/build/openvdb"
OVDB_PREFIX="${SLIM_ROOT}/build/install"

SLIMVDB_SRC="${SLIM_ROOT}/slim-vdb"
VARIANT_ROOT="${SLIM_ROOT}/build-${VARIANT}"
SLIMVDB_BUILD="${VARIANT_ROOT}/slim-vdb"
EXAMPLES_BUILD="${VARIANT_ROOT}/examples-cpp"
VARIANT_PREFIX="${VARIANT_ROOT}/install"

echo "============================================================"
echo " SLIM-VDB inside-docker build"
echo "   NCLASSES : ${NCLASSES}"
echo "   VARIANT  : ${VARIANT}"
echo "   LANGUAGE : ${LANGUAGE}"
echo "   JOBS     : ${JOBS}"
echo "   prefix   : ${VARIANT_PREFIX}"
echo "============================================================"

mkdir -p "${OVDB_BUILD}" "${OVDB_PREFIX}" "${SLIMVDB_BUILD}" "${EXAMPLES_BUILD}" "${VARIANT_PREFIX}"

# ───────────────────────────── 1/3  OpenVDB (+ NanoVDB + CUDA) ──────────────
if [[ ! -f "${OVDB_PREFIX}/lib/libopenvdb.so" ]]; then
    echo ">>> [1/3] Configuring + building OpenVDB fork into ${OVDB_PREFIX}"
    pushd "${OVDB_BUILD}" >/dev/null
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${OVDB_PREFIX}" \
        -DOPENVDB_BUILD_PYTHON_MODULE=OFF \
        -DUSE_NUMPY=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DUSE_ZLIB=OFF \
        -DOPENVDB_BUILD_NANOVDB=ON \
        -DNANOVDB_USE_CUDA=ON \
        -DOPENVDB_BUILD_BINARIES=OFF \
        -DOPENVDB_BUILD_UNITTESTS=OFF \
        -DCMAKE_INSTALL_RPATH="${OVDB_PREFIX}/lib" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        "${OVDB_SRC}"
    make -j"${JOBS}"
    make install
    popd >/dev/null
else
    echo ">>> [1/3] OpenVDB already installed at ${OVDB_PREFIX}, skipping"
fi

# ───────────────────────────── 2/3  SLIM-VDB library ────────────────────────
# SLIM-VDB uses find_package(OpenVDB) in CMake MODULE mode — OpenVDB ships
# FindOpenVDB.cmake (not OpenVDBConfig.cmake), so CMAKE_MODULE_PATH must include
# our OpenVDB prefix's cmake/OpenVDB dir. CMAKE_PREFIX_PATH alone isn't enough.
# Export CPLUS_INCLUDE_PATH because SLIM-VDB's CMakeLists overwrites
# CMAKE_CXX_FLAGS (kills our -I), so CXX compiles can't find openvdb headers.
export CPLUS_INCLUDE_PATH="${OVDB_PREFIX}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${OVDB_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
echo ">>> [2/3] Configuring + building SLIM-VDB core (NCLASSES=${NCLASSES})"
pushd "${SLIMVDB_BUILD}" >/dev/null
# `cuda_lib` is built via the deprecated FindCUDA `cuda_add_library`, which
# ignores CMAKE_CUDA_FLAGS. It reads the CUDA_NVCC_FLAGS cache variable. Pass
# the OpenVDB include via both CUDA_NVCC_FLAGS (nvcc) and CMAKE_CXX_FLAGS
# (host cc) so every compile unit can find <nanovdb/NanoVDB.h>.
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${VARIANT_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${OVDB_PREFIX}" \
    -DCMAKE_MODULE_PATH="${OVDB_PREFIX}/lib/cmake/OpenVDB" \
    -DUSE_SYSTEM_OPENVDB=ON \
    -DCMAKE_CXX_FLAGS="-I${OVDB_PREFIX}/include" \
    -DCMAKE_CUDA_FLAGS="-I${OVDB_PREFIX}/include" \
    -DCUDA_NVCC_FLAGS="-I${OVDB_PREFIX}/include" \
    -DSLIMVDB_LANGUAGE="${LANGUAGE}" \
    -DSLIMVDB_NCLASSES="${NCLASSES}" \
    "${SLIMVDB_SRC}"
make -j"${JOBS}"
make install
popd >/dev/null

# ───────────────────────────── 3/3  Examples (C++) ──────────────────────────
# Same story as step 2: examples/cpp calls find_package(SLIMVDB), whose
# SLIMVDBConfig.cmake find_dependency(OpenVDB)'s through CMAKE_MODULE_PATH.
# Also requires libtorch — downloaded once into ${SLIM_ROOT}/downloads/libtorch.
LIBTORCH_PREFIX="${SLIM_ROOT}/downloads/libtorch"
if [[ ! -f "${LIBTORCH_PREFIX}/share/cmake/Torch/TorchConfig.cmake" ]]; then
    echo "ERROR: libtorch not found at ${LIBTORCH_PREFIX}."
    echo "       Download + unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cu121.zip into that path."
    exit 1
fi

echo ">>> [3/3] Configuring + building example pipelines (${VARIANT})"
# The datasets/ OBJECT library in examples/cpp has a PRIVATE include of
# ${SLIMVDB_SRC}/src only — not deep enough to resolve `slimvdb/utils/Utils.h`.
# Exporting CPLUS_INCLUDE_PATH with the installed SLIM-VDB prefix fixes it
# without patching upstream CMakeLists.
export CPLUS_INCLUDE_PATH="${VARIANT_PREFIX}/include:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="${VARIANT_PREFIX}/lib:${LIBRARY_PATH}"
pushd "${EXAMPLES_BUILD}" >/dev/null
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${OVDB_PREFIX};${VARIANT_PREFIX};${LIBTORCH_PREFIX}" \
    -DCMAKE_MODULE_PATH="${OVDB_PREFIX}/lib/cmake/OpenVDB" \
    -DLIBTORCH_PREFIX="${LIBTORCH_PREFIX}" \
    "${SLIMVDB_SRC}/examples/cpp"
make -j"${JOBS}"
popd >/dev/null

# Upstream examples CMake hardcodes RUNTIME_OUTPUT_DIRECTORY to
# slim-vdb/examples/cpp/. Copy the just-built binaries into a variant-specific
# bin dir so parallel NCLASSES variants don't clobber each other.
VARIANT_BIN="${VARIANT_ROOT}/bin"
mkdir -p "${VARIANT_BIN}"
for exe in kitti_pipeline scenenet_pipeline realworld_pipeline replica_pipeline vdb_to_voxels; do
    src="${SLIMVDB_SRC}/examples/cpp/${exe}"
    if [[ -f "${src}" ]]; then
        cp -p "${src}" "${VARIANT_BIN}/"
    fi
done

echo ""
echo "============================================================"
echo " DONE."
echo " OpenVDB prefix  : ${OVDB_PREFIX}"
echo " SLIM-VDB prefix : ${VARIANT_PREFIX}"
echo " Binaries        : ${VARIANT_BIN}/"
echo "   ↳ kitti_pipeline, scenenet_pipeline, realworld_pipeline, replica_pipeline"
echo "============================================================"
