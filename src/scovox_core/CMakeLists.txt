cmake_minimum_required(VERSION 3.16)
project(scovox_core VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build. This is a real-time mapping library whose hot
# path is the per-voxel DDA + Eigen math; without a build type colcon produces
# an unoptimized binary with Eigen's runtime asserts enabled. Respect an
# explicitly chosen type (Debug, etc.) — only set a default when none is given.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

# Dual build.  scovox_core is declared "zero ROS dependencies" and every one of
# its test binaries links only scovox_core + Eigen, so the ament layer here is
# pure packaging.  Finding it QUIET rather than REQUIRED lets the same file
# build two ways from one source of truth:
#
#   inside a ROS 2 workspace   ament_cmake is found; ament_add_gtest registers
#                              the suites and the package is exported to colcon.
#   plain cmake, no ROS        GTest is found directly and gtest_discover_tests
#                              registers the same suites with ctest.
#
# The library, its sources, its include dirs and its tests are declared exactly
# once.  A standalone mirror of this file previously lived out-of-tree and had
# already drifted -- it silently skipped test_carve_stage, so seven cases were
# reported as passing while never being built.  Structurally that cannot recur.
find_package(ament_cmake QUIET)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)

if(ament_cmake_FOUND)
  message(STATUS "scovox_core: ament build (ROS 2 workspace)")
else()
  message(STATUS "scovox_core: standalone build (plain cmake, no ROS)")
endif()

# ==================================================================
# Core library: scovox_core (zero ROS dependencies)
# ==================================================================

add_library(scovox_core
    src/uncertainty.cpp
    src/tsdf_map.cpp
    src/sem_split_map.cpp)

target_link_libraries(scovox_core
  PUBLIC Eigen3::Eigen
  PUBLIC ${LZ4_LIBRARIES}
)

target_include_directories(scovox_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/bonxai>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/third_party/bonxai>
)

target_include_directories(scovox_core PUBLIC ${LZ4_INCLUDE_DIRS})

# ==================================================================
# Tests (GTest, no ROS dependencies)
# ==================================================================

if(ament_cmake_FOUND)
  find_package(ament_cmake_gtest REQUIRED)
else()
  find_package(GTest REQUIRED)
  include(GoogleTest)
  enable_testing()
endif()

# One call site, two registration backends.  Uses the plain (non-keyword)
# target_link_libraries signature because every call below links its target the
# same way, and cmake forbids mixing the two forms on one target.
function(scovox_add_gtest name)
  if(ament_cmake_FOUND)
    ament_add_gtest(${name} ${ARGN})
  else()
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} GTest::gtest GTest::gtest_main)
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
  endif()
endfunction()

scovox_add_gtest(test_uncertainty test/test_uncertainty.cpp)
target_link_libraries(test_uncertainty scovox_core Eigen3::Eigen)

scovox_add_gtest(test_sparse_add test/test_sparse_add.cpp)
target_link_libraries(test_sparse_add scovox_core Eigen3::Eigen)

# Step-1 gate of the split-grid refactor (see docs/design/slimvdb_like_tsdf_mapping_plan.md):
# forces tsdf_voxel.hpp and sembeta_voxel.hpp through the compiler so the
# header-side static_asserts on layout / sizes / triviality fire.
scovox_add_gtest(test_voxel_layouts test/test_voxel_layouts.cpp)
target_link_libraries(test_voxel_layouts scovox_core Eigen3::Eigen)

# Step-2 gate: TsdfMap parity (Euclidean SDF, voxel CENTRE, per-voxel weight)
# + Curless-Levoy fundamentals + drainTouched semantics.
scovox_add_gtest(test_tsdf_map test/test_tsdf_map.cpp)
target_link_libraries(test_tsdf_map scovox_core Eigen3::Eigen)

# Step-4 gate: TSDF-typed extractMesh / extractZeroCrossing / extractPointCloud
# overloads + free labelMesh / labelPointCloud functions.
scovox_add_gtest(test_mesh_labelling test/test_mesh_labelling.cpp)
target_link_libraries(test_mesh_labelling scovox_core Eigen3::Eigen)

# Step-5 gate: ScovoxMapSplit composer end-to-end (integrate, mesh, drain).
scovox_add_gtest(test_scovox_map_split test/test_scovox_map_split.cpp)
target_link_libraries(test_scovox_map_split scovox_core Eigen3::Eigen)

# Split Beta/Dirichlet substrate (SemSplitMap): de-unifies SemDir into a
# BetaVoxel occupancy grid + DirVoxel semantics grid. Layout/prior, two-stream
# update, STRICT per-grid mass conservation, gate + sparse-semantics memory win.
scovox_add_gtest(test_sem_split_map test/test_sem_split_map.cpp)
target_link_libraries(test_sem_split_map scovox_core Eigen3::Eigen)

# Batched-carve staging substrate (CarveStage): differential gate vs the
# retired unordered_map + per-scan-sort staging — identical staged set /
# max weights / block visit order (efficiency_audit_2026_08_26.md item 1).
scovox_add_gtest(test_carve_stage test/test_carve_stage.cpp)
target_link_libraries(test_carve_stage scovox_core Eigen3::Eigen)

# Wire format (triple-stream: TSDF + BetaVoxel + DirVoxel) round-trip +
# header/size/truncation validation.
scovox_add_gtest(test_binary_serializer test/test_binary_serializer.cpp)
target_link_libraries(test_binary_serializer scovox_core Eigen3::Eigen)

# Consensus merge: independent BetaVoxel + DirVoxel conjugate merges,
# per-grid mass conservation, frame-level prior assertions.
scovox_add_gtest(test_consensus_merge test/test_consensus_merge.cpp)
target_link_libraries(test_consensus_merge scovox_core Eigen3::Eigen)

# Fine TSDF band (fine_tsdf_band_dbh_2026_07_30.md): region registry gate,
# gated two-lattice integration, per-scan anchor re-registration (drift
# absorption), raw-return fine-only path (refineHit), rev-7 fine wire
# stream + merge. Uses the DBH circle fit (dbh_fit.hpp, a post-processing
# utility the mapping runtime never calls) as its accuracy metric.
scovox_add_gtest(test_fine_tsdf test/test_fine_tsdf.cpp)
target_link_libraries(test_fine_tsdf scovox_core Eigen3::Eigen)

# ==================================================================
# Tools (CLI demos / diagnostics, lib-only no ROS)
# ==================================================================

# Memory accounting demo for the split-grid refactor (Phase 5 paper number).
# Integrates synthetic frames into ScovoxMapSplit and prints per-grid memory.
add_executable(split_memory_demo tools/split_memory_demo.cpp)
target_link_libraries(split_memory_demo scovox_core Eigen3::Eigen)
install(TARGETS split_memory_demo DESTINATION lib/${PROJECT_NAME})

# ==================================================================
# Install
# ==================================================================

install(TARGETS scovox_core
  EXPORT scovox_core_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/scovox
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")

install(DIRECTORY include/third_party/bonxai
  DESTINATION include/third_party)

# ==================================================================
# Ament exports
# ==================================================================

if(ament_cmake_FOUND)
  ament_export_include_directories(include include/third_party/bonxai)
  ament_export_libraries(scovox_core)
  ament_export_targets(scovox_core_export HAS_LIBRARY_TARGET)
  ament_export_dependencies(Eigen3)
  ament_package()
else()
  # The same export set, installed by hand so a non-ROS consumer can
  # find_package(scovox_core) and link scovox_core::scovox_core.
  include(CMakePackageConfigHelpers)
  install(EXPORT scovox_core_export
          FILE scovox_coreTargets.cmake
          NAMESPACE scovox_core::
          DESTINATION lib/cmake/scovox_core)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/scovox_coreConfig.cmake"
"include(CMakeFindDependencyMacro)\nfind_dependency(Eigen3)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/scovox_coreTargets.cmake\")\n")
  write_basic_package_version_file(
          "${CMAKE_CURRENT_BINARY_DIR}/scovox_coreConfigVersion.cmake"
          VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/scovox_coreConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/scovox_coreConfigVersion.cmake"
          DESTINATION lib/cmake/scovox_core)
endif()
