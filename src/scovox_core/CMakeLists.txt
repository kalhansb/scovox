cmake_minimum_required(VERSION 3.16)
project(scovox_core VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)

# ==================================================================
# Core library: scovox_core (zero ROS dependencies)
# ==================================================================

add_library(scovox_core
    src/uncertainty.cpp
    src/tsdf_map.cpp
    src/sembeta_map.cpp
    src/semdir_map.cpp
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

find_package(ament_cmake_gtest REQUIRED)

ament_add_gtest(test_uncertainty test/test_uncertainty.cpp)
target_link_libraries(test_uncertainty scovox_core Eigen3::Eigen)

ament_add_gtest(test_sparse_add test/test_sparse_add.cpp)
target_link_libraries(test_sparse_add scovox_core Eigen3::Eigen)

ament_add_gtest(test_binary_serializer test/test_binary_serializer.cpp)
target_link_libraries(test_binary_serializer scovox_core Eigen3::Eigen ${LZ4_LIBRARIES})

# Step-1 gate of the split-grid refactor (see docs/design/slimvdb_like_tsdf_mapping_plan.md):
# forces tsdf_voxel.hpp and sembeta_voxel.hpp through the compiler so the
# header-side static_asserts on layout / sizes / triviality fire.
ament_add_gtest(test_voxel_layouts test/test_voxel_layouts.cpp)
target_link_libraries(test_voxel_layouts scovox_core Eigen3::Eigen)

# Step-2 gate: TsdfMap parity (Euclidean SDF, voxel CENTRE, per-voxel weight)
# + Curless-Levoy fundamentals + drainTouched semantics.
ament_add_gtest(test_tsdf_map test/test_tsdf_map.cpp)
target_link_libraries(test_tsdf_map scovox_core Eigen3::Eigen)

# Step-3 gate: SemBetaMap Beta(1,1) prior + miss-no-occ + Dirichlet gating
# + carve free + wall blocking + evidence saturation + drainTouched.
ament_add_gtest(test_sembeta_map test/test_sembeta_map.cpp)
target_link_libraries(test_sembeta_map scovox_core Eigen3::Eigen)

# Step-4 gate: TSDF-typed extractMesh / extractZeroCrossing / extractPointCloud
# overloads + free labelMesh / labelPointCloud functions.
ament_add_gtest(test_mesh_labelling test/test_mesh_labelling.cpp)
target_link_libraries(test_mesh_labelling scovox_core Eigen3::Eigen)

# Step-5 gate: ScovoxMapSplit composer end-to-end (integrate, mesh, drain).
ament_add_gtest(test_scovox_map_split test/test_scovox_map_split.cpp)
target_link_libraries(test_scovox_map_split scovox_core Eigen3::Eigen)

# Step-6 gate: dual-stream wire format v2 round-trip + magic/version validation.
ament_add_gtest(test_binary_serializer_v2 test/test_binary_serializer_v2.cpp)
target_link_libraries(test_binary_serializer_v2 scovox_core Eigen3::Eigen)

# Step-7 gate: per-voxel + per-frame consensus merge for the v2 dual-stream format.
ament_add_gtest(test_consensus_merge_v2 test/test_consensus_merge_v2.cpp)
target_link_libraries(test_consensus_merge_v2 scovox_core Eigen3::Eigen)

# Step-7.5 gate: unified-Dirichlet SemDirMap (Beta+Dirichlet → single Dir).
# Symmetric prior + mass-conservation strict equality + dirichlet_min_p_occ gate.
ament_add_gtest(test_semdir_map test/test_semdir_map.cpp)
target_link_libraries(test_semdir_map scovox_core Eigen3::Eigen)

# Step-7.5 gate: v3 wire format (semdir-only by default, dual-stream optional).
ament_add_gtest(test_binary_serializer_v3 test/test_binary_serializer_v3.cpp)
target_link_libraries(test_binary_serializer_v3 scovox_core Eigen3::Eigen)

# Step-7.5 gate: v3 consensus merge (TSDF + unified-Dirichlet SemDir).
ament_add_gtest(test_consensus_merge_v3 test/test_consensus_merge_v3.cpp)
target_link_libraries(test_consensus_merge_v3 scovox_core Eigen3::Eigen)

# Split Beta/Dirichlet substrate (SemSplitMap): de-unifies SemDir into a
# BetaVoxel occupancy grid + DirVoxel semantics grid. Layout/prior, two-stream
# update, STRICT per-grid mass conservation, gate + sparse-semantics memory win.
ament_add_gtest(test_sem_split_map test/test_sem_split_map.cpp)
target_link_libraries(test_sem_split_map scovox_core Eigen3::Eigen)

# v4 wire format (triple-stream: TSDF + BetaVoxel + DirVoxel) round-trip +
# header/size/truncation validation.
ament_add_gtest(test_binary_serializer_v4 test/test_binary_serializer_v4.cpp)
target_link_libraries(test_binary_serializer_v4 scovox_core Eigen3::Eigen)

# v4 consensus merge: independent BetaVoxel + DirVoxel conjugate merges,
# per-grid mass conservation, frame-level prior assertions.
ament_add_gtest(test_consensus_merge_v4 test/test_consensus_merge_v4.cpp)
target_link_libraries(test_consensus_merge_v4 scovox_core Eigen3::Eigen)

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

ament_export_include_directories(include include/third_party/bonxai)
ament_export_libraries(scovox_core)
ament_export_targets(scovox_core_export HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

ament_package()
