cmake_minimum_required(VERSION 3.16)
project(scovox_mapping VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build (real-time mapping node hot path). colcon users
# frequently omit -DCMAKE_BUILD_TYPE; without a default the per-ray integration
# runs unoptimized with Eigen asserts on. Respect an explicit choice when given.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(scovox_msgs REQUIRED)
find_package(scovox_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)

# Global include paths
include_directories(include)

# ==================================================================
# Beta-conjugate map library
# ==================================================================

add_library(scovoxmap src/scovoxmap.cpp)
ament_target_dependencies(scovoxmap scovox_core)
target_include_directories(scovoxmap PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(scovoxmap Eigen3::Eigen)

# ==================================================================
# ROS nodes
# ==================================================================

# DSCovoxNode — multi-robot fusion
add_executable(dscovox_mapping_node src/dscovox_node.cpp)
ament_target_dependencies(dscovox_mapping_node
  rclcpp sensor_msgs nav_msgs tf2 tf2_ros geometry_msgs scovox_msgs scovox_core)
target_include_directories(dscovox_mapping_node PRIVATE ${LZ4_INCLUDE_DIRS})
target_link_libraries(dscovox_mapping_node scovoxmap Eigen3::Eigen ${LZ4_LIBRARIES})

# SCovoxNode — single-robot mapping
add_executable(scovox_mapping_node src/scovox_node.cpp)
ament_target_dependencies(scovox_mapping_node
  rclcpp sensor_msgs std_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
  message_filters nav_msgs diagnostic_msgs scovox_msgs scovox_core)
target_link_libraries(scovox_mapping_node scovoxmap ${LZ4_LIBRARIES})
target_include_directories(scovox_mapping_node PRIVATE ${LZ4_INCLUDE_DIRS})

# ==================================================================
# Tests
# ==================================================================

find_package(ament_cmake_gtest REQUIRED)

set(SCOVOX_TEST_SOURCES
  test/test_beta_update.cpp
  test/test_dirichlet_update.cpp
  test/test_consensus.cpp
  test/test_semantic_audit.cpp
  test/test_tsdf_band.cpp
  test/test_marching_cubes.cpp
)

foreach(TEST_SRC ${SCOVOX_TEST_SOURCES})
  get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
  ament_add_gtest(${TEST_NAME} ${TEST_SRC})
  ament_target_dependencies(${TEST_NAME} scovox_core)
  target_link_libraries(${TEST_NAME} scovoxmap Eigen3::Eigen)
  target_include_directories(${TEST_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach()

# TopkProvider — header-only .topk loader/cache (extracted from scovox_node).
# Needs rclcpp for the injected logger/clock; logging works without rclcpp::init.
ament_add_gtest(test_topk_provider test/test_topk_provider.cpp)
ament_target_dependencies(test_topk_provider rclcpp)
target_include_directories(test_topk_provider PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ==================================================================
# Install
# ==================================================================

install(TARGETS scovoxmap
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS dscovox_mapping_node scovox_mapping_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/scovox
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

# ==================================================================
# Ament exports
# ==================================================================

ament_export_include_directories(include)
ament_export_libraries(scovoxmap)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(scovox_core scovox_msgs rclcpp sensor_msgs std_msgs geometry_msgs tf2 tf2_ros)

ament_package()
