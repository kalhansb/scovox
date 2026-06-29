cmake_minimum_required(VERSION 3.16)
project(scovox_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ScovoxSemanticEvidence.msg"
  "msg/ScovoxVoxel.msg"
  "msg/ScovoxMap.msg"
  "msg/ScovoxMapBinary.msg"
  "msg/RobotIntent.msg"
  "srv/GetRegion.srv"
  "srv/GetOccupancyGrid.srv"
  "srv/ExtractMesh.srv"
  DEPENDENCIES std_msgs geometry_msgs nav_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
