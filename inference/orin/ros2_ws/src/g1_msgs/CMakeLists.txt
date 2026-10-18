cmake_minimum_required(VERSION 3.8)
project(g1_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PolicyOutput.msg"
  DEPENDENCIES builtin_interfaces std_msgs trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
