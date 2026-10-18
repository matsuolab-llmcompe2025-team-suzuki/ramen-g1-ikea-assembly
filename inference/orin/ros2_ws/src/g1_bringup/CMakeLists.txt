cmake_minimum_required(VERSION 3.8)
project(g1_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/camera_stream_watchdog.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
