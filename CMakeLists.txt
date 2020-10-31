cmake_minimum_required(VERSION 3.13)

project(zetton_inference)

# set compilation
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-Wall -Wextra -Wno-unused-parameter -O2)

# useful macros
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/util.cmake)

# find dependencies
find_package(Threads)
find_package(OpenCV 4 REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/yolo-tensorrt)

# ROS
find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    rospy
    image_transport
    cv_bridge
    sensor_msgs
    message_generation
    zetton_common
    )
catkin_python_setup()
catkin_package(
    INCLUDE_DIRS include ${CMAKE_CURRENT_SOURCE_DIR}/external/yolo-tensorrt/include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp roslib rospy
    DEPENDS)

# common include directories
include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/yolo-tensorrt/include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# find all source files
file(GLOB_RECURSE
    src_files
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*/*.cc
    )

# common library
add_library(
    ${PROJECT_NAME}
    SHARED
    ${src_files}
)
target_link_libraries(
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
    yolo_trt
)

# installation
install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
    )
# need cmake version >= 3.13
install(TARGETS yolo_trt
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
    )
# catkin_install_python(PROGRAMS app/text_detection_node.py
#     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    PATTERN ".svn" EXCLUDE
    )
install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    PATTERN ".svn" EXCLUDE)
install(DIRECTORY config/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
    PATTERN ".svn" EXCLUDE)
install(DIRECTORY asset/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/asset
    PATTERN ".svn" EXCLUDE)

# examples
add_simple_examples()
