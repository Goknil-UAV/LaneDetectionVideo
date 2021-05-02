CMake_minimum_required(VERSION 3.8)

project(LaneDetection)

set(OpenCV_DIR "C:/Users/Kadir/Downloads/opencv/build")

find_package(OpenCV 4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(app src/main.cpp)

target_link_libraries(app ${OpenCV_LIBS})