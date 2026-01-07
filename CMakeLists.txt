cmake_minimum_required(VERSION 3.10)
project (mono_slam)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the Build type (Release/Debug)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)


set(BUILD_TESTS OFF CHACHE BOOL "Build Sophus tests" FORCE)
set(BUILD_SOPHUS_TESTS OFF CHACHE BOOL "Build Sophus tests" FORCE)
set(BUILD_SOPHUS_EXAMPLES OFF CHACHE BOOL "Build Sophus examples"FORCE)
add_subdirectory(thirdparty/Sophus)

set(SPDLOG_BUILD_EXAMPLES OFF CACHE BOOL "Build spdlog examples" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Build spdlog tests" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Build spdlog benchmarks" FORCE)
add_subdirectory(thirdparty/spdlog)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

file (GLOB_RECURSE SRC_FILES src/*.cpp)




target_compile_options(mono_slam PRIVATE -Wall -Wextra -Wpedantic)

