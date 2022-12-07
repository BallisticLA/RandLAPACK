cmake_minimum_required(VERSION 3.2)

project(RandLAPACK)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# configure the build
enable_testing()

include(rl_build_options)
include(rl_version)

# find dependencies
find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)

# configure RandBLAS subproject
add_subdirectory(RandBLAS)

# compile sources
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmark)

# export the configuration
include(rl_config)
