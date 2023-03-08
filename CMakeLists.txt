cmake_minimum_required(VERSION 3.2)

project(RandLAPACK)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# configure the build
enable_testing()

include(rl_build_options)
include(rl_version)

# find dependencies
find_package(lapackpp REQUIRED)

# configure RandBLAS subproject
add_subdirectory(RandBLAS)

# compile sources
add_subdirectory(RandLAPACK)
add_subdirectory(test)
add_subdirectory(benchmark)

# export the configuration
include(rl_config)
