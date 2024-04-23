cmake_minimum_required(VERSION 3.29)

project(RandLAPACK LANGUAGES CXX CUDA)

set(CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

# cuda stuff
include(find_cuda)

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

# export the configuration
include(rl_config)
