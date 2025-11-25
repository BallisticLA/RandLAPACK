cmake_minimum_required(VERSION 3.27)

# Set project without CUDA language by default
project(RandLAPACK LANGUAGES CXX)

# Set C++ standard for entire project
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optional RequireCUDA flag (defaults to OFF)
option(RequireCUDA "Require CUDA support" OFF)

if (RequireCUDA)
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA found! Enabling CUDA support.")
        enable_language(CUDA)
        
        set(CUDA_VISIBILITY_PRESET hidden)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
        set(CMAKE_CUDA_ARCHITECTURES native)

        # Add general CMake module paths
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
        list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

        # Include CUDA-specific files or flags
        include(compiler_flags)
    else()
        message(STATUS "CUDA not found. Continuing without CUDA support.")
    endif()
else()
    message(STATUS "CUDA support not required.")
    # Add general CMake module paths
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
endif()

# Configure the build
enable_testing()

include(rl_build_options)
include(rl_version)

# Find dependencies
find_package(lapackpp REQUIRED)
# Cache lapackpp location for downstream consumers
set(RandLAPACK_lapackpp_DIR "${lapackpp_DIR}" CACHE INTERNAL "Location of lapackpp for RandLAPACK consumers")

# Configure RandBLAS subproject
add_subdirectory(RandBLAS)

# Compile sources
add_subdirectory(RandLAPACK)
add_subdirectory(test)

# Export the configuration
include(rl_config)