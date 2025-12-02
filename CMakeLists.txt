cmake_minimum_required(VERSION 3.21)

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

        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

        # Set CUDA architectures (default to all-major for portability)
        # Users can override by passing -DCMAKE_CUDA_ARCHITECTURES=... at configure time
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES all-major)
        endif()

        # Add required CUDA flags for RandLAPACK code
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    else()
        message(STATUS "CUDA not found. Continuing without CUDA support.")
    endif()
else()
    message(STATUS "CUDA support not required.")
endif()

# Add general CMake module paths (needed regardless of CUDA)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

# Include compiler flags (needed for both CUDA and non-CUDA builds)
include(compiler_flags)

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