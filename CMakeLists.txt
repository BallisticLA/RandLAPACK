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

# Configure RandBLAS.
#
# RandBLAS ships as a git submodule pinned to an exact commit, and that pinned
# copy is the only RandBLAS configuration RandLAPACK is developed and tested
# against. Do not point RandLAPACK at a RandBLAS working copy, and do not
# develop RandBLAS inside the submodule checkout; clone RandBLAS separately
# for that (see INSTALL.md, section "RandBLAS is a pinned submodule").
#
# RandLAPACK_EXTERNAL_RandBLAS exists for package maintainers only (Spack,
# conda-forge, ...), whose recipes build RandBLAS as its own package. The
# commit gate below makes version skew impossible: the external install must
# have been built from exactly the commit the submodule pins.
option(RandLAPACK_EXTERNAL_RandBLAS
    "Packager-only: consume an installed RandBLAS instead of the pinned submodule" OFF)

# The submodule pin, recorded as a variable so configure-time checks also work
# in tarball builds (no .git directory). Update alongside every submodule
# bump; with a git checkout present, the cross-check below enforces that.
set(RandLAPACK_RandBLAS_PIN "8417f4bb911c6ca3aa83b157a594758d2d4fab8c")

find_package(Git QUIET)
if (Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} ls-tree HEAD RandBLAS
        OUTPUT_VARIABLE randblas_pin_lstree
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (randblas_pin_lstree MATCHES "commit ([0-9a-f]+)")
        set(randblas_pin_actual "${CMAKE_MATCH_1}")
        if (NOT randblas_pin_actual STREQUAL RandLAPACK_RandBLAS_PIN)
            message(FATAL_ERROR
                "RandLAPACK_RandBLAS_PIN is stale: CMakeLists.txt records "
                "${RandLAPACK_RandBLAS_PIN} but the RandBLAS submodule pins "
                "${randblas_pin_actual}. Update the pin variable in the same "
                "commit as the submodule bump.")
        endif()
    endif()
endif()

if (RandLAPACK_EXTERNAL_RandBLAS)
    find_package(RandBLAS REQUIRED)
    if (NOT DEFINED RandBLAS_COMMIT_HASH OR RandBLAS_COMMIT_HASH STREQUAL ""
        OR RandBLAS_COMMIT_HASH MATCHES "see version tag")
        message(FATAL_ERROR
            "RandLAPACK_EXTERNAL_RandBLAS=ON, but the RandBLAS install at "
            "RandBLAS_DIR=${RandBLAS_DIR} does not record a commit hash, so "
            "it cannot be checked against the pinned submodule commit. Build "
            "RandBLAS from a git checkout of commit ${RandLAPACK_RandBLAS_PIN}.")
    endif()
    string(FIND "${RandLAPACK_RandBLAS_PIN}" "${RandBLAS_COMMIT_HASH}" randblas_pin_prefix)
    if (NOT randblas_pin_prefix EQUAL 0)
        message(FATAL_ERROR
            "RandLAPACK_EXTERNAL_RandBLAS=ON, but the RandBLAS install at "
            "RandBLAS_DIR=${RandBLAS_DIR} was built from commit "
            "${RandBLAS_COMMIT_HASH}, while this RandLAPACK is developed and "
            "tested against RandBLAS ${RandLAPACK_RandBLAS_PIN} (the submodule "
            "pin). Version skew between the two is not supported. Build the "
            "external RandBLAS from exactly the pinned commit, or drop "
            "RandLAPACK_EXTERNAL_RandBLAS to use the submodule. This option "
            "is intended for package maintainers; see INSTALL.md, section "
            "\"RandBLAS is a pinned submodule\".")
    endif()
    message(STATUS "RandBLAS: external install at ${RandBLAS_DIR} (commit ${RandBLAS_COMMIT_HASH}, matches the submodule pin)")
else()
    add_subdirectory(RandBLAS)
endif()

# Compile sources
add_subdirectory(RandLAPACK)
if (RandLAPACK_BUILD_TESTS)
    add_subdirectory(test)
endif()

# Export the configuration
include(rl_config)