cmake_minimum_required(VERSION 3.21)
project(RandLAPACK_extras LANGUAGES CXX)

# C++20 standard required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Dependency Detection
# -----------------------------------------------------------------------------

# OpenMP (optional)
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_FOUND)
    set(HAVE_OpenMP TRUE CACHE BOOL "OpenMP support detected")
    message(STATUS "OpenMP detected: ${OpenMP_CXX_VERSION}")
else()
    set(HAVE_OpenMP FALSE CACHE BOOL "OpenMP support not detected")
    message(STATUS "OpenMP not found - extras will run without parallelization")
endif()

# CUDA (optional)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(HAVE_CUDA TRUE CACHE BOOL "CUDA support detected")
    message(STATUS "CUDA detected: ${CUDAToolkit_VERSION}")
else()
    set(HAVE_CUDA FALSE CACHE BOOL "CUDA support not detected")
    message(STATUS "CUDA not found - GPU extras will not be built")
endif()

# RandLAPACK (REQUIRED - this is a consumer project)
find_package(RandLAPACK REQUIRED)
message(STATUS "Found RandLAPACK: ${RandLAPACK_DIR}")

# -----------------------------------------------------------------------------
# External Dependencies via FetchContent
# -----------------------------------------------------------------------------
# NOTE: These dependencies are shared with the benchmark/ project.
# Use identical names and versions to ensure CMake reuses cached downloads.

include(FetchContent)

# Eigen - Matrix operations
message(STATUS "Fetching Eigen library...")
FetchContent_Declare(
    eigen_lib
    # Pinned to Eigen master commit e2ca51a (2026-10-22). To bump: change the hash in the URL
    # and URL_HASH (sha256sum of the downloaded archive). Needs SparseLU::toSparse().
    URL https://gitlab.com/libeigen/eigen/-/archive/e2ca51a02b84a6f6f106039e0ebb15d479eff9b6/eigen-e2ca51a02b84a6f6f106039e0ebb15d479eff9b6.tar.gz
    URL_HASH SHA256=aef6282955cfa57f298d8a65c3ea1c4ab2b11d4ef990bb458798b1daa7217128
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(eigen_lib)

# fast_matrix_market - Matrix file I/O
message(STATUS "Fetching fast_matrix_market library...")
FetchContent_Declare(
    fast_matrix_market
    # Pinned to fast_matrix_market commit b6172c9 (2026-10). To bump: change the hash
    # in the URL and URL_HASH (sha256sum of the downloaded archive).
    URL https://github.com/alugowski/fast_matrix_market/archive/b6172c96ef73d16b2c10917373bacec1a7583a31.tar.gz
    URL_HASH SHA256=02dff7b419d94378bd93cec3a05cf6184f073d5a6d47c75495c71cc0987ebb31
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

# -----------------------------------------------------------------------------
# Library Sets
# -----------------------------------------------------------------------------

# Basic extras (RandLAPACK only)
set(Extras_libs_basic
    RandLAPACK::RandLAPACK
)

# External library extras (Eigen + fast_matrix_market)
set(Extras_libs_external
    RandLAPACK::RandLAPACK
    Eigen3::Eigen
    fast_matrix_market
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# Function to add a CPU extra target
function(add_extra)
    cmake_parse_arguments(
        TGT
        ""  # No boolean options
        "NAME"  # Single-value arguments
        "CXX_SOURCES;LINK_LIBS"  # Multi-value arguments
        ${ARGN}
    )

    add_executable(${TGT_NAME} ${TGT_CXX_SOURCES})
    target_link_libraries(${TGT_NAME} PRIVATE ${TGT_LINK_LIBS})

    if(HAVE_OpenMP)
        target_link_libraries(${TGT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# Function to add a GPU extra target (.cu files)
function(add_extra_gpu)
    cmake_parse_arguments(
        TGT
        ""
        "NAME"
        "CU_SOURCES;LINK_LIBS"
        ${ARGN}
    )

    add_executable(${TGT_NAME} ${TGT_CU_SOURCES})
    set_target_properties(${TGT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
    )
    target_link_libraries(${TGT_NAME} PRIVATE ${TGT_LINK_LIBS})

    if(HAVE_OpenMP)
        target_link_libraries(${TGT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Extra Target Declarations
# -----------------------------------------------------------------------------

message(STATUS "Configuring extras...")

# No extra targets currently configured
# Add targets here as needed

message(STATUS "Configured ${CMAKE_PROJECT_NAME}:")
message(STATUS "  - (none configured)")

if(HAVE_CUDA)
    message(STATUS "GPU extras: Available (CUDA ${CUDAToolkit_VERSION})")
else()
    message(STATUS "GPU extras: Skipped (no CUDA)")
endif()

# -----------------------------------------------------------------------------
# Testing Support
# -----------------------------------------------------------------------------

option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
