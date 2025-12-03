cmake_minimum_required(VERSION 3.21)
project(RandLAPACK_demos LANGUAGES CXX)

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
    message(STATUS "OpenMP not found - demos will run without parallelization")
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
    message(STATUS "CUDA not found - GPU demos will not be built")
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
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen_lib)

# Spectra - Eigenvalue solvers (depends on Eigen)
message(STATUS "Fetching Spectra library...")
FetchContent_Declare(
    spectra_lib
    GIT_REPOSITORY https://github.com/yixuan/spectra/
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spectra_lib)

# Link Spectra to Eigen (Spectra is header-only, uses INTERFACE)
target_link_libraries(Spectra INTERFACE Eigen3::Eigen)

# fast_matrix_market - Matrix file I/O (optional)
message(STATUS "Fetching fast_matrix_market library...")
FetchContent_Declare(
    fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

# -----------------------------------------------------------------------------
# Library Sets
# -----------------------------------------------------------------------------

# Basic demos (RandLAPACK only)
set(Demo_libs_basic
    RandLAPACK::RandLAPACK
)

# External library demos (Eigen + Spectra)
set(Demo_libs_external
    RandLAPACK::RandLAPACK
    Eigen3::Eigen
    Spectra
    fast_matrix_market
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# Function to add a CPU demo
function(add_demo)
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

    # Demos are not installed system-wide - they run from build directory
    # Uncomment to enable installation:
    # install(TARGETS ${TGT_NAME} DESTINATION bin)
endfunction()

# Function to add a GPU demo (.cu files)
function(add_demo_gpu)
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

    # Uncomment to enable installation:
    # install(TARGETS ${TGT_NAME} DESTINATION bin)
endfunction()

# -----------------------------------------------------------------------------
# Demo Declarations
# -----------------------------------------------------------------------------

# Note: functions/ directory headers should be included with relative paths
# from demo source files, e.g., #include "../../functions/misc/format_conversions.hh"

message(STATUS "Configuring demos...")

# linops_external demos (from demos/linops_external/)
add_demo(NAME demo_eigen_linops CXX_SOURCES demos/linops_external/demo_eigen_linops.cc LINK_LIBS ${Demo_libs_external})
add_demo(NAME demo_spectra_eigensolvers CXX_SOURCES demos/linops_external/demo_spectra_eigensolvers.cc LINK_LIBS ${Demo_libs_external})

# Dummy demo (from demos/)
add_demo(NAME dummy CXX_SOURCES demos/dummy.cc LINK_LIBS ${Demo_libs_basic})

message(STATUS "Configured ${CMAKE_PROJECT_NAME} with demos:")
message(STATUS "  - demo_eigen_linops")
message(STATUS "  - demo_spectra_eigensolvers")
message(STATUS "  - dummy")

if(HAVE_CUDA)
    message(STATUS "GPU demos: Available (CUDA ${CUDAToolkit_VERSION})")
    # Add GPU demos here when ready:
    # add_demo_gpu(
    #     NAME demo_gpu_example
    #     CU_SOURCES linops_external/demo_gpu_example.cu
    #     LINK_LIBS ${Demo_libs_external} CUDA::cusolver
    # )
else()
    message(STATUS "GPU demos: Skipped (no CUDA)")
endif()
