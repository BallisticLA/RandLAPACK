cmake_minimum_required(VERSION 3.21)

project(benchmark)

# Benchmark is a standalone project, needs its own C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include GNUInstallDirs for standard installation directories
include(GNUInstallDirs)

message(STATUS "Checking for OpenMP ... ")
find_package(OpenMP COMPONENTS CXX)
set(tmp FALSE)
if (OpenMP_CXX_FOUND)
    set(tmp TRUE)
elseif(APPLE)
    message(STATUS
        "\n\tOpenMP was not found! Try installing it with \
        \n\n\t    brew install libomp \
        \n\tor \
        \n\t    brew reinstall libomp \
        \n\n\tWhen you do this, Homebrew might suggest that you run \
        \n\n\t    export CXXFLAGS=-I<absolute path to OpenMP include files> \
        \n\t    export LDFLAGS=-L<absolute path to OpenMP binaries> \
        \n\n\tBe sure to follow that suggestion! Once these environment \
        \n\tvariables are set, CMake should be able to find OpenMP.
    ")
endif()
set(HAVE_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP")
message(STATUS "Checking for OpenMP ... ${HAVE_OpenMP}")

message(STATUS "Checking for CUDA ... ")
find_package(CUDAToolkit QUIET)
set(tmp FALSE)
if (CUDAToolkit_FOUND)
    set(tmp TRUE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(HAVE_CUDA ${tmp} CACHE BOOL "Set if we have a working CUDA")
message(STATUS "Checking for CUDA ... ${HAVE_CUDA}")

message(STATUS "Checking for RandLAPACK ... ")
find_package(RandLAPACK REQUIRED)
message(STATUS "Done checking for RandLAPACK.")

function(add_benchmark)
    set(OPTS)
    set(NVPO NAME)
    set(MVO CXX_SOURCES LINK_LIBS)
    cmake_parse_arguments(PARSE_ARGV 0 TGT "${OPTS}" "${NVPO}" "${MVO}")
    add_executable(${TGT_NAME} ${TGT_CXX_SOURCES})
    target_compile_options(${TGT_NAME} PRIVATE "-g")
    # Include directories come from target_link_libraries, no need to set manually
    target_link_libraries(${TGT_NAME} ${TGT_LINK_LIBS})
    message(STATUS "RandLAPACK: added ${TGT_NAME} benchmark")
endfunction()

# Helper function for GPU benchmarks (.cu files)
function(add_benchmark_gpu)
    set(OPTS)
    set(NVPO NAME)
    set(MVO CU_SOURCES LINK_LIBS)
    cmake_parse_arguments(PARSE_ARGV 0 TGT "${OPTS}" "${NVPO}" "${MVO}")
    add_executable(${TGT_NAME} ${TGT_CU_SOURCES})
    target_compile_options(${TGT_NAME} PRIVATE "-g")
    target_link_libraries(${TGT_NAME} ${TGT_LINK_LIBS})
    message(STATUS "RandLAPACK: added ${TGT_NAME} GPU benchmark")
endfunction()

# External Dependencies via FetchContent
# NOTE: These dependencies are shared with the demos/ project.
# If demos was built first, these targets will already exist and won't be re-downloaded.
include(FetchContent)

# EIGEN dependency - only fetch if not already available
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Fetching Eigen library for benchmarks...")
    FetchContent_Declare(
            eigen_lib
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen
            GIT_TAG master
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen_lib)
    # Exporting Eigen target.
    install(EXPORT Eigen3Targets NAMESPACE Eigen3:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Eigen3)
else()
    message(STATUS "Eigen library already available (likely from demos project)")
endif()

# SPECTRA dependency - only fetch if not already available
if(NOT TARGET Spectra)
    message(STATUS "Fetching Spectra library for benchmarks...")
    FetchContent_Declare(
            spectra_lib
            GIT_REPOSITORY https://github.com/yixuan/spectra/
            GIT_TAG master
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(spectra_lib)
    # Linking Spectra and Eigen.
    target_link_libraries(Spectra INTERFACE Eigen3::Eigen)
else()
    message(STATUS "Spectra library already available (likely from demos project)")
endif()

# fast_matrix_market dependency - only fetch if not already available
if(NOT TARGET fast_matrix_market)
    message(STATUS "Fetching fast_matrix_market library for benchmarks...")
    FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(
        fast_matrix_market
    )
else()
    message(STATUS "fast_matrix_market library already available (likely from demos project)")
endif()

set(
    Benchmark_libs
    RandLAPACK::RandLAPACK
)

# Declaring a full set of libs needed for one of ABRIK benchmarks.
set(
    Benchmark_libs_external
    RandLAPACK::RandLAPACK
    Eigen3::Eigen
    Spectra
    fast_matrix_market
)

# Performance profiling through GEMM
add_benchmark(NAME GEMM_flop_count  CXX_SOURCES bench_general/GEMM_flop_count.cc       LINK_LIBS ${Benchmark_libs})
# Performance profiling through GEQRF
add_benchmark(NAME LAPACK_flop_count CXX_SOURCES bench_general/LAPACK_flop_count.cc    LINK_LIBS ${Benchmark_libs})
# Lapack functionality benchmark
add_benchmark(NAME Chol_check       CXX_SOURCES bench_general/Chol_check.cc            LINK_LIBS ${Benchmark_libs})
# Data conversion helper script
add_benchmark(NAME convert_time     CXX_SOURCES bench_general/convert_time.cc          LINK_LIBS ${Benchmark_libs})
# Compare GEMM and ORMQR performance
add_benchmark(NAME Gemm_vs_ormqr    CXX_SOURCES bench_general/Gemm_vs_ormqr.cc         LINK_LIBS ${Benchmark_libs})
# BLAS levels performance
add_benchmark(NAME basic_blas_speed   CXX_SOURCES bench_general/basic_blas_speed.cc    LINK_LIBS ${Benchmark_libs})

# CQRRPT benchmarks
add_benchmark(NAME CQRRPT_speed_comparisons CXX_SOURCES bench_CQRRPT/CQRRPT_speed_comparisons.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_runtime_breakdown CXX_SOURCES bench_CQRRPT/CQRRPT_runtime_breakdown.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_pivot_quality     CXX_SOURCES bench_CQRRPT/CQRRPT_pivot_quality.cc     LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_error_analysis    CXX_SOURCES bench_CQRRPT/CQRRPT_error_analysis.cc     LINK_LIBS ${Benchmark_libs})

# BQRRP benchmarks
add_benchmark(NAME find_test_mat_spectrum             CXX_SOURCES bench_BQRRP/find_test_mat_spectrum.cc             LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME BQRRP_speed_comparisons_block_size CXX_SOURCES bench_BQRRP/BQRRP_speed_comparisons_block_size.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME BQRRP_speed_comparisons_mat_size   CXX_SOURCES bench_BQRRP/BQRRP_speed_comparisons_mat_size.cc   LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME BQRRP_runtime_breakdown            CXX_SOURCES bench_BQRRP/BQRRP_runtime_breakdown.cc            LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME BQRRP_pivot_quality                CXX_SOURCES bench_BQRRP/BQRRP_pivot_quality.cc                LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME HQRRP_runtime_breakdown            CXX_SOURCES bench_BQRRP/HQRRP_runtime_breakdown.cc            LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME BQRRP_subroutines_speed            CXX_SOURCES bench_BQRRP/BQRRP_subroutines_speed.cc            LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME BQRRP_error_analysis               CXX_SOURCES bench_BQRRP/BQRRP_error_analysis.cc               LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME HQRRP_sanity_check                 CXX_SOURCES bench_BQRRP/HQRRP_sanity_check.cc                 LINK_LIBS ${Benchmark_libs})

# ABRIK benchmarks
add_benchmark(NAME ABRIK_runtime_breakdown        CXX_SOURCES bench_ABRIK/ABRIK_runtime_breakdown.cc        LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME ABRIK_runtime_breakdown_sparse CXX_SOURCES bench_ABRIK/ABRIK_runtime_breakdown_sparse.cc LINK_LIBS ${Benchmark_libs_external})
add_benchmark(NAME ABRIK_speed_comparisons        CXX_SOURCES bench_ABRIK/ABRIK_speed_comparisons.cc        LINK_LIBS ${Benchmark_libs_external})
add_benchmark(NAME ABRIK_speed_comparisons_sparse CXX_SOURCES bench_ABRIK/ABRIK_speed_comparisons_sparse.cc LINK_LIBS ${Benchmark_libs_external})

# GPU benchmarks (if CUDA is available)
if (HAVE_CUDA)
    message(STATUS "Building GPU benchmarks...")

    # BQRRP GPU benchmark - single executable with command-line options
    add_benchmark_gpu(
        NAME BQRRP_GPU_benchmark
        CU_SOURCES bench_BQRRP/BQRRP_GPU_benchmark.cu
        LINK_LIBS ${Benchmark_libs} CUDA::cusolver
    )
else()
    message(STATUS "Skipping GPU benchmarks (CUDA not available)")
endif()
