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

# EIGEN dependency.
include(FetchContent)
FetchContent_Declare(
        eigen_lib
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen
        GIT_TAG master
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen_lib)
# Exporting Eigen target.
install(EXPORT Eigen3Targets NAMESPACE Eigen3:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Eigen3)

# SPECTRA dependency
FetchContent_Declare(
        spectra_lib
        GIT_REPOSITORY https://github.com/yixuan/spectra/
        GIT_TAG master
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spectra_lib)
# Linking Spectra and Eigen.
target_link_libraries(Spectra INTERFACE Eigen3::Eigen)

# Adding fast matri matrket for benchmarking matrices examples
include(FetchContent)
FetchContent_Declare(
    fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(
    fast_matrix_market
)

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

# CQRRT linops benchmarks (require Eigen + fast_matrix_market + extras headers)
set(
    Benchmark_libs_cqrrt_linops
    RandLAPACK::RandLAPACK
    Eigen3::Eigen
    fast_matrix_market
)
add_benchmark(NAME CQRRT_linop_applications           CXX_SOURCES bench_CQRRT_linops/CQRRT_linop_applications.cc           LINK_LIBS ${Benchmark_libs_cqrrt_linops})
add_benchmark(NAME CQRRT_linop_basic                  CXX_SOURCES bench_CQRRT_linops/CQRRT_linop_basic.cc                  LINK_LIBS ${Benchmark_libs_cqrrt_linops})
add_benchmark(NAME CQRRT_diagnostic                   CXX_SOURCES bench_CQRRT_linops/CQRRT_diagnostic.cc                   LINK_LIBS ${Benchmark_libs_cqrrt_linops})

# Toeplitz LS benchmark (autoregression/system-id experiment). The matrix-free
# prolate-Toeplitz operator (extras/linops/ext_toeplitz_linop.hh) uses MKL's FFT (DFTI),
# so this target additionally needs the MKL include dir + the ILP64 define (the MKL libs
# themselves are already linked transitively via RandLAPACK -> blaspp -> MKL). RandLAPACK
# also supports OpenBLAS, so on a non-MKL machine mkl_dfti.h will not exist; skip only
# this target with a warning rather than aborting the whole benchmark configure.
# Locate mkl_dfti.h. MKLROOT is the usual signal, but batch builds (e.g. ISAAC's
# fresh-clone install job) may not export it, so also probe the other oneAPI env
# vars and common install roots. As a last resort, derive it from a BLAS/LAPACK
# library dir that CMake already resolved (MKL's lib and include are siblings).
find_path(MKL_DFTI_INCLUDE_DIR mkl_dfti.h
    HINTS
        $ENV{MKLROOT}/include
        $ENV{MKL_ROOT}/include
        $ENV{ONEAPI_ROOT}/mkl/latest/include
        $ENV{CMPLR_ROOT}/../mkl/latest/include
        /opt/intel/oneapi/mkl/latest/include
        /opt/intel/oneapi/mkl/2026.0/include
    PATH_SUFFIXES include)
if(NOT MKL_DFTI_INCLUDE_DIR)
    # Derive from a resolved MKL/LAPACK library path (…/mkl/<ver>/lib -> …/include).
    foreach(_lib ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${blaspp_libraries})
        if(_lib MATCHES "mkl")
            get_filename_component(_libdir "${_lib}" DIRECTORY)
            find_path(MKL_DFTI_INCLUDE_DIR mkl_dfti.h HINTS "${_libdir}/../include" "${_libdir}/../../include")
        endif()
    endforeach()
endif()
if(NOT MKL_DFTI_INCLUDE_DIR)
    message(WARNING "mkl_dfti.h not found; skipping toeplitz_ls_benchmark target. Pass "
                    "-DMKL_DFTI_INCLUDE_DIR=<MKL>/include or export MKLROOT to build it.")
else()
    message(STATUS "Toeplitz benchmark: mkl_dfti.h in ${MKL_DFTI_INCLUDE_DIR}")
    add_benchmark(NAME toeplitz_ls_benchmark              CXX_SOURCES bench_toeplitz_ls/toeplitz_ls_benchmark.cc              LINK_LIBS ${Benchmark_libs_cqrrt_linops})
    target_include_directories(toeplitz_ls_benchmark PRIVATE ${MKL_DFTI_INCLUDE_DIR})

    # Derive MKL_ILP64 from the BLAS interface blaspp was actually built against,
    # instead of assuming ILP64 (an LP64 MKL build would otherwise silently mismatch
    # the DFTI header's ABI). blaspp's installed config records this in
    # blas/defines.h (BLAS_ILP64 defined iff the ILP64 interface was linked);
    # blaspp_DIR locates that install tree. Fall back to matching ilp64/lp64 in the
    # resolved library names if that header cannot be found.
    set(_toeplitz_ilp64 "")
    if(blaspp_DIR)
        get_filename_component(_blaspp_prefix "${blaspp_DIR}/../../.." ABSOLUTE)
        set(_blaspp_defines_h "${_blaspp_prefix}/include/blas/defines.h")
        if(EXISTS "${_blaspp_defines_h}")
            file(READ "${_blaspp_defines_h}" _blaspp_defines_contents)
            if(_blaspp_defines_contents MATCHES "#define[ \t]+BLAS_ILP64")
                set(_toeplitz_ilp64 TRUE)
            else()
                set(_toeplitz_ilp64 FALSE)
            endif()
            message(STATUS "Toeplitz benchmark: BLAS interface from ${_blaspp_defines_h}: "
                            "ILP64=${_toeplitz_ilp64}")
        endif()
    endif()
    if(_toeplitz_ilp64 STREQUAL "")
        foreach(_lib ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${blaspp_libraries})
            if(_lib MATCHES "ilp64")
                set(_toeplitz_ilp64 TRUE)
            elseif(_lib MATCHES "lp64")
                set(_toeplitz_ilp64 FALSE)
            endif()
        endforeach()
        if(NOT _toeplitz_ilp64 STREQUAL "")
            message(STATUS "Toeplitz benchmark: BLAS interface from linked library names: "
                            "ILP64=${_toeplitz_ilp64}")
        endif()
    endif()
    if(_toeplitz_ilp64 STREQUAL "")
        message(WARNING "Toeplitz benchmark: could not determine whether the resolved "
                        "BLAS/MKL interface is ILP64 or LP64; defaulting to MKL_ILP64.")
        set(_toeplitz_ilp64 TRUE)
    endif()
    if(_toeplitz_ilp64)
        target_compile_definitions(toeplitz_ls_benchmark PRIVATE MKL_ILP64)
    endif()
endif()

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
