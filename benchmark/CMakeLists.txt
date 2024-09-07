cmake_minimum_required(VERSION 3.10)

# cmake -DCMAKE_BUILD_TYPE=Release -DRandom123_DIR=`pwd`/../../install/random123-install/include/  -Dblaspp_DIR=`pwd`/../../install/blaspp_GPU-install/lib64/blaspp/  -DRandLAPACK_DIR=`pwd`/../../install/RandLAPACK-install/lib64/cmake/ -Dlapackpp_DIR=`pwd`/../../install/lapackpp-install/lib64/lapackpp/  -DCMAKE_BINARY_DIR=`pwd`     ../../randlibs/RandLAPACK/benchmark/
# ^ Example CMake configuration line with "install," "build" and "randlibs" folders.

project(benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
    target_include_directories(${TGT_NAME} PUBLIC ${Benchmark_include_dirs})
    target_link_libraries(${TGT_NAME} ${TGT_LINK_LIBS})
    message(STATUS "RandLAPACK: added ${TGT_NAME} benchmark")
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
install(EXPORT Eigen3Targets NAMESPACE Eigen3:: DESTINATION lib/cmake/Eigen3)

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

set(
    Benchmark_libs
    RandLAPACK
)

# Declaring a full set of libs needed for one of RBKI benchmarks.
set(
    Benchmark_libs_external
    RandLAPACK
    Eigen3::Eigen
    Spectra
)

# Performance profiling through GEMM
add_benchmark(NAME GEMM_flop_count CXX_SOURCES bench_general/GEMM_flop_count.cc LINK_LIBS ${Benchmark_libs})
# Lapack functionality benchmark
add_benchmark(NAME Chol_check      CXX_SOURCES bench_general/Chol_check.cc      LINK_LIBS ${Benchmark_libs})
# Data conversion helper script
add_benchmark(NAME convert_time    CXX_SOURCES bench_general/convert_time.cc    LINK_LIBS ${Benchmark_libs})
# Compare GEMM and ORMQR performance
add_benchmark(NAME Gemm_vs_ormqr   CXX_SOURCES bench_general/Gemm_vs_ormqr.cc   LINK_LIBS ${Benchmark_libs})

# CQRRPT benchmarks
add_benchmark(NAME CQRRPT_speed_comparisons CXX_SOURCES bench_CQRRPT/CQRRPT_speed_comparisons.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_runtime_breakdown CXX_SOURCES bench_CQRRPT/CQRRPT_runtime_breakdown.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_pivot_quality     CXX_SOURCES bench_CQRRPT/CQRRPT_pivot_quality.cc     LINK_LIBS ${Benchmark_libs})

# CQRRP benchmarks
add_benchmark(NAME CQRRP_speed_comparisons       CXX_SOURCES bench_CQRRP/CQRRP_speed_comparisons.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRP_runtime_breakdown       CXX_SOURCES bench_CQRRP/CQRRP_runtime_breakdown.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRP_single_precision        CXX_SOURCES bench_CQRRP/CQRRP_single_precision.cc  LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRP_pivot_quality           CXX_SOURCES bench_CQRRP/CQRRP_pivot_quality.cc     LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME HQRRP_runtime_breakdown       CXX_SOURCES bench_CQRRP/HQRRP_runtime_breakdown.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME QR_speed_comp                 CXX_SOURCES bench_CQRRP/QR_speed_comp.cc           LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME ICQRRP_subroutines_speed      CXX_SOURCES bench_CQRRP/ICQRRP_subroutines_speed.cc LINK_LIBS ${Benchmark_libs})

# RBKI benchmarks
add_benchmark(NAME RBKI_speed_comparisons      CXX_SOURCES bench_RBKI/RBKI_speed_comparisons.cc      LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME RBKI_runtime_breakdown      CXX_SOURCES bench_RBKI/RBKI_runtime_breakdown.cc      LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME RBKI_speed_comparisons_SVDS CXX_SOURCES bench_RBKI/RBKI_speed_comparisons_SVDS.cc LINK_LIBS ${Benchmark_libs_external})
