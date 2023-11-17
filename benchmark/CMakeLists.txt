cmake_minimum_required(VERSION 3.10)
project(benchmark)

set(CMAKE_CXX_STANDARD 17)
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

set(
    Benchmark_libs
    RandLAPACK
)
# Lapack functionality benchmarks
add_benchmark(NAME Chol_check CXX_SOURCES Chol_check.cc LINK_LIBS ${Benchmark_libs})

# Data conversion helper scripts
add_benchmark(NAME convert_time CXX_SOURCES convert_time.cc LINK_LIBS ${Benchmark_libs})

# QR speed comparisons with CQRRP benchmarks
add_benchmark(NAME CQRRP_speed_comparisons CXX_SOURCES CQRRP_speed_comparisons.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CHOLQR_vs_GEQRF CXX_SOURCES CHOLQR_vs_GEQRF.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRP_block_per_time CXX_SOURCES CQRRP_block_per_time.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRP_inner_speed CXX_SOURCES CQRRP_inner_speed.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRP_accuracy CXX_SOURCES CQRRP_accuracy.cc LINK_LIBS ${Benchmark_libs})

add_benchmark(NAME Gemm_vs_ormqr CXX_SOURCES Gemm_vs_ormqr.cc LINK_LIBS ${Benchmark_libs})

add_benchmark(NAME RBKI_speed_comparisons CXX_SOURCES bench_RBKI/RBKI_speed_comparisons.cc LINK_LIBS ${Benchmark_libs})
