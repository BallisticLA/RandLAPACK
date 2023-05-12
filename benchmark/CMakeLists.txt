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

# Bencmarks for Stab/Orth tech report
add_benchmark(NAME GEMM_flop_count CXX_SOURCES GEMM_flop_count.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME Orth_speed CXX_SOURCES Orth_speed.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME QB_cond_nums CXX_SOURCES QB_cond_nums.cc LINK_LIBS ${Benchmark_libs})

# Benchmarks for CQRRPT paper
add_benchmark(NAME QR_speed CXX_SOURCES QR_speed.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_accuracy CXX_SOURCES CQRRPT_accuracy.cc  LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_inner_speed CXX_SOURCES CQRRPT_inner_speed.cc  LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_embedding_speed_effect CXX_SOURCES CQRRPT_embedding_speed_effect.cc  LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_vs_HQRRP_speed CXX_SOURCES CQRRPT_vs_HQRRP_speed.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME QR_accuracy_stability_analysis CXX_SOURCES QR_accuracy_stability_analysis.cc LINK_LIBS ${Benchmark_libs})

# Data processing scripts for CQRRPT paper
add_benchmark(NAME CQRRPT_vs_HQRRP_flops CXX_SOURCES CQRRPT_vs_HQRRP_flops.cc  LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME QR_flops CXX_SOURCES QR_flops.cc LINK_LIBS ${Benchmark_libs})
add_benchmark(NAME CQRRPT_inner_speed_process CXX_SOURCES CQRRPT_inner_speed_process.cc LINK_LIBS ${Benchmark_libs})

# Data conversion helper scripts
add_benchmark(NAME convert_time CXX_SOURCES convert_time.cc LINK_LIBS ${Benchmark_libs})
