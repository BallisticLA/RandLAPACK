
function(add_benchmark)
    set(OPTS)
    set(NVPO NAME)
    set(MVO CXX_SOURCES LINK_LIBS)
    cmake_parse_arguments(PARSE_ARGV 0 TGT "${OPTS}" "${NVPO}" "${MVO}")
    add_executable(${TGT_NAME} ${TGT_CXX_SOURCES})
    target_compile_options(${TGT_NAME} PRIVATE "-O3")
    target_link_libraries(${TGT_NAME} ${TGT_LINK_LIBS})
    message(STATUS "RandLAPACK: added ${TGT_NAME} benchmark")
endfunction()

# Bencmarks for Stab/Orth tech report
add_benchmark(NAME GEMM_flop_count CXX_SOURCES GEMM_flop_count.cc LINK_LIBS RandLAPACK)
add_benchmark(NAME Orth_speed CXX_SOURCES Orth_speed.cc LINK_LIBS RandLAPACK)
add_benchmark(NAME QB_cond_nums CXX_SOURCES QB_cond_nums.cc LINK_LIBS RandLAPACK)

# Benchmarks for cqrrpt paper
add_benchmark(NAME cqrrpt_speed CXX_SOURCES cqrrpt_speed.cc LINK_LIBS RandLAPACK)
add_benchmark(NAME cqrrpt_accuracy CXX_SOURCES cqrrpt_accuracy.cc  LINK_LIBS RandLAPACK)
add_benchmark(NAME cqrrpt_inner_speed CXX_SOURCES cqrrpt_inner_speed.cc  LINK_LIBS RandLAPACK)
add_benchmark(NAME cqrrpt_embedding_speed_effect CXX_SOURCES cqrrpt_embedding_speed_effect.cc  LINK_LIBS RandLAPACK)
add_benchmark(NAME cqrrpt_vs_HQRRP_speed CXX_SOURCES cqrrpt_vs_HQRRP_speed.cc LINK_LIBS RandLAPACK)

# Data processing scripts for cqrrpt paper
add_benchmark(NAME FLOP_data_hqrrp CXX_SOURCES FLOP_data_hqrrp.cc  LINK_LIBS RandLAPACK)
add_benchmark(NAME FLOP_data CXX_SOURCES FLOP_data.cc LINK_LIBS RandLAPACK)
add_benchmark(NAME cqrrpt_inner_speed_process CXX_SOURCES cqrrpt_inner_speed_process.cc LINK_LIBS RandLAPACK)
