# TODO stop using gtest
#message(STATUS "Checking for benchmarks ...")
#find_package(GTest)
#set(tmp FALSE)
#if (GTest_FOUND)
#    set(tmp TRUE)
#
#    add_executable(RandLAPACK_benchmark experiments/comps/QB_cond_nums.cc experiments/comps/Orth_speed.cc)
#    target_link_libraries(RandLAPACK_benchmark OpenMP::OpenMP_CXX RandLAPACK GTest::GTest GTest::Main)
#    gtest_discover_tests(RandLAPACK_benchmark)
#
#endif()
#message(STATUS "Checking for benchmarks ... ${tmp}")

cmake_minimum_required(VERSION 3.10)

project(benchmark)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Checking for RandLAPACK ... ")
find_package(RandLAPACK REQUIRED)
message(STATUS "Done checking for RandLAPACK. ...")

message(STATUS "Looking for RandBLAS ... ")
find_package(RandBLAS REQUIRED)
message(STATUS "Done looking for RandBLAS.")

set(
    FLOP_count_cxx
    FLOP_count.cc
)
add_executable(
    FLOP_count ${FLOP_count_cxx}
)
target_include_directories(
    FLOP_count PUBLIC ${Random123_DIR}
)
target_link_libraries(
    FLOP_count PUBLIC RandLAPACK RandBLAS
)
