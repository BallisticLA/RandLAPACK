cmake_minimum_required(VERSION 3.10)

project(benchmark)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Checking for RandLAPACK ... ")
find_package(RandLAPACK REQUIRED)
message(STATUS "Done checking for RandLAPACK. ...")

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
    FLOP_count PUBLIC RandLAPACK
)
