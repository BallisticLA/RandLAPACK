cmake_minimum_required(VERSION 3.10)

project(benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_FIND_DEBUG_MODE TRUE)
message(STATUS "Checking for RandLAPACK ... ")
find_package(RandLAPACK REQUIRED)
message(STATUS "Done checking for RandLAPACK.")
#set(CMAKE_FIND_DEBUG_MODE FALSE)


message(STATUS "Checking for RandBLAS ... ")
find_package(RandBLAS REQUIRED)
message(STATUS "Done checking for RandBLAS.")

message(STATUS "Looking for BLAS++ ... ")
find_package(blaspp REQUIRED)
message(STATUS "Done looking for BLAS++.")

message(STATUS "Looking for LAPACK++ ...")
find_package(lapackpp REQUIRED)
message(STATUS "Done looking for LAPACK++.")


set(
    FLOP_count_cxx
    FLOP_count.cc
)
add_executable(
    FLOP_count ${FLOP_count_cxx}
)
set(
    RandLAPACK_include_dirs
    ${RandLAPACK_DIR}/../../include
    ${RandLAPACK_DIR}/../../include/RandLAPACK
)
target_include_directories(
    FLOP_count PUBLIC ${Random123_DIR} ${RandLAPACK_include_dirs}
)
set(
    RandLAPACK_lib_file
    ${RandLAPACK_DIR}/../../lib/libRandLAPACK.a
)
target_link_libraries(
    FLOP_count PUBLIC RandBLAS blaspp lapackpp ${RandLAPACK_lib_file}
)

set(
    Orth_speed_cxx
    Orth_speed.cc
)
add_executable(
    Orth_speed ${Orth_speed_cxx}
)
set(
    RandLAPACK_include_dirs
    ${RandLAPACK_DIR}/../../include
    ${RandLAPACK_DIR}/../../include/RandLAPACK
)
target_include_directories(
    Orth_speed PUBLIC ${Random123_DIR} ${RandLAPACK_include_dirs}
)
set(
    RandLAPACK_lib_file
    ${RandLAPACK_DIR}/../../lib/libRandLAPACK.a
)
target_link_libraries(
    Orth_speed PUBLIC RandBLAS blaspp lapackpp ${RandLAPACK_lib_file}
)
