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
    GEMM_flop_count_cxx
    GEMM_flop_count.cc
)
add_executable(
    GEMM_flop_count ${GEMM_flop_count_cxx}
)
set(
    RandLAPACK_include_dirs
    ${RandLAPACK_DIR}/../../include
    ${RandLAPACK_DIR}/../../include/RandLAPACK
)
target_include_directories(
    GEMM_flop_count PUBLIC ${Random123_DIR} ${RandLAPACK_include_dirs}
)
set(
    RandLAPACK_lib_file
    ${RandLAPACK_DIR}/../../lib/libRandLAPACK.a
)
target_link_libraries(
    GEMM_flop_count PUBLIC ${RandLAPACK_lib_file} RandBLAS blaspp lapackpp
)

set(
    Orth_speed_cxx
    Orth_speed.cc
)
add_executable(
    Orth_speed ${Orth_speed_cxx}
)
target_compile_options(
    Orth_speed PRIVATE "-g"
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
    Orth_speed PUBLIC ${RandLAPACK_lib_file} RandBLAS blaspp lapackpp
)

set(
    CholQRCP_speed_cxx
    CholQRCP_speed.cc
)
add_executable(
    CholQRCP_speed ${CholQRCP_speed_cxx}
)
target_compile_options(
    CholQRCP_speed PRIVATE "-g"
)
set(
    RandLAPACK_include_dirs
    ${RandLAPACK_DIR}/../../include
    ${RandLAPACK_DIR}/../../include/RandLAPACK
)
target_include_directories(
    CholQRCP_speed PUBLIC ${Random123_DIR} ${RandLAPACK_include_dirs}
)
set(
    RandLAPACK_lib_file
    ${RandLAPACK_DIR}/../../lib/libRandLAPACK.a
)
target_link_libraries(
    CholQRCP_speed PUBLIC ${RandLAPACK_lib_file} RandBLAS blaspp lapackpp
)

set(
    QR_speed_basic_cxx
    QR_speed_basic.cc
)
add_executable(
    QR_speed_basic ${QR_speed_basic_cxx}
)
target_compile_options(
    QR_speed_basic PRIVATE "-g"
)
set(
    RandLAPACK_include_dirs
    ${RandLAPACK_DIR}/../../include
    ${RandLAPACK_DIR}/../../include/RandLAPACK
)
target_include_directories(
    QR_speed_basic PUBLIC ${Random123_DIR} ${RandLAPACK_include_dirs}
)
set(
    RandLAPACK_lib_file
    ${RandLAPACK_DIR}/../../lib/libRandLAPACK.a
)
target_link_libraries(
    QR_speed_basic PUBLIC ${RandLAPACK_lib_file} RandBLAS blaspp lapackpp
)

set(
    CholQRCP_accuracy_cxx
    CholQRCP_accuracy.cc
)
add_executable(
    CholQRCP_accuracy ${CholQRCP_accuracy_cxx}
)
target_compile_options(
    CholQRCP_accuracy PRIVATE "-g"
)
set(
    RandLAPACK_include_dirs
    ${RandLAPACK_DIR}/../../include
    ${RandLAPACK_DIR}/../../include/RandLAPACK
)
target_include_directories(
    CholQRCP_accuracy PUBLIC ${Random123_DIR} ${RandLAPACK_include_dirs}
)
set(
    RandLAPACK_lib_file
    ${RandLAPACK_DIR}/../../lib/libRandLAPACK.a
)
target_link_libraries(
    CholQRCP_accuracy PUBLIC ${RandLAPACK_lib_file} RandBLAS blaspp lapackpp
)

set(
    FLOP_data_cxx
    FLOP_data.cc
)
add_executable(
    FLOP_data ${FLOP_data_cxx}
)
target_compile_options(
    FLOP_data PRIVATE "-g"
)
set(
    RandLAPACK_include_dirs
    ${RandLAPACK_DIR}/../../include
    ${RandLAPACK_DIR}/../../include/RandLAPACK
)
target_include_directories(
    FLOP_data PUBLIC ${Random123_DIR} ${RandLAPACK_include_dirs}
)
set(
    RandLAPACK_lib_file
    ${RandLAPACK_DIR}/../../lib/libRandLAPACK.a
)
target_link_libraries(
    FLOP_data PUBLIC ${RandLAPACK_lib_file} RandBLAS blaspp lapackpp
)

