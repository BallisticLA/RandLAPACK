cmake_minimum_required(VERSION 3.10)

# cmake -DCMAKE_BUILD_TYPE=Release -DRandLAPACK_DIR=`pwd`/../../../RandLAPACK-install/lib/cmake/ -DAMD_DIR=`pwd`/../../../SuiteSparse/lib/cmake/AMD -DSuiteSparse_config_DIR=`pwd`/../../../SuiteSparse/lib/cmake/SuiteSparse_config  -DCMAKE_BINARY_DIR=`pwd`  ..
# cmake -DCMAKE_BUILD_TYPE=Debug -DRandLAPACK_DIR=`pwd`/../../../RandLAPACK-install/lib64/cmake/ -DCMAKE_CXX_FLAGS="-Xarch_host -fsanitize=address"  -DCMAKE_BINARY_DIR=`pwd`  ..

# cmake -DCMAKE_BUILD_TYPE=Debug -DRandLAPACK_DIR=`pwd`/../../../RandLAPACK-install/lib/cmake/ -DCMAKE_CXX_FLAGS="-Xarch_host -fsanitize=address" -DAMD_DIR=`pwd`/../../../SuiteSparse/lib/cmake/AMD -DSuiteSparse_config_DIR=`pwd`/../../../SuiteSparse/lib/cmake/SuiteSparse_config  -DCMAKE_BINARY_DIR=`pwd`  ..

project(demos)

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

message(STATUS "Checking for SuiteSparse::AMD ... ")
find_package(AMD REQUIRED)
include_directories(${AMD_INCLUDE_DIRS})
message(STATUS "AMD include directories: ${AMD_INCLUDE_DIRS}")
message(STATUS "Done checking for SuiteSparse::AMD.")

function(add_demo)
    set(OPTS)
    set(NVPO NAME)
    set(MVO CXX_SOURCES LINK_LIBS)
    cmake_parse_arguments(PARSE_ARGV 0 TGT "${OPTS}" "${NVPO}" "${MVO}")
    add_executable(${TGT_NAME} ${TGT_CXX_SOURCES})
    target_compile_options(${TGT_NAME} PRIVATE "-O2")
    target_link_libraries(${TGT_NAME} ${TGT_LINK_LIBS})
    message(STATUS "RandLAPACK: added ${TGT_NAME} benchmark")
endfunction()

include(FetchContent)
FetchContent_Declare(
    fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)
#add_subdirectory(build/_deps/fast_matrix_market-src)

set(
    main_libs
    RandLAPACK
    fast_matrix_market::fast_matrix_market
    SuiteSparse::AMD
)

if (DEFINED USE_BOOST)
FetchContent_Declare(
  boost_config
  GIT_REPOSITORY https://github.com/boostorg/config.git
  GIT_TAG        boost-1.89.0
  GIT_SHALLOW    TRUE
)
FetchContent_Declare(
  boost_multiprecision
  GIT_REPOSITORY https://github.com/boostorg/multiprecision.git
  GIT_TAG        boost-1.89.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(boost_config boost_multiprecision)
set(
    main_libs ${main_libs} boost_config boost_multiprecision
)
endif()

set(richol_headers
    richol/richol_core.hh
    richol/richol_linops.hh
)

add_demo(NAME demo_factor   CXX_SOURCES richol/richol_core.hh  richol/demo_factor.cc    LINK_LIBS ${main_libs} )
add_demo(NAME demo_pinv     CXX_SOURCES ${richol_headers} richol/demo_pinv.cc      LINK_LIBS ${main_libs} )
add_demo(NAME demo_solve    CXX_SOURCES ${richol_headers} richol/demo_solve.cc  LINK_LIBS ${main_libs} )
