set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Configure to build shared or static libraries" OFF)
option(RandLAPACK_BUILD_TESTS "Build RandLAPACK's test suite" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
  CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(SANITIZE_ADDRESS OFF CACHE BOOL "Add address sanitizer flags to the library")

message(STATUS "Checking for OpenMP ... ")

# This find_package runs BEFORE add_subdirectory(RandBLAS), so the guard in
# RandBLAS/CMake/OpenMP.cmake ("if NOT DEFINED OpenMP_CXX_FLAGS") never
# fires: whatever flavor is cached here is what every target gets. MSVC's
# classic /openmp implements OpenMP 2.0 only and silently *ignores* the
# collapse clause (warning C4849) that rl_rpchol relies on; /openmp:llvm
# supports 64-bit loop indices and collapse. Mirror RandBLAS's guard here.
# Callers can still override with -DOpenMP_CXX_FLAGS=... at configure time.
if (MSVC AND NOT DEFINED OpenMP_CXX_FLAGS)
    set(OpenMP_CXX_FLAGS "/openmp:llvm" CACHE STRING
        "OpenMP compiler flags for C++")
endif()
if (MSVC)
    set(RandLAPACK_OpenMP_MSVC_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenMP COMPONENTS CXX)

# FindOpenMP may replace OpenMP_CXX_FLAGS while probing the compiler. Ensure
# the imported target carries the requested MSVC mode.
if (MSVC AND OpenMP_CXX_FOUND AND TARGET OpenMP::OpenMP_CXX)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
        INTERFACE_COMPILE_OPTIONS "${RandLAPACK_OpenMP_MSVC_FLAGS}")
    set(OpenMP_CXX_FLAGS "${RandLAPACK_OpenMP_MSVC_FLAGS}")
endif()

set(tmp FALSE)
if (OpenMP_CXX_FOUND)
    set(tmp TRUE)
endif()

set(RandBLAS_HAS_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP" FORCE)
message(STATUS "Checking for OpenMP ... ${RandBLAS_HAS_OpenMP}")

include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

