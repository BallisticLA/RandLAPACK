set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS OFF "Configure to build shared or static libraries")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
  CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(SANITIZE_ADDRESS OFF CACHE BOOL "Add address sanitizer flags to the library")

message(STATUS "Checking for OpenMP ... ")
find_package(OpenMP COMPONENTS CXX)

set(tmp FALSE)
if (OpenMP_CXX_FOUND)
    set(tmp TRUE)
endif()

set(RandBLAS_HAS_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP")
message(STATUS "Checking for OpenMP ... ${RandBLAS_HAS_OpenMP}")

include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

