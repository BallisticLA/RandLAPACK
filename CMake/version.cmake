set(tmp)
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE}
        --git-dir=${CMAKE_SOURCE_DIR}/.git describe --tags
        OUTPUT_VARIABLE tmp OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()
if(NOT tmp)
    set(tmp "0.0.0")
endif()
set(RandLAPACK_VERSION ${tmp} CACHE STRING "RandLAPACK version" FORCE)

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\1" RandLAPACK_VERSION_MAJOR ${RandLAPACK_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\2" RandLAPACK_VERSION_MINOR ${RandLAPACK_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\3" RandLAPACK_VERSION_PATCH ${RandLAPACK_VERSION})

message(STATUS "RandLAPACK_VERSION_MAJOR=${RandLAPACK_VERSION_MAJOR}")
message(STATUS "RandLAPACK_VERSION_MINOR=${RandLAPACK_VERSION_MINOR}")
message(STATUS "RandLAPACK_VERSION_PATCH=${RandLAPACK_VERSION_PATCH}")
