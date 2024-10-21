# Initialize tmp variable
set(tmp)

# Find Git executable
find_package(Git QUIET)
if(GIT_FOUND)
    message(STATUS "Git found: ${GIT_EXECUTABLE}")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_SOURCE_DIR}/.git describe --tags --match "[0-9]*.[0-9]*.[0-9]*"
        OUTPUT_VARIABLE tmp
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE git_error
        RESULT_VARIABLE git_result
    )
    
    # Print the result of the Git command
    message(STATUS "Git command result: ${git_result}")
    message(STATUS "Git command output: ${tmp}")
    if(NOT git_result EQUAL 0)
        message(WARNING "Git command failed with error: ${git_error}")
        set(tmp "0.0.0")
    endif()
else()
    message(WARNING "Git not found, using fallback version 0.0.0")
    set(tmp "0.0.0")
endif()

# Check if tmp is empty and set a fallback version if necessary
if(NOT tmp)
    message(WARNING "Git describe output is empty, using fallback version 0.0.0")
    set(tmp "0.0.0")
endif()

# Debugging: Print tmp before setting RandLAPACK_VERSION
message(STATUS "tmp before setting RandLAPACK_VERSION: ${tmp}")

# Set RandLAPACK_VERSION without CACHE option
set(RandLAPACK_VERSION "${tmp}")
message(STATUS "RandLAPACK_VERSION after setting: ${RandLAPACK_VERSION}")

# Ensure RandLAPACK_VERSION is not empty
if(NOT RandLAPACK_VERSION)
    message(FATAL_ERROR "RandLAPACK_VERSION is empty")
endif()

# Extract major, minor, and patch versions
string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*)$" "\\1" RandLAPACK_VERSION_MAJOR "${RandLAPACK_VERSION}")
string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*)$" "\\2" RandLAPACK_VERSION_MINOR "${RandLAPACK_VERSION}")
string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*)$" "\\3" RandLAPACK_VERSION_PATCH "${RandLAPACK_VERSION}")

# Print extracted version components
message(STATUS "RandLAPACK_VERSION_MAJOR=${RandLAPACK_VERSION_MAJOR}")
message(STATUS "RandLAPACK_VERSION_MINOR=${RandLAPACK_VERSION_MINOR}")
message(STATUS "RandLAPACK_VERSION_PATCH=${RandLAPACK_VERSION_PATCH}")
