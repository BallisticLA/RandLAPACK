# Native Windows builds use TARGET_RUNTIME_DLLS to stage imported shared-library
# dependencies (BLAS++/LAPACK++ DLLs) beside executables, so that test discovery
# and test runs find them without PATH edits. The generator expression requires
# CMake 3.21, which is already this project's minimum. Note this only covers
# CMake imported targets: oneMKL enters through raw library paths recorded by
# BLAS++, so the MKL bin directory must be on PATH at build and test time.
function(randlapack_stage_runtime_dlls target)
    if (WIN32)
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:${target}>
                    $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
    endif()
endfunction()
