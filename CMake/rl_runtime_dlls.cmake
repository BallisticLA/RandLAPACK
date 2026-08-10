# Native Windows builds stage every runtime DLL an executable needs beside
# that executable (app-local deployment, the idiomatic Windows layout: the
# exe's own directory is the first place the loader searches). Two sources:
#
#   1. TARGET_RUNTIME_DLLS (CMake >= 3.21, this project's minimum) covers
#      imported SHARED targets -- BLAS++/LAPACK++ DLLs.
#   2. RANDLAPACK_RUNTIME_DLL_DIRS covers what the generator expression
#      cannot see: the BLAS backend (oneMKL, OpenBLAS, ...) enters BLAS++ as
#      raw library paths, i.e. UNKNOWN imported targets, which
#      TARGET_RUNTIME_DLLS documentedly ignores. The installer and CI set
#      this to the backend's DLL directory; its *.dll contents are staged
#      alongside each executable.
#
# With both in place, staged executables run without any PATH preparation.
set(RANDLAPACK_RUNTIME_DLL_DIRS "" CACHE STRING
    "Semicolon-separated directories whose DLLs are staged beside RandLAPACK executables on Windows.")

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
        foreach(dll_dir IN LISTS RANDLAPACK_RUNTIME_DLL_DIRS)
            file(GLOB dlls "${dll_dir}/*.dll")
            if (dlls)
                add_custom_command(
                    TARGET ${target}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            ${dlls}
                            $<TARGET_FILE_DIR:${target}>
                    VERBATIM
                )
            endif()
        endforeach()
    endif()
endfunction()
