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
#
# Declare the cache entry only when no value exists yet. RandLAPACKConfig.cmake
# sets a normal variable to the directories recorded when RandLAPACK was
# configured and then includes this file; in a consumer whose
# cmake_minimum_required is below 3.21 (policy CMP0126 OLD), declaring the
# cache entry on a fresh configure deletes that normal variable and the
# backend DLLs are silently not staged.
if (NOT DEFINED RANDLAPACK_RUNTIME_DLL_DIRS)
    set(RANDLAPACK_RUNTIME_DLL_DIRS "" CACHE STRING
        "Semicolon-separated directories whose DLLs are staged beside RandLAPACK executables on Windows.")
endif()

function(randlapack_stage_runtime_dlls target)
    if (WIN32)
        # TARGET_RUNTIME_DLLS is empty when every dependency is static, and
        # copy_if_different given only a destination fails the build; run a
        # no-op instead in that case.
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E
                    $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${target}>>,copy_if_different,true>
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
