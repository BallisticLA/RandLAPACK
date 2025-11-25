# Modern compiler flags configuration using generator expressions
# These should be applied per-target using target_compile_options

# Function to add default compiler flags to a target
function(add_randlapack_compile_options target_name)
    # Common flags for all configurations
    target_compile_options(${target_name} PRIVATE
        -fPIC
        -Wall
        -Wextra
        -Wno-unknown-pragmas
    )

    # Apple Clang specific flags
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target_name} PRIVATE -stdlib=libc++)
    endif()

    # Release-specific optimizations
    target_compile_options(${target_name} PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-mtune=native>
        $<$<CONFIG:Release>:-fno-trapping-math>
        $<$<CONFIG:Release>:-fno-math-errno>
    )

    # Non-Clang specific flags for Release
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target_name} PRIVATE
            $<$<CONFIG:Release>:-fno-signaling-nans>
        )
    endif()
endfunction()

# Function to add default CUDA compiler flags to a target
function(add_randlapack_cuda_compile_options target_name)
    # Common CUDA flags
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # Release-specific CUDA flags
    target_compile_options(${target_name} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fopenmp,-Wall,-Wextra,-O3,-march=native,-mtune=native,-fno-trapping-math,-fno-math-errno>
    )

    # Non-Clang specific CUDA flags for Release
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fno-signaling-nans>
        )
    endif()

    # Debug-specific CUDA flags
    target_compile_options(${target_name} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fopenmp,-Wall,-Wextra,-O0,-g>
    )
endfunction()
