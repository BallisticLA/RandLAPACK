# set default compiler flags
if (NOT CMAKE_CXX_FLAGS)
    set(tmp "-fPIC -std=c++20 -Wall -Wextra -Wno-unknown-pragmas")
    if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
        set(tmp "${tmp} -stdlib=libc++")
    endif()
    if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
        set(tmp "${tmp} -O3 -march=native -mtune=native -fno-trapping-math -fno-math-errno")
        if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            set(tmp "${tmp} -fno-signaling-nans")
        endif()
    endif()
    set(CMAKE_CXX_FLAGS "${tmp}"
            CACHE STRING "RandLAPACK build defaults"
        FORCE)
endif()
if (NOT CMAKE_CUDA_FLAGS)
    set(tmp "--default-stream per-thread --expt-relaxed-constexpr")
    if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
        set(tmp "${tmp} -Xcompiler -fopenmp,-Wall,-Wextra,-O3,-march=native,-mtune=native,-fno-trapping-math,-fno-math-errno")
        if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            set(tmp "${tmp},-fno-signaling-nans")
        endif()
    elseif ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
        set(tmp "${tmp} -g -G -Xcompiler -fopenmp,-Wall,-Wextra,-O0,-g")
    endif()
    set(CMAKE_CUDA_FLAGS "${tmp}"
        CACHE STRING "CUDA compiler build defaults"
        FORCE)
    string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${tmp}"
        CACHE STRING "CUDA compiler build defaults"
        FORCE)
    string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CUDA_FLAGS_RELEASE}")
    set(CMAKE_CUDA_FLAGS_RELEASE "${tmp}"
        CACHE STRING "CUDA compiler build defaults"
        FORCE)
endif()