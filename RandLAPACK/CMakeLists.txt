
set(RandLAPACK_cxx_sources
    rl_abrik.hh
    rl_lapackpp.hh
    rl_cqrrt.hh
    rl_cqrrpt.hh
    rl_bqrrp.hh
    rl_rsvd.hh
    rl_revd2.hh
    rl_qb.hh
    rl_orth.hh
    rl_util.hh
    rl_determiter.hh
    rl_rs.hh
    rl_rf.hh
    rl_syps.hh
    rl_syrf.hh
    rl_rpchol.hh
    rl_blaspp.hh
    rl_pdkernels.hh
    rl_exceptions.hh

    rl_cusolver.hh
    rl_cuda_kernels.cuh
    rl_cuda_macros.hh
    rl_cqrrpt_gpu.hh
    rl_bqrrp_gpu.hh
)

add_library(RandLAPACK INTERFACE)
target_compile_features(RandLAPACK INTERFACE cxx_std_20)

target_link_libraries(RandLAPACK INTERFACE
    RandBLAS
    lapackpp
    blaspp
    Random123::Random123
)

if (RandBLAS_HAS_OpenMP)
    target_link_libraries(RandLAPACK INTERFACE OpenMP::OpenMP_CXX)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rl_config.hh.in rl_config.hh)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rl_config.hh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK)


# RandLAPACK is header-only, so MSVC requirements must propagate to every
# translation unit that includes its headers: /utf-8 because several headers
# contain non-ASCII characters in string literals (Greek letters in timing
# and diagnostic messages), which MSVC otherwise interprets in the local
# codepage. GCC and Clang read source as UTF-8 already.
target_compile_options(RandLAPACK INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>)

# Build-time only: warning and sanitizer flags should not leak into the installed target.
target_compile_options(RandLAPACK INTERFACE
    $<BUILD_INTERFACE:$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall>>
    $<BUILD_INTERFACE:$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wextra>>
    $<BUILD_INTERFACE:$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>>)

if (SANITIZE_ADDRESS)
    if (MSVC)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("/fsanitize=address" RANDLAPACK_MSVC_ASAN_SUPPORTED)
        if (NOT RANDLAPACK_MSVC_ASAN_SUPPORTED)
            message(FATAL_ERROR "SANITIZE_ADDRESS=ON requires an MSVC toolchain with "
                "/fsanitize=address support and its AddressSanitizer runtime installed.")
        endif()
        # /fsanitize=address is a compile option, not a raw link option, on MSVC.
        target_compile_options(RandLAPACK INTERFACE
            $<BUILD_INTERFACE:/fsanitize=address> $<BUILD_INTERFACE:/Zi>)
        target_link_options(RandLAPACK INTERFACE $<BUILD_INTERFACE:/DEBUG>)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(RandLAPACK INTERFACE $<BUILD_INTERFACE:-fsanitize=address>)
        target_link_options(RandLAPACK    INTERFACE $<BUILD_INTERFACE:-fsanitize=address>)
    else()
        message(FATAL_ERROR "SANITIZE_ADDRESS=ON is not supported with ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

target_include_directories(
    RandLAPACK INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/comps>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/drivers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/testing>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/linops>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu_functions>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/comps>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/drivers>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/misc>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/linops>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/gpu_functions>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/testing>
)

install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/../RandLAPACK.hh"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.hh"
        PATTERN "*.cuh"
)

install(
    TARGETS RandLAPACK
    EXPORT RandLAPACK
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    EXPORT RandLAPACK
    NAMESPACE RandLAPACK::
    FILE RandLAPACKTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK
)
