
set(RandLAPACK_cxx_sources
    rl_rbki.hh
    rl_lapackpp.hh
    rl_cqrrpt.hh
    rl_cqrrp.hh
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
    rl_gen.hh
    rl_blaspp.hh
    rl_linops.hh

    rl_cusolver.hh
    rl_cuda_kernels.cuh
    rl_cuda_macros.hh
    rl_cqrrpt_gpu.hh
    rl_cqrrp_gpu.hh
)

add_library(RandLAPACK INTERFACE)

target_link_libraries(RandLAPACK INTERFACE 
    RandBLAS
    lapackpp
    blaspp
    Random123
)

if (RandLAPACK_HAS_OpenMP)
    target_link_libraries(RandLAPACK INTERFACE OpenMP::OpenMP_CXX)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rl_config.hh.in rl_config.hh)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rl_config.hh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK)


set(RandLAPACK_cxx_opts -Wall -Wextra)
if (SANITIZE_ADDRESS)
    list(APPEND RandLAPACK_cxx_opts -fsanitize=address)
    target_link_options(RandLAPACK INTERFACE -fsanitize=address)
endif()
target_compile_options(RandLAPACK INTERFACE ${RandLAPACK_cxx_opts})

#set_source_files_properties(${RandLAPACK_cxx_sources}
#    PROPERTIES COMPILE_OPTIONS "${RandLAPACK_cxx_opts}"
#)

target_include_directories(
    RandLAPACK INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/comps>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/drivers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu_functions>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/comps>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/drivers>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/misc>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK/gpu_functions>
)

install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/../RandLAPACK.hh"
    DESTINATION include
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DESTINATION include FILES_MATCHING PATTERN "*.hh"
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
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
)
