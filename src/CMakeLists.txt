set(RandLAPACK_cxx_srcs
    comps/determiter.cc
    comps/qb.cc
    comps/rf.cc
    comps/rs.cc
    comps/util.cc
    comps/orth.cc
)

set(RandLAPACK_libs
    RandBLAS
    lapackpp
    blaspp
    Random123
    OpenMP::OpenMP_CXX
)

add_library(RandLAPACK ${RandLAPACK_cxx_srcs})
target_compile_options(RandLAPACK PUBLIC -Wall -Wextra)

target_include_directories(
    RandLAPACK PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/RandLAPACK>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandLAPACK>
)

target_link_libraries(RandLAPACK PUBLIC ${RandLAPACK_libs})

install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
    DESTINATION include FILES_MATCHING PATTERN "*.hh"
)

install(
    TARGETS RandLAPACK
    EXPORT RandLAPACK
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    EXPORT RandLAPACK
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
)
