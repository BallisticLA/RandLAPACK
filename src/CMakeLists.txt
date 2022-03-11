set(RandLAPACK_cxx_srcs
    comps/determiter.cc
)

set(RandLAPACK_libs
    RandLAPACK_RandBLAS
    RandLAPACK_lapackpp
    RandBLAS_blaspp
    RandBLAS_Random123
    RandBLAS_OpenMP
) 

add_library(RandLAPACK ${RandLAPACK_cxx_srcs})

target_include_directories(
    RandLAPACK PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/RandLAPACK>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/RandLAPACK>
)

target_link_libraries(RandLAPACK PUBLIC ${RandLAPACK_libs})

install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
    DESTINATION include FILES_MATCHING PATTERN "*.hh"
)

install(
    TARGETS RandLAPACK
    EXPORT RandLAPACK
    INCLUDES DESTINATION include/RandLAPACK
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)