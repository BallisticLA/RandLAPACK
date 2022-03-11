# interface library for RBLAS

message(
    STATUS "RandLAPACK checking for RandBLAS ... "
)
find_package(
    RandBLAS REQUIRED
)
message(
    STATUS "RandLAPACK found RandBLAS ${RandBLAS_VERSION}\n"
)
add_library(
    RandLAPACK_RandBLAS INTERFACE
)
target_link_libraries(
    RandLAPACK_RandBLAS INTERFACE RandBLAS
)
install(
    TARGETS RandLAPACK_RandBLAS EXPORT RandLAPACK_RandBLAS
)
install(
    EXPORT RandLAPACK_RandBLAS
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES
)

# interface library for BLAS++, inherited from RBLAS.

message(
    STATUS "RandLAPACK checking for BLAS++, inherited from RBLAS ..."
)
find_package(
    blaspp REQUIRED
)
message(
    STATUS "RandLAPACK found BLAS++ ${blaspp_VERSION}.\n"
)
add_library(
    RandLAPACK_blaspp INTERFACE
)
target_link_libraries(
    RandLAPACK_blaspp INTERFACE blaspp
)
install(
    TARGETS RandLAPACK_blaspp EXPORT RandLAPACK_blaspp
)
install(
    EXPORT RandLAPACK_blaspp
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES
)

# interface library for Random123, inherited from RBLAS.

message(
    STATUS "RandLAPACK checking for Random123, inherited from RBLAS ... "
)
find_package(
    Random123 REQUIRED
)
message(
    STATUS "RandLAPACK found Random123 ${Random123_VERSION}.\n"
)
add_library(
    RandLAPACK_Random123 INTERFACE
)
target_include_directories(
    RandLAPACK_Random123
    SYSTEM INTERFACE "${Random123_INCLUDE_DIR}"
)
install(
    TARGETS RandLAPACK_Random123 EXPORT RandLAPACK_Random123
)
install(
    EXPORT RandLAPACK_Random123
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES
)