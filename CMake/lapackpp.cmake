message(STATUS "\n\n-- RandLAPACK checking for lapackpp ... ")
find_package(lapackpp REQUIRED)
message(STATUS "RandLAPACK found lapackpp ${lapackpp_VERSION}\n")

# interface library for use elsewhere in the project
add_library(RandLAPACK_lapackpp INTERFACE)

target_link_libraries(RandLAPACK_lapackpp INTERFACE lapackpp)

install(TARGETS RandLAPACK_lapackpp EXPORT RandLAPACK_lapackpp)

install(EXPORT RandLAPACK_lapackpp
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES)
