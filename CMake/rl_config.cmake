
# Values substituted into an installed CMake package file must use CMake path
# syntax. Native Windows backslashes would otherwise be parsed as escape
# sequences when a downstream project loads RandLAPACKConfig.cmake.
file(TO_CMAKE_PATH "${RandLAPACK_lapackpp_DIR}" RandLAPACK_lapackpp_DIR)

configure_file(CMake/RandLAPACKConfig.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK/RandLAPACKConfig.cmake @ONLY)

configure_file(CMake/RandLAPACKConfigVersion.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK/RandLAPACKConfigVersion.cmake @ONLY)

install(FILES
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK/RandLAPACKConfig.cmake
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK/RandLAPACKConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK)
