
configure_file(CMake/RandLAPACKConfig.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK/RandLAPACKConfig.cmake @ONLY)

configure_file(CMake/RandLAPACKConfigVersion.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK/RandLAPACKConfigVersion.cmake @ONLY)

install(FILES
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK/RandLAPACKConfig.cmake
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK/RandLAPACKConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RandLAPACK)
