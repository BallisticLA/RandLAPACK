message(STATUS "Checking for RandLAPACK extras tests ...")
find_package(GTest)

set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    # Include path to extras subdirectories for utilities
    include_directories(${CMAKE_SOURCE_DIR})

    # RandBLAS test utilities (comparison.hh) are not installed;
    # resolve them from the source tree via the RandBLAS submodule.
    set(RANDBLAS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../RandBLAS")
    if(EXISTS "${RANDBLAS_SOURCE_DIR}/test/comparison.hh")
        include_directories(${RANDBLAS_SOURCE_DIR})
        message(STATUS "RandBLAS source tree found at: ${RANDBLAS_SOURCE_DIR}")
    else()
        message(WARNING "RandBLAS source tree not found - test comparison utilities unavailable")
    endif()

    set(RandLAPACK_extras_test_srcs
        linops/test_ext_solver_linop_unified.cc
        linops/test_ext_composite_linop.cc
    )

    # Create test executable
    add_executable(RandLAPACK_extras_tests ${RandLAPACK_extras_test_srcs})
    target_link_libraries(RandLAPACK_extras_tests
        RandLAPACK::RandLAPACK
        Eigen3::Eigen
        fast_matrix_market
        GTest::GTest
        GTest::Main
    )

    # Enable test discovery
    gtest_discover_tests(RandLAPACK_extras_tests)
else()
    message(WARNING "GTest not found - extras tests will be disabled. Install GTest to enable regression tests.")
endif()

message(STATUS "Checking for RandLAPACK extras tests ... ${tmp}")
