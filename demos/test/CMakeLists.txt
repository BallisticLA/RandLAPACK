message(STATUS "Checking for RandLAPACK demos tests ...")
find_package(GTest)

set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    # Include path to functions/ directory for demos utilities
    include_directories(${CMAKE_SOURCE_DIR}/functions)

    set(RandLAPACK_demos_test_srcs
        drivers/test_dm_cqrrt_linops.cc
        misc/test_dm_cholsolver_linop.cc
        misc/test_linop_dm_composite.cc
    )

    # Create test executable
    add_executable(RandLAPACK_demos_tests ${RandLAPACK_demos_test_srcs})
    target_link_libraries(RandLAPACK_demos_tests
        RandLAPACK::RandLAPACK
        Eigen3::Eigen
        fast_matrix_market
        GTest::GTest
        GTest::Main
    )

    # Enable test discovery
    gtest_discover_tests(RandLAPACK_demos_tests)
else()
    message(WARNING "GTest not found - demos tests will be disabled. Install GTest to enable regression tests.")
endif()

message(STATUS "Checking for RandLAPACK demos tests ... ${tmp}")
