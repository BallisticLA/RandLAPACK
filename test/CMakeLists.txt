message(STATUS "Checking for regression tests ...")
find_package(GTest)

set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    set(RandLAPACK_test_srcs
        comps/test_determiter.cc
        comps/test_util.cc
        comps/test_orth.cc
        comps/test_qb.cc
        drivers/test_rsvd.cc
        drivers/test_cqrrpt.cc
        drivers/test_revd2.cc
    )

    add_executable(RandLAPACK_tests ${RandLAPACK_test_srcs})

    target_link_libraries(RandLAPACK_tests RandLAPACK GTest::GTest GTest::Main)

    gtest_discover_tests(RandLAPACK_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")

add_executable(test_pcgls comps/test_pcgls.cpp)
target_link_libraries(test_pcgls RandLAPACK)
add_test(NAME test_pcgls COMMAND test_pcgls 100 15)

