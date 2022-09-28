message(STATUS "Checking for regression tests ...")
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    add_executable(RandLAPACK_tests src/comps/test_determiter.cc src/comps/test_util.cc src/comps/test_orth.cc src/comps/test_qb.cc src/drivers/test_rsvd.cc)
    target_link_libraries(RandLAPACK_tests OpenMP::OpenMP_CXX RandLAPACK GTest::GTest GTest::Main)
    gtest_discover_tests(RandLAPACK_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")
