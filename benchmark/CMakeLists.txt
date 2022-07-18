# TODO stop using gtest
message(STATUS "Checking for benchmarks ...")
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    add_executable(RandLAPACK_benchmark experiments/comps/QB_cond_nums.cc experiments/comps/Orth_speed.cc)
    target_link_libraries(RandLAPACK_benchmark OpenMP::OpenMP_CXX RandLAPACK GTest::GTest GTest::Main)
    gtest_discover_tests(RandLAPACK_benchmark)

endif()
message(STATUS "Checking for benchmarks ... ${tmp}")