message(STATUS "Checking for regression tests ...")
find_package(GTest)
find_package(CUDA QUIET)  # Add this to check for CUDA

set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    set(RandLAPACK_test_srcs
        comps/test_determiter.cc
        comps/test_util.cc
        comps/test_orth.cc
        comps/test_qb.cc
        comps/test_preconditioners.cc
        comps/test_rf.cc
        comps/test_syrf.cc
        drivers/test_rsvd.cc
        drivers/test_cqrrpt.cc
        drivers/test_cqrrp.cc
        drivers/test_revd2.cc
        drivers/test_hqrrp.cc
        drivers/test_rbki.cc
    )

    add_executable(RandLAPACK_tests ${RandLAPACK_test_srcs})
    target_link_libraries(RandLAPACK_tests RandLAPACK GTest::GTest GTest::Main)

    # Conditionally handle CUDA-related tests and benchmarks
    if (CUDA_FOUND)
        message(STATUS "CUDA found! Enabling GPU tests and benchmarks.")

        set(RandLAPACK_test_cu_srcs
            comps/test_util_gpu.cu
            drivers/test_cqrrpt_gpu.cu
            drivers/test_cqrrp_gpu.cu
        )
        set_source_files_properties(${RandLAPACK_test_cu_srcs} PROPERTIES LANGUAGE CUDA)
        
        add_executable(RandLAPACK_tests_gpu ${RandLAPACK_test_cu_srcs})
        target_link_libraries(RandLAPACK_tests_gpu RandLAPACK GTest::GTest GTest::Main CUDA::cusolver)
        set_property(TARGET RandLAPACK_tests_gpu PROPERTY CUDA_ARCHITECTURES native)

        add_executable(RandLAPACK_bench_gpu drivers/bench_cqrrp_gpu.cu)
        target_link_libraries(RandLAPACK_bench_gpu RandLAPACK GTest::GTest GTest::Main CUDA::cusolver)
        set_property(TARGET RandLAPACK_bench_gpu PROPERTY CUDA_ARCHITECTURES native)

        gtest_discover_tests(RandLAPACK_bench_gpu)
        gtest_discover_tests(RandLAPACK_tests_gpu)
    else()
        message(STATUS "CUDA not found! Skipping GPU tests and benchmarks.")
    endif()

    gtest_discover_tests(RandLAPACK_tests)

    add_executable(test_pcgls comps/test_pcgls.cc)
    target_link_libraries(test_pcgls RandLAPACK GTest::GTest)
    add_test(NAME test_pcgls COMMAND test_pcgls 15 100)
endif()

message(STATUS "Checking for regression tests ... ${tmp}")