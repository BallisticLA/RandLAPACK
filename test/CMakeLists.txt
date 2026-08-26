message(STATUS "Checking for regression tests ...")
find_package(GTest)

set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    set(RandLAPACK_test_srcs
        comps/test_determiter.cc
        comps/test_orth.cc
        comps/test_qb.cc
        comps/test_preconditioners.cc
        comps/test_rf.cc
        comps/test_syrf.cc
        comps/test_rpchol.cc
        comps/test_bk.cc
        drivers/test_krill.cc
        drivers/test_rsvd.cc
        drivers/test_cqrrpt.cc
        drivers/test_cqrrt.cc
        drivers/test_bqrrp.cc
        drivers/test_revd2.cc
        drivers/test_hqrrp.cc
        drivers/test_abrik.cc
        drivers/test_orth_linop.cc
        misc/test_util.cc
        misc/test_pdkernels.cc
        linops/test_linops.cc
        linops/test_linop_unified.cc
        misc/test_gen.cc
        misc/test_memory_tracker.cc
        linops/test_linop_block_views.cc
    )
    
    # Create non-CUDA test executable
    add_executable(RandLAPACK_tests ${RandLAPACK_test_srcs})
    target_link_libraries(RandLAPACK_tests RandLAPACK GTest::GTest GTest::Main)
    # The template-heavy linop test translation units exceed MSVC's default
    # object-file section limit.
    target_compile_options(RandLAPACK_tests PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>)
    # Stage dependency DLLs before gtest_discover_tests runs the binary at build time.
    randlapack_stage_runtime_dlls(RandLAPACK_tests)
    # An explicit timeout is the only mechanism that converts a non-termination regression
    # into a test FAILURE rather than a hung CI job: gtest has no per-test timeout, and the
    # CTest default is 1500s. BK's default budget is INT_MAX and its termination now rests
    # on norm_converged plus the saturation guard, so this is a real risk, not a formality.
    gtest_discover_tests(RandLAPACK_tests PROPERTIES TIMEOUT 300)

    # Check if CUDA is required and setup CUDA-specific tests
    if (RequireCUDA AND CUDAToolkit_FOUND)
        set(RandLAPACK_test_cu_srcs
            comps/test_util_gpu.cu
            drivers/test_cqrrpt_gpu.cu
            drivers/test_bqrrp_gpu.cu
        )

        # Create CUDA test executables
        set_source_files_properties(${RandLAPACK_test_cu_srcs} PROPERTIES LANGUAGE CUDA)
        add_executable(RandLAPACK_tests_gpu ${RandLAPACK_test_cu_srcs})
        target_link_libraries(RandLAPACK_tests_gpu RandLAPACK GTest::GTest GTest::Main CUDA::cusolver)
        randlapack_stage_runtime_dlls(RandLAPACK_tests_gpu)

        # Discover CUDA tests
        gtest_discover_tests(RandLAPACK_tests_gpu)
    endif()

    # Create additional tests
    add_executable(test_pcgls comps/test_pcgls.cc)
    target_link_libraries(test_pcgls RandLAPACK GTest::GTest)
    randlapack_stage_runtime_dlls(test_pcgls)
    add_test(NAME test_pcgls COMMAND test_pcgls 15 100)
else()
    message(WARNING "GTest not found - tests will be disabled. Install GTest to enable regression tests.")
endif()

message(STATUS "Checking for regression tests ... ${tmp}")