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
        comps/test_cholqr.cc
        comps/test_syrf.cc
        comps/test_rpchol.cc
        comps/test_bk.cc
        comps/test_restarted_pcg_ne.cc
        drivers/test_krill.cc
        drivers/test_rsvd.cc
        drivers/test_rsvd_linop.cc
        drivers/test_cqrrpt.cc
        drivers/test_cqrrt.cc
        drivers/test_bqrrp.cc
        drivers/test_revd2.cc
        drivers/test_hqrrp.cc
        drivers/test_abrik.cc
        drivers/test_abrik_projector.cc
        drivers/test_orth_linop.cc
        drivers/test_lsqr_blendenpik.cc
        drivers/test_iter_refine_lsq.cc
        drivers/test_refined_blendenpik.cc
        misc/test_config.cc
        misc/test_util.cc
        misc/test_pdkernels.cc
        misc/test_thread_caps.cc
        linops/test_svd_residual.cc
        linops/test_linops.cc
        linops/test_downdatable_linop.cc
        linops/test_power_linop.cc
        linops/test_transposed_linop.cc
        linops/test_vstack_linop.cc
        linops/test_linop_unified.cc
        misc/test_gen.cc
        misc/test_matrix_io.cc
        misc/test_memory_tracker.cc
        misc/test_bench_common.cc
        linops/test_linop_block_views.cc
    )
    
    # Benchmark helpers tested here live in benchmark/, outside the library.
    set_source_files_properties(drivers/test_refined_blendenpik.cc misc/test_bench_common.cc
        PROPERTIES INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/benchmark/bench_CQRRTO_linops")

    # Create non-CUDA test executable
    add_executable(RandLAPACK_tests ${RandLAPACK_test_srcs})
    target_link_libraries(RandLAPACK_tests RandLAPACK GTest::GTest GTest::Main)
    # The template-heavy linop test translation units exceed MSVC's default
    # object-file section limit.
    target_compile_options(RandLAPACK_tests PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>)
    # Stage dependency DLLs before gtest_discover_tests runs the binary at build time.
    randlapack_stage_runtime_dlls(RandLAPACK_tests)
    gtest_discover_tests(RandLAPACK_tests)
    # Bound BK/ABRIK non-termination regressions after test discovery, including
    # parameterized tests, without changing the other suites' timeout behavior.
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/bk_abrik_timeout.cmake")

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

        # Discover CUDA tests. Every GPU entry holds the lock "gpu", so under ctest -j no two run at
        # once: the leak checks read device-wide free memory, which another GPU test would change.
        gtest_discover_tests(RandLAPACK_tests_gpu PROPERTIES RESOURCE_LOCK gpu)
        # The CQRRPT_GPU timing test (which also runs alone) and the CQRRPT_GPU and BQRRP_GPU leak checks assume no
        # other process loads the host or uses the GPU. They carry the label exclusive: skip them with ctest -LE exclusive.
        set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/gpu_exclusive_tests.cmake")
        # Discovered tests run one per process. This entry runs them in one process, shuffled and twice, so a test
        # that leaves CUDA state behind fails here. It runs alone, as it would double their memory peak, and leaves
        # out the exclusive tests, so that ctest -LE exclusive skips them everywhere.
        set(gpu_exclusive_tests "TestCQRRPT.CQRRPT_GPU_timing_covers_device_work:TestCQRRPT.CQRRPT_GPU_repeated_calls_no_leak:TestBQRRP.BQRRP_GPU_repeated_calls_no_leak:TestBQRRP.BQRRP_GPU_empty_input")
        add_test(NAME RandLAPACK_tests_gpu_shuffle_repeat
                 COMMAND RandLAPACK_tests_gpu --gtest_shuffle --gtest_repeat=2 --gtest_filter=-${gpu_exclusive_tests})
        set_tests_properties(RandLAPACK_tests_gpu_shuffle_repeat PROPERTIES RUN_SERIAL TRUE RESOURCE_LOCK gpu)

        # A program built with per-thread default streams. Its own executable: the drivers are
        # header templates, so one shared with legacy-stream sources could link either version.
        set_source_files_properties(drivers/test_cqrrpt_gpu_ptds.cu PROPERTIES LANGUAGE CUDA)
        add_executable(RandLAPACK_tests_gpu_ptds drivers/test_cqrrpt_gpu_ptds.cu)
        target_compile_options(RandLAPACK_tests_gpu_ptds PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        target_link_libraries(RandLAPACK_tests_gpu_ptds RandLAPACK GTest::GTest GTest::Main CUDA::cusolver)
        randlapack_stage_runtime_dlls(RandLAPACK_tests_gpu_ptds)
        gtest_discover_tests(RandLAPACK_tests_gpu_ptds PROPERTIES RESOURCE_LOCK gpu)
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
