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
        drivers/test_krill.cc
        drivers/test_rsvd.cc
        drivers/test_cqrrpt.cc
        drivers/test_cqrrt.cc
        drivers/test_bqrrp.cc
        drivers/test_revd2.cc
        drivers/test_hqrrp.cc
        drivers/test_abrik.cc
        drivers/test_orth_linop.cc
        drivers/test_cholqr_dense.cc
        drivers/test_iter_refine_lsq.cc
        drivers/test_lsqr_blendenpik.cc
        drivers/test_refined_blendenpik.cc
        misc/test_util.cc
        misc/test_pdkernels.cc
        misc/test_sparse_axpby_shared_pattern.cc
        misc/test_blas2_threads.cc
        linops/test_linops.cc
        linops/test_power_linop.cc
        linops/test_transposed_linop.cc
        linops/test_vstack_linop.cc
        linops/test_linop_unified.cc
        misc/test_gen.cc
        misc/test_memory_tracker.cc
        linops/test_linop_block_views.cc
    )

    # ToeplitzLinOp lives under extras/ (it depends on MKL's DFTI FFT interface,
    # which core RandLAPACK does not require); only register its test when MKL is
    # actually available AND mkl_dfti.h can be located. Do the header search before
    # building the source list so a missing header skips the test file instead of
    # landing in a target that then fails to compile.
    set(_toeplitz_test_dfti_found FALSE)
    if (RandBLAS_HAS_MKL)
        # Locate mkl_dfti.h for test_toeplitz_linop.cc. MKLROOT is the usual signal;
        # also probe the other oneAPI env vars and common install roots (mirrors
        # benchmark/CMakeLists.txt's toeplitz_ls_benchmark target).
        find_path(MKL_DFTI_INCLUDE_DIR mkl_dfti.h
            HINTS
                $ENV{MKLROOT}/include
                $ENV{MKL_ROOT}/include
                $ENV{ONEAPI_ROOT}/mkl/latest/include
                $ENV{CMPLR_ROOT}/../mkl/latest/include
                /opt/intel/oneapi/mkl/latest/include
                /opt/intel/oneapi/mkl/2026.0/include
            PATH_SUFFIXES include)
        if (MKL_DFTI_INCLUDE_DIR)
            set(_toeplitz_test_dfti_found TRUE)
            message(STATUS "test_toeplitz_linop: mkl_dfti.h in ${MKL_DFTI_INCLUDE_DIR}")
            list(APPEND RandLAPACK_test_srcs linops/test_toeplitz_linop.cc)
        else()
            message(WARNING "mkl_dfti.h not found; skipping test_toeplitz_linop.cc. Pass "
                            "-DMKL_DFTI_INCLUDE_DIR=<MKL>/include or export MKLROOT to "
                            "build it.")
        endif()
    endif()

    # Create non-CUDA test executable
    add_executable(RandLAPACK_tests ${RandLAPACK_test_srcs})
    target_link_libraries(RandLAPACK_tests RandLAPACK GTest::GTest GTest::Main)

    # extras/ headers (e.g. ext_toeplitz_linop.hh, ext_sparse_axpy.hh) are addressed
    # relative to the repo root by the tests that exercise them.
    target_include_directories(RandLAPACK_tests PRIVATE ${PROJECT_SOURCE_DIR})

    if (_toeplitz_test_dfti_found)
        # Derive MKL_ILP64 from the BLAS interface blaspp was actually built
        # against, instead of assuming ILP64 (an LP64 MKL build would otherwise
        # silently mismatch the DFTI header's ABI). blaspp's installed config
        # records this in blas/defines.h (BLAS_ILP64 defined iff the ILP64
        # interface was linked); blaspp_DIR locates that install tree. Fall back
        # to matching ilp64/lp64 in the resolved library names if that header
        # cannot be found.
        set(_toeplitz_test_ilp64 "")
        if (blaspp_DIR)
            get_filename_component(_blaspp_prefix "${blaspp_DIR}/../../.." ABSOLUTE)
            set(_blaspp_defines_h "${_blaspp_prefix}/include/blas/defines.h")
            if (EXISTS "${_blaspp_defines_h}")
                file(READ "${_blaspp_defines_h}" _blaspp_defines_contents)
                if (_blaspp_defines_contents MATCHES "#define[ \t]+BLAS_ILP64")
                    set(_toeplitz_test_ilp64 TRUE)
                else()
                    set(_toeplitz_test_ilp64 FALSE)
                endif()
                message(STATUS "test_toeplitz_linop: BLAS interface from "
                                "${_blaspp_defines_h}: ILP64=${_toeplitz_test_ilp64}")
            endif()
        endif()
        if (_toeplitz_test_ilp64 STREQUAL "")
            foreach(_lib ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${blaspp_libraries})
                if (_lib MATCHES "ilp64")
                    set(_toeplitz_test_ilp64 TRUE)
                elseif(_lib MATCHES "lp64")
                    set(_toeplitz_test_ilp64 FALSE)
                endif()
            endforeach()
            if (NOT _toeplitz_test_ilp64 STREQUAL "")
                message(STATUS "test_toeplitz_linop: BLAS interface from linked "
                                "library names: ILP64=${_toeplitz_test_ilp64}")
            endif()
        endif()
        if (_toeplitz_test_ilp64 STREQUAL "")
            message(WARNING "test_toeplitz_linop: could not determine whether the "
                            "resolved BLAS/MKL interface is ILP64 or LP64; "
                            "defaulting to MKL_ILP64.")
            set(_toeplitz_test_ilp64 TRUE)
        endif()
        if (_toeplitz_test_ilp64)
            set_source_files_properties(linops/test_toeplitz_linop.cc PROPERTIES
                INCLUDE_DIRECTORIES "${MKL_DFTI_INCLUDE_DIR}"
                COMPILE_DEFINITIONS "MKL_ILP64")
        else()
            set_source_files_properties(linops/test_toeplitz_linop.cc PROPERTIES
                INCLUDE_DIRECTORIES "${MKL_DFTI_INCLUDE_DIR}")
        endif()
    endif()

    gtest_discover_tests(RandLAPACK_tests)

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

        # Discover CUDA tests
        gtest_discover_tests(RandLAPACK_tests_gpu)
    endif()

    # Create additional tests
    add_executable(test_pcgls comps/test_pcgls.cc)
    target_link_libraries(test_pcgls RandLAPACK GTest::GTest)
    add_test(NAME test_pcgls COMMAND test_pcgls 15 100)
else()
    message(WARNING "GTest not found - tests will be disabled. Install GTest to enable regression tests.")
endif()

message(STATUS "Checking for regression tests ... ${tmp}")