// Tests for RandLAPACK/misc/rl_thread_caps.hh.
//
// The environment overrides are read once per process into function-local statics, so the
// cached paths cannot be toggled from within one binary; tests that assume the calibrated
// defaults skip when an override is exported.

// The header first, so a lost RandBLAS/config.h include would compile the guard as a no-op
// here and fail the MKL cases below.
#include "RandLAPACK/misc/rl_thread_caps.hh"
#include "RandLAPACK.hh"
#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>

#if defined(RandBLAS_HAS_MKL)
#include <mkl_service.h>
#endif

static bool thread_overrides_exported() {
    return std::getenv("RANDLAPACK_BLAS2_THREADS") != nullptr
        || std::getenv("RANDLAPACK_SOLVE_WIDTH_MATCH") != nullptr;
}

static void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

TEST(TestThreadCapFromEnv, parses_integers_and_rejects_everything_else) {
    const char* name = "RANDLAPACK_TEST_THREAD_CAP_PARSE";
    set_env(name, "");
    EXPECT_FALSE(RandLAPACK::thread_cap_from_env(name).has_value());
    set_env(name, "5");
    EXPECT_EQ(RandLAPACK::thread_cap_from_env(name), 5);
    set_env(name, "0");
    EXPECT_EQ(RandLAPACK::thread_cap_from_env(name), 0);
    set_env(name, "-1");
    EXPECT_EQ(RandLAPACK::thread_cap_from_env(name), -1);
    set_env(name, "8x");
    EXPECT_FALSE(RandLAPACK::thread_cap_from_env(name).has_value());
    set_env(name, "auto");
    EXPECT_FALSE(RandLAPACK::thread_cap_from_env(name).has_value());
    set_env(name, "99999999999999999999");
    EXPECT_FALSE(RandLAPACK::thread_cap_from_env(name).has_value());
    EXPECT_FALSE(RandLAPACK::thread_cap_from_env("RANDLAPACK_TEST_THREAD_CAP_NEVER_SET").has_value());
}

TEST(TestBlas2ThreadCap, calibrated_tiers_match_dimension_threshold) {
    if (thread_overrides_exported()) GTEST_SKIP() << "a thread-cap override is exported";
    ASSERT_EQ(RandLAPACK::blas2_thread_cap(RandLAPACK::kBlas2SmallDim), RandLAPACK::kBlas2ThreadsSmall);
    ASSERT_EQ(RandLAPACK::blas2_thread_cap(RandLAPACK::kBlas2SmallDim + 1), RandLAPACK::kBlas2ThreadsLarge);
}

TEST(TestSolveWidthScope, single_scope_sets_then_restores_width) {
    if (thread_overrides_exported()) GTEST_SKIP() << "a thread-cap override is exported";
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
    {
        RandLAPACK::SolveWidthScope scope(20000);
        ASSERT_EQ(RandLAPACK::solve_context_width(), RandLAPACK::blas2_thread_cap(20000));
    }
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
}

TEST(TestSolveWidthScope, nested_narrower_inner_restores_outer_on_exit) {
    if (thread_overrides_exported()) GTEST_SKIP() << "a thread-cap override is exported";
    const int64_t n_outer = 20000, n_inner = 100;
    ASSERT_LT(RandLAPACK::blas2_thread_cap(n_inner), RandLAPACK::blas2_thread_cap(n_outer));
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
    {
        RandLAPACK::SolveWidthScope outer(n_outer);
        ASSERT_EQ(RandLAPACK::solve_context_width(), RandLAPACK::blas2_thread_cap(n_outer));
        {
            RandLAPACK::SolveWidthScope inner(n_inner);
            ASSERT_EQ(RandLAPACK::solve_context_width(), RandLAPACK::blas2_thread_cap(n_inner));
        }
        ASSERT_EQ(RandLAPACK::solve_context_width(), RandLAPACK::blas2_thread_cap(n_outer));
    }
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
}

#if defined(RandBLAS_HAS_MKL)

// Caps sit below the caller's width (cases that need more threads skip), so each narrowing
// case narrows on any runner. After each scope the thread must be back to following the global
// setting: mkl_set_num_threads_local(0) returns the previous local value, which must be 0.

TEST(TestThreadCapGuard, narrows_then_restores_follow_global) {
    const int baseline = mkl_get_max_threads();
    if (baseline < 2) GTEST_SKIP() << "needs at least 2 threads";
    {
        RandLAPACK::ThreadCapGuard guard(baseline - 1);
        ASSERT_EQ(mkl_get_max_threads(), baseline - 1);
    }
    ASSERT_EQ(mkl_get_max_threads(), baseline);
    ASSERT_EQ(mkl_set_num_threads_local(0), 0);
}

TEST(TestThreadCapGuard, never_widens_beyond_the_callers_width) {
    const int baseline = mkl_get_max_threads();
    {
        RandLAPACK::ThreadCapGuard guard(baseline + 5);
        ASSERT_EQ(mkl_get_max_threads(), baseline);
    }
    ASSERT_EQ(mkl_get_max_threads(), baseline);
    ASSERT_EQ(mkl_set_num_threads_local(0), 0);
}

TEST(TestThreadCapGuard, nonpositive_cap_is_a_noop) {
    const int baseline = mkl_get_max_threads();
    {
        RandLAPACK::ThreadCapGuard guard(0);
        ASSERT_EQ(mkl_get_max_threads(), baseline);
    }
    ASSERT_EQ(mkl_set_num_threads_local(0), 0);
}

TEST(TestThreadCapGuard, nested_narrower_inner_restores_outer_on_exit) {
    const int baseline = mkl_get_max_threads();
    if (baseline < 3) GTEST_SKIP() << "needs at least 3 threads";
    {
        RandLAPACK::ThreadCapGuard outer(2);
        ASSERT_EQ(mkl_get_max_threads(), 2);
        {
            RandLAPACK::ThreadCapGuard inner(1);
            ASSERT_EQ(mkl_get_max_threads(), 1);
        }
        ASSERT_EQ(mkl_get_max_threads(), 2);
    }
    ASSERT_EQ(mkl_get_max_threads(), baseline);
    ASSERT_EQ(mkl_set_num_threads_local(0), 0);
}

TEST(TestThreadCapGuard, restores_during_exception_unwinding) {
    const int baseline = mkl_get_max_threads();
    if (baseline < 2) GTEST_SKIP() << "needs at least 2 threads";
    try {
        RandLAPACK::ThreadCapGuard guard(1);
        ASSERT_EQ(mkl_get_max_threads(), 1);
        throw std::runtime_error("unwind");
    } catch (const std::runtime_error&) {}
    ASSERT_EQ(mkl_get_max_threads(), baseline);
    ASSERT_EQ(mkl_set_num_threads_local(0), 0);
}

TEST(TestThreadCapGuard, blas2_guard_applies_the_calibrated_cap) {
    if (thread_overrides_exported()) GTEST_SKIP() << "a thread-cap override is exported";
    const int baseline = mkl_get_max_threads();
    if (baseline <= RandLAPACK::kBlas2ThreadsSmall) GTEST_SKIP() << "needs more threads than the cap";
    {
        RandLAPACK::Blas2ThreadGuard guard(RandLAPACK::kBlas2SmallDim);
        ASSERT_EQ(mkl_get_max_threads(), RandLAPACK::kBlas2ThreadsSmall);
    }
    ASSERT_EQ(mkl_get_max_threads(), baseline);
}

#endif  // RandBLAS_HAS_MKL
