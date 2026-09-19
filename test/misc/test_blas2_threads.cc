// Tests for RandLAPACK::Blas2ThreadGuard and RandLAPACK::SolveWidthScope
// (RandLAPACK/misc/rl_blas2_threads.hh).
//
// Blas2ThreadGuard is exercised through MKL's mkl_get_max_threads() /
// mkl_set_num_threads_local() return-value conventions; it is a no-op when MKL is
// not present, so those cases are compiled out under RandBLAS_HAS_MKL.
//
// SolveWidthScope's save/restore/nesting logic is pure (MKL-free) and is checked
// directly against blas2_thread_cap(). The RANDLAPACK_SOLVE_FFT_MATCH=0 kill switch
// reads its env var into a function-local static on first use in the process, so
// flipping it mid-suite is not reliably observable from a shared gtest binary; per
// the F9 audit convention it is benchmark-validated only, not exercised here.

#include <RandBLAS.hh>
#include "rl_blas2_threads.hh"
#include <gtest/gtest.h>

#if defined(RandBLAS_HAS_MKL)
#include <mkl_service.h>
#endif

TEST(TestSolveWidthScope, default_width_is_zero_outside_any_scope) {
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
}

TEST(TestSolveWidthScope, single_scope_sets_then_restores_width) {
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
    {
        RandLAPACK::SolveWidthScope scope(20000);
        ASSERT_EQ(RandLAPACK::solve_context_width(), RandLAPACK::blas2_thread_cap(20000));
    }
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
}

TEST(TestSolveWidthScope, nested_narrower_inner_restores_outer_on_exit) {
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
    const int64_t n_outer = 20000, n_inner = 100;
    ASSERT_LT(RandLAPACK::blas2_thread_cap(n_inner), RandLAPACK::blas2_thread_cap(n_outer))
        << "test assumes the inner scope is calibrated narrower than the outer one";
    {
        RandLAPACK::SolveWidthScope outer(n_outer);
        ASSERT_EQ(RandLAPACK::solve_context_width(), RandLAPACK::blas2_thread_cap(n_outer));
        {
            RandLAPACK::SolveWidthScope inner(n_inner);
            ASSERT_EQ(RandLAPACK::solve_context_width(), RandLAPACK::blas2_thread_cap(n_inner));
        }
        // Inner destructs: the enclosing (outer) context is restored, not zero.
        ASSERT_EQ(RandLAPACK::solve_context_width(), RandLAPACK::blas2_thread_cap(n_outer));
    }
    ASSERT_EQ(RandLAPACK::solve_context_width(), 0);
}

TEST(TestSolveWidthScope, calibrated_tiers_match_dimension_threshold) {
    ASSERT_EQ(RandLAPACK::blas2_thread_cap(RandLAPACK::kBlas2SmallDim), RandLAPACK::kBlas2ThreadsSmall);
    ASSERT_EQ(RandLAPACK::blas2_thread_cap(RandLAPACK::kBlas2SmallDim + 1), RandLAPACK::kBlas2ThreadsLarge);
}

#if defined(RandBLAS_HAS_MKL)

TEST(TestBlas2ThreadGuard, sets_then_restores_thread_count) {
    int baseline = mkl_get_max_threads();
    {
        RandLAPACK::Blas2ThreadGuard guard(7, 0);
        ASSERT_EQ(mkl_get_max_threads(), 7);
    }
    ASSERT_EQ(mkl_get_max_threads(), baseline);
}

TEST(TestBlas2ThreadGuard, nonpositive_cap_is_a_noop) {
    int baseline = mkl_get_max_threads();
    {
        RandLAPACK::Blas2ThreadGuard guard(0, 0);
        ASSERT_EQ(mkl_get_max_threads(), baseline);
    }
    ASSERT_EQ(mkl_get_max_threads(), baseline);
}

TEST(TestBlas2ThreadGuard, nested_narrower_inner_restores_outer_on_exit) {
    int baseline = mkl_get_max_threads();
    {
        RandLAPACK::Blas2ThreadGuard outer(12, 0);
        ASSERT_EQ(mkl_get_max_threads(), 12);
        {
            RandLAPACK::Blas2ThreadGuard inner(4, 0);
            ASSERT_EQ(mkl_get_max_threads(), 4);
        }
        ASSERT_EQ(mkl_get_max_threads(), 12);
    }
    ASSERT_EQ(mkl_get_max_threads(), baseline);
}

#endif  // RandBLAS_HAS_MKL
