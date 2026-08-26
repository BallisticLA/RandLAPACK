#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>

class TestGeneratorsMutateState : public ::testing::Test
{
    protected:

    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Verify that a generator function advances the RNG state.
    /// If the state is unchanged after calling the generator, then
    /// successive calls would produce identical output — a silent bug.

    template <typename T>
    static void test_gen_sparse_cond_coo_mutates_state() {
        int64_t m = 100, n = 10;
        T cond_num = 1e4;
        T target_density = 0.5;
        RandBLAS::RNGState<> state(42);
        auto state_before = state;

        auto coo = RandLAPACK::gen::gen_sparse_cond_coo<T>(
            m, n, cond_num, state, target_density
        );

        ASSERT_NE(state.counter, state_before.counter)
            << "gen_sparse_cond_coo must advance the RNG state";
    }

    template <typename T>
    static void test_gen_sparse_cond_coo_no_density_mutates_state() {
        // target_density = 0 skips Givens rotations; state should still advance
        int64_t m = 100, n = 10;
        T cond_num = 1e4;
        RandBLAS::RNGState<> state(42);
        auto state_before = state;

        auto coo = RandLAPACK::gen::gen_sparse_cond_coo<T>(
            m, n, cond_num, state
        );

        ASSERT_NE(state.counter, state_before.counter)
            << "gen_sparse_cond_coo (density=0) must advance the RNG state";
    }

    template <typename T>
    static void test_gen_random_dense_mutates_state() {
        int64_t m = 50, n = 10;
        std::vector<T> A(m * n);
        RandBLAS::RNGState<> state(42);
        auto state_before = state;

        RandLAPACK::gen::gen_random_dense<T>(
            m, n, A.data(), blas::Layout::ColMajor, state
        );

        ASSERT_NE(state.counter, state_before.counter)
            << "gen_random_dense must advance the RNG state";
    }

    template <typename T>
    static void test_gen_sparse_coo_mutates_state() {
        int64_t m = 50, n = 10;
        T density = 0.3;
        RandBLAS::RNGState<> state(42);
        auto state_before = state;

        auto coo = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state);

        ASSERT_NE(state.counter, state_before.counter)
            << "gen_sparse_coo must advance the RNG state";
    }

    template <typename T>
    static void test_gen_sparse_from_singvals_mutates_state() {
        int64_t m = 100, n = 10;
        std::vector<T> sigma(n);
        for (int64_t i = 0; i < n; ++i) sigma[i] = (T)(n - i);
        RandBLAS::RNGState<> state(42);
        auto state_before = state;

        auto coo = RandLAPACK::gen::gen_sparse_from_singvals<T>(
            m, n, sigma.data(), state, (T)0.5
        );

        ASSERT_NE(state.counter, state_before.counter)
            << "gen_sparse_from_singvals must advance the RNG state";
    }

    template <typename T>
    static void test_gen_spd_from_eigvals_mutates_state() {
        int64_t n = 20;
        std::vector<T> eigvals(n);
        for (int64_t i = 0; i < n; ++i) eigvals[i] = (T)(i + 1);
        std::vector<T> A(n * n);
        RandBLAS::RNGState<> state(42);
        auto state_before = state;

        RandLAPACK::gen::gen_spd_from_eigvals<T>(n, eigvals.data(), A.data(), state);

        ASSERT_NE(state.counter, state_before.counter)
            << "gen_spd_from_eigvals must advance the RNG state";
    }

    template <typename T>
    static void test_gen_spd_mat_mutates_state() {
        int64_t n = 20;
        T cond_num = 1e3;
        std::vector<T> A(n * n);
        RandBLAS::RNGState<> state(42);
        auto state_before = state;

        RandLAPACK::gen::gen_spd_mat<T>(n, cond_num, A.data(), state);

        ASSERT_NE(state.counter, state_before.counter)
            << "gen_spd_mat must advance the RNG state";
    }
};

TEST_F(TestGeneratorsMutateState, sparse_cond_coo_mutates_state)           { test_gen_sparse_cond_coo_mutates_state<double>(); }
TEST_F(TestGeneratorsMutateState, sparse_cond_coo_no_density_mutates_state) { test_gen_sparse_cond_coo_no_density_mutates_state<double>(); }
TEST_F(TestGeneratorsMutateState, sparse_from_singvals_mutates_state)      { test_gen_sparse_from_singvals_mutates_state<double>(); }
TEST_F(TestGeneratorsMutateState, random_dense_mutates_state)              { test_gen_random_dense_mutates_state<double>(); }
TEST_F(TestGeneratorsMutateState, sparse_coo_mutates_state)                { test_gen_sparse_coo_mutates_state<double>(); }
TEST_F(TestGeneratorsMutateState, spd_from_eigvals_mutates_state)          { test_gen_spd_from_eigvals_mutates_state<double>(); }
TEST_F(TestGeneratorsMutateState, spd_mat_mutates_state)                   { test_gen_spd_mat_mutates_state<double>(); }


/// Spectrum-level coverage for gen_bad_cholqr_singvals, which until 2026 returned all ones
/// for every requested condition number and had no test at all. Its only caller is
/// gen_bad_cholqr_mat, so the fault was invisible: the matrix documented as "supposed to
/// make QB fail with CholQR" was the most benign input possible.
///
/// bad_cholqr_singvals_realises_cond is the load-bearing one. Before the fix s.back() was
/// 1.0 regardless of cond, so the realised condition number was 1.
class TestGenSpectra : public ::testing::Test
{
    protected:
        virtual void SetUp() {};
        virtual void TearDown() {};

    /// The spectrum must be non-increasing across the whole vector, not just within blocks.
    template <typename T>
    static void test_bad_cholqr_singvals_is_monotone() {
        int64_t k = 1000;
        for (T cond : {(T) 1e8, (T) 1e10, (T) 1e12}) {
            auto s = RandLAPACK::gen::gen_bad_cholqr_singvals<T>(k, (T) 0.1, cond);
            ASSERT_EQ((int64_t) s.size(), k);
            for (int64_t i = 1; i < k; ++i)
                ASSERT_LE(s[i], s[i - 1]) << "not monotone at i=" << i << " for cond=" << cond;
        }
    }

    /// s[0] is exactly 1 and s[k-1] is 1/cond, so the realised condition number is the
    /// requested one. This is the assertion that fails outright without the fix.
    template <typename T>
    static void test_bad_cholqr_singvals_realises_cond() {
        int64_t k = 1000;
        for (T cond : {(T) 1e8, (T) 1e10, (T) 1e12}) {
            auto s = RandLAPACK::gen::gen_bad_cholqr_singvals<T>(k, (T) 0.1, cond);
            ASSERT_EQ(s.front(), (T) 1.0);
            T realised = s.front() / s.back();
            // A few ulps of slack: the trailing endpoint comes out of std::pow.
            ASSERT_NEAR(realised / cond, (T) 1.0, (T) 1e-12)
                << "requested cond=" << cond << " but realised " << realised;
        }
    }

    /// The block structure is what makes the Gram matrix numerically indefinite, so pin
    /// both the leading count and the size of the cliff between the blocks.
    template <typename T>
    static void test_bad_cholqr_singvals_block_sizes() {
        int64_t k = 1000;
        T frac = (T) 0.1;
        auto s = RandLAPACK::gen::gen_bad_cholqr_singvals<T>(k, frac, (T) 1e10);
        int64_t offset = (int64_t) std::floor((double) k * (double) frac);

        for (int64_t i = 0; i < offset; ++i)
            ASSERT_EQ(s[i], (T) 1.0) << "leading block not all ones at i=" << i;
        ASSERT_LT(s[offset], (T) 1.0) << "trailing block did not drop";
        // The cliff is 1 -> 1e-8 by construction.
        ASSERT_NEAR(s[offset - 1] / s[offset], (T) 1e8, (T) 1e8 * (T) 1e-12);
    }

    /// Degenerate shapes throw rather than returning a silently wrong spectrum, which is
    /// how the original fault went unnoticed.
    template <typename T>
    static void test_bad_cholqr_singvals_rejects_degenerate_shapes() {
        // cond below 1e8: the trailing block would rise rather than decay.
        ASSERT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<T>(1000, (T) 0.1, (T) 1e4),
                     RandLAPACK::Error);
        // frac too small: no leading block of ones.
        ASSERT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<T>(5, (T) 0.1, (T) 1e10),
                     RandLAPACK::Error);
        // frac too large: fewer than two decaying values.
        ASSERT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<T>(10, (T) 1.0, (T) 1e10),
                     RandLAPACK::Error);
    }
};

TEST_F(TestGenSpectra, bad_cholqr_singvals_is_monotone)                  { test_bad_cholqr_singvals_is_monotone<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_realises_cond)                { test_bad_cholqr_singvals_realises_cond<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_block_sizes)                  { test_bad_cholqr_singvals_block_sizes<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_rejects_degenerate_shapes)    { test_bad_cholqr_singvals_rejects_degenerate_shapes<double>(); }
