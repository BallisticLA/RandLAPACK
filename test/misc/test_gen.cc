#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

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
        T min_cond = T(1) / std::sqrt(std::numeric_limits<T>::epsilon());
        for (T cond : {min_cond, T(100) * min_cond, T(10000) * min_cond}) {
            auto s = RandLAPACK::gen::gen_bad_cholqr_singvals<T>(k, (T) 0.1, cond);
            ASSERT_EQ((int64_t) s.size(), k);
            ASSERT_GT(s.back(), T(0));
            for (int64_t i = 1; i < k; ++i)
                ASSERT_LE(s[i], s[i - 1]) << "not monotone at i=" << i << " for cond=" << cond;
        }
    }

    /// s[0] is exactly 1 and s[k-1] is 1/cond, so the realised condition number is the
    /// requested one. This is the assertion that fails outright without the fix.
    template <typename T>
    static void test_bad_cholqr_singvals_realises_cond() {
        int64_t k = 1000;
        T eps = std::numeric_limits<T>::epsilon();
        T min_cond = T(1) / std::sqrt(eps);
        for (T cond : {min_cond, T(100) * min_cond, T(10000) * min_cond}) {
            auto s = RandLAPACK::gen::gen_bad_cholqr_singvals<T>(k, (T) 0.1, cond);
            ASSERT_EQ(s.front(), (T) 1.0);
            T realised = s.front() / s.back();
            // Allow a few ulps for constructing the endpoint and taking its reciprocal.
            ASSERT_NEAR(realised / cond, T(1), T(8) * eps)
                << "requested cond=" << cond << " but realised " << realised;
        }
    }

    /// The block structure is what makes the Gram matrix numerically indefinite, so pin
    /// both the leading count and the size of the cliff between the blocks.
    template <typename T>
    static void test_bad_cholqr_singvals_block_sizes() {
        int64_t k = 1000;
        T frac = (T) 0.1;
        T eps = std::numeric_limits<T>::epsilon();
        T cliff = std::sqrt(eps);
        auto s = RandLAPACK::gen::gen_bad_cholqr_singvals<T>(k, frac, (T) 1e10);
        int64_t offset = (int64_t) std::floor((double) k * (double) frac);

        for (int64_t i = 0; i < offset; ++i)
            ASSERT_EQ(s[i], (T) 1.0) << "leading block not all ones at i=" << i;
        ASSERT_LT(s[offset], (T) 1.0) << "trailing block did not drop";
        ASSERT_NEAR(s[offset] / cliff, T(1), T(4) * eps);
    }

    template <typename T>
    static void test_bad_cholqr_singvals_rejects_cond_below_bound() {
        T min_cond = T(1) / std::sqrt(std::numeric_limits<T>::epsilon());
        T below_bound = std::nextafter(min_cond, T(0));
        EXPECT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<T>(10, T(0.5), below_bound),
                     RandLAPACK::Error);
    }

    /// Degenerate shapes throw rather than returning a silently wrong spectrum, which is
    /// how the original fault went unnoticed.
    template <typename T>
    static void test_bad_cholqr_singvals_rejects_degenerate_shapes() {
        for (int64_t k : {0, 1, 2}) {
            EXPECT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<T>(k, T(0.5), T(1e10)),
                         RandLAPACK::Error);
        }
        // frac too small: no leading block of ones.
        ASSERT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<T>(5, (T) 0.1, (T) 1e10),
                     RandLAPACK::Error);
        // frac too large: fewer than two decaying values.
        ASSERT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<T>(8, T(0.875), T(1e10)),
                     RandLAPACK::Error);
        ASSERT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<T>(10, T(1), T(1e10)),
                     RandLAPACK::Error);
    }

    template <typename T>
    static void test_bad_cholqr_mat_random_vectors_preserve_spectrum() {
        int64_t m = 6, n = 5;
        T eps = std::numeric_limits<T>::epsilon();
        T cliff = std::sqrt(eps);
        RandLAPACK::gen::mat_gen_info<T> info(m, n, RandLAPACK::gen::bad_cholqr);
        info.rank = 3;
        info.frac_spectrum_one = T(0.5);
        info.cond_num = T(16) / cliff;
        std::vector<T> A(m * n, T(-7));
        RandBLAS::RNGState<> state(7);
        auto state_before = state;
        RandLAPACK::gen::mat_gen(info, A.data(), state);
        EXPECT_NE(state.counter, state_before.counter);

        std::vector<T> s(n);
        ASSERT_EQ(lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec,
            m, n, A.data(), m, s.data(), nullptr, 1, nullptr, 1), 0);
        const std::vector<T> expected = {T(1), cliff, cliff / T(16), T(0), T(0)};
        // SVD error is absolute at the scale of the largest singular value (one).
        for (int64_t i = 0; i < n; ++i)
            EXPECT_NEAR(s[i], expected[i], T(32) * eps) << "i=" << i;
    }
};

TEST_F(TestGenSpectra, bad_cholqr_singvals_is_monotone)                  { test_bad_cholqr_singvals_is_monotone<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_is_monotone_float)            { test_bad_cholqr_singvals_is_monotone<float>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_realises_cond)                { test_bad_cholqr_singvals_realises_cond<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_realises_cond_float)          { test_bad_cholqr_singvals_realises_cond<float>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_block_sizes)                  { test_bad_cholqr_singvals_block_sizes<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_block_sizes_float)            { test_bad_cholqr_singvals_block_sizes<float>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_rejects_cond_below_bound)      { test_bad_cholqr_singvals_rejects_cond_below_bound<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_rejects_cond_below_bound_float) { test_bad_cholqr_singvals_rejects_cond_below_bound<float>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_rejects_degenerate_shapes)    { test_bad_cholqr_singvals_rejects_degenerate_shapes<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_singvals_rejects_degenerate_shapes_float) { test_bad_cholqr_singvals_rejects_degenerate_shapes<float>(); }
TEST_F(TestGenSpectra, bad_cholqr_mat_random_vectors_preserve_spectrum) { test_bad_cholqr_mat_random_vectors_preserve_spectrum<double>(); }
TEST_F(TestGenSpectra, bad_cholqr_mat_random_vectors_preserve_spectrum_float) { test_bad_cholqr_mat_random_vectors_preserve_spectrum<float>(); }

TEST_F(TestGenSpectra, bad_cholqr_singvals_rejects_nonfinite_parameters) {
    double inf = std::numeric_limits<double>::infinity();
    double nan = std::numeric_limits<double>::quiet_NaN();
    for (double frac : {inf, -inf, nan}) {
        EXPECT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<double>(10, frac, 1e10),
                     RandLAPACK::Error);
    }
    for (double cond : {inf, nan}) {
        EXPECT_THROW(RandLAPACK::gen::gen_bad_cholqr_singvals<double>(10, 0.1, cond),
                     RandLAPACK::Error);
    }
}

TEST_F(TestGenSpectra, bad_cholqr_mat_gen_uses_requested_spectrum) {
    int64_t m = 6, n = 5;
    double cliff = std::sqrt(std::numeric_limits<double>::epsilon());
    RandLAPACK::gen::mat_gen_info<double> info(m, n, RandLAPACK::gen::bad_cholqr);
    info.rank = n;
    info.frac_spectrum_one = 0.4;
    info.cond_num = 1e10;
    info.diag = true;
    std::vector<double> A(m * n, -7.0);
    RandBLAS::RNGState<> state(7);
    RandLAPACK::gen::mat_gen(info, A.data(), state);
    const std::vector<double> expected = {1.0, 1.0, cliff, std::sqrt(cliff * 1e-10), 1e-10};
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i)
            EXPECT_DOUBLE_EQ(A[i + j * m], i == j ? expected[i] : 0.0);
    }
    EXPECT_DOUBLE_EQ(A[0], 1.0);
    EXPECT_NEAR(A[n - 1 + (n - 1) * m], 1e-10, 1e-25);
}

TEST_F(TestGenSpectra, bad_cholqr_diagonal_rank_deficient_rectangles) {
    double cliff = std::sqrt(std::numeric_limits<double>::epsilon());
    const std::vector<double> expected = {1.0, cliff, cliff / 16.0, 0.0, 0.0};
    for (int64_t m : {5, 6}) {
        int64_t n = 11 - m;
        RandLAPACK::gen::mat_gen_info<double> info(m, n, RandLAPACK::gen::bad_cholqr);
        info.rank = 3;
        info.frac_spectrum_one = 0.5;
        info.cond_num = 16.0 / cliff;
        info.diag = true;
        std::vector<double> A(m * n, -7.0);
        RandBLAS::RNGState<> state(7);
        RandLAPACK::gen::mat_gen(info, A.data(), state);
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < m; ++i)
                EXPECT_DOUBLE_EQ(A[i + j * m], i == j ? expected[i] : 0.0);
        }
    }
}

TEST_F(TestGenSpectra, bad_cholqr_legacy_call_signatures) {
    int64_t m = 24, n = 20;
    auto expected = RandLAPACK::gen::gen_bad_cholqr_singvals<double>(n, 0.1, 1e10);
    EXPECT_EQ(RandLAPACK::gen::gen_bad_cholqr_singvals(n, n, 1e10), expected);
    EXPECT_EQ(RandLAPACK::gen::gen_bad_cholqr_singvals<double>(n, 20, 1e10), expected);
    std::vector<double> A(m * n, -7.0);
    RandBLAS::RNGState<> state(9);
    RandLAPACK::gen::gen_bad_cholqr_mat(m, n, A.data(), n, 1e10, true, state);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i)
            EXPECT_DOUBLE_EQ(A[i + j * m], i == j ? expected[i] : 0.0);
    }
}
