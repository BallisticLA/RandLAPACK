#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>

class TestGen : public ::testing::Test
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

TEST_F(TestGen, sparse_cond_coo_mutates_state)           { test_gen_sparse_cond_coo_mutates_state<double>(); }
TEST_F(TestGen, sparse_cond_coo_no_density_mutates_state) { test_gen_sparse_cond_coo_no_density_mutates_state<double>(); }
TEST_F(TestGen, sparse_from_singvals_mutates_state)      { test_gen_sparse_from_singvals_mutates_state<double>(); }
TEST_F(TestGen, random_dense_mutates_state)              { test_gen_random_dense_mutates_state<double>(); }
TEST_F(TestGen, sparse_coo_mutates_state)                { test_gen_sparse_coo_mutates_state<double>(); }
TEST_F(TestGen, spd_from_eigvals_mutates_state)          { test_gen_spd_from_eigvals_mutates_state<double>(); }
TEST_F(TestGen, spd_mat_mutates_state)                   { test_gen_spd_mat_mutates_state<double>(); }
