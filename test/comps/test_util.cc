#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>

#include <math.h>
#include <chrono>
#include <gtest/gtest.h>
/*
TODO #1: Resizing tests.

TODO #2: Diagonalization tests.

TODO #4: L & pivotig tests.
*/
using namespace std::chrono;

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestUtil : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T, typename RNG>
    static void 
    test_spectral_norm(int64_t m, int64_t n, std::tuple<int, T, bool> mat_type, RandBLAS::base::RNGState<RNG> state) {
        
        std::vector<T> A(m * n, 0.0);
        std::vector<T> A_cpy(m * n, 0.0);
        std::vector<T> s(n, 0.0);
        RandLAPACK::util::gen_mat_type(m, n, A, n, state, mat_type);

        T norm = RandLAPACK::util::estimate_spectral_norm(m, n, A.data(), 10000, state);

        // Get an SVD -> first singular value == 2_norm
        lapack::lacpy(MatrixType::General, m, n, A.data(), m, A_cpy.data(), m);
        lapack::gesdd(Job::NoVec, m, n, A_cpy.data(), m, s.data(), NULL, m, NULL, n);

        printf("Computed norm:  %e\nComputed s_max: %e\n", norm, s[0]);
        ASSERT_NEAR(norm, s[0], std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_normc(int64_t m) {
        
        std::vector<T> A(m, 0.0);
        std::vector<T> A_norm(m, 0.0);
        int cnt = 0;
        // we have a vector with entries 10:m, first 10 entries = 0
        std::for_each(A.begin() + 10, A.end(), [&cnt](T &entry) { entry = ++cnt;});

        // We expect A_norm to have all 1's
        RandLAPACK::util::normc(1, m, A, A_norm);

        // We expect this to be 1;
        T norm = blas::nrm2(m, A_norm.data(), 1);
        printf("norm is%f\n", norm);
        ASSERT_NEAR(norm, std::sqrt(m - 10), std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_binary_rank_search_zero_mat(int64_t m, int64_t n) {
        std::vector<T> A(m * n, 0.0);
        
        int64_t k = RandLAPACK::util::rank_search_binary(0, m, 0, n, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75), A.data());

        printf("K IS %ld\n", k);
        ASSERT_EQ(k, 0);
    }
};

TEST_F(TestUtil, test_spectral_norm) {
    auto state = RandBLAS::base::RNGState(0, 0);
    test_spectral_norm<double, r123::Philox4x32>(1000, 100, std::make_tuple(0, 2, false), state);
    test_spectral_norm<double, r123::Philox4x32>(1000, 100, std::make_tuple(9, std::pow(10, 15), false), state);
    test_spectral_norm<float, r123::Philox4x32>(1000, 100, std::make_tuple(0, 2, false), state);
    test_spectral_norm<float, r123::Philox4x32>(1000, 100, std::make_tuple(9, std::pow(10, 7), false), state);
}

TEST_F(TestUtil, test_normc) {
    test_normc<double>(1000);
}

TEST_F(TestUtil, test_binary_rank_search_zero_mat) {
    test_binary_rank_search_zero_mat<double>(1000, 100);
}
