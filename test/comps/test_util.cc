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

    template <typename T>
    struct UtilTestData {
        std::vector<T> A;
        std::vector<T> A_cpy;
        std::vector<T> s;
        std::vector<T> A_norm;

        UtilTestData(int64_t m, int64_t n) :
        A(m * n, 0.0), 
        A_cpy(m * n, 0.0), 
        s(n, 0.0),
        A_norm(m * n, 0.0) 
        {}
    };

    template <typename T, typename RNG>
    static void computational_helper(int64_t m, int64_t n, UtilTestData<T>& all_data) {
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    }

    template <typename T, typename RNG>
    static void 
    test_spectral_norm(int64_t m, int64_t n, RandBLAS::base::RNGState<RNG> state, UtilTestData<T>& all_data) {

        T norm = RandLAPACK::util::estimate_spectral_norm(m, n, all_data.A.data(), 10000, state);
        // Get an SVD -> first singular value == 2_norm
        lapack::gesdd(Job::NoVec, m, n, all_data.A_cpy.data(), m, all_data.s.data(), NULL, m, NULL, n);

        printf("Computed norm:  %e\nComputed s_max: %e\n", norm, all_data.s[0]);
        ASSERT_NEAR(norm, all_data.s[0], std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_normc(int64_t m, UtilTestData<T>& all_data) {
        
        int cnt = 0;
        // we have a vector with entries 10:m, first 10 entries = 0
        std::for_each(all_data.A.begin() + 10, all_data.A.end(), [&cnt](T &entry) { entry = ++cnt;});

        // We expect A_norm to have all 1's
        RandLAPACK::util::normc(1, m, all_data.A, all_data.A_norm);

        // We expect this to be 1;
        T norm = blas::nrm2(m, all_data.A_norm.data(), 1);
        printf("norm is%f\n", norm);
        ASSERT_NEAR(norm, std::sqrt(m - 10), std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_binary_rank_search_zero_mat(int64_t m, int64_t n, UtilTestData<T>& all_data) {
        
        int64_t k = RandLAPACK::util::rank_search_binary(0, m, 0, n, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75), all_data.A.data());

        printf("K IS %ld\n", k);
        ASSERT_EQ(k, 0);
    }
};

TEST_F(TestUtil, test_spectral_norm_polynomial_decay_double_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::base::RNGState();
    UtilTestData<double> all_data(m, n);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, n, state, std::make_tuple(0, 2025, false));
    computational_helper<double, r123::Philox4x32>(m, n, all_data);
    test_spectral_norm<double, r123::Philox4x32>(m, n, state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_rank_def_mat_double_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::base::RNGState();
    UtilTestData<double> all_data(m, n);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, n, state, std::make_tuple(9, std::pow(10, 15), false));
    computational_helper<double, r123::Philox4x32>(m, n, all_data);
    test_spectral_norm<double, r123::Philox4x32>(m, n, state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_polynomial_decay_single_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::base::RNGState();
    UtilTestData<float> all_data(m, n);

    RandLAPACK::util::gen_mat_type<float, r123::Philox4x32>(m, n, all_data.A, n, state, std::make_tuple(0, 2, false));
    computational_helper<float, r123::Philox4x32>(m, n, all_data);
    test_spectral_norm<float, r123::Philox4x32>(m, n, state, all_data);
}

TEST_F(TestUtil, test_spectral_norm_rank_def_mat_single_precision) {
    
    int64_t m = 1000;
    int64_t n = 100;
    auto state = RandBLAS::base::RNGState();
    UtilTestData<float> all_data(m, n);

    RandLAPACK::util::gen_mat_type<float, r123::Philox4x32>(m, n, all_data.A, n, state, std::make_tuple(9, std::pow(10, 7), false));
    computational_helper<float, r123::Philox4x32>(m, n, all_data);
    test_spectral_norm<float, r123::Philox4x32>(m, n, state, all_data);
}

TEST_F(TestUtil, test_normc) {
    int64_t m = 1000;
    int64_t n = 1;
    UtilTestData<double> all_data(m, n);

    test_normc<double>(m, all_data);
}

TEST_F(TestUtil, test_binary_rank_search_zero_mat) {
    int64_t m = 1000;
    int64_t n = 100;
    UtilTestData<double> all_data(m, n);

    test_binary_rank_search_zero_mat<double>(m, n, all_data);
}
