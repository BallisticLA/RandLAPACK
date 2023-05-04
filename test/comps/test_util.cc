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
    static void 
    test_2_norm(int64_t m, int64_t n, std::tuple<int, T, bool> mat_type, RandBLAS::base::RNGState<r123::Philox4x32> state) {
        
        std::vector<T> A(m * n, 0.0);
        std::vector<T> A_cpy(m * n, 0.0);
        std::vector<T> s(n, 0.0);
        RandLAPACK::util::gen_mat_type(m, n, A, n, state, mat_type);

        T norm = RandLAPACK::util::get_2_norm(m, n, A.data(), 1000, state);

        // Get an SVD -> first singular value == 2_norm
        lapack::lacpy(MatrixType::General, m, n, A.data(), m, A_cpy.data(), m);
        lapack::gesdd(Job::NoVec, m, n, A_cpy.data(), m, s.data(), NULL, m, NULL, n);

        printf("Computed norm: %e\nComputed s_max: %e\n", norm, s[0]);
        ASSERT_NEAR(norm, s[0], std::pow(std::numeric_limits<double>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    test_normc(int64_t m) {
        
        std::vector<T> A(m, 0.0);
        std::vector<T> A_norm(m, 0.0);
        int cnt = 0;
        // we have a vector withentries 1:m
        std::for_each(A.begin(), A.end(), [&cnt](T &entry) { entry = ++cnt;});

        // We expect A_norm to have all 1's
        RandLAPACK::util::normc(1, m, A, A_norm);

        // We expect this to be 1;
        T norm = blas::nrm2(m, A_norm.data(), 1);

        ASSERT_NEAR(norm, std::sqrt(m), std::pow(std::numeric_limits<double>::epsilon(), 0.75));
    }
};

TEST_F(TestUtil, test_2_norm) {
    auto state = RandBLAS::base::RNGState(0, 0);
    test_2_norm<double>(1000, 100, std::make_tuple(0, 2, false), state);
    test_2_norm<double>(1000, 100, std::make_tuple(9, std::pow(10, 15), false), state);
}

TEST_F(TestUtil, test_normc) {
    auto state = RandBLAS::base::RNGState(0, 0);
    test_normc<double>(1000);
}
