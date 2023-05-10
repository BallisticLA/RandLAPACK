#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

class TestREVD2 : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    /// General test for RSVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T, typename RNG>
    static void test_REVD2_general(int64_t m, int64_t k, int64_t k_start, std::tuple<int, T, bool> mat_type, RandBLAS::base::RNGState<RNG> state, int rank_expectation, T err_expectation) {

        printf("|==================================TEST REVD2 GENERAL BEGIN==================================|\n");

        // For running QB
        std::vector<T> A_buf(m * m, 0.0);
        RandLAPACK::util::gen_mat_type(m, m, A_buf, k, state, mat_type);

        std::vector<T> A(m * m, 0.0);
        T* A_dat = A.data();

        // We're using Nystrom, the original must be positive semidefinite
        blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, m, m, 1.0, A_buf.data(), m, 0.0, A.data(), m);
        for(int i = 1; i < m; ++i)
            blas::copy(m - i, &A_dat[i + ((i-1) * m)], 1, &A_dat[(i - 1) + (i * m)], m);

        // For results comparison
        std::vector<T> A_cpy (m * m, 0.0);
        std::vector<T> V(m * k, 0.0);
        std::vector<T> eigvals(k, 0.0);

        T* A_cpy_dat = A_cpy.data();

        // Create a copy of the original matrix
        blas::copy(m * m, A_dat, 1, A_cpy_dat, 1);
        T norm_A = lapack::lange(Norm::Fro, m, m, A_cpy_dat, m);

        //Subroutine parameters 
        bool verbosity = false;
        bool cond_check = false;
        int64_t p = 10;
        int64_t passes_per_iteration = 10;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        RandLAPACK::PLUL<T> Stab(cond_check, verbosity);
        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::RS<T, RNG> RS(Stab, state, p, passes_per_iteration, verbosity, cond_check);
        // Orthogonalization Constructor - Choose HouseholderQR
        RandLAPACK::HQRQ<T> Orth_RF(cond_check, verbosity);
        // RangeFinder constructor - Choose default (rf1)
        RandLAPACK::RF<T> RF(RS, Orth_RF, verbosity, cond_check);
        // REVD2 constructor
        RandLAPACK::REVD2<T, RNG> REVD2(RF, std::numeric_limits<double>::epsilon(), 10, state, verbosity);

        k = k_start;
        REVD2.tol = std::pow(10, -14);
        REVD2.call(m, blas::Uplo::Upper, A, k, V, eigvals);

        std::vector<T> E(k * k, 0.0);
        std::vector<T> Buf (m * k, 0.0);

        T* V_dat = V.data();
        T* E_dat = E.data();
        T* Buf_dat = Buf.data();

        // Construnct A_hat = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, eigvals, k, E);
        // V * E = Buf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, V_dat, m, E_dat, k, 0.0, Buf_dat, m);
        // A - Buf * V' - should be close to 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, m, k, 1.0, Buf_dat, m, V_dat, m, -1.0, A_cpy_dat, m);

        T norm_0 = lapack::lange(Norm::Fro, m, m, A_cpy_dat, m);
        printf("||A - VEV'||_F / ||A||_F:  %e\n", norm_0 / norm_A);
        ASSERT_NEAR(norm_0 / norm_A, err_expectation, 10 * err_expectation);
        ASSERT_NEAR(k, rank_expectation, std::numeric_limits<double>::epsilon());

        printf("|===================================TEST REVD2 GENERAL END===================================|\n");
    }
};

TEST_F(TestREVD2, Underestimation1) { 
    auto state = RandBLAS::base::RNGState();
    // Rank estimation must be 80 - underestimation - starting with very small rank
    test_REVD2_general<double, r123::Philox4x32>(1000, 100, 1, std::make_tuple(0, std::pow(10, 8), false), state, 64, std::pow(10, -13));
}
TEST_F(TestREVD2, Underestimation2) { 
    auto state = RandBLAS::base::RNGState();
    // Rank estimation must be 80 - underestimation
    test_REVD2_general<double, r123::Philox4x32>(1000, 100, 10, std::make_tuple(0, std::pow(10, 8), false), state, 80, std::pow(10, -13));
}
TEST_F(TestREVD2, Overestimation1) { 
    auto state = RandBLAS::base::RNGState();
    // Rank estimation must be 60 - overestimation
    test_REVD2_general<double, r123::Philox4x32>(1000, 100, 10, std::make_tuple(0, std::pow(10, 2), false), state, 160, std::pow(10, -13));
}
TEST_F(TestREVD2, Oversetimation2) { 
    auto state = RandBLAS::base::RNGState();
    // Rank estimation must be 160 - slight overestimation
    test_REVD2_general<double, r123::Philox4x32>(1000, 159, 10, std::make_tuple(0, std::pow(10, 2), false), state, 160, std::pow(10, -13));
}
TEST_F(TestREVD2, Exactness) { 
    auto state = RandBLAS::base::RNGState();
    // Numerically rank deficient matrix - expecting rank estimate = m
    test_REVD2_general<double, r123::Philox4x32>(100, 100, 10, std::make_tuple(0, std::pow(10, 2), false), state, 100, std::pow(10, -13));
}
