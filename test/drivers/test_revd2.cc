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

    template <typename T>
    struct REVD2TestData {
        std::vector<T> A_buf;
        std::vector<T> A;
        std::vector<T> A_cpy;
        std::vector<T> V;
        std::vector<T> eigvals;
        std::vector<T> E;
        std::vector<T> Buf;

        REVD2TestData(int64_t m, int64_t k) :
        A_buf(m * m, 0.0), 
        A(m * m, 0.0), 
        A_cpy(m * m, 0.0),  
        V(m * k, 0.0), 
        eigvals(k, 0.0), 
        E(k * k, 0.0), 
        Buf(m * k, 0.0)
        {}
    };

    template <typename T, typename RNG>
    struct algorithm_objects {
        RandLAPACK::PLUL<T> Stab;
        RandLAPACK::RS<T, RNG> RS;
        RandLAPACK::HQRQ<T> Orth_RF;
        RandLAPACK::RF<T> RF;
        RandLAPACK::REVD2<T, RNG> REVD2;

        algorithm_objects(bool verbosity, 
                            bool cond_check, 
                            int64_t p, 
                            int64_t passes_per_iteration, 
                            RandBLAS::base::RNGState<RNG> state) :
                            Stab(cond_check, verbosity),
                            RS(Stab, state, p, passes_per_iteration, verbosity, cond_check),
                            Orth_RF(cond_check, verbosity),
                            RF(RS, Orth_RF, verbosity, cond_check),
                            REVD2(RF, state, verbosity)
                            {}
    };

    template <typename T, typename RNG>
    static void symm_mat_and_copy_computational_helper(int64_t m, T& norm_A, REVD2TestData<T>& all_data) {

        // We're using Nystrom, the original must be positive semidefinite
        blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, m, m, 1.0, all_data.A_buf.data(), m, 0.0, all_data.A.data(), m);
        
        T* A_dat = all_data.A.data();

        for(int i = 1; i < m; ++i)
            blas::copy(m - i, &A_dat[i + ((i-1) * m)], 1, &A_dat[(i - 1) + (i * m)], m);

                    // Create a copy of the original matrix
        blas::copy(m * m, all_data.A.data(), 1, all_data.A_cpy.data(), 1);
        norm_A = lapack::lange(Norm::Fro, m, m, all_data.A_cpy.data(), m);
    }

    /// General test for REVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T, typename RNG>
    static void test_REVD2_general(
        int64_t m, 
        int64_t k, 
        int64_t k_start, 
        T tol,
        int p,
        int rank_expectation, 
        T err_expectation, 
        T& norm_A, 
        REVD2TestData<T>& all_data,
        algorithm_objects<T, RNG>& all_algs) {

        k = k_start;
        all_algs.REVD2.call(m, blas::Uplo::Upper, all_data.A, k, tol, p, all_data.V, all_data.eigvals);

        RandLAPACK::util::upsize(k * k, all_data.E);
        RandLAPACK::util::upsize(m * k, all_data.Buf);

        T* A_cpy_dat = all_data.A_cpy.data();
        T* V_dat = all_data.V.data();
        T* E_dat = all_data.E.data();
        T* Buf_dat = all_data.Buf.data();

        // Construnct A_hat = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, all_data.eigvals, k, all_data.E);
        // V * E = Buf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, V_dat, m, E_dat, k, 0.0, Buf_dat, m);
        // A - Buf * V' - should be close to 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, m, k, 1.0, Buf_dat, m, V_dat, m, -1.0, A_cpy_dat, m);

        T norm_0 = lapack::lange(Norm::Fro, m, m, A_cpy_dat, m);
        printf("||A - VEV'||_F / ||A||_F:  %e\n", norm_0 / norm_A);
        ASSERT_NEAR(norm_0 / norm_A, err_expectation, 10 * err_expectation);
        ASSERT_NEAR(k, rank_expectation, std::numeric_limits<double>::epsilon());
    }
};

TEST_F(TestREVD2, Underestimation1) { 
    int64_t m = 1000;
    int64_t k = 100;
    int64_t k_start = 1;
    int64_t rank_expectation = 64;
    double norm_A = 0;
    int64_t p = 10;
    double tol = std::pow(10, -14);
    int64_t passes_per_iteration = 10;
    double err_expectation =std::pow(10, -13);
    auto state = RandBLAS::base::RNGState();

    //Subroutine parameters 
    bool verbosity = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration, state);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, m, all_data.A_buf, k, state, std::make_tuple(0, std::pow(10, 8), false));
    symm_mat_and_copy_computational_helper<double, r123::Philox4x32>(m, norm_A, all_data);
    test_REVD2_general<double, r123::Philox4x32>(m, k, k_start, tol, p, rank_expectation, err_expectation, norm_A, all_data, all_algs);
}

TEST_F(TestREVD2, Underestimation2) { 
    int64_t m = 1000;
    int64_t k = 100;
    int64_t k_start = 10;
    int64_t rank_expectation = 80;
    double norm_A = 0;
    int64_t p = 10;
    double tol = std::pow(10, -14);
    int64_t passes_per_iteration = 10;
    double err_expectation =std::pow(10, -13);
    auto state = RandBLAS::base::RNGState();
    //Subroutine parameters 
    bool verbosity = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration, state);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, m, all_data.A_buf, k, state, std::make_tuple(0, std::pow(10, 8), false));
    symm_mat_and_copy_computational_helper<double, r123::Philox4x32>(m, norm_A, all_data);
    test_REVD2_general<double, r123::Philox4x32>(m, k, k_start, tol, p, rank_expectation, err_expectation, norm_A, all_data, all_algs);
}

TEST_F(TestREVD2, Overestimation1) { 
    int64_t m = 1000;
    int64_t k = 100;
    int64_t k_start = 10;
    int64_t rank_expectation = 160;
    double norm_A = 0;
    int64_t p = 10;
    double tol = std::pow(10, -14);
    int64_t passes_per_iteration = 10;
    double err_expectation =std::pow(10, -13);
    auto state = RandBLAS::base::RNGState();
    //Subroutine parameters 
    bool verbosity = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration, state);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, m, all_data.A_buf, k, state, std::make_tuple(0, std::pow(10, 2), false));
    symm_mat_and_copy_computational_helper<double, r123::Philox4x32>(m, norm_A, all_data);
    test_REVD2_general<double, r123::Philox4x32>(m, k, k_start, tol, p, rank_expectation, err_expectation, norm_A, all_data, all_algs);
}

TEST_F(TestREVD2, Oversetimation2) { 
    int64_t m = 1000;
    int64_t k = 159;
    int64_t k_start = 10;
    int64_t rank_expectation = 160;
    double norm_A = 0;
    int64_t p = 10;
    double tol = std::pow(10, -14);
    int64_t passes_per_iteration = 10;
    double err_expectation =std::pow(10, -13);
    auto state = RandBLAS::base::RNGState();
    //Subroutine parameters 
    bool verbosity = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration, state);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, m, all_data.A_buf, k, state, std::make_tuple(0, std::pow(10, 2), false));
    symm_mat_and_copy_computational_helper<double, r123::Philox4x32>(m, norm_A, all_data);
    test_REVD2_general<double, r123::Philox4x32>(m, k, k_start, tol, p, rank_expectation, err_expectation, norm_A, all_data, all_algs);
}

TEST_F(TestREVD2, Exactness) { 
    int64_t m = 100;
    int64_t k = 100;
    int64_t k_start = 10;
    int64_t rank_expectation = 100;
    double norm_A = 0;
    int64_t p = 10;
    double tol = std::pow(10, -14);
    int64_t passes_per_iteration = 10;
    double err_expectation =std::pow(10, -13);
    auto state = RandBLAS::base::RNGState();
    //Subroutine parameters 
    bool verbosity = false;
    bool cond_check = false;

    REVD2TestData<double> all_data(m, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, p, passes_per_iteration, state);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, m, all_data.A_buf, k, state, std::make_tuple(0, std::pow(10, 2), false));
    symm_mat_and_copy_computational_helper<double, r123::Philox4x32>(m, norm_A, all_data);
    test_REVD2_general<double, r123::Philox4x32>(m, k, k_start, tol, p, rank_expectation, err_expectation, norm_A, all_data, all_algs);
}
