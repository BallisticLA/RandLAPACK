#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>

#include <fstream>
#include <gtest/gtest.h>

class TestQB : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct QBTestData {
        int64_t row;
        int64_t col;
        int64_t rank;
        std::vector<T> A;
        std::vector<T> Q;
        std::vector<T> B;
        std::vector<T> B_cpy;
        std::vector<T> A_hat;
        std::vector<T> A_k;
        std::vector<T> A_cpy;
        std::vector<T> A_cpy_2;
        std::vector<T> A_cpy_3;
        std::vector<T> s;
        std::vector<T> S;
        std::vector<T> U;
        std::vector<T> VT;

        QBTestData(int64_t m, int64_t n, int64_t k) :
            A(m * n, 0.0), 
            Q(m * k, 0.0), 
            B(k * n, 0.0), 
            B_cpy(k * n, 0.0), 
            A_hat(m * n, 0.0),
            A_k(m * n, 0.0),  
            A_cpy(m * n, 0.0),  
            A_cpy_2(m * n, 0.0),
            A_cpy_3(m * n, 0.0),
            s(n, 0.0),
            S(n * n, 0.0),
            U(m * n, 0.0),
            VT(n * n, 0.0)
            {
                row = m;
                col = n;
                rank = k;
            }
    };

    template <typename T, typename RNG>
    struct algorithm_objects {
        RandLAPACK::PLUL<T> Stab;
        RandLAPACK::RS<T, RNG> RS;
        RandLAPACK::CholQRQ<T> Orth_RF;
        RandLAPACK::RF<T, RNG> RF;
        RandLAPACK::CholQRQ<T> Orth_QB;
        RandLAPACK::QB<T, RNG> QB;

        algorithm_objects(bool verbosity, 
                            bool cond_check, 
                            bool orth_check, 
                            int64_t p, 
                            int64_t passes_per_iteration) :
                            Stab(cond_check, verbosity),
                            RS(Stab, p, passes_per_iteration, verbosity, cond_check),
                            Orth_RF(cond_check, verbosity),
                            RF(RS, Orth_RF, verbosity, cond_check),
                            Orth_QB(cond_check, verbosity),
                            QB(RF, Orth_QB, verbosity, orth_check)
                            {}
    };

    template <typename T>
    static void svd_and_copy_computational_helper(QBTestData<T> &all_data) {
        
        auto m = all_data.row;
        auto n = all_data.col;

        // Create a copy of the original matrix
        blas::copy(m * n, all_data.A.data(), 1, all_data.A_cpy.data(), 1);
        blas::copy(m * n, all_data.A.data(), 1, all_data.A_cpy_2.data(), 1);
        blas::copy(m * n, all_data.A.data(), 1, all_data.A_cpy_3.data(), 1);

        // Get low-rank SVD
        lapack::gesdd(Job::SomeVec, m, n, all_data.A_cpy.data(), m, all_data.s.data(), all_data.U.data(), m, all_data.VT.data(), n);
    }

    /// General test for QB:
    /// Computes QB factorzation, and checks:
    /// 1. A - QB
    /// 2. B - \transpose{Q}A
    /// 3. I - \transpose{Q}Q
    /// 4. A_k - QB = U_k\Sigma_k\transpose{V_k} - QB
    template <typename T, typename RNG, typename alg_type>
    static void test_QB2_low_exact_rank(
        int64_t block_sz, 
        T tol,  
        QBTestData<T> &all_data,
        alg_type &all_algs,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        T* A_dat = all_data.A.data();
        T* A_hat_dat = all_data.A_hat.data();
        T* A_k_dat = all_data.A_k.data();
        T* A_cpy_2_dat = all_data.A_cpy_2.data();

        T* U_dat = all_data.U.data();
        T* s_dat = all_data.s.data();
        T* S_dat = all_data.S.data();
        T* VT_dat = all_data.VT.data();

        // Regular QB2 call
        all_algs.QB.call(m, n, all_data.A, k, block_sz, tol, all_data.Q, all_data.B, state);

        // Reassing pointers because Q, B have been resized
        T* Q_dat = all_data.Q.data();
        T* B_dat = all_data.B.data();
        T* B_cpy_dat = all_data.B_cpy.data();

        printf("Inner dimension of QB: %-25ld\n", k);

        std::vector<T> Ident(k * k, 0.0);
        T* Ident_dat = Ident.data();
        // Generate a reference identity
        RandLAPACK::util::eye(k, k, Ident);
        // Buffer for testing B
        blas::copy(k * n, B_dat, 1, B_cpy_dat, 1);

        // A_hat = Q * B
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, B_dat, k, 0.0, A_hat_dat, m);
        // TEST 1: A = A - Q * B = 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, -1.0, Q_dat, m, B_dat, k, 1.0, A_dat, m);
        // TEST 2: B - Q'A = 0
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, -1.0, Q_dat, m, A_cpy_2_dat, m, 1.0, B_cpy_dat, k);
        // TEST 3: Q'Q = I
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, Ident_dat, k);

        // zero out the trailing singular values
        std::fill(s_dat + k, s_dat + n, 0.0);
        RandLAPACK::util::diag(n, n, all_data.s.data(), n, all_data.S.data());

        // TEST 4: Below is A_k - A_hat = A_k - QB
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U_dat, m, S_dat, n, 1.0, A_k_dat, m);
        // A_k * VT -  A_hat == 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k_dat, m, VT_dat, n, -1.0, A_hat_dat, m);

        T test_tol = std::pow(std::numeric_limits<T>::epsilon(), 0.625);
        // Test 1 Output
        T norm_test_1 = lapack::lange(Norm::Fro, m, n, A_dat, m);
        printf("FRO NORM OF A - QB:    %e\n", norm_test_1);
        ASSERT_NEAR(norm_test_1, 0, test_tol);
        // Test 2 Output
        T norm_test_2 = lapack::lange(Norm::Fro, k, n, B_cpy_dat, k);
        printf("FRO NORM OF B - Q'A:   %e\n", norm_test_2);
        ASSERT_NEAR(norm_test_2, 0, test_tol);
        // Test 3 Output
        T norm_test_3 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, Ident_dat, k);
        printf("FRO NORM OF Q'Q - I:   %e\n", norm_test_3);
        ASSERT_NEAR(norm_test_3, 0, test_tol);
        // Test 4 Output
        T norm_test_4 = lapack::lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);
        ASSERT_NEAR(norm_test_4, 0, test_tol);
    }

    /// k = min(m, n) test for CholQRCP:
    /// Checks for whether the factorization is exact with tol = 0.
    // Checks for whether ||A-QB||_F <= tol * ||A||_F if tol > 0.
    template <typename T, typename RNG>
    static void test_QB2_k_eq_min(
        int64_t block_sz, 
        T tol, 
        T &norm_A, 
        QBTestData<T> &all_data,
        algorithm_objects<T, RNG> &all_algs,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        int64_t k_est = std::min(m, n);

        T* A_dat = all_data.A.data();
        T* Q_dat = all_data.Q.data();
        T* B_dat = all_data.B.data();
        T* A_hat_dat = all_data.A_hat.data();

        // Regular QB2 call
        all_algs.QB.call(m, n, all_data.A, k_est, block_sz, tol, all_data.Q, all_data.B, state);

        // Reassing pointers because Q, B have been resized
        Q_dat = all_data.Q.data();
        B_dat = all_data.B.data();

        printf("Inner dimension of QB: %ld\n", k_est);

        // A_hat = Q * B
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_est, 1.0, Q_dat, m, B_dat, k_est, 0.0, A_hat_dat, m);
        // TEST 1: A = A - Q * B = 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_est, -1.0, Q_dat, m, B_dat, k_est, 1.0, A_dat, m);

        T norm_test_1 = lapack::lange(Norm::Fro, m, n, A_dat, m);
        T test_tol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        if(tol == 0.0) {
            // Test Zero Tol Output
            printf("FRO NORM OF A - QB:    %e\n", norm_test_1);
            ASSERT_NEAR(norm_test_1, 0, test_tol);
        }
        else {
            // Test Nonzero Tol Output
            printf("FRO NORM OF A - QB:    %e\n", norm_test_1);
            printf("FRO NORM OF A:         %e\n", norm_A);
            EXPECT_TRUE(norm_test_1 <= (tol * norm_A));
        }
    }
};

TEST_F(TestQB, Polynomial_Decay_general1)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    int64_t block_sz = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::RNGState();
    
    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;
    bool orth_check = true;

    auto all_data = new QBTestData<double>(m, n, k);
    auto all_algs = new algorithm_objects<double, r123::Philox4x32>(verbosity, cond_check, orth_check, p, passes_per_iteration);
    
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, (*all_data).A.data(), state);

    svd_and_copy_computational_helper(*all_data);
    test_QB2_low_exact_rank(block_sz, tol, *all_data, *all_algs, state);

    delete all_data;
    delete all_algs;
}

TEST_F(TestQB, Polynomial_Decay_general2)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    int64_t block_sz = 2;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::RNGState();
    
    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;
    bool orth_check = true;

    auto all_data = new QBTestData<double>(m, n, k);
    auto all_algs = new algorithm_objects<double, r123::Philox4x32>(verbosity, cond_check, orth_check, p, passes_per_iteration);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 6.7;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, (*all_data).A.data(), state);
    
    svd_and_copy_computational_helper(*all_data);
    test_QB2_low_exact_rank(block_sz, tol, *all_data, *all_algs, state);

    delete all_data;
    delete all_algs;
}

TEST_F(TestQB, Polynomial_Decay_zero_tol1)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    int64_t block_sz = 10;
    double tol = (double) 0.1;
    auto state = RandBLAS::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;
    bool orth_check = true;

    auto all_data = new QBTestData<double>(m, n, k);
    auto all_algs = new algorithm_objects<double, r123::Philox4x32>(verbosity, cond_check, orth_check, p, passes_per_iteration);
  
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, (*all_data).A.data(), state);

    double norm_A = lapack::lange(Norm::Fro, m, n, (*all_data).A.data(), m);
    test_QB2_k_eq_min(block_sz, tol, norm_A, *all_data, *all_algs, state);

    delete all_data;
    delete all_algs;
}

TEST_F(TestQB, Polynomial_Decay_zero_tol2)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    int64_t block_sz = 10;
    double tol = 0.0;
    auto state = RandBLAS::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;
    bool orth_check = true;

    auto all_data = new QBTestData<double>(m, n, k);
    auto all_algs = new algorithm_objects<double, r123::Philox4x32>(verbosity, cond_check, orth_check, p, passes_per_iteration);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, (*all_data).A.data(), state);

    double norm_A = lapack::lange(Norm::Fro, m, n, (*all_data).A.data(), m);
    test_QB2_k_eq_min(block_sz, tol, norm_A, *all_data, *all_algs, state);

    delete all_data;
    delete all_algs;
}