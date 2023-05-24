#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>

#include <fstream>
#include <gtest/gtest.h>

class TestQB : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    /// General test for QB:
    /// Computes QB factorzation, and checks:
    /// 1. A - QB
    /// 2. B - \transpose{Q}A
    /// 3. I - \transpose{Q}Q
    /// 4. A_k - QB = U_k\Sigma_k\transpose{V_k} - QB

    template <typename T, typename RNG>
    static void computational_helper(int64_t m, int64_t n,
    T& norm_A,
    std::vector<T>& A, 
    std::vector<T>& A_cpy, 
    std::vector<T>& A_cpy_2, 
    std::vector<T>& A_cpy_3,
    std::vector<T>& s, 
    std::vector<T>& U, 
    std::vector<T>& VT) {
        
        // Create a copy of the original matrix
        blas::copy(m * n, A.data(), 1, A_cpy.data(), 1);
        blas::copy(m * n, A.data(), 1, A_cpy_2.data(), 1);
        blas::copy(m * n, A.data(), 1, A_cpy_3.data(), 1);

        // Get low-rank SVD
        lapack::gesdd(Job::SomeVec, m, n, A_cpy.data(), m, s.data(), U.data(), m, VT.data(), n);

        // pre-compute norm
        norm_A = lapack::lange(Norm::Fro, m, n, A.data(), m);
    }

    template <typename T, typename RNG>
    static void test_QB2_low_exact_rank(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, RandBLAS::base::RNGState<RNG> state,
    std::vector<T>& A,
    std::vector<T>& Q,
    std::vector<T>& B, 
    std::vector<T>& B_cpy, 
    std::vector<T>& A_hat, 
    std::vector<T>& A_k, 
    std::vector<T>& A_cpy_2, 
    std::vector<T>& s, 
    std::vector<T>& S, 
    std::vector<T>& U, 
    std::vector<T>& VT) {

        printf("|==================================TEST QB2 GENERAL BEGIN==================================|\n");

        T* A_dat = A.data();
        T* A_hat_dat = A_hat.data();
        T* A_k_dat = A_k.data();
        T* A_cpy_2_dat = A_cpy_2.data();

        T* U_dat = U.data();
        T* s_dat = s.data();
        T* S_dat = S.data();
        T* VT_dat = VT.data();

        //Subroutine parameters
        bool verbosity = false;
        bool cond_check = true;
        bool orth_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        RandLAPACK::PLUL<T> Stab(cond_check, verbosity);
        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::RS<T, RNG> RS(Stab, state, p, passes_per_iteration, verbosity, cond_check);
        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::CholQRQ<T> Orth_RF(cond_check, verbosity);
        // RangeFinder constructor - Choose default (rf1)
        RandLAPACK::RF<T> RF(RS, Orth_RF, verbosity, cond_check);
        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::CholQRQ<T> Orth_QB(cond_check, verbosity);
        // QB constructor - Choose defaut (QB2)
        RandLAPACK::QB<T> QB(RF, Orth_QB, verbosity, orth_check);
        // Regular QB2 call
        QB.call(m, n, A, k, block_sz, tol, Q, B);

        // Reassing pointers because Q, B have been resized
        T* Q_dat = Q.data();
        T* B_dat = B.data();
        T* B_cpy_dat = B_cpy.data();

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

        // buffer zero vector
        // buffer zero vector
        std::vector<T> z_buf(n, 0.0);
        // zero out the trailing singular values
        blas::copy(n - k, z_buf.data(), 1, s_dat + k, 1);
        RandLAPACK::util::diag(n, n, s, n, S);

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
        printf("|===================================TEST QB2 GENERAL END===================================|\n");
    }

    /// k = min(m, n) test for CholQRCP:
    /// Checks for whether the factorization is exact with tol = 0.
    // Checks for whether ||A-QB||_F <= tol * ||A||_F if tol > 0.
    template <typename T, typename RNG>
    static void test_QB2_k_eq_min(
        int64_t m, 
        int64_t n, 
        int64_t p, 
        int64_t block_sz, 
        T tol, 
        RandBLAS::base::RNGState<RNG> state,
        T& norm_A, 
        std::vector<T>& A,
        std::vector<T>& Q,
        std::vector<T>& B,
        std::vector<T>& A_hat) {

        printf("|===============================TEST QB2 K = min(M, N) BEGIN===============================|\n");

        int64_t k_est = std::min(m, n);

        T* A_dat = A.data();
        T* Q_dat = Q.data();
        T* B_dat = B.data();
        T* A_hat_dat = A_hat.data();

        //Subroutine parameters
        bool verbosity = false;
        bool cond_check = true;
        bool orth_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose CholQR
        RandLAPACK::PLUL<T> Stab(cond_check, verbosity);
        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::RS<T, RNG> RS(Stab, state, p, passes_per_iteration, verbosity, cond_check);
        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::CholQRQ<T> Orth_RF(cond_check, verbosity);
        // RangeFinder constructor - Choose default (rf1)
        RandLAPACK::RF<T> RF(RS, Orth_RF, verbosity, cond_check);
        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::CholQRQ<T> Orth_QB(cond_check, verbosity);
        // QB constructor - Choose defaut (QB2)
        RandLAPACK::QB<T> QB(RF, Orth_QB, verbosity, orth_check);
        // Regular QB2 call
        QB.call(m, n, A, k_est, block_sz, tol, Q, B);

        // Reassing pointers because Q, B have been resized
        Q_dat = Q.data();
        B_dat = B.data();

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
        printf("|================================TEST QB2 K = min(M, N) END================================|\n");
    }
};
/*
TEST_F(TestQB, Polynomial_Decay)
{
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::base::RNGState();
    // Fast polynomial decay test
    test_QB2_low_exact_rank<double, r123::Philox4x32>(100, 100, 50, 5, 10, tol, std::make_tuple(0, 2025, false), state);
    // Slow polynomial decay test
    test_QB2_low_exact_rank<double, r123::Philox4x32>(100, 100, 50, 5, 2, tol, std::make_tuple(0, 6.7, false), state);
    // Superfast exponential decay test
    test_QB2_low_exact_rank<double, r123::Philox4x32>(100, 100, 50, 5, 2, tol, std::make_tuple(1, 2025, false), state);
}
TEST_F(TestQB, Zero_Mat)
{
    auto state = RandBLAS::base::RNGState();
    // A = 0
    test_QB2_low_exact_rank<double, r123::Philox4x32>(100, 100, 50, 5, 2, std::pow(std::numeric_limits<double>::epsilon(), 0.75), std::make_tuple(3, 0, false), state);
}
TEST_F(TestQB, Rand_Diag)
{
    auto state = RandBLAS::base::RNGState();
    // Random diagonal matrix test
    test_QB2_low_exact_rank<double, r123::Philox4x32>(100, 100, 50, 5, 2, std::pow(std::numeric_limits<double>::epsilon(), 0.75), std::make_tuple(4, 0, false), state);
}
TEST_F(TestQB, Diag_Drop)
{
    auto state = RandBLAS::base::RNGState();
    // A = diag(sigma), where sigma_1 = ... = sigma_l > sigma_{l + 1} = ... = sigma_n
    test_QB2_low_exact_rank<double, r123::Philox4x32>(100, 100, 0, 5, 2, std::pow(std::numeric_limits<double>::epsilon(), 0.75), std::make_tuple(5, 0, false), state);
}
TEST_F(TestQB, Varying_Tol)
{
    auto state = RandBLAS::base::RNGState();
    // test zero tol
    test_QB2_k_eq_min<double, r123::Philox4x32>(100, 100, 10, 5, 2, 0.0, std::make_tuple(0, 1.23, false), state);
    // test nonzero tol
    test_QB2_k_eq_min<double, r123::Philox4x32>(100, 100, 10, 5, 2, (double) 0.1, std::make_tuple(0, 1.23, false), state);
}
*/
TEST_F(TestQB, Polynomial_Decay)
{
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 50;
    int64_t p = 5;
    int64_t block_sz = 10;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::base::RNGState();

    // For running QB
    std::vector<double> A(m * n, 0.0);
    std::vector<double> Q    (m * k, 0.0);
    std::vector<double> B    (k * n, 0.0);
    std::vector<double> B_cpy(k * n, 0.0);
    // For results comparison
    std::vector<double> A_hat   (m * n, 0.0);
    std::vector<double> A_k     (m * n, 0.0);
    std::vector<double> A_cpy   (m * n, 0.0);
    std::vector<double> A_cpy_2 (m * n, 0.0);
    std::vector<double> A_cpy_3 (m * n, 0.0);
    // For low-rank SVD
    std::vector<double> s (n, 0.0);
    std::vector<double> S (n * n, 0.0);
    std::vector<double> U (m * n, 0.0);
    std::vector<double> VT(n * n, 0.0);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, A, k, state, std::make_tuple(0, 2025, false));
    computational_helper<double, r123::Philox4x32>(m, n, norm_A, A, A_cpy, A_cpy_2, A_cpy_3, s, U, VT);
    test_QB2_low_exact_rank<double, r123::Philox4x32>(m, n, k, p, block_sz, tol, state, A, Q, B, B_cpy, A_hat, A_k, A_cpy_2, s, S, U, VT);
    
    // Reset data - mandatory
    Q.clear();
    B.clear();
    std::fill(A_hat.begin(), A_hat.end(), 0.0);

    test_QB2_k_eq_min<double, r123::Philox4x32>(m, n, p, block_sz, tol, state, norm_A, A_cpy_3, Q, B, A_hat);

}