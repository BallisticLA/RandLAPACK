#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestRSVD : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    /// General test for RSVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T>
    static void test_RSVD1_general(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, std::tuple<int, T, bool> mat_type, RandBLAS::base::RNGState<r123::Philox4x32> state) {
        
        printf("|==================================TEST QB2 GENERAL BEGIN==================================|\n");

        // For running QB
        std::vector<T> A(m * n, 0.0);
        RandLAPACK::util::gen_mat_type(m, n, A, k, state, mat_type);

        int64_t size = m * n;

        // For results comparison
        std::vector<T> A_hat(size, 0.0);
        std::vector<T> A_hat_buf (m * k, 0.0);
        std::vector<T> A_k(size, 0.0);
        std::vector<T> A_cpy (m * n, 0.0);

        // For RSVD
        std::vector<T> s1(n, 0.0);
        std::vector<T> S1(n * n, 0.0);
        std::vector<T> U1(m * n, 0.0);
        std::vector<T> VT1(n * n, 0.0);

        // For low-rank SVD
        std::vector<T> s(n, 0.0);
        std::vector<T> S(n * n, 0.0);
        std::vector<T> U(m * n, 0.0);
        std::vector<T> VT(n * n, 0.0);

        T* A_dat = A.data();

        T* A_hat_dat = A_hat.data();
        T* A_hat_buf_dat = A_hat_buf.data();
        T* A_k_dat = A_k.data();
        T* A_cpy_dat = A_cpy.data();

        T* U1_dat = U1.data();
        T* S1_dat = S1.data();
        T* VT1_dat = VT1.data();

        T* U_dat = U.data();
        T* s_dat = s.data();
        T* S_dat = S.data();
        T* VT_dat = VT.data();

        // Create a copy of the original matrix
        blas::copy(size, A_dat, 1, A_cpy_dat, 1);

        //Subroutine parameters 
        bool verbosity = false;
        bool cond_check = true;
        bool orth_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        RandLAPACK::PLUL<T> Stab(cond_check, verbosity);

        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::RS<T> RS(Stab, state, p, passes_per_iteration, verbosity, cond_check);

        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::CholQRQ<T> Orth_RF(cond_check, verbosity);

        // RangeFinder constructor - Choose default (rf1)
        RandLAPACK::RF<T> RF(RS, Orth_RF, verbosity, cond_check);

        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::CholQRQ<T> Orth_QB(cond_check, verbosity);

        // QB constructor - Choose defaut (QB2)
        RandLAPACK::QB<T> QB(RF, Orth_QB, verbosity, orth_check);

        // RSVD constructor - Choose defaut (RSVD1)
        RandLAPACK::RSVD<T> RSVD(QB, verbosity, block_sz);

        // Regular QB2 call
        RSVD.call(m, n, A, k, tol, U1, s1, VT1);
        
        // Construnct A_hat = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, s1, k, S1);
        // U1 * S1 = A_hat_buf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U1_dat, m, S1_dat, k, 1.0, A_hat_buf_dat, m);
        // A_hat_buf * VT1 =  A_hat
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_hat_buf_dat, m, VT1_dat, k, 0.0, A_hat_dat, m);

        //T norm_test_4 = lapack::lange(Norm::Fro, m, n, A_cpy_dat, m);
        //printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);


        // Reconstruct  a standard low-rank SVD
        lapack::gesdd(Job::SomeVec, m, n, A_cpy_dat, m, s_dat, U_dat, m, VT_dat, n);
        // buffer zero vector
        std::vector<T> z_buf(n, 0.0);
        T* z_buf_dat = z_buf.data();
        // zero out the trailing singular values
        blas::copy(n - k, z_buf_dat, 1, s_dat + k, 1);
        RandLAPACK::util::diag(n, n, s, n, S);

        // TEST 4: Below is A_k - A_hat = A_k - QB
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U_dat, m, S_dat, n, 1.0, A_k_dat, m);
        // A_k * VT -  A_hat == 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k_dat, m, VT_dat, n, -1.0, A_hat_dat, m);

        T norm_test_4 = lapack::lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);
        //ASSERT_NEAR(norm_test_4, 0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
        printf("|===================================TEST QB2 GENERAL END===================================|\n");
    }
};

TEST_F(TestRSVD, SimpleTest)
{ 
    // Generate a random state
    auto state = RandBLAS::base::RNGState(0, 0);
    test_RSVD1_general<double>(100, 100, 50, 5, 10, std::pow(std::numeric_limits<double>::epsilon(), 0.5625), std::make_tuple(0, 2, false), state);
}
