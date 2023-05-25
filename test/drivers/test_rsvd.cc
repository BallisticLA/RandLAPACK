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

    template <typename T>
    struct RSVDTestData {

        std::vector<T> A;
        // For results comparison
        std::vector<T> A_hat;
        std::vector<T> A_hat_buf;
        std::vector<T> A_k;
        std::vector<T> A_cpy;

        // For RSVD
        std::vector<T> s1;
        std::vector<T> S1;
        std::vector<T> U1;
        std::vector<T> VT1;

        // For low-rank SVD
        std::vector<T> s;
        std::vector<T> S;
        std::vector<T> U;
        std::vector<T> VT;

        RSVDTestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0),
        // For results comparison
        A_hat(m * n, 0.0),
        A_hat_buf (m * k, 0.0),
        A_k(m * n, 0.0),
        A_cpy (m * n, 0.0),

        // For RSVD
        s1(n, 0.0),
        S1(n * n, 0.0),
        U1(m * n, 0.0),
        VT1(n * n, 0.0),

        // For low-rank SVD
        s(n, 0.0),
        S(n * n, 0.0),
        U(m * n, 0.0),
        VT(n * n, 0.0)
        {}
    };

    template <typename T, typename RNG>
    static void computational_helper(int64_t m, int64_t n, RSVDTestData<T>& all_data) {
        
        // Create a copy of the original matrix
        blas::copy(m * n, all_data.A.data(), 1, all_data.A_cpy.data(), 1);

        // Get low-rank SVD
        lapack::gesdd(Job::SomeVec, m, n, all_data.A_cpy.data(), m, all_data.s.data(), all_data.U.data(), m, all_data.VT.data(), n);
    }

    /// General test for RSVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T, typename RNG>
    static void test_RSVD1_general(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, RandBLAS::base::RNGState<RNG> state, RSVDTestData<T>& all_data) {

        T* A_hat_dat = all_data.A_hat.data();
        T* A_hat_buf_dat = all_data.A_hat_buf.data();
        T* A_k_dat = all_data.A_k.data();

        T* U1_dat = all_data.U1.data();
        T* S1_dat = all_data.S1.data();
        T* VT1_dat = all_data.VT1.data();

        T* U_dat = all_data.U.data();
        T* s_dat = all_data.s.data();
        T* S_dat = all_data.S.data();
        T* VT_dat = all_data.VT.data();

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

        // RSVD constructor - Choose defaut (RSVD1)
        RandLAPACK::RSVD<T> RSVD(QB, verbosity, block_sz);

        // Regular QB2 call
        RSVD.call(m, n, all_data.A, k, tol, all_data.U1, all_data.s1, all_data.VT1);
        
        // Construnct A_hat = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, all_data.s1, k, all_data.S1);
        // U1 * S1 = A_hat_buf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U1_dat, m, S1_dat, k, 1.0, A_hat_buf_dat, m);
        // A_hat_buf * VT1 =  A_hat
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_hat_buf_dat, m, VT1_dat, k, 0.0, A_hat_dat, m);

        //T norm_test_4 = lapack::lange(Norm::Fro, m, n, A_cpy_dat, m);
        //printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);

        // zero out the trailing singular values
        std::fill(s_dat + k, s_dat + n, 0.0);
        RandLAPACK::util::diag(n, n, all_data.s, n, all_data.S);

        // TEST 4: Below is A_k - A_hat = A_k - QB
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U_dat, m, S_dat, n, 1.0, A_k_dat, m);
        // A_k * VT -  A_hat == 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k_dat, m, VT_dat, n, -1.0, A_hat_dat, m);

        T norm_test_4 = lapack::lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);
        //ASSERT_NEAR(norm_test_4, 0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    }
};

TEST_F(TestRSVD, SimpleTest)
{ 
    int64_t m = 100;
    int64_t n = 100;
    int64_t k = 5;
    int64_t p = 10;
    int64_t block_sz = 2;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.5625);
    auto state = RandBLAS::base::RNGState();
    RSVDTestData<double> all_data(m, n, k);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2, false));
    computational_helper<double, r123::Philox4x32>(m, n, all_data);
    test_RSVD1_general<double, r123::Philox4x32>(m, n, k, p, block_sz, tol, state, all_data);
}
