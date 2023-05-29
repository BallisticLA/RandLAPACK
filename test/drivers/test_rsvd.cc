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
        std::vector<T> A_approx_determ;
        std::vector<T> A_approx_determ_duf;
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
        A_approx_determ(m * n, 0.0),
        A_approx_determ_duf (m * k, 0.0),
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
    struct algorithm_objects {
        RandLAPACK::PLUL<T> Stab;
        RandLAPACK::RS<T, RNG> RS;
        RandLAPACK::CholQRQ<T> Orth_RF;
        RandLAPACK::RF<T> RF;
        RandLAPACK::CholQRQ<T> Orth_QB;
        RandLAPACK::QB<T> QB;
        RandLAPACK::RSVD<T> RSVD;

        algorithm_objects(
            bool verbosity, 
            bool cond_check, 
            bool orth_check, 
            int64_t p, 
            int64_t passes_per_iteration, 
            int64_t block_sz,
            RandBLAS::base::RNGState<RNG> state
        ) :
            Stab(cond_check, verbosity),
            RS(Stab, state, p, passes_per_iteration, verbosity, cond_check),
            Orth_RF(cond_check, verbosity),
            RF(RS, Orth_RF, verbosity, cond_check),
            Orth_QB(cond_check, verbosity),
            QB(RF, Orth_QB, verbosity, orth_check),
            RSVD(QB, verbosity, block_sz)
            {}
    };

    template <typename T>
    static void computational_helper(int64_t m, int64_t n, RSVDTestData<T>& all_data) {
        
        // Create a copy of the original matrix
        blas::copy(m * n, all_data.A.data(), 1, all_data.A_cpy.data(), 1);

        // Get low-rank SVD
        lapack::gesdd(Job::SomeVec, m, n, all_data.A_cpy.data(), m, all_data.s.data(), all_data.U.data(), m, all_data.VT.data(), n);
    }

    /// General test for RSVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T, typename RNG>
    static void test_RSVD1_general(
        int64_t m, 
        int64_t n, 
        int64_t k, 
        T tol, 
        RSVDTestData<T>& all_data,
        algorithm_objects<T, RNG>& all_algs) {

        T* A_approx_determ_dat = all_data.A_approx_determ.data();
        T* A_approx_determ_duf_dat = all_data.A_approx_determ_duf.data();
        T* A_k_dat = all_data.A_k.data();

        T* U1_dat = all_data.U1.data();
        T* S1_dat = all_data.S1.data();
        T* VT1_dat = all_data.VT1.data();

        T* U_dat = all_data.U.data();
        T* s_dat = all_data.s.data();
        T* S_dat = all_data.S.data();
        T* VT_dat = all_data.VT.data();

        // Regular QB2 call
        all_algs.RSVD.call(m, n, all_data.A, k, tol, all_data.U1, all_data.s1, all_data.VT1);
        
        // Construnct A_approx_determ = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, all_data.s1, k, all_data.S1);
        // U1 * S1 = A_approx_determ_duf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U1_dat, m, S1_dat, k, 1.0, A_approx_determ_duf_dat, m);
        // A_approx_determ_duf * VT1 =  A_approx_determ
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_approx_determ_duf_dat, m, VT1_dat, k, 0.0, A_approx_determ_dat, m);

        //T norm_test_4 = lapack::lange(Norm::Fro, m, n, A_cpy_dat, m);
        //printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);

        // zero out the trailing singular values
        std::fill(s_dat + k, s_dat + n, 0.0);
        RandLAPACK::util::diag(n, n, all_data.s, n, all_data.S);

        // TEST 4: Below is A_k - A_approx_determ = A_k - QB
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U_dat, m, S_dat, n, 1.0, A_k_dat, m);
        // A_k * VT -  A_hat == 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k_dat, m, VT_dat, n, -1.0, A_approx_determ_dat, m);

        T norm_test_4 = lapack::lange(Norm::Fro, m, n, A_approx_determ_dat, m);
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
    int64_t passes_per_iteration = 1;
    int64_t block_sz = 2;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.5625);
    auto state = RandBLAS::base::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;
    bool orth_check = true;

    RSVDTestData<double> all_data(m, n, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, orth_check, p, passes_per_iteration, block_sz, state);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2, false));
    computational_helper<double>(m, n, all_data);
    test_RSVD1_general<double, r123::Philox4x32>(m, n, k, tol, all_data, all_algs);
}
