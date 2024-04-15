#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


class TestRSVD : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct RSVDTestData {
        int64_t row;
        int64_t col;
        int64_t rank;
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
        RandLAPACK::RSVD<T, RNG> RSVD;

        algorithm_objects(
            bool verbosity, 
            bool cond_check, 
            bool orth_check, 
            int64_t p, 
            int64_t passes_per_iteration, 
            int64_t block_sz
        ) :
            Stab(cond_check, verbosity),
            RS(Stab, p, passes_per_iteration, verbosity, cond_check),
            Orth_RF(cond_check, verbosity),
            RF(RS, Orth_RF, verbosity, cond_check),
            Orth_QB(cond_check, verbosity),
            QB(RF, Orth_QB, verbosity, orth_check),
            RSVD(QB, verbosity, block_sz)
            {}
    };

    template <typename T>
    static void computational_helper(RSVDTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        
        // Create a copy of the original matrix
        blas::copy(m * n, all_data.A.data(), 1, all_data.A_cpy.data(), 1);
        // Get low-rank SVD
        lapack::gesdd(Job::SomeVec, m, n, all_data.A_cpy.data(), m, all_data.s.data(), all_data.U.data(), m, all_data.VT.data(), n);
    }

    /// General test for RSVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T, typename RNG>
    static void test_RSVD1_general(
        T tol, 
        RSVDTestData<T> &all_data,
        algorithm_objects<T, RNG> &all_algs,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

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
        all_algs.RSVD.call(m, n, all_data.A, k, tol, all_data.U1, all_data.s1, all_data.VT1, state);
        
        // Construnct A_approx_determ = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, all_data.s1.data(), k, all_data.S1.data());
        // U1 * S1 = A_approx_determ_duf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U1_dat, m, S1_dat, k, 1.0, A_approx_determ_duf_dat, m);
        // A_approx_determ_duf * VT1 =  A_approx_determ
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_approx_determ_duf_dat, m, VT1_dat, k, 0.0, A_approx_determ_dat, m);

        //T norm_test_4 = lapack::lange(Norm::Fro, m, n, A_cpy_dat, m);
        //printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);

        // zero out the trailing singular values
        std::fill(s_dat + k, s_dat + n, 0.0);
        RandLAPACK::util::diag(n, n, all_data.s.data(), n, all_data.S.data());

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
    auto state = RandBLAS::RNGState();

    //Subroutine parameters
    bool verbosity = false;
    bool cond_check = true;
    bool orth_check = true;

    RSVDTestData<double> all_data(m, n, k);
    algorithm_objects<double, r123::Philox4x32> all_algs(verbosity, cond_check, orth_check, p, passes_per_iteration, block_sz);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    computational_helper(all_data);
    test_RSVD1_general(tol, all_data, all_algs, state);
}
