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
        std::vector<T> V1;
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
        V1(n * n, 0.0),

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
            bool verbose, 
            bool cond_check, 
            bool orth_check, 
            int64_t p, 
            int64_t passes_per_iteration, 
            int64_t block_sz
        ) :
            Stab(cond_check, verbose),
            RS(Stab, p, passes_per_iteration, verbose, cond_check),
            Orth_RF(cond_check, verbose),
            RF(RS, Orth_RF, verbose, cond_check),
            Orth_QB(cond_check, verbose),
            QB(RF, Orth_QB, verbose, orth_check),
            RSVD(QB, block_sz)
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

        T* U1_dat  = nullptr;
        T* s1_dat  = nullptr;
        T* V1_dat = nullptr;
        T* S1_dat  = all_data.S1.data();

        T* U_dat = all_data.U.data();
        T* s_dat = all_data.s.data();
        T* S_dat = all_data.S.data();
        T* VT_dat = all_data.VT.data();

        // Regular QB2 call
        all_algs.RSVD.call(m, n, all_data.A.data(), k, tol, U1_dat, s1_dat, V1_dat, state);

        // Construnct A_approx_determ = U1 * S1 * V1^T
        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, s1_dat, k, S1_dat);
        // U1 * S1 = A_approx_determ_duf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U1_dat, m, S1_dat, k, 1.0, A_approx_determ_duf_dat, m);
        // A_approx_determ_duf * V1^T =  A_approx_determ
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, k, 1.0, A_approx_determ_duf_dat, m, V1_dat, n, 0.0, A_approx_determ_dat, m);

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

        free(U1_dat);
        free(s1_dat);
        free(V1_dat);
    }
};

TEST_F(TestRSVD, SimpleTest)
{ 
    int64_t m = 10;
    int64_t n = 10;
    int64_t k = 5;
    int64_t p = 10;
    int64_t passes_per_iteration = 1;
    int64_t block_sz = 2;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.5625);
    auto state = RandBLAS::RNGState();

    //Subroutine parameters
    bool verbose = false;
    bool cond_check = true;
    bool orth_check = true;

    auto all_data = new RSVDTestData<double>(m, n, k);
    auto all_algs = new algorithm_objects<double, r123::Philox4x32>(verbose, cond_check, orth_check, p, passes_per_iteration, block_sz);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    RandLAPACK::gen::mat_gen(m_info, (*all_data).A.data(), state);

    computational_helper(*all_data);
    test_RSVD1_general(tol, *all_data, *all_algs, state);

    delete all_data;
    delete all_algs;
}
