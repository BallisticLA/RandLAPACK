#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestREVD2 : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    /// General test for RSVD:
    /// Computes the decomposition factors, then checks A-U\Sigma\transpose{V}.
    template <typename T>
    static void test_REVD2_general(int64_t m, int64_t k, int64_t k_start, std::tuple<int, T, bool> mat_type, RandBLAS::base::RNGState<r123::Philox4x32> state) {
        
        printf("|==================================TEST QB2 GENERAL BEGIN==================================|\n");

        // For running QB
        std::vector<T> A_buf(m * m, 0.0);
        RandLAPACK::util::gen_mat_type(m, m, A_buf, k, state, mat_type);

        std::vector<T> A(m * m, 0.0);

        // We're using Nystrom, the original must be positive semidefinite
        blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, m, m, 1.0, A_buf.data(), m, 0.0, A.data(), m);
        for(int i = 1; i < m; ++i)
            blas::copy(m - i, A.data() + i + ((i-1) * m), 1, A.data() + (i - 1) + (i * m), m);

        //char name [] = "A";
        //RandBLAS::util::print_colmaj(m, m, A.data(), name);

        // For results comparison
        std::vector<T> A_cpy (m * m, 0.0);
        std::vector<T> V(m * k, 0.0);
        std::vector<T> e(k, 0.0);

        T* A_dat = A.data();
        T* A_cpy_dat = A_cpy.data();

        // Create a copy of the original matrix
        blas::copy(m * m, A.data(), 1, A_cpy_dat, 1);
        T norm_A = lapack::lange(Norm::Fro, m, m, A_cpy_dat, m);

        //Subroutine parameters 
        bool verbosity = false;
        bool cond_check = false;
        int64_t p = 0;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        RandLAPACK::PLUL<T> Stab(cond_check, verbosity);
        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::RS<T> RS(Stab, state, p, passes_per_iteration, verbosity, cond_check);
        // Orthogonalization Constructor - Choose HouseholderQR
        RandLAPACK::HQRQ<T> Orth_RF(cond_check, verbosity);
        // RangeFinder constructor - Choose default (rf1)
        RandLAPACK::RF<T> RF(RS, Orth_RF, verbosity, cond_check);
        // REVD2 constructor
        RandLAPACK::REVD2<T> REVD2(RF, state, verbosity);

        k = k_start;
        // Regular QB2 call
        REVD2.call(m, A, k, V, e);
        printf("COMPUTED K IS %ld\n", k);

        std::vector<T> E(k * k, 0.0);
        std::vector<T> Buf (m * k, 0.0);

        T* V_dat = V.data();
        T* E_dat = E.data();
        T* Buf_dat = Buf.data();
        
        // Construnct A_hat = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        RandLAPACK::util::diag(k, k, e, k, E);
        // V * E = Buf
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, V_dat, m, E_dat, k, 0.0, Buf_dat, m);
        // A - Buf * V' - should be close to 0
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, m, k, 1.0, Buf_dat, m, V_dat, m, -1.0, A_cpy_dat, m);

        T norm_0 = lapack::lange(Norm::Fro, m, m, A_cpy_dat, m);
        printf("FRO NORM OF A - VEV':  %e\n", norm_0 / norm_A);
        
        printf("|===================================TEST QB2 GENERAL END===================================|\n");
    }
};

TEST_F(TestREVD2, SimpleTest)
{ 
    // Generate a random state
    auto state = RandBLAS::base::RNGState(0, 0);
    test_REVD2_general<double>(1000, 159, 10, std::make_tuple(0, 2, false), state);
}