#include <gtest/gtest.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace RandLAPACK::comps::util;
using namespace RandLAPACK::comps::orth;
using namespace RandLAPACK::comps::rs;
using namespace RandLAPACK::comps::rf;
using namespace RandLAPACK::comps::qb;
using namespace RandLAPACK::drivers::rsvd;

class TestRSVD : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void test_RSVD1_general(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|==================================TEST QB2 GENERAL BEGIN==================================|\n");
        using namespace blas;
        using namespace lapack;
        
        // For running QB
        std::vector<T> A(m * n, 0.0);
        gen_mat_type<T>(m, n, A, k, seed, mat_type);

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
        T* s1_dat = s1.data();
        T* S1_dat = S1.data();
        T* VT1_dat = VT1.data();

        T* U_dat = U.data();
        T* s_dat = s.data();
        T* S_dat = S.data();
        T* VT_dat = VT.data();

        // Create a copy of the original matrix
        copy(size, A_dat, 1, A_cpy_dat, 1);

        //Subroutine parameters 
        bool verbosity = false;
        bool cond_check = true;
        bool orth_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        Stab<T> Stab(use_PLUL, cond_check, verbosity);

        // RowSketcher constructor - Choose default (rs1)
        RS<T> RS(Stab, seed, p, passes_per_iteration, verbosity, cond_check, use_rs1);

        // Orthogonalization Constructor - Choose CholQR
        Orth<T> Orth_RF(use_CholQRQ, cond_check, verbosity);

        // RangeFinder constructor - Choose default (rf1)
        RF<T> RF(RS, Orth_RF, verbosity, cond_check, use_rf1);

        // Orthogonalization Constructor - Choose CholQR
        Orth<T> Orth_QB(use_CholQRQ, cond_check, verbosity);

        // QB constructor - Choose defaut (QB2)
        QB<T> QB(RF, Orth_QB, verbosity, orth_check, use_qb2);

        // RSVD constructor - Choose defaut (RSVD1)
        RSVD<T> RSVD(QB, verbosity, use_rsvd1);

        // Regular QB2 call
        RSVD.call(
            m,
            n,
            A,
            k,
            block_sz,
            tol,
            U1,
            s1,
            VT1
        );
        
        // Construnct A_hat = U1 * S1 * VT1

        // Turn vector into diagonal matrix
        diag<T>(k, k, s1, k, S1);
        // U1 * S1 = A_hat_buf
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, U1_dat, m, S1_dat, k, 1.0, A_hat_buf_dat, m);
        // A_hat_buf * VT1 =  A_hat
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_hat_buf_dat, m, VT1_dat, k, 0.0, A_hat_dat, m);

        //T norm_test_4 = lange(Norm::Fro, m, n, A_cpy_dat, m);
        //printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);


        // Reconstruct  a standard low-rank SVD
        gesdd(Job::SomeVec, m, n, A_cpy_dat, m, s_dat, U_dat, m, VT_dat, n);
        // buffer zero vector
        std::vector<T> z_buf(n, 0.0);
        T* z_buf_dat = z_buf.data();
        // zero out the trailing singular values
        copy(n - k, z_buf_dat, 1, s_dat + k, 1);
        diag<T>(n, n, s, n, S);

        // TEST 4: Below is A_k - A_hat = A_k - QB
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U_dat, m, S_dat, n, 1.0, A_k_dat, m);
        // A_k * VT -  A_hat == 0
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k_dat, m, VT_dat, n, -1.0, A_hat_dat, m);

        T norm_test_4 = lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);
        //ASSERT_NEAR(norm_test_4, 0, 1e-10);
        printf("|===================================TEST QB2 GENERAL END===================================|\n");
    }
};

TEST_F(TestRSVD, SimpleTest)
{ 
    for (uint32_t seed : {2})//, 1, 2})
    {
        test_RSVD1_general<double>(100, 100, 50, 5, 10, 1.0e-9, std::make_tuple(0, 2, false), seed);
    }
}