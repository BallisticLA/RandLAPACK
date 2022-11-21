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
using namespace RandLAPACK::drivers::cholqrcp;

class TestCholQRCP : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void test_CholQRCP1_general(int64_t m, int64_t n, int64_t d, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|================================TEST CholQRCP GENERAL BEGIN===============================|\n");
        using namespace blas;
        using namespace lapack;
        
        // For running QB
        std::vector<T> A(m * n, 0.0);
        
        gen_mat_type<T>(m, n, A, 30, seed, mat_type);
        
        int64_t size = m * n;

        // For results comparison
        std::vector<T> A_hat(size, 0.0);
        std::copy(A.data(), A.data() + size, A_hat.data());

        // For RSVD
        std::vector<T> Q;
        std::vector<T> R;
        std::vector<int64_t> J(n, 0.0);

        T* A_dat = A.data();
        T* A_hat_dat = A_hat.data();

        //Subroutine parameters 
        bool verbosity = false;
        bool timimg = true;
        
        // CholQRCP constructor - Choose defaut (CholQRCP1)
        CholQRCP<T> CholQRCP(verbosity, timing, seed, tol, use_cholqrcp1);
        
        // Regular QB2 call
        CholQRCP.call(
            m,
            n,
            A,
            d,
            R,
            J
        );
        
        T* Q_dat = Q.data();
        T* R_dat = R.data();
        int64_t* J_dat = J.data();
        int64_t k = CholQRCP.rank;
        
        // AP
        col_swap(m, n, n, A, J);

        // AP - QR
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_dat, m, R_dat, k, -1.0, A_hat_dat, m);

        T norm_test = lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF A - QR:  %e\n", norm_test);
        //ASSERT_NEAR(norm_test, 0, 1e-10);
        printf("|=================================TEST CholQRCP GENERAL END================================|\n");
    }
};

TEST_F(TestCholQRCP, SimpleTest)
{ 
    for (uint32_t seed : {2})//, 1, 2})
    {
        test_CholQRCP1_general<double>(100, 60, 70,  1.0e-9, std::make_tuple(0, 2, false), seed);
    }
}
