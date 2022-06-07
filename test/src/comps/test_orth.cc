#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>
#include <lapack.hh>


#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestOrth : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};


    template <typename T>
    static void test_Chol_QR(int64_t m, int64_t n, uint32_t seed) {
    
        using namespace blas;

        int64_t size = m * n;
        std::vector<T> A(size);
        std::vector<T> I_ref(n * n, 0.0);
        // Generate a random matrix of std normal distribution
        RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);
        // Generate a reference identity
        RandLAPACK::comps::util::eye<T>(n, n, I_ref.data());  
        // Orthonormalize A
        RandLAPACK::comps::orth::chol_QR(m, n, A.data());;

        // Q' * Q  - I = 0
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A.data(), m, A.data(), m, -1.0, I_ref.data(), n);

        T norm_fro = lapack::lange(lapack::Norm::Fro, n, n, I_ref.data(), n);	
        printf("FRO NORM OF Q' * Q - I %f\n", norm_fro);
        ASSERT_NEAR(norm_fro, 0.0, 1e-12);
    }
};

TEST_F(TestOrth, SimpleTest)
{
    for (uint32_t seed : {0, 1, 2})
    {
        test_Chol_QR<double>(1000, 10, seed);
        //test_Chol_QR<double>(500, 500, seed);
    }
}