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

    template <typename T>
    static void test_orth_sketch(int64_t m, int64_t n, int64_t k, int64_t mat_type, uint32_t seed) {
    
        using namespace blas;

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);
        std::vector<T> Y(m * k, 0.0);
        std::vector<T> Omega(n * k, 0.0);
        std::vector<T> I_ref(k * k, 0.0);
        
        switch(mat_type) 
        {
            case 1:
                // Generating matrix with exponentially decaying singular values
                RandLAPACK::comps::util::gen_exp_mat<T>(m, n, A.data(), k, 0.5, seed); 
                break;
            case 2:
                // Generating matrix with s-shaped singular values plot
                RandLAPACK::comps::util::gen_s_mat<T>(m, n, A.data(), k, seed); 
                break;
            case 3:
                // Full-rank random A
                RandBLAS::dense_op::gen_rmat_norm<T>(m, k, A.data(), seed);
                if (2 * k <= n)
                {
                    // Add entries without increasing the rank
                    std::copy(A.data(), A.data() + (n / 2) * m, A.data() + (n / 2) * m);
                }
                break;
        }
        
        // Fill the gaussian random matrix
        RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega.data(), seed);
        // Generate a reference identity
        RandLAPACK::comps::util::eye<T>(k, k, I_ref.data());  
        
        // Y = A * Omega
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Y.data(), m);
        
        // Orthonormalize sketch Y
        RandLAPACK::comps::orth::chol_QR(m, k, Y.data());;

        // Q' * Q  - I = 0
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Y.data(), m, Y.data(), m, -1.0, I_ref.data(), k);

        T norm_fro = lapack::lange(lapack::Norm::Fro, k, k, I_ref.data(), k);	

        printf("FRO NORM OF Q' * Q - I: %f\n", norm_fro);
        ASSERT_NEAR(norm_fro, 0.0, 1e-12);

    }
};

TEST_F(TestOrth, SimpleTest)
{
    //for (uint32_t seed : {0, 1, 2})
    //{
        test_orth_sketch<double>(10, 1, 1, 1, 0);
    //}
}