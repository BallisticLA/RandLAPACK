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
    static void test_Chol_QR(int64_t m, int64_t n, int mat_type, uint32_t seed) {
    
        using namespace blas;

        int64_t size = m * n;
        std::vector<T> A(size);
        std::vector<T> I_ref(n * n, 0.0);

        T* A_dat = A.data();
        T* I_ref_dat = I_ref.data();
        
        switch(mat_type) 
        {
            case 1:
                // Generating matrix with exponentially decaying singular values
                RandLAPACK::comps::util::gen_exp_mat<T>(m, n, A, n, 0.5, seed); 
                break;
            case 2:
                // Generating matrix with s-shaped singular values plot
                RandLAPACK::comps::util::gen_s_mat<T>(m, n, A, n, seed); 
                break;
            case 3:
                // Full-rank random A
                RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A_dat, seed);
                break;
        }

        // Generate a reference identity
        RandLAPACK::comps::util::eye<T>(n, n, I_ref);  

        // Orthonormalize A
        if (RandLAPACK::comps::orth::chol_QR(m, n, A) != 0)
        {
            EXPECT_TRUE(false) << "\nPOTRF FAILED DURE TO ILL-CONDITIONED DATA\n";
            return;
        }
        // Call the scheme twice for better orthogonality
        RandLAPACK::comps::orth::chol_QR(m, n, A);

        // Q' * Q  - I = 0
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A_dat, m, A_dat, m, -1.0, I_ref_dat, n);

        T norm_fro = lapack::lange(lapack::Norm::Fro, n, n, I_ref_dat, n);	
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

        T* A_dat = A.data();
        T* Y_dat = Y.data();
        T* Omega_dat = Omega.data();
        T* I_ref_dat = I_ref.data();
        
        switch(mat_type) 
        {
            case 1:
                // Generating matrix with exponentially decaying singular values
                RandLAPACK::comps::util::gen_exp_mat<T>(m, n, A, k, 0.5, seed); 
                break;
            case 2:
                // Generating matrix with s-shaped singular values plot
                RandLAPACK::comps::util::gen_s_mat<T>(m, n, A, k, seed); 
                break;
            case 3:
                // Full-rank random A
                RandBLAS::dense_op::gen_rmat_norm<T>(m, k, A_dat, seed);
                if (2 * k <= n)
                {
                    // Add entries without increasing the rank
                    std::copy(A_dat, A_dat + (n / 2) * m, A_dat + (n / 2) * m);
                }
                break;
        }
        
        // Fill the gaussian random matrix
        RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega_dat, seed);
        // Generate a reference identity
        RandLAPACK::comps::util::eye<T>(k, k, I_ref);  
        
        // Y = A * Omega
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Y_dat, m);
        
        // Orthonormalize sketch Y
        if(RandLAPACK::comps::orth::chol_QR(m, k, Y) != 0)
        {
            EXPECT_TRUE(false) << "\nPOTRF FAILED DURE TO ILL-CONDITIONED DATA\n";
            return;
        }
        // Call the scheme twice for better orthogonality
        RandLAPACK::comps::orth::chol_QR(m, k, Y);

        // Q' * Q  - I = 0
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Y_dat, m, Y_dat, m, -1.0, I_ref_dat, k);

        T norm_fro = lapack::lange(lapack::Norm::Fro, k, k, I_ref_dat, k);	

        printf("FRO NORM OF Q' * Q - I: %f\n", norm_fro);
        ASSERT_NEAR(norm_fro, 0.0, 1e-10);
    }
};

TEST_F(TestOrth, SimpleTest)
{
    //for (uint32_t seed : {0, 1, 2})
    //{
        //test_orth_sketch<double>(10, 10, 9, 1, 1);
        test_Chol_QR<double>(12, 12, 1, 0);
    //}
}