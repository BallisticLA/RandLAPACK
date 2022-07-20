/*
TODO #1: Switch tuples to vectors.
*/

#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>
#include <lapack.hh>

#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono;

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace RandLAPACK::comps::util;
using namespace RandLAPACK::comps::orth;

class TestOrth : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};


    template <typename T>
    static void test_Chol_QR(int64_t m, int64_t n, std::tuple<int, T, bool> mat_type, uint32_t seed) {
    
        using namespace blas;

        int64_t size = m * n;
        std::vector<T> A(size);
        std::vector<T> I_ref(n * n, 0.0);

        T* A_dat = A.data();
        T* I_ref_dat = I_ref.data();
        
        gen_mat_type<T>(m, n, A, n, seed, mat_type);

        // Generate a reference identity
        eye<T>(n, n, I_ref);  

        // Orthogonalization Constructor
        Orth<T> Orth(0, false, false);

        // Orthonormalize A
        if (Orth.call(m, n, A) != 0)
        {
            EXPECT_TRUE(false) << "\nPOTRF FAILED DURE TO ILL-CONDITIONED DATA\n";
            return;
        }
        // Call the scheme twice for better orthogonality
        Orth.call(m, n, A);

        // Q' * Q  - I = 0
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A_dat, m, A_dat, m, -1.0, I_ref_dat, n);

        T norm_fro = lapack::lange(lapack::Norm::Fro, n, n, I_ref_dat, n);	
        printf("FRO NORM OF Q' * Q - I %f\n", norm_fro);
        ASSERT_NEAR(norm_fro, 0.0, 1e-12);
    }

    template <typename T>
    static void test_orth_sketch(int64_t m, int64_t n, int64_t k, std::tuple<int, T, bool> mat_type, uint32_t seed) {
    
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
        
        gen_mat_type<T>(m, n, A, k, seed, mat_type);
        
        // Fill the gaussian random matrix
        RandBLAS::dense::DenseDist D{.n_rows = n, .n_cols = k};
        auto state = RandBLAS::base::RNGState(seed, 0);
        state = RandBLAS::dense::fill_buff<T>(Omega_dat, D, state);
        // Generate a reference identity
        eye<T>(k, k, I_ref);  
        
        // Y = A * Omega
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Y_dat, m);
        
        // Orthogonalization Constructor
        Orth<T> Orth(0, false, false);

        // Orthonormalize sketch Y
        if(Orth.call(m, k, Y) != 0)
        {
            EXPECT_TRUE(false) << "\nPOTRF FAILED DURE TO ILL-CONDITIONED DATA\n";
            return;
        }
        // Call the scheme twice for better orthogonality
        Orth.call(m, k, Y);

        // Q' * Q  - I = 0
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Y_dat, m, Y_dat, m, -1.0, I_ref_dat, k);

        T norm_fro = lapack::lange(lapack::Norm::Fro, k, k, I_ref_dat, k);	

        printf("FRO NORM OF Q' * Q - I: %f\n", norm_fro);
        ASSERT_NEAR(norm_fro, 0.0, 1e-10);
    }
};

/*
TEST_F(TestOrth, SimpleTest)
{
}
*/

