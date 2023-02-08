#include "RandBLAS.hh"
#include "RandLAPACK.hh"
#include "blaspp.h"
#include "lapackpp.h"

#include <math.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <gtest/gtest.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace std::chrono;
using namespace RandLAPACK::comps::util;
using namespace RandLAPACK::comps::orth;

class TestOrth : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    /// Tests orthogonality of a matrix Q, obtained by orthogonalizing a Gaussian sketch.
    /// Checks I - \transpose{Q}Q.
    template <typename T>
    static void test_orth_sketch(int64_t m, int64_t n, int64_t k, std::tuple<int, T, bool> mat_type, uint32_t seed) {
    
        int64_t size = m * n;
        std::vector<T> A(size, 0.0);
        std::vector<T> Y(m * k, 0.0);
        std::vector<T> Omega(n * k, 0.0);
        std::vector<T> I_ref(k * k, 0.0);

        T* A_dat = A.data();
        T* Y_dat = Y.data();
        T* Omega_dat = Omega.data();
        T* I_ref_dat = I_ref.data();
        
        gen_mat_type(m, n, A, k, seed, mat_type);
        
        // Fill the gaussian random matrix
        RandBLAS::dense::DenseDist D{.n_rows = n, .n_cols = k};
        auto state = RandBLAS::base::RNGState(seed, 0);
        state = RandBLAS::dense::fill_buff(Omega_dat, D, state);
        // Generate a reference identity
        eye(k, k, I_ref);  
        // Y = A * Omega
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_dat, m, Omega_dat, n, 0.0, Y_dat, m);
        // Orthogonalization Constructor
        CholQRQ<T> CholQRQ(false, false);

        // Orthonormalize sketch Y
        if(CholQRQ.call(m, k, Y) != 0) {
            EXPECT_TRUE(false) << "\nPOTRF FAILED DURE TO ILL-CONDITIONED DATA\n";
            return;
        }
        // Call the scheme twice for better orthogonality
        CholQRQ.call(m, k, Y);
        // Q' * Q  - I = 0
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Y_dat, m, Y_dat, m, -1.0, I_ref_dat, k);

        T norm_fro = lapack::lange(lapack::Norm::Fro, k, k, I_ref_dat, k);	
        printf("FRO NORM OF Q' * Q - I: %f\n", norm_fro);
        ASSERT_NEAR(norm_fro, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    }
};

TEST_F(TestOrth, SimpleTest)
{
}
