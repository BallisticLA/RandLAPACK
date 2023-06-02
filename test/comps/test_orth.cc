#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>

#include <math.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <gtest/gtest.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace std::chrono;

class TestOrth : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct OrthTestData {
        int64_t row;
        int64_t col;
        int64_t rank;
        std::vector<T> A;
        std::vector<T> Y;
        std::vector<T> Omega;
        std::vector<T> I_ref;

        OrthTestData(
            int64_t m, int64_t n, int64_t k
        ) :
            A(m * n, 0.0),
            Y(m * k, 0.0),
            Omega(n * k, 0.0),
            I_ref(k * k, 0.0)
        {
            row = m;
            col = n;
            rank = k;
        }
    };

    template <typename T, typename RNG>
    static void sketch_and_copy_computational_helper(
        RandBLAS::base::RNGState<RNG> state,
        OrthTestData<T>& all_data
    ) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        // Fill the gaussian random matrix
        RandBLAS::dense::DenseDist D{.n_rows = n, .n_cols = k};
        state = RandBLAS::dense::fill_buff(all_data.Omega.data(), D, state);
        
        // Generate a reference identity
        RandLAPACK::util::eye(k, k, all_data.I_ref);

        // Y = A * Omega
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, all_data.A.data(), m, all_data.Omega.data(), n, 0.0, all_data.Y.data(), m);
    }

    /// Tests orthogonality of a matrix Q, obtained by orthogonalizing a Gaussian sketch.
    /// Checks I - \transpose{Q}Q.
    template <typename T, typename RNG>
    static void test_orth_sketch(
        OrthTestData<T>& all_data, 
        RandLAPACK::CholQRQ<T>& CholQRQ
    ) {

        auto m = all_data.row;
        auto k = all_data.rank;

        T* Y_dat = all_data.Y.data();
        T* I_ref_dat = all_data.I_ref.data();

        // Orthonormalize sketch Y
        if(CholQRQ.call(m, k, all_data.Y) != 0) {
            EXPECT_TRUE(false) << "\nPOTRF FAILED DURE TO ILL-CONDITIONED DATA\n";
            return;
        }
        // Call the scheme twice for better orthogonality
        CholQRQ.call(m, k, all_data.Y);
        // Q' * Q  - I = 0
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Y_dat, m, -1.0, I_ref_dat, k);
        T norm_fro = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);


        printf("FRO NORM OF Q' * Q - I: %f\n", norm_fro);
        ASSERT_NEAR(norm_fro, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    }
};

TEST_F(TestOrth, Test_CholQRQ)
{
    int64_t m = 1000;
    int64_t n = 200;
    int64_t k = 200;
    auto state = RandBLAS::base::RNGState();
    
    OrthTestData<double> all_data(m, n, k);
    // Orthogonalization Constructor
    RandLAPACK::CholQRQ<double> CholQRQ(false, false);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2, false));
    sketch_and_copy_computational_helper<double, r123::Philox4x32>(state, all_data);
    test_orth_sketch<double, r123::Philox4x32>(all_data, CholQRQ);
}
