#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>
#include <lapack.hh>

#include <numeric>
#include <iostream>
#include <fstream>
using namespace std::chrono;

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

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
        
        RandLAPACK::comps::util::gen_mat_type(m, n, A, n, seed, mat_type);

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
        
        RandLAPACK::comps::util::gen_mat_type(m, n, A, k, seed, mat_type);
        
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

    template <typename T>
    static std::tuple<long, long, long> 
    test_speed_helper(int64_t m, int64_t n, uint32_t seed) {
    
        using namespace blas;
        using namespace lapack;

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);
        std::vector<T> A_cpy(size, 0.0);
        std::vector<T> A_cpy_2(size, 0.0);
	    std::vector<int64_t> ipiv(n, 0);
        std::vector<T> tau(n, 2.0);
	    
        T* A_dat = A.data();
        T* A_cpy_dat = A_cpy.data();
        T* A_cpy_2_dat = A_cpy_2.data();
        int64_t* ipiv_dat = ipiv.data();
        T* tau_dat = tau.data();

        // Random Gaussian test matrix
        RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A_dat, seed);
        // Make a copy
        std::copy(A_dat, A_dat + size, A_cpy_dat);
        std::copy(A_dat, A_dat + size, A_cpy_2_dat);

        // PIV LU
        // Stores L, U into Omega
        auto start_lu = high_resolution_clock::now();
        getrf(m, n, A_cpy_dat, m, ipiv_dat);
        // Addresses pivoting
        RandLAPACK::comps::util::row_swap<T>(m, n, A_cpy, ipiv);
        // Extracting L
        RandLAPACK::comps::util::get_L<T>(m, n, A_cpy);
        auto stop_lu = high_resolution_clock::now();
        long dur_lu = duration_cast<microseconds>(stop_lu - start_lu).count();

        // CHOL QR
        // Orthonormalize A
        auto start_chol = high_resolution_clock::now();
        RandLAPACK::comps::orth::chol_QR(m, n, A);
        // Call the scheme twice for better orthogonality
        //RandLAPACK::comps::orth::chol_QR(m, n, A);
        auto stop_chol = high_resolution_clock::now();
        long dur_chol = duration_cast<microseconds>(stop_chol - start_chol).count();

        auto start_qr = high_resolution_clock::now();
        geqrf(m, n, A_cpy_2_dat, m, tau_dat);
        ungqr(m, n, n, A_cpy_2_dat, m, tau_dat);
        auto stop_qr = high_resolution_clock::now();
        long dur_qr = duration_cast<microseconds>(stop_qr - start_qr).count();

        return std::make_tuple(dur_chol, dur_lu, dur_qr);
    }


    template <typename T>
    static void 
    test_speed(int r_pow, int r_pow_max, int c_pow, int c_pow_max, int runs)
    {
        int64_t rows = 0;
        int64_t cols = 0;

        double chol_avg = 0;
        double lu_avg = 0;
        double qr_avg = 0;

        for(; r_pow <= r_pow_max; ++r_pow)
        {
            rows = std::pow(2, r_pow);
            int c_buf = c_pow;

            for (; c_buf <= c_pow_max; ++c_buf)
            {
                cols = std::pow(2, c_buf);

                std::tuple<long, long, long> res;
                long t_chol = 0;
                long t_lu = 0;
                long t_qr = 0;

                long curr_t_chol = 0;
                long curr_t_lu = 0;
                long curr_t_qr = 0;

                std::ofstream file("../../build/test_plots/test_speed/raw_data/test_" + std::to_string(rows) + "_" + std::to_string(cols) + ".dat");
                for(int i = 0; i < runs; ++i)
                {
                    res = test_speed_helper<double>(rows, cols, i);
                    curr_t_chol = std::get<0>(res);
                    curr_t_lu = std::get<1>(res);
                    curr_t_qr = std::get<2>(res);

                    // Save the output into .dat file
                    file << curr_t_chol << "  " << curr_t_lu << "  " << curr_t_qr << "\n";
                
                    t_chol += curr_t_chol;
                    t_lu += curr_t_lu;
                    t_qr += curr_t_qr;
                }

                chol_avg = (double)t_chol / (double)runs;
                lu_avg = (double)t_lu / (double)runs;
                qr_avg = (double)t_qr / (double)runs;

                printf("\nMatrix size: %ld by %ld.\n", rows, cols);
                printf("Average timing of Chol QR for %d runs: %f μs.\n", runs, chol_avg);
                printf("Average timing of Pivoted LU for %d runs: %f μs.\n", runs, lu_avg);
                printf("Average timing of Householder QR for %d runs: %f μs.\n", runs, qr_avg);
                printf("\nResult: cholQR is %f times faster then HQR and %f times faster then PLU.\n", qr_avg / chol_avg, lu_avg / chol_avg);
            }
        }
    }
};

/*
TEST_F(TestOrth, SimpleTest)
{
    test_orth_sketch<double>(10, 10, 9, std::make_tuple(0, 2, true), 1);
    test_Chol_QR<double>(12, 12, std::make_tuple(1, 0, false), 0);
}
*/
/*
// Cache size of my processor is 24 megs
TEST_F(TestOrth, InOutCacheSpeedTest)
{
    test_speed<double>(14, 15, 7, 9, 10);
}
*/