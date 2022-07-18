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

class BenchmarkOrth : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static std::tuple<long, long, long, long> 
    test_speed_helper(int64_t m, int64_t n, uint32_t seed) {
    
        using namespace blas;
        using namespace lapack;

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);
        std::vector<T> A_cpy(size, 0.0);
        std::vector<T> A_cpy_2(size, 0.0);
        std::vector<T> A_cpy_3(size, 0.0);
	    
        T* A_dat = A.data();
        T* A_cpy_dat = A_cpy.data();
        T* A_cpy_2_dat = A_cpy_2.data();
        T* A_cpy_3_dat = A_cpy_3.data();

        // Random Gaussian test matrix
        RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A_dat, seed);
        // Make a copy
        std::copy(A_dat, A_dat + size, A_cpy_dat);
        std::copy(A_dat, A_dat + size, A_cpy_2_dat);
        std::copy(A_dat, A_dat + size, A_cpy_3_dat);

        // Stabilization Constructor
        Stab<T> Stab_PLU(1);
        Orth<T> Orth_CholQR(0);
        Orth<T> Orth_HQR(1);
        Orth<T> Orth_GEQR(2);

        // PIV LU
        // Stores L, U into Omega
        auto start_lu = high_resolution_clock::now();
        Stab_PLU.call(m, n, A_cpy);
        auto stop_lu = high_resolution_clock::now();
        long dur_lu = duration_cast<microseconds>(stop_lu - start_lu).count();

        // HQR
        auto start_qr = high_resolution_clock::now();
        Orth_HQR.call(m, n, A_cpy_2);
        auto stop_qr = high_resolution_clock::now();
        long dur_qr = duration_cast<microseconds>(stop_qr - start_qr).count();

        // CHOL QR
        // Orthonormalize A
        auto start_chol = high_resolution_clock::now();
        Orth_CholQR.call(m, n, A);
        auto stop_chol = high_resolution_clock::now();
        long dur_chol = duration_cast<microseconds>(stop_chol - start_chol).count();

        // GEQR
        auto start_geqr = high_resolution_clock::now();
        Orth_GEQR.call(m, n, A);
        auto stop_geqr = high_resolution_clock::now();
        long dur_geqr = duration_cast<microseconds>(stop_geqr - start_geqr).count();

        return std::make_tuple(dur_chol, dur_lu, dur_qr, dur_geqr);
    }

    template <typename T>
    static void 
    test_speed(int r_pow, int r_pow_max, int c_pow, int c_pow_max, int runs)
    {
        int64_t rows = 0;
        int64_t cols = 0;

        T chol_avg = 0;
        T lu_avg = 0;
        T qr_avg = 0;
        T geqr_avg = 0;

        for(; r_pow <= r_pow_max; ++r_pow)
        {
            rows = std::pow(2, r_pow);
            int c_buf = c_pow;

            for (; c_buf <= c_pow_max; ++c_buf)
            {
                cols = std::pow(2, c_buf);

                std::tuple<long, long, long, long> res;
                long t_chol = 0;
                long t_lu   = 0;
                long t_qr   = 0;
                long t_geqr = 0;

                long curr_t_chol = 0;
                long curr_t_lu   = 0;
                long curr_t_qr   = 0;
                long curr_t_geqr = 0;

                std::ofstream file("../../build/test_plots/test_speed/raw_data/test_" + std::to_string(rows) + "_" + std::to_string(cols) + ".dat");
                for(int i = 0; i < runs; ++i)
                {
                    res = test_speed_helper<T>(rows, cols, 1);
                    curr_t_chol = std::get<0>(res);
                    curr_t_lu   = std::get<1>(res);
                    curr_t_qr   = std::get<2>(res);
                    curr_t_geqr = std::get<3>(res);

                    // Skip first iteration, as it tends to produce garbage results
                    if (i != 0)
                    {
                        // Save the output into .dat file
                        file << curr_t_chol << "  " << curr_t_lu << "  " << curr_t_qr << "  " << curr_t_geqr << "\n";
                
                        t_chol += curr_t_chol;
                        t_lu   += curr_t_lu;
                        t_qr   += curr_t_qr;
                        t_geqr += curr_t_geqr;
                    }
                }

                chol_avg = (T)t_chol / (T)(runs - 1);
                lu_avg   = (T)t_lu   / (T)(runs - 1);
                qr_avg   = (T)t_qr   / (T)(runs - 1);
                geqr_avg   = (T)t_geqr   / (T)(runs - 1);

                printf("\nMatrix size: %ld by %ld.\n", rows, cols);
                printf("Average timing of Chol QR for %d runs: %f μs.\n", runs, chol_avg);
                printf("Average timing of Pivoted LU for %d runs: %f μs.\n", runs, lu_avg);
                printf("Average timing of Householder QR for %d runs: %f μs.\n", runs, qr_avg);
                printf("Average timing of GEQR for %d runs: %f μs.\n", runs, geqr_avg);
                printf("\nResult: cholQR is %f times faster then HQR, %f times faster then GEQR and %f times faster then PLU.\n", qr_avg / chol_avg, geqr_avg / chol_avg, lu_avg / chol_avg);
            }
        }
    }

    template <typename T>
    static void 
    test_speed_mean(int r_pow, int r_pow_max, int col, int col_max, int runs)
    {

        // Clear all files
        for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf)
        {
            int rows = std::pow(2, r_buf);
            std::ofstream ofs;
            ofs.open("../../build/test_plots/test_speed/raw_data/test_mean_time_QR_" + std::to_string(rows) + ".dat", std::ofstream::out | std::ofstream::trunc);
            ofs.close();
        }

        int64_t rows = 0;
        int64_t cols = 0;

        T chol_avg = 0;
        T lu_avg = 0;
        T qr_avg = 0;
        T geqr_avg = 0;

        for(; r_pow <= r_pow_max; ++r_pow)
        {
            rows = std::pow(2, r_pow);
            int64_t cols = col;

            for (; cols <= col_max; cols += 64)
            {
                std::tuple<long, long, long, long> res;
                long t_chol = 0;
                long t_lu   = 0;
                long t_qr   = 0;
                long t_geqr = 0;

                long curr_t_chol = 0;
                long curr_t_lu   = 0;
                long curr_t_qr   = 0;
                long curr_t_geqr = 0;

                for(int i = 0; i < runs; ++i)
                {
                    res = test_speed_helper<T>(rows, cols, 1);
                    curr_t_chol = std::get<0>(res);
                    curr_t_lu   = std::get<1>(res);
                    curr_t_qr   = std::get<2>(res);
                    curr_t_geqr = std::get<3>(res);

                    // Skip first iteration, as it tends to produce garbage results
                    if (i != 0)
                    {
                        t_chol += curr_t_chol;
                        t_lu   += curr_t_lu;
                        t_qr   += curr_t_qr;
                        t_geqr += curr_t_geqr;
                    }
                }

                chol_avg = (T)t_chol / (T)(runs - 1);
                lu_avg   = (T)t_lu   / (T)(runs - 1);
                qr_avg   = (T)t_qr   / (T)(runs - 1);
                geqr_avg   = (T)t_geqr   / (T)(runs - 1);

                // Save the output into .dat file
                //std::ofstream file("../../build/test_plots/test_speed/raw_data/test_mean_time_" + std::to_string(rows) + ".dat");
                std::fstream file;
                file.open("../../build/test_plots/test_speed/raw_data/test_mean_time_QR_" + std::to_string(rows) + ".dat", std::fstream::app);
                file << chol_avg << "  " << lu_avg << "  " << qr_avg << "  " << geqr_avg << "\n";

                printf("\nMatrix size: %ld by %ld.\n", rows, cols);
                printf("Average timing of Chol QR for %d runs: %f μs.\n", runs, chol_avg);
                printf("Average timing of Pivoted LU for %d runs: %f μs.\n", runs, lu_avg);
                printf("Average timing of Householder QR for %d runs: %f μs.\n", runs, qr_avg);
                printf("Average timing of GEQR for %d runs: %f μs.\n", runs, geqr_avg);
                printf("\nResult: cholQR is %f times faster then HQR, %f times faster then GEQR and %f times faster then PLU.\n", qr_avg / chol_avg, geqr_avg / chol_avg, lu_avg / chol_avg);
            }
        }
    }
};

/*
// Cache size of my processor is 24 megs
TEST_F(BenchmarkOrth, InOutCacheSpeedTest)
{
    // 4096 : 64 to 256
    // rest : 64 to 512
    //test_speed_mean<double>(12, 12, 64, 256, 100);
    //test_speed_mean<double>(13, 14, 64, 512, 100);
    //test_speed<double>(12, 14, 7, 9, 100);
}
*/