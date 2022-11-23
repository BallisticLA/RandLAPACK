/*
TODO #1: Switch tuples to vectors.
*/

#include <blas.hh>
#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>
#include <math.h>

#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
using namespace std::chrono;

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace RandLAPACK::comps::util;
using namespace RandLAPACK::drivers::cholqrcp;

template <typename T>
static std::vector<long> 
test_speed_helper(int64_t m, int64_t n, int64_t nnz, int64_t num_threads, uint32_t seed) {
    
    using namespace blas;
    using namespace lapack;

    int64_t size = m * n;
    std::vector<T> A_1(size, 0.0);
    std::vector<T> A_2(size, 0.0);
    std::vector<T> A_3(size, 0.0);
    std::vector<T> A_4(size, 0.0);
    
    std::vector<T> R_1;
    std::vector<int64_t> J_1(n, 0);
    std::vector<T> D_1(n, 0.0);
    std::vector<T> T_1(n * n, 0.0);

    std::vector<int64_t> J_2(n, 0.0);
    std::vector<T> tau_2(n, 0);

    std::vector<T> R_3(m * n, 0);
    std::vector<T> t_3(5, 0);
    std::vector<T> tau_3(n, 0);
    std::vector<int64_t> J_3(n, 0);

    std::vector<T> tau_4(n, 0);

    // Random Gaussian test matrix
    RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A_1.data(), seed);


    // Make copies
    std::copy(A_1.data(), A_1.data() + size, A_2.data());
    std::copy(A_1.data(), A_1.data() + size, A_3.data());
    std::copy(A_1.data(), A_1.data() + size, A_4.data());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // CholQRCP constructor
    int64_t d = 2 * n;
    bool log_times = true;
    CholQRCP<T> CholQRCP(false, log_times, seed, 1.0e-16, use_cholqrcp1);
    CholQRCP.nnz = nnz;
    CholQRCP.num_threads = num_threads;
    // Upsizing buffers

    auto start_alloc = high_resolution_clock::now();
    if(log_times)
    {
        (CholQRCP.times).resize(10);
    }
    upsize(d * n, (CholQRCP.A_hat));
    upsize(n, (CholQRCP.A_hat));
    J_1.resize(n);
    upsize(n * n, (CholQRCP.R_sp));
    upsize(n * n, R_1);
    upsize(n * n, (CholQRCP.R_buf));
    auto stop_alloc = high_resolution_clock::now();
    long dur_alloc = duration_cast<microseconds>(stop_alloc - start_alloc).count();
    
    // CholQRCP
    auto start_cholqrcp = high_resolution_clock::now();
    CholQRCP.call(m, n, A_1, d, R_1, J_1);
    auto stop_cholqrcp = high_resolution_clock::now();
    long dur_cholqrcp = duration_cast<microseconds>(stop_cholqrcp - start_cholqrcp).count();

    // CholQRCP verbose info print
    if(log_times)
    {
        printf("\n\n/------------CholQRCP1 TIMING RESULTS BEGIN------------/\n");
        printf("SASO time: %33ld μs,\n",                    (CholQRCP.times)[0]);
        printf("QRCP time: %33ld μs,\n",                    (CholQRCP.times)[1]);
        printf("Rank revealing time: %23ld μs,\n",          (CholQRCP.times)[2]);
        printf("CholQRCP time: %29ld μs,\n",                (CholQRCP.times)[3]);
        printf("A modification pivoting time: %14ld μs,\n", (CholQRCP.times)[4]);
        printf("A modification TRSM time: %18ld μs,\n",     (CholQRCP.times)[5]);
        printf("Copying time: %30ld μs,\n",                 (CholQRCP.times)[6]);
        printf("Resizing time: %29ld μs,\n",                (CholQRCP.times)[7]);
        printf("Other routines time: %23ld μs,\n",          (CholQRCP.times)[8]);
        printf("Total time: %32ld μs.\n",                   (CholQRCP.times)[9]);

        printf("\nSASO generation and application takes %.1f%% of runtime.\n", 100 * ((double) (CholQRCP.times)[0] / (double) (CholQRCP.times)[9]));
        printf("QRCP takes %31.1f%% of runtime.\n",                            100 * ((double) (CholQRCP.times)[1] / (double) (CholQRCP.times)[9]));
        printf("Rank revealing takes %21.1f%% of runtime.\n",                  100 * ((double) (CholQRCP.times)[2] / (double) (CholQRCP.times)[9]));
        printf("Modifying matrix (pivoting) A %12.1f%% of runtime.\n",         100 * ((double) (CholQRCP.times)[3] / (double) (CholQRCP.times)[9]));
        printf("Modifying matrix (trsm) A %16.1f%% of runtime.\n",             100 * ((double) (CholQRCP.times)[4] / (double) (CholQRCP.times)[9]));
        printf("Cholqrcp takes %27.1f%% of runtime.\n",                        100 * ((double) (CholQRCP.times)[5] / (double) (CholQRCP.times)[9]));
        printf("Copying takes %28.1f%% of runtime.\n",                         100 * ((double) (CholQRCP.times)[6] / (double) (CholQRCP.times)[9]));
        printf("Resizing takes %27.1f%% of runtime.\n",                        100 * ((double) (CholQRCP.times)[7] / (double) (CholQRCP.times)[9]));
        printf("Everything else takes %20.1f%% of runtime.\n",                 100 * ((double) (CholQRCP.times)[8] / (double) (CholQRCP.times)[9]));
        printf("/-------------CholQRCP1 TIMING RESULTS END-------------/\n\n");
    }

    // Householder reflectors restoring
    auto start_rest = high_resolution_clock::now();
    orhr_col(m, n, n, A_1.data(), m, T_1.data(), n, D_1.data());
    auto stop_rest = high_resolution_clock::now();
    long dur_rest = duration_cast<microseconds>(stop_rest - start_rest).count();

    // GEQP3
    auto start_geqp3 = high_resolution_clock::now();
    geqp3(m, n, A_2.data(), m, J_2.data(), tau_2.data());
    auto stop_geqp3 = high_resolution_clock::now();
    long dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();

    // GEQR + GEQP3
    auto start_tsqrp = high_resolution_clock::now();
    // GEQR part
    auto sart_geqr = high_resolution_clock::now();
    geqr(m, n, A_3.data(), m, t_3.data(), -1);
    int64_t tsize = (int64_t) t_3[0]; 
    t_3.resize(tsize);
    geqr(m, n, A_3.data(), m, t_3.data(), tsize);
    auto stop_geqr = high_resolution_clock::now();
    long dur_geqr = duration_cast<microseconds>(stop_geqr - sart_geqr).count();

    // GEQP3 on R part
    get_U(m, n, A_3, R_3);
    geqp3(m, n, A_3.data(), m, J_3.data(), tau_3.data());

    auto stop_tsqrp = high_resolution_clock::now();
    long dur_tsqrp = duration_cast<microseconds>(stop_tsqrp - start_tsqrp).count();

    // GEQRF
    auto start_geqrf = high_resolution_clock::now();
    geqrf(m, n, A_4.data(), m, tau_4.data());
    auto stop_geqrf = high_resolution_clock::now();
    long dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();

    std::vector<long> res{dur_alloc, dur_cholqrcp, dur_rest, dur_geqp3, dur_geqrf, dur_geqr, dur_tsqrp}; 
 
    return res;
}


template <typename T>
static void 
test_speed_mean(int r_pow, int r_pow_max, int col, int col_max, int runs, int nnz, int num_threads)
{

    // Clear all files
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf)
    {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open("../../../test_plots/test_speed_full_Q/raw_data/test_mean_time_" + std::to_string(rows) + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    int64_t rows = 0;
    int64_t cols = 0;

    T alloc_avg    = 0;
    T cholqrcp_avg = 0;
    T rest_avg     = 0;
    T geqp3_avg    = 0;
    T geqrf_avg    = 0;
    T geqr_avg     = 0;
    T tsqrp_avg    = 0;

    for(; r_pow <= r_pow_max; ++r_pow)
    {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols += 64)
        {
            std::vector<long> res;
            long t_alloc    = 0;
            long t_cholqrcp = 0;
            long t_rest     = 0;
            long t_geqp3    = 0;
            long t_geqrf    = 0;
            long t_geqr     = 0;
            long t_tsqrp    = 0;

            long curr_t_alloc    = 0;
            long curr_t_cholqrcp = 0;
            long curr_t_rest     = 0;
            long curr_t_geqp3    = 0;
            long curr_t_geqrf    = 0;
            long curr_t_geqr     = 0;
            long curr_t_tsqrp    = 0;

            for(int i = 0; i < runs; ++i)
            {
                res = test_speed_helper<T>(rows, cols, nnz, num_threads, i);

                //res{dur_alloc, dur_cholqrcp, dur_rest, dur_geqp3, dur_geqrf, dur_geqr, dur_tsqrp}; 

                curr_t_alloc    = res[0];
                curr_t_cholqrcp = res[1];
                curr_t_rest     = res[2];
                curr_t_geqp3    = res[3];
                curr_t_geqrf    = res[4];
                curr_t_geqr     = res[5];
                curr_t_tsqrp    = res[6];

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0)
                {
                    t_alloc    += curr_t_alloc;
                    t_cholqrcp += curr_t_cholqrcp;
                    t_rest     += curr_t_rest;
                    t_geqp3    += curr_t_geqp3;
                    t_geqrf    += curr_t_geqrf;
                    t_geqr     += curr_t_geqr;
                    t_tsqrp    += curr_t_tsqrp;
                }
            }

            alloc_avg    = (T)t_alloc    / (T)(runs - 1);
            cholqrcp_avg = (T)t_cholqrcp / (T)(runs - 1);
            rest_avg     = (T)t_rest     / (T)(runs - 1);
            geqp3_avg    = (T)t_geqp3    / (T)(runs - 1);
            geqrf_avg    = (T)t_geqrf    / (T)(runs - 1);
            geqr_avg     = (T)t_geqr     / (T)(runs - 1);
            tsqrp_avg    = (T)t_tsqrp    / (T)(runs - 1);

            // Save the output into .dat file
            std::fstream file("../../../test_plots/test_speed_full_Q/raw_data/test_mean_time_" + std::to_string(rows) + ".dat", std::fstream::app);
            file << cholqrcp_avg << "  " << geqp3_avg << tsqrp_avg << "\n";

            printf("\n/--------------------------------------QRCP MEAN TIMING BEGIN--------------------------------------/\n");
            printf("\nMatrix size: %ld by %ld.\n", rows, cols);
            printf("Number of nonzeros per column in SASO: %d\n", nnz);
            printf("Number of threads used in SASO application: %d\n", num_threads);

            printf("\nAverage timing of workspace pre-allocation for CholQRCP for %d runs: %15f μs.\n", runs - 1, alloc_avg);
            printf("Average timing of CholQRCP for %d runs: %45f μs.\n",                                runs - 1, cholqrcp_avg);
            printf("Average timing Householder vector restoration for %d runs: %24f μs.\n",             runs - 1, rest_avg);
            printf("Average timing of GEQP3 for %d runs: %48f μs.\n",                                   runs - 1, geqp3_avg);
            printf("Average timing of GEQRF for %d runs: %48f μs.\n",                                   runs - 1, geqrf_avg);
            printf("Average timing of GEQR for %d runs: %49f μs.\n",                                    runs - 1, geqr_avg);
            printf("Average timing of TSQRP for %d runs: %48f μs.\n\n",                                 runs - 1, tsqrp_avg);

            printf("Result: CholQRCP is %f times faster than GEQP3.\n",        geqp3_avg / cholqrcp_avg);
            printf("With space allocation: %f.\n",                             geqp3_avg / (cholqrcp_avg + alloc_avg));
            printf("With Householder restoration: %f.\n",                      geqp3_avg / (cholqrcp_avg + rest_avg));
            printf("With space allocation + Householder restoration: %f.\n\n", geqp3_avg / (cholqrcp_avg + alloc_avg + rest_avg));

            printf("Result: CholQRCP is %f times faster than TSQRP.\n",        tsqrp_avg / cholqrcp_avg);
            printf("With space allocation: %f.\n",                             tsqrp_avg / (cholqrcp_avg + alloc_avg));
            printf("With Householder restoration: %f.\n",                      tsqrp_avg / (cholqrcp_avg + rest_avg));
            printf("With space allocation + Householder restoration: %f.\n\n", tsqrp_avg / (cholqrcp_avg + alloc_avg + rest_avg));

            printf("Result: CholQRCP is %f times faster than GEQRF.\n",        geqrf_avg / cholqrcp_avg);
            printf("With space allocation: %f.\n",                             geqrf_avg / (cholqrcp_avg + alloc_avg));
            printf("With Householder restoration: %f.\n",                      geqrf_avg / (cholqrcp_avg + rest_avg));
            printf("With space allocation + Householder restoration: %f.\n\n", geqrf_avg / (cholqrcp_avg + alloc_avg + rest_avg));

            printf("Result: CholQRCP is %f times faster than GEQR.\n",         geqr_avg / cholqrcp_avg);
            printf("With space allocation: %f.\n",                             geqr_avg / (cholqrcp_avg + alloc_avg));
            printf("With Householder restoration: %f.\n",                      geqr_avg / (cholqrcp_avg + rest_avg));
            printf("With space allocation + Householder restoration: %f.\n\n", geqr_avg / (cholqrcp_avg + alloc_avg + rest_avg));
            printf("\n/---------------------------------------QRCP MEAN TIMING END---------------------------------------/\n\n");
        }
    }
}

int main(int argc, char **argv){
    for (int nnz : {4})
    {
        test_speed_mean<double>(17, 17, 32, 32, 3, nnz, 32);
        //test_speed_mean<double>(17, 17, 2000, 2000, 3, nnz, 32);
        //test_speed_mean<double>(18, 18, 4000, 4000, 3, nnz, 32);
        //test_speed_mean<double>(19, 19, 8000, 8000, 3, nnz, 32);
    }
    return 0;
}
