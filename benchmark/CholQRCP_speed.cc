/*
Compares speed of CholQRCP to other pivoted and unpivoted QR factorizations
*/
#include<stdio.h>
#include<string.h>

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
using std::string;

template <typename T>
static void 
print_info(int64_t rows, 
           int64_t cols,
           T d_multiplier, 
           int64_t nnz, 
           int64_t num_threads, 
           T alloc_time, 
           T cholqrcp_time, 
           T rest_time, 
           T geqp3_time, 
           T geqrf_time, 
           T geqr_time, 
           T tsqrp_time, 
           string test_type,
           int runs)
{
    const char * test_type_print = test_type.c_str();

    printf("\n/-------------------------------------QR TIMING INFO BEGIN-------------------------------------/\n");
            printf("\nMatrix size: %ld by %ld.\n", rows, cols);
            printf("Embedding size: %f.\n", d_multiplier * cols);
            printf("Number of nonzeros per column in SASO: %ld\n", nnz);
            printf("Number of threads used in SASO application: %ld\n", num_threads);

            printf("\n%s timing of workspace pre-allocation for CholQRCP for %d runs: %25.2f μs.\n", test_type_print, runs - 1, alloc_time);
            printf("%s timing of CholQRCP for %d runs: %54.2f μs.\n",                                test_type_print, runs - 1, cholqrcp_time);
            printf("%s timing Householder vector restoration for %d runs: %35.2f μs.\n",             test_type_print, runs - 1, rest_time);
            printf("%s timing of GEQP3 for %d runs: %57.2f μs.\n",                                   test_type_print, runs - 1, geqp3_time);
            printf("%s timing of GEQRF for %d runs: %57.2f μs.\n",                                   test_type_print, runs - 1, geqrf_time);
            printf("%s timing of GEQR for %d runs: %58.2f μs.\n",                                    test_type_print, runs - 1, geqr_time);
            printf("%s timing of TSQRP for %d runs: %57.2f μs.\n\n",                                 test_type_print, runs - 1, tsqrp_time);

            /*CholQRCP vs GEQP3*/
            if(cholqrcp_time < geqp3_time)
            {
                printf("Result: CholQRCP is %33.2f times faster than GEQP3.\n",                               geqp3_time / cholqrcp_time);
            }
            else
            {
                printf("Result: CholQRCP is %33.2f times slower than GEQP3.\n",                               cholqrcp_time / geqp3_time);
            }
            if((cholqrcp_time + alloc_time) < geqp3_time)
            {
            printf("With space allocation: %30.2f times faster than GEQP3.\n",                                geqp3_time / (cholqrcp_time + alloc_time));
            }
            else
            {
                printf("With space allocation: %30.2f times slower than GEQP3.\n",                            (cholqrcp_time + alloc_time) / geqp3_time);
            }
            if((cholqrcp_time + rest_time) < geqp3_time)
            {
                printf("With Householder restoration: %23.2f times faster than GEQP3.\n",                     geqp3_time / (cholqrcp_time + rest_time));
            }
            else
            {
                printf("With Householder restoration: %23.2f times slower than GEQP3.\n",                     (cholqrcp_time + rest_time) / geqp3_time);
            }
            if((cholqrcp_time + alloc_time + rest_time) < geqp3_time)
            {
                printf("With space allocation + Householder restoration: %3.2f times faster than GEQP3.\n\n", geqp3_time / (cholqrcp_time + alloc_time + rest_time));
            }
            else
            {
                printf("With space allocation + Householder restoration: %3.2f times slower than GEQP3.\n\n", (cholqrcp_time + alloc_time + rest_time) / geqp3_time);
            }

            /*CholQRCP vs TSQRP*/
            if(cholqrcp_time < tsqrp_time)
            {
                printf("Result: CholQRCP is %33.2f times faster than TSQRP.\n",                               tsqrp_time / cholqrcp_time);
            }
            else
            {
                printf("Result: CholQRCP is %33.2f times slower than TSQRP.\n",                               cholqrcp_time / tsqrp_time);
            }
            if((cholqrcp_time + alloc_time) < tsqrp_time)
            {
            printf("With space allocation: %30.2f times faster than TSQRP.\n",                                tsqrp_time / (cholqrcp_time + alloc_time));
            }
            else
            {
                printf("With space allocation: %30.2f times slower than TSQRP.\n",                            (cholqrcp_time + alloc_time) / tsqrp_time);
            }
            if((cholqrcp_time + rest_time) < tsqrp_time)
            {
                printf("With Householder restoration: %23.2f times faster than TSQRP.\n",                     tsqrp_time / (cholqrcp_time + rest_time));
            }
            else
            {
                printf("With Householder restoration: %23.2f times slower than TSQRP.\n",                     (cholqrcp_time + rest_time) / tsqrp_time);
            }
            if((cholqrcp_time + alloc_time + rest_time) < tsqrp_time)
            {
                printf("With space allocation + Householder restoration: %3.2f times faster than TSQRP.\n\n", tsqrp_time / (cholqrcp_time + alloc_time + rest_time));
            }
            else
            {
                printf("With space allocation + Householder restoration: %3.2f times slower than TSQRP.\n\n", (cholqrcp_time + alloc_time + rest_time) / tsqrp_time);
            }

            /*CholQRCP vs GEQRF*/
            if(cholqrcp_time < geqrf_time)
            {
                printf("Result: CholQRCP is %33.2f times faster than GEQRF.\n",                               geqrf_time / cholqrcp_time);
            }
            else
            {
                printf("Result: CholQRCP is %33.2f times slower than GEQRF.\n",                               cholqrcp_time / geqrf_time);
            }
            if((cholqrcp_time + alloc_time) < geqrf_time)
            {
            printf("With space allocation: %30.2f times faster than GEQRF.\n",                                geqrf_time / (cholqrcp_time + alloc_time));
            }
            else
            {
                printf("With space allocation: %30.2f times slower than GEQRF.\n",                            (cholqrcp_time + alloc_time) / geqrf_time);
            }
            if((cholqrcp_time + rest_time) < geqrf_time)
            {
                printf("With Householder restoration: %23.2f times faster than GEQRF.\n",                     geqrf_time / (cholqrcp_time + rest_time));
            }
            else
            {
                printf("With Householder restoration: %23.2f times slower than GEQRF.\n",                     (cholqrcp_time + rest_time) / geqrf_time);
            }
            if((cholqrcp_time + alloc_time + rest_time) < geqrf_time)
            {
                printf("With space allocation + Householder restoration: %3.2f times faster than GEQRF.\n\n", geqrf_time / (cholqrcp_time + alloc_time + rest_time));
            }
            else
            {
                printf("With space allocation + Householder restoration: %3.2f times slower than GEQRF.\n\n", (cholqrcp_time + alloc_time + rest_time) / geqrf_time);
            }

            /*CholQRCP vs GEQR*/
            if(cholqrcp_time < geqr_time)
            {
                printf("Result: CholQRCP is %33.2f times faster than GEQR.\n",                               geqr_time / cholqrcp_time);
            }
            else
            {
                printf("Result: CholQRCP is %33.2f times slower than GEQR.\n",                               cholqrcp_time / geqr_time);
            }
            if((cholqrcp_time + alloc_time) < geqr_time)
            {
            printf("With space allocation: %30.2f times faster than GEQR.\n",                                geqr_time / (cholqrcp_time + alloc_time));
            }
            else
            {
                printf("With space allocation: %30.2f times slower than GEQR.\n",                            (cholqrcp_time + alloc_time) / geqr_time);
            }
            if((cholqrcp_time + rest_time) < geqr_time)
            {
                printf("With Householder restoration: %23.2f times faster than GEQR.\n",                     geqr_time / (cholqrcp_time + rest_time));
            }
            else
            {
                printf("With Householder restoration: %23.2f times slower than GEQR.\n",                     (cholqrcp_time + rest_time) / geqr_time);
            }
            if((cholqrcp_time + alloc_time + rest_time) < geqr_time)
            {
                printf("With space allocation + Householder restoration: %3.2f times faster than GEQR.\n\n", geqr_time / (cholqrcp_time + alloc_time + rest_time));
            }
            else
            {
                printf("With space allocation + Householder restoration: %3.2f times slower than GEQR.\n\n", (cholqrcp_time + alloc_time + rest_time) / geqr_time);
            }

            printf("\n/---------------------------------------QR TIMING INFO END---------------------------------------/\n\n");

}



template <typename T>
static std::vector<long> 
test_speed_helper(int64_t m, 
                  int64_t n, 
                  int64_t d, 
                  int64_t k, 
                  T tol, 
                  int64_t nnz, 
                  int64_t num_threads, 
                  std::tuple<int, T, bool> mat_type, 
                  uint32_t seed) 
{
    
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

    std::vector<T> R_3(n * n, 0);
    std::vector<T> t_3(5, 0);
    std::vector<T> tau_3(n, 0);
    std::vector<int64_t> J_3(n, 0);

    std::vector<T> tau_4(n, 0);

    // Generate random matrix
    gen_mat_type<T>(m, n, A_1, k, seed, mat_type);

    // Make copies
    std::copy(A_1.data(), A_1.data() + size, A_2.data());
    std::copy(A_1.data(), A_1.data() + size, A_3.data());
    std::copy(A_1.data(), A_1.data() + size, A_4.data());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // CholQRCP constructor
    bool log_times = true;
    CholQRCP<T> CholQRCP(false, log_times, seed, tol, use_cholqrcp1);
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

        printf("\nSASO generation and application takes %2.2f%% of runtime.\n", 100 * ((double) (CholQRCP.times)[0] / (double) (CholQRCP.times)[9]));
        printf("QRCP takes %32.2f%% of runtime.\n",                            100 * ((double) (CholQRCP.times)[1] / (double) (CholQRCP.times)[9]));
        printf("Rank revealing takes %22.2f%% of runtime.\n",                  100 * ((double) (CholQRCP.times)[2] / (double) (CholQRCP.times)[9]));
        printf("Cholqrcp takes %28.2f%% of runtime.\n",                        100 * ((double) (CholQRCP.times)[3] / (double) (CholQRCP.times)[9]));
        printf("Modifying matrix (pivoting) A %13.2f%% of runtime.\n",         100 * ((double) (CholQRCP.times)[4] / (double) (CholQRCP.times)[9]));
        printf("Modifying matrix (trsm) A %17.2f%% of runtime.\n",             100 * ((double) (CholQRCP.times)[5] / (double) (CholQRCP.times)[9]));
        printf("Copying takes %29.2f%% of runtime.\n",                         100 * ((double) (CholQRCP.times)[6] / (double) (CholQRCP.times)[9]));
        printf("Resizing takes %28.2f%% of runtime.\n",                        100 * ((double) (CholQRCP.times)[7] / (double) (CholQRCP.times)[9]));
        printf("Everything else takes %21.2f%% of runtime.\n",                 100 * ((double) (CholQRCP.times)[8] / (double) (CholQRCP.times)[9]));
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
    // We are not timing the pre-allocation of R, as it expected to take very small time
    get_U(m, n, A_3, R_3);
    geqp3(n, n, R_3.data(), n, J_3.data(), tau_3.data());

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
test_speed(int r_pow, 
           int r_pow_max, 
           int col, 
           int col_max, 
           int runs, 
           int nnz, 
           int num_threads, 
           T tol, 
           T k_multiplier, 
           T d_multiplier, 
           std::tuple<int, T, bool> mat_type, 
           string test_type)
{
    printf("\n/-----------------------------------------SPEED TEST START-----------------------------------------/\n");
    // Clear all files
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf)
    {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open("../../../testing/test_benchmark/QR/speed/raw_data/CholQRCP_comp_time_" + test_type 
                                                                      + "_m_"            + std::to_string(rows) 
                                                                      + "_d_multiplier_" + std::to_string(d_multiplier)
                                                                      + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                      + "_log10(tol)_"   + std::to_string(int(log10(tol)))
                                                                      + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                      + "_cond_"         + std::to_string(int(std::get<1>(mat_type)))
                                                                      + "_nnz_"          + std::to_string(nnz)
                                                                      + "_runs_per_sz_"  + std::to_string(runs)
                                                                      + "_OMP_threads_"  + std::to_string(num_threads) 
                                                                      + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    int64_t rows = 0;
    int64_t cols = 0;

    T alloc_total    = 0;
    T cholqrcp_total = 0;
    T rest_total     = 0;
    T geqp3_total    = 0;
    T geqrf_total    = 0;
    T geqr_total     = 0;
    T tsqrp_total    = 0;

    for(; r_pow <= r_pow_max; ++r_pow)
    {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols *= 2)
        {
            std::vector<long> res;
            long t_alloc    = 0;
            long t_cholqrcp = 0;
            long t_rest     = 0;
            long t_geqp3    = 0;
            long t_geqrf    = 0;
            long t_geqr     = 0;
            long t_tsqrp    = 0;

            for(int i = 0; i < runs; ++i)
            {
                res = test_speed_helper<T>(rows, cols, d_multiplier * cols, k_multiplier * cols, tol, nnz, num_threads, mat_type, i);

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0)
                {
                    if(!test_type.compare("Mean"))
                    {
                        t_alloc    += res[0];
                        t_cholqrcp += res[1];
                        t_rest     += res[2];
                        t_geqp3    += res[3];
                        t_geqrf    += res[4];
                        t_geqr     += res[5];
                        t_tsqrp    += res[6];
                    }
                    else
                    {
                        if(alloc_total > res[0] || alloc_total == 0)
                        {
                            alloc_total = res[0];
                        }
                        if(cholqrcp_total > res[1] || cholqrcp_total == 0)
                        {
                            cholqrcp_total = res[1];
                        }
                        if(rest_total > res[2] || rest_total == 0)
                        {
                            rest_total = res[2];
                        }
                        if(geqp3_total > res[3] || geqp3_total == 0)
                        {
                            geqp3_total = res[3];
                        }
                        if(geqrf_total > res[4] || geqrf_total == 0)
                        {
                            geqrf_total = res[4];
                        }
                        if(geqr_total > res[5] || geqr_total == 0)
                        {
                            geqr_total = res[5];
                        }
                        if(tsqrp_total > res[6] || tsqrp_total == 0)
                        {
                            tsqrp_total = res[6];
                        }
                    }
                }
            }

            if(!test_type.compare("Mean"))
            {
                alloc_total    = (T)t_alloc    / (T)(runs - 1);
                cholqrcp_total = (T)t_cholqrcp / (T)(runs - 1);
                rest_total     = (T)t_rest     / (T)(runs - 1);
                geqp3_total    = (T)t_geqp3    / (T)(runs - 1);
                geqrf_total    = (T)t_geqrf    / (T)(runs - 1);
                geqr_total     = (T)t_geqr     / (T)(runs - 1);
                tsqrp_total    = (T)t_tsqrp    / (T)(runs - 1);
            }

            // Save the output into .dat file
            std::fstream file("../../../testing/test_benchmark/QR/speed/raw_data/CholQRCP_comp_time_" + test_type 
                                                                                   + "_m_"            + std::to_string(rows) 
                                                                                   + "_d_multiplier_" + std::to_string(d_multiplier)
                                                                                   + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                   + "_log10(tol)_"   + std::to_string(int(log10(tol)))
                                                                                   + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                   + "_cond_"         + std::to_string(int(std::get<1>(mat_type)))
                                                                                   + "_nnz_"          + std::to_string(nnz)
                                                                                   + "_runs_per_sz_"  + std::to_string(runs)
                                                                                   + "_OMP_threads_"  + std::to_string(num_threads) 
                                                                                   + ".dat", std::fstream::app);
            file << cholqrcp_total                            << "  " 
                 << geqp3_total                               << "  " 
                 << tsqrp_total                               << "  " 
                 << geqrf_total                               << "  " 
                 << geqr_total                                << "  " 
                 << cholqrcp_total + alloc_total              << "  " 
                 << cholqrcp_total + rest_total               << "  "
                 << cholqrcp_total + alloc_total + rest_total << "\n";


            print_info(rows, cols, d_multiplier, nnz, num_threads, alloc_total, cholqrcp_total, rest_total, geqp3_total, geqrf_total, geqr_total, tsqrp_total, test_type, runs);
        }
    }
    printf("\n/-----------------------------------------SPEED TEST STOP-----------------------------------------/\n\n");
}

int main(int argc, char **argv){

    // Run with env OMP_NUM_THREADS=36 numactl --interleave all ./filename
    test_speed<double>(14, 14, 64, 1024, 5, 1, 32, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), "Mean"); 
    test_speed<double>(14, 14, 64, 1024, 5, 1, 32, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), "Best"); 

    test_speed<double>(16, 16, 256, 4096, 5, 1, 32, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), "Mean"); 
    test_speed<double>(16, 16, 256, 4096, 5, 1, 32, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), "Best");

    test_speed<double>(17, 17, 512, 8192, 5, 1, 32, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), "Mean");
    test_speed<double>(17, 17, 512, 8192, 5, 1, 32, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), "Best"); 

    return 0;
}
