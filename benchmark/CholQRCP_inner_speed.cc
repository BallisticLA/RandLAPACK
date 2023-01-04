/*This is only concerned with what's INSIDE of cholqrcp*/

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
log_info(int64_t rows, 
           int64_t cols,
           T d_multiplier,
           T k_multiplier, 
           T tol,
           int64_t nnz, 
           int64_t num_threads,
           std::tuple<int, T, bool> mat_type,
           T saso_time, 
           T qrcp_time, 
           T rank_reveal_time, 
           T cholqr_time, 
           T A_pivot_time, 
           T A_trsm_time,
           T copy_time,
           T resize_time,
           T other_time,
           T total_time,
           string test_type,
           int runs)
{
    // Save the output into .dat file
    std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_inner_time_" + test_type 
                                                                                              + "_m_"            + std::to_string(rows) 
                                                                                              + "_d_multiplier_" + std::to_string(d_multiplier)
                                                                                              + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                              + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                                                                              + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                              + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                                                                              + "_nnz_"          + std::to_string(nnz)
                                                                                              + "_runs_per_sz_"  + std::to_string(runs)
                                                                                              + "_OMP_threads_"  + std::to_string(num_threads) 
                                                                                              + ".dat", std::fstream::app);
    file << saso_time   << "  " 
         << qrcp_time      << "  " 
         << rank_reveal_time       << "  "
         << cholqr_time      << "  " 
         << A_pivot_time      << "  "  
         << A_trsm_time  << "  " 
         << copy_time << "  " 
         << resize_time  << "  "
         << other_time << "  " 
         << total_time << "\n";
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

    int64_t size  = m * n;

    std::vector<T>       A_1(size, 0.0);
    std::vector<T>       R_1;
    std::vector<int64_t> J_1;

    // Generate random matrix
    gen_mat_type<T>(m, n, A_1, k, seed, mat_type);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // CholQRCP constructor
    CholQRCP<T> CholQRCP(false, true, seed, tol, use_cholqrcp1);
    CholQRCP.nnz         = nnz;
    CholQRCP.num_threads = num_threads;

    //-TEST POINT 1 BEGIN-------------------------------------------------------------------------------------------------------------------------------------------/
    
    upsize(d * n, (CholQRCP.A_hat));
    upsize(n, (CholQRCP.A_hat));
    J_1.resize(n);
    upsize(n * n, (CholQRCP.R_sp));
    upsize(n * n, R_1);
    upsize(n * n, (CholQRCP.R_buf));

    // CholQRCP
    CholQRCP.call(m, n, A_1, d, R_1, J_1);
    
    // CholQRCP verbose info print
    printf("\n\n/------------CholQRCP1 TIMING RESULTS BEGIN------------/\n");
    printf("SASO time: %33ld μs,\n",                    (CholQRCP.times)[0]);
    printf("QRCP time: %33ld μs,\n",                    (CholQRCP.times)[1]);
    printf("Rank revealing time: %23ld μs,\n",          (CholQRCP.times)[2]);
    printf("CholQR time: %31ld μs,\n",                  (CholQRCP.times)[3]);
    printf("A modification pivoting time: %14ld μs,\n", (CholQRCP.times)[4]);
    printf("A modification TRSM time: %18ld μs,\n",     (CholQRCP.times)[5]);
    printf("Copying time: %30ld μs,\n",                 (CholQRCP.times)[6]);
    printf("Resizing time: %29ld μs,\n",                (CholQRCP.times)[7]);
    printf("Other routines time: %23ld μs,\n",          (CholQRCP.times)[8]);
    printf("Total time: %32ld μs.\n",                   (CholQRCP.times)[9]);

    printf("\nSASO generation and application takes %2.2f%% of runtime.\n", 100 * ((double) (CholQRCP.times)[0] / (double) (CholQRCP.times)[9]));
    printf("QRCP takes %32.2f%% of runtime.\n",                            100 * ((double) (CholQRCP.times)[1] / (double) (CholQRCP.times)[9]));
    printf("Rank revealing takes %22.2f%% of runtime.\n",                  100 * ((double) (CholQRCP.times)[2] / (double) (CholQRCP.times)[9]));
    printf("Cholqr takes %30.2f%% of runtime.\n",                          100 * ((double) (CholQRCP.times)[3] / (double) (CholQRCP.times)[9]));
    printf("Modifying matrix (pivoting) A %13.2f%% of runtime.\n",         100 * ((double) (CholQRCP.times)[4] / (double) (CholQRCP.times)[9]));
    printf("Modifying matrix (trsm) A %17.2f%% of runtime.\n",             100 * ((double) (CholQRCP.times)[5] / (double) (CholQRCP.times)[9]));
    printf("Copying takes %29.2f%% of runtime.\n",                         100 * ((double) (CholQRCP.times)[6] / (double) (CholQRCP.times)[9]));
    printf("Resizing takes %28.2f%% of runtime.\n",                        100 * ((double) (CholQRCP.times)[7] / (double) (CholQRCP.times)[9]));
    printf("Everything else takes %21.2f%% of runtime.\n",                 100 * ((double) (CholQRCP.times)[8] / (double) (CholQRCP.times)[9]));
    printf("/-------------CholQRCP1 TIMING RESULTS END-------------/\n\n");

    return CholQRCP.times;
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
           std::tuple<int, T, bool> mat_type)
{
    printf("\n/-----------------------------------------SPEED TEST START-----------------------------------------/\n");
    // We are now filling 3 types of data - best, mean and raw
    
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf)
    {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_inner_time_Best_m_"
                                                                                                            + std::to_string(rows) 
                                                                                         + "_d_multiplier_" + std::to_string(d_multiplier)
                                                                                         + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                         + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                                                                         + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                         + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                                                                         + "_nnz_"          + std::to_string(nnz)
                                                                                         + "_runs_per_sz_"  + std::to_string(runs)
                                                                                         + "_OMP_threads_"  + std::to_string(num_threads) 
                                                                                         + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();

        ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_inner_time_Mean_m_"
                                                                                                            + std::to_string(rows) 
                                                                                         + "_d_multiplier_" + std::to_string(d_multiplier)
                                                                                         + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                         + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                                                                         + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                         + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                                                                         + "_nnz_"          + std::to_string(nnz)
                                                                                         + "_runs_per_sz_"  + std::to_string(runs)
                                                                                         + "_OMP_threads_"  + std::to_string(num_threads) 
                                                                                         + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();

        ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_inner_time_Raw_m_"
                                                                                                            + std::to_string(rows) 
                                                                                         + "_d_multiplier_" + std::to_string(d_multiplier)
                                                                                         + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                         + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                                                                         + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                         + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                                                                         + "_nnz_"          + std::to_string(nnz)
                                                                                         + "_runs_per_sz_"  + std::to_string(runs)
                                                                                         + "_OMP_threads_"  + std::to_string(num_threads) 
                                                                                         + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }
    
    int64_t rows = 0;
    int64_t cols = 0;

    for(; r_pow <= r_pow_max; ++r_pow)
    {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols *= 2)
        {
            std::vector<long> res;

            long t_saso        = 0;
            long t_qrcp        = 0;
            long t_rank_reveal = 0;
            long t_cholqr      = 0;
            long t_A_pivot     = 0;
            long t_A_trsm      = 0;
            long t_copy        = 0;
            long t_resize      = 0;
            long t_other       = 0;
            long t_total       = 0;

            T saso_best        = 0;
            T qrcp_best        = 0;
            T rank_reveal_best = 0;
            T cholqr_best      = 0;
            T A_pivot_best     = 0;
            T A_trsm_best      = 0;
            T copy_best        = 0;
            T resize_best      = 0;
            T other_best       = 0;
            T total_best       = 0;

            T saso_mean        = 0;
            T qrcp_mean        = 0;
            T rank_reveal_mean = 0;
            T cholqr_mean      = 0;
            T A_pivot_mean     = 0;
            T A_trsm_mean      = 0;
            T copy_mean        = 0;
            T resize_mean      = 0;
            T other_mean       = 0;
            T total_mean       = 0;

            for(int i = 0; i < runs; ++i)
            {
                res = test_speed_helper<T>(rows, cols, d_multiplier * cols, k_multiplier * cols, tol, nnz, num_threads, mat_type, i);

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0)
                {
                    t_saso        += res[0];
                    t_qrcp        += res[1];
                    t_rank_reveal += res[2];
                    t_cholqr      += res[3];
                    t_A_pivot     += res[4];
                    t_A_trsm      += res[5];
                    t_copy        += res[6];
                    t_resize      += res[7];
                    t_other       += res[8];
                    t_total       += res[9];
                    
                    // Log every run in the raw data file
                    std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_inner_time_Raw_m_" 
                                                                                                                         + std::to_string(rows) 
                                                                                                      + "_d_multiplier_" + std::to_string(d_multiplier)
                                                                                                      + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                                      + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                                                                                      + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                                      + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                                                                                      + "_nnz_"          + std::to_string(nnz)
                                                                                                      + "_runs_per_sz_"  + std::to_string(runs)
                                                                                                      + "_OMP_threads_"  + std::to_string(num_threads) 
                                                                                                      + ".dat", std::fstream::app);
                    file << res[0]  << "  " 
                         << res[1]  << "  " 
                         << res[2]  << "  "
                         << res[3]  << "  " 
                         << res[4]  << "  "  
                         << res[5]  << "  " 
                         << res[6]  << "  " 
                         << res[7]  << "  "
                         << res[8]  << "  " 
                         << res[9]  << "\n";
                    
                    // For best timing
                    if(total_best > res[9] || total_best == 0)
                    {
                        saso_best        = res[0];
                        qrcp_best        = res[1];
                        qrcp_best        = res[1];
                        rank_reveal_best = res[2];
                        cholqr_best      = res[3];
                        A_pivot_best     = res[4];
                        A_trsm_best      = res[5];
                        copy_best        = res[6];
                        resize_best      = res[7];
                        resize_best      = res[7];
                        other_best       = res[8];
                        total_best       = res[9];
                    }
                }
            }

            // For mean timing
            saso_mean        = (T)t_saso        / (T)(runs - 1);
            qrcp_mean        = (T)t_qrcp        / (T)(runs - 1);
            rank_reveal_mean = (T)t_rank_reveal / (T)(runs - 1);
            cholqr_mean      = (T)t_cholqr      / (T)(runs - 1);
            A_pivot_mean     = (T)t_A_pivot     / (T)(runs - 1);
            A_trsm_mean      = (T)t_A_trsm      / (T)(runs - 1);
            copy_mean        = (T)t_copy        / (T)(runs - 1);
            resize_mean      = (T)t_resize      / (T)(runs - 1);
            other_mean       = (T)t_other       / (T)(runs - 1);
            total_mean       = (T)t_total       / (T)(runs - 1);
            
            log_info(rows, cols, d_multiplier, k_multiplier, tol, nnz, num_threads, mat_type, 
                     saso_best,
                     qrcp_best,
                     rank_reveal_best,
                     cholqr_best,
                     A_pivot_best,  
                     A_trsm_best, 
                     copy_best, 
                     resize_best,
                     other_best, 
                     total_best,
                     "Best", runs);

            log_info(rows, cols, d_multiplier, k_multiplier, tol, nnz, num_threads, mat_type, 
                     saso_mean,
                     qrcp_mean,
                     rank_reveal_mean,
                     cholqr_mean,
                     A_pivot_mean,  
                     A_trsm_mean, 
                     copy_mean, 
                     resize_mean,
                     other_mean, 
                     total_mean,
                     "Mean", runs);
        }
    }
    printf("\n/-----------------------------------------SPEED TEST STOP-----------------------------------------/\n\n");
}

int main(int argc, char **argv){

    // Run with env OMP_NUM_THREADS=36 numactl --interleave all ./filename 
    
    //test_speed<double>(14, 14, 64, 1024, 5, 1, 36, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false)); 

    //test_speed<double>(16, 16, 256, 4096, 5, 1, 36, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false)); 

    test_speed<double>(17, 17, 32, 16384, 5, 1, 36, std::pow(1.0e-16, 0.75), 1.0, 2.0, std::make_tuple(6, 0, false));

    //test_speed<double>(18, 18, 2048, 8192, 5, 1, 36, std::pow(1.0e-16, 0.75), 1.0, 1.0, std::make_tuple(6, 0, false));

    return 0;
}