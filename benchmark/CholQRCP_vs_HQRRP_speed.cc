/*
Note: this benchmark attempts to save files into a specific location.
If the required folder structure does not exist, the files will not be saved.
*/
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
#include <fstream>

using namespace std::chrono;
using namespace RandLAPACK::comps::util;
using namespace RandLAPACK::comps::orth;
using namespace RandLAPACK::drivers::cholqrcp;

template <typename T>
static void 
log_info(int64_t rows, 
           int64_t cols,
           T d_multiplier,
           T k_multiplier, 
           T tol,
           int64_t block_sz,
           int num_omp_threads,
           int64_t nnz, 
           int64_t num_threads,
           const std::tuple<int, T, bool>& mat_type,
           T cholqrcp_time, 
           T cholqrcp_hqrrp_time, 
           T chol_full_time,
           T cholqrcp_hqrrp_full_time,
           const std::string& test_type,
           int runs) {
    // Save the output into .dat file
    std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/apply_Q_to_large/CholQRCP_vs_HQRRP_time_" + test_type 
                                                                                         + "_m_"              + std::to_string(rows) 
                                                                                         + "_d_multiplier_"   + std::to_string(d_multiplier)
                                                                                         + "_k_multiplier_"   + std::to_string(k_multiplier)
                                                                                         + "_log10(tol)_"     + std::to_string(long(log10(tol)))
                                                                                         + "_hqrrp_block_sz_" + std::to_string(block_sz)
                                                                                         + "_mat_type_"       + std::to_string(std::get<0>(mat_type))
                                                                                         + "_cond_"           + std::to_string(long(std::get<1>(mat_type)))
                                                                                         + "_nnz_"            + std::to_string(nnz)
                                                                                         + "_runs_per_sz_"    + std::to_string(runs)
                                                                                         + "_OMP_threads_"    + std::to_string(num_omp_threads) 
                                                                                         + "_SASO_threads_"   + std::to_string(num_threads)
                                                                                         + ".dat", std::fstream::app);
    file << cholqrcp_time       << "  " 
         << cholqrcp_hqrrp_time << "  "  
         << chol_full_time      << "  " 
         << cholqrcp_hqrrp_full_time << "\n";
}

template <typename T>
static std::vector<long> 
test_speed_helper(int64_t m, 
                  int64_t n, 
                  int64_t d, 
                  int64_t k, 
                  T tol, 
                  int64_t block_sz,
                  int64_t nnz, 
                  int64_t num_threads, 
                  const std::tuple<int, T, bool>& mat_type, 
                  uint32_t seed) {
    using namespace blas;
    using namespace lapack;

    int64_t size = m * n;
    int64_t b_dim = n;
    std::vector<T> A_1(size, 0.0);
    std::vector<T> B_1(b_dim * m, 0.0);
    std::vector<T> R_1;
    std::vector<int64_t> J_1;
    std::vector<T> Res_1;
    upsize(b_dim * n, Res_1);

    std::vector<T> R_2;
    std::vector<int64_t> J_2;
    std::vector<T> Res_2;
    upsize(b_dim * n, Res_2);
    
    // Generate random matrix
    gen_mat_type<T>(m, n, A_1, k, seed, mat_type);

    // Generate random matrix that we will apply Q to
    gen_mat_type<T>(b_dim, m, B_1, b_dim, seed + 1, mat_type);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // CholQRCP constructor
    bool log_times = true;
    CholQRCP<T> CholQRCP_basic(false, log_times, seed, tol);
    CholQRCP_basic.nnz = nnz;
    CholQRCP_basic.num_threads = num_threads;

    // CholQRCP constructor
    CholQRCP<T> CholQRCP_HQRRP(false, log_times, seed, tol);
    
    CholQRCP_HQRRP.nnz = nnz;
    CholQRCP_HQRRP.num_threads = num_threads;
    CholQRCP_HQRRP.no_hqrrp = 0;
    CholQRCP_HQRRP.nb_alg = block_sz;

    //-TEST POINT 2 BEGIN-------------------------------------------------------------------------------------------------------------------------------------------/
    
    // Pre-allocation for CholQRCP
    auto start_alloc2 = high_resolution_clock::now();
    if(log_times) {
        (CholQRCP_HQRRP.times).resize(10);
    }
    upsize(d * n, (CholQRCP_HQRRP.A_hat));
    upsize(n, (CholQRCP_HQRRP.A_hat));
    J_2.resize(n);
    upsize(n * n, (CholQRCP_HQRRP.R_sp));
    upsize(n * n, R_2);
    auto stop_alloc2 = high_resolution_clock::now();
    long dur_alloc2 = duration_cast<microseconds>(stop_alloc2 - start_alloc2).count();
    
    // CholQRCP
    auto start_cholqrcp_hqrrp = high_resolution_clock::now();
    CholQRCP_HQRRP.call(m, n, A_1, d, R_2, J_2);
    auto stop_cholqrcp_hqrrp = high_resolution_clock::now();
    long dur_cholqrcp_hqrrp = duration_cast<microseconds>(stop_cholqrcp_hqrrp - start_cholqrcp_hqrrp).count();

    // Apply Q_1
    auto start_appl2 = high_resolution_clock::now();
    gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_dim, n, m, 1.0, B_1.data(), b_dim, A_1.data(), m, 0.0, Res_2.data(), b_dim);
    auto stop_appl2 = high_resolution_clock::now();
    long dur_appl2 = duration_cast<microseconds>(stop_appl2 - start_appl2).count();



    //-TEST POINT 1 BEGIN-------------------------------------------------------------------------------------------------------------------------------------------/
    gen_mat_type<T>(m, n, A_1, k, seed, mat_type);
    gen_mat_type<T>(b_dim, m, B_1, b_dim, seed + 1, mat_type);
    // Pre-allocation for CholQRCP
    auto start_alloc1 = high_resolution_clock::now();
    if(log_times) {
        (CholQRCP_basic.times).resize(10);
    }
    upsize(d * n, (CholQRCP_basic.A_hat));
    upsize(n, (CholQRCP_basic.A_hat));
    J_1.resize(n);
    upsize(n * n, (CholQRCP_basic.R_sp));
    upsize(n * n, R_1);
    auto stop_alloc1 = high_resolution_clock::now();
    long dur_alloc1 = duration_cast<microseconds>(stop_alloc1 - start_alloc1).count();
    
    // CholQRCP
    auto start_cholqrcp = high_resolution_clock::now();
    CholQRCP_basic.call(m, n, A_1, d, R_1, J_1);
    auto stop_cholqrcp = high_resolution_clock::now();
    long dur_cholqrcp = duration_cast<microseconds>(stop_cholqrcp - start_cholqrcp).count();

    // Apply Q_1
    auto start_appl1 = high_resolution_clock::now();
    gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_dim, n, m, 1.0, B_1.data(), b_dim, A_1.data(), m, 0.0, Res_1.data(), b_dim);
    auto stop_appl1 = high_resolution_clock::now();
    long dur_appl1 = duration_cast<microseconds>(stop_appl1 - start_appl1).count();

    //-TEST POINT 1 END---------------------------------------------------------------------------------------------------------------------------------------------/

    std::vector<long> res{dur_alloc1, dur_cholqrcp,       dur_appl1, 
                          dur_alloc2, dur_cholqrcp_hqrrp, dur_appl2}; 

    return res;
}

template <typename T>
static void 
test_speed(int r_pow, 
           int r_pow_max, 
           int col, 
           int col_max, 
           int runs, 
           int block_sz,
           int num_omp_threads,
           int nnz, 
           int num_threads, 
           T tol, 
           T k_multiplier, 
           T d_multiplier, 
           const std::tuple<int, T, bool> & mat_type) {

    printf("\n/-----------------------------------------SPEED TEST START-----------------------------------------/\n");
    // We are now filling 3 types of data - best, mean and raw
    
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf) {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/apply_Q_to_large/CholQRCP_vs_HQRRP_time_Best_m_"
                                                                                                              + std::to_string(rows) 
                                                                                         + "_d_multiplier_"   + std::to_string(d_multiplier)
                                                                                         + "_k_multiplier_"   + std::to_string(k_multiplier)
                                                                                         + "_log10(tol)_"     + std::to_string(long(log10(tol)))
                                                                                         + "_hqrrp_block_sz_" + std::to_string(block_sz)
                                                                                         + "_mat_type_"       + std::to_string(std::get<0>(mat_type))
                                                                                         + "_cond_"           + std::to_string(long(std::get<1>(mat_type)))
                                                                                         + "_nnz_"            + std::to_string(nnz)
                                                                                         + "_runs_per_sz_"    + std::to_string(runs)
                                                                                         + "_OMP_threads_"    + std::to_string(num_omp_threads) 
                                                                                         + "_SASO_threads_"   + std::to_string(num_threads)
                                                                                         + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();

        ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/apply_Q_to_large/CholQRCP_vs_HQRRP_time_Mean_m_"
                                                                                                              + std::to_string(rows) 
                                                                                         + "_d_multiplier_"   + std::to_string(d_multiplier)
                                                                                         + "_k_multiplier_"   + std::to_string(k_multiplier)
                                                                                         + "_log10(tol)_"     + std::to_string(long(log10(tol)))
                                                                                         + "_hqrrp_block_sz_" + std::to_string(block_sz)
                                                                                         + "_mat_type_"       + std::to_string(std::get<0>(mat_type))
                                                                                         + "_cond_"           + std::to_string(long(std::get<1>(mat_type)))
                                                                                         + "_nnz_"            + std::to_string(nnz)
                                                                                         + "_runs_per_sz_"    + std::to_string(runs)
                                                                                         + "_OMP_threads_"    + std::to_string(num_omp_threads) 
                                                                                         + "_SASO_threads_"   + std::to_string(num_threads)
                                                                                         + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();

        ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/apply_Q_to_large/CholQRCP_vs_HQRRP_time_Raw_m_"
                                                                                                              + std::to_string(rows) 
                                                                                         + "_d_multiplier_"   + std::to_string(d_multiplier)
                                                                                         + "_k_multiplier_"   + std::to_string(k_multiplier)
                                                                                         + "_log10(tol)_"     + std::to_string(long(log10(tol)))
                                                                                         + "_hqrrp_block_sz_" + std::to_string(block_sz)
                                                                                         + "_mat_type_"       + std::to_string(std::get<0>(mat_type))
                                                                                         + "_cond_"           + std::to_string(long(std::get<1>(mat_type)))
                                                                                         + "_nnz_"            + std::to_string(nnz)
                                                                                         + "_runs_per_sz_"    + std::to_string(runs)
                                                                                         + "_OMP_threads_"    + std::to_string(num_omp_threads) 
                                                                                         + "_SASO_threads_"   + std::to_string(num_threads) 
                                                                                         + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }
    
    int64_t rows = 0;
    for(; r_pow <= r_pow_max; ++r_pow) {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols *= 2) {
            std::vector<long> res;

            long t_alloc1         = 0;
            long t_cholqrcp       = 0;
            long t_appl1          = 0;
            long t_alloc2         = 0;
            long t_cholqrcp_hqrrp = 0;
            long t_appl2          = 0;

            T alloc1_best         = 0;
            T cholqrcp_best       = 0;
            T appl1_best          = 0;
            T alloc2_best         = 0;
            T cholqrcp_hqrrp_best = 0;
            T appl2_best          = 0;

            T alloc1_mean         = 0;
            T cholqrcp_mean       = 0;
            T appl1_mean          = 0;
            T alloc2_mean         = 0;
            T cholqrcp_hqrrp_mean = 0;
            T appl2_mean          = 0;

            for(int i = 0; i < runs; ++i) {
                res = test_speed_helper<T>(rows, cols, d_multiplier * cols, k_multiplier * cols, tol, block_sz, nnz, num_threads, mat_type, i);

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0) {
                    t_alloc1         += res[0];
                    t_cholqrcp       += res[1];
                    t_appl1          += res[2];
                    t_alloc2         += res[3];
                    t_cholqrcp_hqrrp += res[4];
                    t_appl2          += res[5];
                    
                    // Log every run in the raw data file
                    std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/apply_Q_to_large/CholQRCP_vs_HQRRP_time_Raw_m_" 
                                                                                                              + std::to_string(rows) 
                                                                                         + "_d_multiplier_"   + std::to_string(d_multiplier)
                                                                                         + "_k_multiplier_"   + std::to_string(k_multiplier)
                                                                                         + "_log10(tol)_"     + std::to_string(long(log10(tol)))
                                                                                         + "_hqrrp_block_sz_" + std::to_string(block_sz)
                                                                                         + "_mat_type_"       + std::to_string(std::get<0>(mat_type))
                                                                                         + "_cond_"           + std::to_string(long(std::get<1>(mat_type)))
                                                                                         + "_nnz_"            + std::to_string(nnz)
                                                                                         + "_runs_per_sz_"    + std::to_string(runs)
                                                                                         + "_OMP_threads_"    + std::to_string(num_omp_threads) 
                                                                                         + "_SASO_threads_"   + std::to_string(num_threads)
                                                                                                      + ".dat", std::fstream::app);
                    file << res[0]  << "  " 
                         << res[1]  << "  " 
                         << res[2]  << "  "
                         << res[3]  << "  " 
                         << res[4]  << "  "  
                         << res[5]  << "\n";
                    
                    // For best timing
                    if(alloc1_best > res[0] || alloc1_best == 0) {
                        alloc1_best = res[0];
                    }
                    if(cholqrcp_best > res[1] || cholqrcp_best == 0) {
                        cholqrcp_best = res[1];
                    }
                    if(appl1_best > res[2] || appl1_best == 0) {
                        appl1_best = res[2];
                    }
                    if(alloc2_best > res[3] || alloc2_best == 0) {
                        alloc2_best = res[3];
                    }
                    if(cholqrcp_hqrrp_best > res[4] || cholqrcp_hqrrp_best == 0) {
                        cholqrcp_hqrrp_best = res[4];
                    }
                    if(appl2_best > res[5] || appl2_best == 0) {
                        appl2_best = res[5];
                    }
                }
            }

            // For mean timing
            alloc1_mean          = (T)t_alloc1         / (T)(runs - 1);
            cholqrcp_mean        = (T)t_cholqrcp       / (T)(runs - 1);
            appl1_mean           = (T)t_appl1          / (T)(runs - 1);
            alloc2_mean          = (T)t_alloc2         / (T)(runs - 1);
            cholqrcp_hqrrp_mean  = (T)t_cholqrcp_hqrrp / (T)(runs - 1);
            appl2_mean           = (T)t_appl2          / (T)(runs - 1);
            
            log_info(rows, cols, d_multiplier, k_multiplier, tol, block_sz, num_omp_threads, nnz, num_threads, mat_type, 
                     cholqrcp_best,
                     cholqrcp_hqrrp_best,
                     cholqrcp_best       + alloc1_best + appl1_best, 
                     cholqrcp_hqrrp_best + alloc2_best + appl2_best, 
                     "Best", runs);

            log_info(rows, cols, d_multiplier, k_multiplier, tol, block_sz, num_omp_threads, nnz, num_threads, mat_type, 
                     cholqrcp_mean,
                     cholqrcp_hqrrp_mean,
                     cholqrcp_mean       + alloc1_mean + appl1_mean, 
                     cholqrcp_hqrrp_mean + alloc2_mean + appl2_mean, 
                     "Mean", runs);

            printf("Done with size %ld by %ld\n", rows, cols);
        }
    }
    printf("\n/-----------------------------------------SPEED TEST STOP-----------------------------------------/\n\n");
}

int main(){
    // Run with env OMP_NUM_THREADS=36 numactl --interleave all ./filename 

    for(int num_omp_threads = 36; num_omp_threads <= 36; ++num_omp_threads)
    {
        test_speed<double>(17, 17, 512, 8192, 5, 256, num_omp_threads, 1, 36, std::pow(std::numeric_limits<double>::epsilon(), 0.75), 1.0, 1.0, std::make_tuple(6, 0, false));
    }
    return 0;
}