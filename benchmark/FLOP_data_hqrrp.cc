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
#include <iterator>

using namespace std::chrono;

template <typename T>
static void 
compute_and_log(
    const std::string& test_type,
    int64_t rows, 
    int64_t cols,
    const std::string& d_multiplier,
    const std::string& k_multiplier, 
    const std::string& log10tol,
    const std::string& block_sz,
    const std::string& mat_type, 
    const std::string& cond,
    const std::string& nnz,
    const std::string& runs,
    const std::string& omp_num_threads,
    const std::string& num_threads,
    T cholqrcp_time, 
    T cholqrcp_hqrrp_time)
{

    T geqrf_gflop = (2 * rows * std::pow(cols, 2) - (2 / 3)* std::pow(cols, 3) + rows * cols + std::pow(cols, 2) + (14 / 3) * cols) / 1e+9;

    printf("\n/----------------------------------------FLOP ITER START----------------------------------------/\n");

    // This version finds flop RATES, pretending that geqrf_gflop is the standard num flops
    T cholqrcp_flop_rate = geqrf_gflop / (cholqrcp_time / 1e+6);
    T cholqrcp_hqrrp_flop_rate = geqrf_gflop / (cholqrcp_hqrrp_time / 1e+6);

    printf("/-----------------------------------------FLOP ITER END-----------------------------------------/\n");
    
    std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/flops/raw_data/CholQRCP_HQRRP_FLOP_RATE_" + test_type 
                                                                                          + "_m_"              + std::to_string(rows) 
                                                                                          + "_d_multiplier_"   + d_multiplier
                                                                                          + "_k_multiplier_"   + k_multiplier
                                                                                          + "_log10(tol)_"     + log10tol
                                                                                          + "_hqrrp_block_sz_" + block_sz
                                                                                          + "_mat_type_"       + mat_type
                                                                                          + "_cond_"           + cond
                                                                                          + "_nnz_"            + nnz
                                                                                          + "_runs_per_sz_"    + runs
                                                                                          + "_OMP_threads_"    + omp_num_threads
                                                                                          + "_SASO_threads_"    + num_threads 
                                                                                          + ".dat", std::fstream::app);
    file << cholqrcp_flop_rate       << "  " 
         << cholqrcp_hqrrp_flop_rate << "\n";
}

template <typename T>
static void 
process_dat() {
    std::vector<std::string> test_type       = {"Best"};
    std::vector<std::string> rows            = {"131072"}; 
    std::vector<std::string> d_multiplier    = {"1.000000"};
    std::vector<std::string> k_multiplier    = {"1.000000"};
    std::vector<std::string> log10tol        = {"-11"};
    std::vector<std::string> mat_type        = {"6"};
    std::vector<std::string> cond            = {"0"};
    std::vector<std::string> nnz             = {"1"};
    std::vector<std::string> runs            = {"5"};
    std::vector<std::string> num_threads     = {"36"};
    std::vector<std::string> block_sz        = {"16"};
    std::vector<std::string> omp_num_threads = {"36"};
    printf("HERE\n");

    for (int i = 0; i < (int) test_type.size(); ++i) {
        for (int j = 0; j < (int) rows.size(); ++j) {
            for (int k = 0; k < (int) d_multiplier.size(); ++k) {
                for (int l = 0; l < (int) k_multiplier.size(); ++l) {
                    for (int m = 0; m < (int) log10tol.size(); ++m) {
                        for (int n = 0; n < (int) mat_type.size(); ++n) {
                            for (int o = 0; o < (int) cond.size(); ++o) {
                                for (int p = 0; p < (int) nnz.size(); ++p) {
                                    for (int q = 0; q < (int) runs.size(); ++q) {
                                        for (int r = 0; r < (int) num_threads.size(); ++r) {
                                            for (int s = 0; s < (int) block_sz.size(); ++s) {
                                                for (int t = 0; t < (int) omp_num_threads.size(); ++t) {
                                                    printf("HERE\n");
                                                    // Clear old flop file   
                                                    std::ofstream ofs;
                                                    ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/flops/raw_data/CholQRCP_HQRRP_FLOP_RATE_"  + test_type[i] 
                                                                                                                                        + "_m_"              + rows[j] 
                                                                                                                                        + "_d_multiplier_"   + d_multiplier[k]
                                                                                                                                        + "_k_multiplier_"   + k_multiplier[l]
                                                                                                                                        + "_log10(tol)_"     + log10tol[m]
                                                                                                                                        + "_hqrrp_block_sz_" + block_sz[s]
                                                                                                                                        + "_mat_type_"       + mat_type[n]
                                                                                                                                        + "_cond_"           + cond[o]
                                                                                                                                        + "_nnz_"            + nnz[p]
                                                                                                                                        + "_runs_per_sz_"    + runs[q]
                                                                                                                                        + "_OMP_threads_"    + omp_num_threads[t]
                                                                                                                                        + "_SASO_threads_"   + num_threads[r]
                                                                                                                                        + ".dat", std::ofstream::out | std::ofstream::trunc);
                                                    ofs.close();
                                                    
                                                    //CholQRCP_vs_HQRRP_time_Best_m_131072_d_multiplier_1.000000_k_multiplier_1.000000_log10(tol)_-11_hqrrp_block_sz_256_mat_type_6_cond_0_nnz_1_runs_per_sz_5_OMP_threads_36_SASO_threads_36
                                                    
                                                    // Open data file
                                                    std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/apply_Q_to_large/CholQRCP_vs_HQRRP_time_" + test_type[i] 
                                                                                                                                        + "_m_"              + rows[j] 
                                                                                                                                        + "_d_multiplier_"   + d_multiplier[k]
                                                                                                                                        + "_k_multiplier_"   + k_multiplier[l]
                                                                                                                                        + "_log10(tol)_"     + log10tol[m]
                                                                                                                                        + "_hqrrp_block_sz_" + block_sz[s]
                                                                                                                                        + "_mat_type_"       + mat_type[n]
                                                                                                                                        + "_cond_"           + cond[o]
                                                                                                                                        + "_nnz_"            + nnz[p]
                                                                                                                                        + "_runs_per_sz_"    + runs[q]
                                                                                                                                        + "_OMP_threads_"    + omp_num_threads[t]
                                                                                                                                        + "_SASO_threads_"   + num_threads[r]
                                                                                                                                        + ".dat");
                                                    
                                                    int64_t numrows = stoi(rows[j]);
                                                    int col_multiplier = 1;
                                                    int start_col_ratio = 256;
                                                    for( std::string l; getline(file, l);)
                                                    {
                                                        printf("IM HERE\n");
                                                        std::stringstream ss(l);
                                                        std::istream_iterator<std::string> begin(ss);
                                                        std::istream_iterator<std::string> end;
                                                        std::vector<std::string> times_per_col_sz(begin, end);
                                                        //std::copy(times_per_col_sz.begin(), times_per_col_sz.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

                                                        compute_and_log(test_type[i],
                                                            numrows, 
                                                            numrows / (start_col_ratio / col_multiplier),
                                                            //numrows / (128 / col_multiplier),
                                                            d_multiplier[k],
                                                            k_multiplier[k], 
                                                            log10tol[m],
                                                            block_sz[s],
                                                            mat_type[n], 
                                                            cond[o],
                                                            nnz[p],
                                                            runs[q],
                                                            omp_num_threads[t],
                                                            num_threads[r],
                                                            stod(times_per_col_sz[0]), 
                                                            stod(times_per_col_sz[1]));
                                                        col_multiplier *= 2;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(){ 
    process_dat<double>();
    return 0;
}