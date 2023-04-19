#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <fstream>
#include <iterator>
/*
Note: this benchmark attempts to save files into a specific location.
If the required folder structure does not exist, the files will not be saved.
*/
/*
Compares speed of CQRRPT to other pivoted and unpivoted QR factorizations
*/

using namespace std::chrono;

template <typename T>
static void 
compute_and_log(
    int64_t rows, 
    int64_t cols,
    T cqrrpt_time, 
    T geqp3_time, 
    T geqr_time, 
    T tsqrp_time, 
    T geqrf_time,
    T scholqr_time,
    std::string path_out,
    std::string file_params)
{
    T geqrf_gflop = (2 * rows * std::pow(cols, 2) - (2 / 3)* std::pow(cols, 3) + rows * cols + std::pow(cols, 2) + (14 / 3) * cols) / 1e+9;

    printf("\n/----------------------------------------FLOP ITER START----------------------------------------/\n");

    // This version finds flop RATES, pretending that geqrf_gflop is the standard num flops
    T geqrf_flop_rate   = geqrf_gflop / (geqrf_time / 1e+6);
    T cqrrpt_flop_rate  = geqrf_gflop / (cqrrpt_time / 1e+6);
    T geqp3_flop_rate   = geqrf_gflop / (geqp3_time / 1e+6);
    T geqr_flop_rate    = geqrf_gflop / (geqr_time / 1e+6);
    T tsqrp_flop_rate   = geqrf_gflop / (tsqrp_time / 1e+6);
    T scholqr_flop_rate = geqrf_gflop / (scholqr_time / 1e+6);

    printf("CQRRPT GFLOP RATE %12.1f\n",   cqrrpt_flop_rate);
    printf("GEQP3 GFLOP RATE    %12.1f\n",   geqp3_flop_rate);
    printf("GEQR GFLOP RATE     %12.1f\n",   geqr_flop_rate);
    printf("TSQRP GFLOP RATE    %12.1f\n",   tsqrp_flop_rate);
    printf("GEQRF GFLOP RATE    %12.1f\n",   geqrf_flop_rate);
    printf("SCHOLQR GFLOP RATE    %12.1f\n",   scholqr_flop_rate);

    printf("/-----------------------------------------FLOP ITER END-----------------------------------------/\n");
    
    std::fstream file(path_out + "CQRRPT_FLOP_RATE_" + file_params + ".dat", std::fstream::app);
    file << cqrrpt_flop_rate  << "  " 
         << geqp3_flop_rate   << "  " 
         << geqr_flop_rate    << "  "
         << tsqrp_flop_rate   << "  " 
         << geqrf_flop_rate   << "  "
         << scholqr_flop_rate << "\n";
}

template <typename T>
static int 
process_dat() {
    std::vector<std::string> test_type    = {"Best"};
    std::vector<std::string> rows         = {"131072"}; // {"262144"};
    std::vector<std::string> d_multiplier = {"1.000000"};
    std::vector<std::string> k_multiplier = {"1.000000"};
    std::vector<std::string> log10tol     = {"-11"};
    std::vector<std::string> mat_type     = {"6"};
    std::vector<std::string> cond         = {"0"};
    std::vector<std::string> nnz          = {"1"};
    std::vector<std::string> runs         = {"5"};
    std::vector<std::string> num_threads  = {"36"};
    std::vector<std::string> apply_to_large  = {"0"};
    std::string path_in = "../../testing/RandLAPACK-benchmarking/QR/speed/raw_data/";
    std::string path_out = "../../testing/RandLAPACK-benchmarking/QR/flops/raw_data/";

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
                                            
                                            std::string file_params =    test_type[i] 
                                                    + "_m_"            + rows[j] 
                                                    + "_d_multiplier_" + d_multiplier[k]
                                                    + "_k_multiplier_" + k_multiplier[l]
                                                    + "_log10(tol)_"   + log10tol[m]
                                                    + "_mat_type_"     + mat_type[n]
                                                    + "_cond_"         + cond[o]
                                                    + "_nnz_"          + nnz[p]
                                                    + "_runs_per_sz_"  + runs[q]
                                                    + "_OMP_threads_"  + num_threads[r];
                                            
                                            // Clear old flop file
                                            std::ofstream ofs;
                                            ofs.open(path_out + "CQRRPT_FLOP_RATE_"  + file_params + ".dat", std::ofstream::out | std::ofstream::trunc);
                                            ofs.close();
                                            
                                            // Open data file
                                            std::string filename_in = path_in + "QR_time_"   + file_params + "_apply_to_large_" + apply_to_large[0] + ".dat";
                                            std::fstream file(filename_in);
                                            if(!file)
                                            {
                                                printf("Looking for filename:\n%s\n", filename_in.c_str());
                                                return 1;
                                            }

                                            int64_t numrows = stoi(rows[j]);
                                            int col_multiplier = 1;
                                            // depends on numrows
                                            int start_col_ratio = 256;
                                            for( std::string l; getline(file, l);)
                                            {
                                                std::stringstream ss(l);
                                                std::istream_iterator<std::string> begin(ss);
                                                std::istream_iterator<std::string> end;
                                                std::vector<std::string> times_per_col_sz(begin, end);
                                                //std::copy(times_per_col_sz.begin(), times_per_col_sz.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

                                                compute_and_log(
                                                    numrows, 
                                                    numrows / (start_col_ratio / col_multiplier),
                                                    stod(times_per_col_sz[0]), 
                                                    stod(times_per_col_sz[1]), 
                                                    stod(times_per_col_sz[2]),
                                                    stod(times_per_col_sz[3]), 
                                                    stod(times_per_col_sz[4]),
                                                    stod(times_per_col_sz[5]),
                                                    path_out,
                                                    file_params);

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
    return 0;
}

int main(){ 
    switch(process_dat<double>())
        {
        case 1:
            printf("\nTERMINATED VIA: input file not found.\n");
            break;
        case 0:
            printf("\nTERMINATED VIA: normal termination.\n");
            break;
        }
    return 0;
}
