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

using namespace std::chrono;

template <typename T>
static void 
process_dat() {
    std::vector<std::string> test_type    = {"Best"};
    std::vector<std::string> rows         = {"131072"};
    std::vector<std::string> d_multiplier = {"1.000000"};
    std::vector<std::string> k_multiplier = {"1.000000"};
    std::vector<std::string> log10tol     = {"-12"};
    std::vector<std::string> mat_type     = {"6"};
    std::vector<std::string> cond         = {"0"};
    std::vector<std::string> nnz          = {"1"};
    std::vector<std::string> runs         = {"5"};
    std::vector<std::string> num_threads  = {"36"};
    std::string path = "../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/";

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

                                            // Clear old file
                                            std::ofstream ofs;
                                            ofs.open(path + "CholQRCP_inner_time_processed_" + test_type[i] 
                                                                         + "_m_"             + rows[j] 
                                                                         + "_d_multiplier_"  + d_multiplier[k]
                                                                         + "_k_multiplier_"  + k_multiplier[l]
                                                                         + "_log10(tol)_"    + log10tol[m]
                                                                         + "_mat_type_"      + mat_type[n]
                                                                         + "_cond_"          + cond[o]
                                                                         + "_nnz_"           + nnz[p]
                                                                         + "_runs_per_sz_"   + runs[q]
                                                                         + "_OMP_threads_"   + num_threads[r] 
                                                                         + ".dat", std::ofstream::out | std::ofstream::trunc);
                                            ofs.close();
                                            // Open data file
                                            std::fstream file(path + "CholQRCP_inner_time_" + test_type[i] 
                                                                         + "_m_"            + rows[j] 
                                                                         + "_d_multiplier_" + d_multiplier[k]
                                                                         + "_k_multiplier_" + k_multiplier[l]
                                                                         + "_log10(tol)_"   + log10tol[m]
                                                                         + "_mat_type_"     + mat_type[n]
                                                                         + "_cond_"         + cond[o]
                                                                         + "_nnz_"          + nnz[p]
                                                                         + "_runs_per_sz_"  + runs[q]
                                                                         + "_OMP_threads_"  + num_threads[r]
                                                                         + ".dat");
                                            int col_multiplier = 1;
                                            for( std::string l1; getline(file, l1);) {
                                                std::stringstream ss(l1);
                                                std::istream_iterator<std::string> begin(ss);
                                                std::istream_iterator<std::string> end;
                                                std::vector<std::string> times_per_col_sz(begin, end);

                                                std::fstream file(path + "CholQRCP_inner_time_processed_" + test_type[i] 
                                                                                      + "_m_"             + rows[j] 
                                                                                      + "_d_multiplier_"  + d_multiplier[k]
                                                                                      + "_k_multiplier_"  + k_multiplier[l]
                                                                                      + "_log10(tol)_"    + log10tol[m]
                                                                                      + "_mat_type_"      + mat_type[n]
                                                                                      + "_cond_"          + cond[o]
                                                                                      + "_nnz_"           + nnz[p]
                                                                                      + "_runs_per_sz_"   + runs[q]
                                                                                      + "_OMP_threads_"   + num_threads[r] 
                                                                                      + ".dat", std::fstream::app);
                                                file << 100 * (stod(times_per_col_sz[0]) / stod(times_per_col_sz[9])) << "  " 
                                                     << 100 * (stod(times_per_col_sz[1]) / stod(times_per_col_sz[9])) << "  "
                                                     << 100 * (stod(times_per_col_sz[3]) / stod(times_per_col_sz[9])) << "  "
                                                     << 100 * (stod(times_per_col_sz[4]) / stod(times_per_col_sz[9])) << "  "
                                                     << 100 * (stod(times_per_col_sz[5]) / stod(times_per_col_sz[9])) << "  "
                                                     << 100 * ((stod(times_per_col_sz[9]) - (stod(times_per_col_sz[0]) + stod(times_per_col_sz[1]) + stod(times_per_col_sz[3]) + stod(times_per_col_sz[4]) + stod(times_per_col_sz[5]))) / stod(times_per_col_sz[9])) << "\n";

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

int main(){ 
    process_dat<double>();
    return 0;
}
