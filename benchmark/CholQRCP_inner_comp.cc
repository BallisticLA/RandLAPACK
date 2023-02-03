/*
Note: this benchmark attempts to save files into a specific location.
If the required folder structure does not exist, the files will not be saved.
*/
/*
Compares speeds of inner components within CholQRCP
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
process_dat() {
    std::vector<std::string> test_type    = {"Best"};
    std::vector<std::string> rows         = {"131072"}; // {"262144"};
    std::vector<std::string> d_multiplier = {"1.000000"};
    std::vector<std::string> k_multiplier = {"1.000000"};
    std::vector<std::string> log10tol     = {"-12"};
    std::vector<std::string> mat_type     = {"6"};
    std::vector<std::string> cond         = {"0"};
    std::vector<std::string> nnz          = {"1"};
    std::vector<std::string> runs         = {"5"};
    std::vector<std::string> num_threads  = {"36"};

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
                                            ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQR_vs_GEQP3_" + test_type[i] 
                                                                                          + "_m_"             + rows[j] 
                                                                                          + "_d_multiplier1_" + d_multiplier[k]
                                                                                          + "_d_multiplier2_" + std::to_string(2.000000)
                                                                                          + "_d_multiplier3_" + std::to_string(4.000000)
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
                                            std::fstream file1("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_inner_time_" + test_type[i] 
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
                                            std::fstream file2("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_inner_time_" + test_type[i] 
                                                                                                                                  + "_m_"            + rows[j] 
                                                                                                                                  + "_d_multiplier_" + std::to_string(2.000000)
                                                                                                                                  + "_k_multiplier_" + k_multiplier[l]
                                                                                                                                  + "_log10(tol)_"   + log10tol[m]
                                                                                                                                  + "_mat_type_"     + mat_type[n]
                                                                                                                                  + "_cond_"         + cond[o]
                                                                                                                                  + "_nnz_"          + nnz[p]
                                                                                                                                  + "_runs_per_sz_"  + runs[q]
                                                                                                                                  + "_OMP_threads_"  + num_threads[r]
                                                                                                                                  + ".dat");
                                            std::fstream file3("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_inner_time_" + test_type[i] 
                                                                                                                                  + "_m_"            + rows[j] 
                                                                                                                                  + "_d_multiplier_" + std::to_string(4.000000)
                                                                                                                                  + "_k_multiplier_" + k_multiplier[l]
                                                                                                                                  + "_log10(tol)_"   + log10tol[m]
                                                                                                                                  + "_mat_type_"     + mat_type[n]
                                                                                                                                  + "_cond_"         + cond[o]
                                                                                                                                  + "_nnz_"          + nnz[p]
                                                                                                                                  + "_runs_per_sz_"  + runs[q]
                                                                                                                                  + "_OMP_threads_"  + num_threads[r]
                                                                                                                                  + ".dat");
                                            
                                            int col_multiplier = 1;
                                            for( std::string l1; getline(file1, l1);) {
                                                std::stringstream ss1(l1);
                                                std::istream_iterator<std::string> begin1(ss1);
                                                std::istream_iterator<std::string> end1;
                                                std::vector<std::string> times_per_col_sz1(begin1, end1);

                                                std::string l2;
                                                getline(file2, l2);
                                                std::stringstream ss2(l2);
                                                std::istream_iterator<std::string> begin2(ss2);
                                                std::istream_iterator<std::string> end2;
                                                std::vector<std::string> times_per_col_sz2(begin2, end2);

                                                std::string l3;
                                                getline(file3, l3);
                                                std::stringstream ss3(l3);
                                                std::istream_iterator<std::string> begin3(ss3);
                                                std::istream_iterator<std::string> end3;
                                                std::vector<std::string> times_per_col_sz3(begin3, end3);

                                                std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQR_vs_GEQP3_" + test_type[i] 
                                                                                          + "_m_"             + rows[j] 
                                                                                          + "_d_multiplier1_" + d_multiplier[k]
                                                                                          + "_d_multiplier2_" + std::to_string(2.000000)
                                                                                          + "_d_multiplier3_" + std::to_string(4.000000)
                                                                                          + "_k_multiplier_"  + k_multiplier[l]
                                                                                          + "_log10(tol)_"    + log10tol[m]
                                                                                          + "_mat_type_"      + mat_type[n]
                                                                                          + "_cond_"          + cond[o]
                                                                                          + "_nnz_"           + nnz[p]
                                                                                          + "_runs_per_sz_"   + runs[q]
                                                                                          + "_OMP_threads_"   + num_threads[r] 
                                                                                          + ".dat", std::fstream::app);
                                                file << stod(times_per_col_sz1[1]) / (stod(times_per_col_sz1[1]) + stod(times_per_col_sz1[3])) << "  " 
                                                     << stod(times_per_col_sz2[1]) / (stod(times_per_col_sz2[1]) + stod(times_per_col_sz2[3])) << "  "
                                                     << stod(times_per_col_sz3[1]) / (stod(times_per_col_sz3[1]) + stod(times_per_col_sz3[3])) << "\n";

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