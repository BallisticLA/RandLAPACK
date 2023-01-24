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
using namespace std::chrono;

#include <fstream>
#include <iterator>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using std::string;
using std::vector;

template <typename T>
static void 
process_dat()
{
    vector<string> test_type    = {"Best"};
    vector<string> rows         = {"131072"}; // {"262144"};
    vector<string> d_multiplier = {"1.000000"};
    vector<string> k_multiplier = {"1.000000"};
    vector<string> log10tol     = {"-12"};
    vector<string> mat_type     = {"6"};
    vector<string> cond         = {"0"};
    vector<string> nnz          = {"1"};
    vector<string> runs         = {"5"};
    vector<string> num_threads  = {"36"};

    for (int i = 0; i < test_type.size(); ++i) {
        for (int j = 0; j < rows.size(); ++j) {
            for (int k = 0; k < d_multiplier.size(); ++k) {
                for (int l = 0; l < k_multiplier.size(); ++l) {
                    for (int m = 0; m < log10tol.size(); ++m) {
                        for (int n = 0; n < mat_type.size(); ++n) {
                            for (int o = 0; o < cond.size(); ++o) {
                                for (int p = 0; p < nnz.size(); ++p) {
                                    for (int q = 0; q < runs.size(); ++q) {
                                        for (int r = 0; r < num_threads.size(); ++r) {
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
                                            
                                            int64_t numrows = stoi(rows[j]);
                                            int col_multiplier = 1;
                                            int start_col_ratio = 256;
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



int main(int argc, char **argv){ 
    process_dat<double>();
    return 0;
}