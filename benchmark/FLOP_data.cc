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
#include <iterator>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using std::string;
using std::vector;

template <typename T>
static void 
compute_and_log(
           string test_type,
           int64_t rows, 
           int64_t cols,
           string d_multiplier,
           string k_multiplier, 
           string log10tol,
           string mat_type, 
           string cond,
           string nnz,
           string runs,
           string num_threads,
           T cholqrcp_time, 
           T geqp3_time, 
           T geqr_time, 
           T tsqrp_time, 
           T geqrf_time, 
           T chol_full_time,
           T geqp3_full_time,
           T geqr_full_time,
           T tsqrp_full_time,
           T geqrf_full_time)
{

    T geqrf_gflop = ((2 * rows * std::pow(cols, 2)) - (2 / (3 * std::pow(cols, 3))) + (3 * rows * cols) - std::pow(cols, 2) + (14 / (3 * rows))) / 1e+9;

    T system_gflops  = geqrf_gflop   / (geqrf_time/ 1e+6);

    printf("SYSTEM GFLOPS %f\n", system_gflops);

    T cholqrcp_gflop = system_gflops * cholqrcp_time;
    T geqp3_gflop    = system_gflops * geqp3_time;
    T geqr_gflop     = system_gflops * geqr_time;
    T tsqrp_gflop    = system_gflops * tsqrp_time;

    std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_FLOPS_" + test_type 
                                                                                          + "_m_"            + std::to_string(rows) 
                                                                                          + "_d_multiplier_" + d_multiplier
                                                                                          + "_k_multiplier_" + k_multiplier
                                                                                          + "_log10(tol)_"   + log10tol
                                                                                          + "_mat_type_"     + mat_type
                                                                                          + "_cond_"         + cond
                                                                                          + "_nnz_"          + nnz
                                                                                          + "_runs_per_sz_"  + runs
                                                                                          + "_OMP_threads_"  + num_threads 
                                                                                          + ".dat", std::fstream::app);
    file << cholqrcp_gflop   << "  " 
         << geqp3_gflop      << "  " 
         << geqr_gflop       << "  "
         << tsqrp_gflop      << "  " 
         << geqrf_gflop      << "\n";

}

template <typename T>
static void 
process_dat()
{
    vector<string> test_type    = {"Best"};
    vector<string> rows         = {"131072"};
    vector<string> d_multiplier = {"1.000000"};
    vector<string> k_multiplier = {"1.000000"};
    vector<string> log10tol     = {"-12"};
    vector<string> mat_type     = {"6"};
    vector<string> cond         = {"0"};
    vector<string> nnz          = {"1"};
    vector<string> runs         = {"5"};
    vector<string> num_threads  = {"36"};

    for (int i = 0; i < test_type.size(); ++i)
    {
        for (int j = 0; j < rows.size(); ++j)
        {
            for (int k = 0; k < d_multiplier.size(); ++k)
            {
                for (int l = 0; l < k_multiplier.size(); ++l)
                {
                    for (int m = 0; m < log10tol.size(); ++m)
                    {
                        for (int n = 0; n < mat_type.size(); ++n)
                        {
                            for (int o = 0; o < cond.size(); ++o)
                            {
                                for (int p = 0; p < nnz.size(); ++p)
                                {
                                    for (int q = 0; q < runs.size(); ++q)
                                    {
                                        for (int r = 0; r < num_threads.size(); ++r)
                                        {
                                            // Clear old flop file
                                            std::ofstream ofs;
                                            ofs.open("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_FLOPS_" + test_type[i] 
                                                                                                                                  + "_m_"            + rows[j] 
                                                                                                                                  + "_d_multiplier_" + d_multiplier[k]
                                                                                                                                  + "_k_multiplier_" + k_multiplier[l]
                                                                                                                                  + "_log10(tol)_"   + log10tol[m]
                                                                                                                                  + "_mat_type_"     + mat_type[n]
                                                                                                                                  + "_cond_"         + cond[o]
                                                                                                                                  + "_nnz_"          + nnz[p]
                                                                                                                                  + "_runs_per_sz_"  + runs[q]
                                                                                                                                  + "_OMP_threads_"  + num_threads[r]
                                                                                                                                  + ".dat", std::ofstream::out | std::ofstream::trunc);
                                            ofs.close();

                                            // Open data file
                                            std::fstream file("../../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/CholQRCP_comp_time_" + test_type[i] 
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
                                            
                                            int64_t numrows = stoi(rows[j]);
                                            int col_multiplier = 1;
                                            for( std::string l; getline(file, l);)
                                            {
                                                std::stringstream ss(l);
                                                std::istream_iterator<std::string> begin(ss);
                                                std::istream_iterator<std::string> end;
                                                std::vector<std::string> times_per_col_sz(begin, end);
                                                std::copy(times_per_col_sz.begin(), times_per_col_sz.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

                                                compute_and_log(test_type[i],
                                                                numrows, 
                                                                numrows / (256 / col_multiplier),
                                                                d_multiplier[k],
                                                                k_multiplier[k], 
                                                                log10tol[m],
                                                                mat_type[n], 
                                                                cond[o],
                                                                nnz[p],
                                                                runs[q],
                                                                num_threads[r],
                                                                stod(times_per_col_sz[0]), 
                                                                stod(times_per_col_sz[1]), 
                                                                stod(times_per_col_sz[2]),
                                                                stod(times_per_col_sz[3]), 
                                                                stod(times_per_col_sz[4]),
                                                                stod(times_per_col_sz[5]),
                                                                stod(times_per_col_sz[6]),
                                                                stod(times_per_col_sz[7]),
                                                                stod(times_per_col_sz[8]),
                                                                stod(times_per_col_sz[9]));

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