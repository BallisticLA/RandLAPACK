#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <fstream>

/*
Note: this benchmark attempts to save files into a specific location.
If the required folder structure does not exist, the files will not be saved.
*/

template <typename T>
    static void test_CholQRCP1_approx_qual(int64_t m, int64_t n, int64_t k, int64_t d, int64_t nnz, T tol, const std::tuple<int, T, bool>& mat_type, uint32_t seed, int test_num, std::string path) {
        
        printf("/-----------------------------------------CholQRCP ACCURACY BENCHMARK START-----------------------------------------/\n");

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);

        std::vector<T> R;
        std::vector<int64_t> J(n, 0);

        // Random Gaussian test matrix
        RandLAPACK::util::gen_mat_type(m, n, A, k, seed, mat_type);

        std::vector<T> A_hat(size, 0.0);
        std::vector<T> A_1(size, 0.0);
        std::vector<T> A_2(size, 0.0);

        // For QRP
        std::vector<int64_t> J_1(n, 0.0);
        std::vector<T> tau_1(n, 0.0);
        std::vector<T> R_1(n * n, 0.0);

        std::copy(A.data(), A.data() + size, A_hat.data());
        std::copy(A.data(), A.data() + size, A_1.data());
        std::copy(A.data(), A.data() + size, A_2.data());

        RandLAPACK::CholQRCP<T> CholQRCP(false, false, seed, tol);
        CholQRCP.nnz = nnz;
        CholQRCP.num_threads = 32;

        CholQRCP.call(m, n, A, d, R, J);
        k = CholQRCP.rank;

        // Deterministic QRP, explicit extraction of R
        lapack::geqp3(m, n, A_1.data(), m, J_1.data(), tau_1.data());
        RandLAPACK::util::get_U(m, n, A_1, R_1);

        switch (test_num) {
            case 1: {
                // Clear the file if it exists
                std::ofstream ofs;
                ofs.open(path + "R_R_norm_ratio_m_" + std::to_string(m) 
                                + "_n_"           + std::to_string(n) 
                                + "_k_"           + std::to_string(k) 
                                + "_d_"           + std::to_string(d) 
                                + "_log10(tol)_"  + std::to_string(long(log10(tol)))
                                + "_mat_type_"    + std::to_string(std::get<0>(mat_type))
                                + "_cond_"        + std::to_string(long(std::get<1>(mat_type)))
                                + "_nnz_"         + std::to_string(nnz)
                                + "_OMP_threads_" + std::to_string(36) 
                                + ".dat", std::ofstream::out | std::ofstream::trunc);
                ofs.close();

                // Open a new file
                std::fstream file(path + "R_R_norm_ratio_m_" + std::to_string(m) 
                                            + "_n_"           + std::to_string(n) 
                                            + "_k_"           + std::to_string(k) 
                                            + "_d_"           + std::to_string(d) 
                                            + "_log10(tol)_"  + std::to_string(long(log10(tol)))
                                            + "_mat_type_"    + std::to_string(std::get<0>(mat_type))
                                            + "_cond_"        + std::to_string(long(std::get<1>(mat_type)))
                                            + "_nnz_"         + std::to_string(nnz)
                                            + "_OMP_threads_" + std::to_string(36) 
                                            + ".dat", std::fstream::app);

                std::vector<T> z_buf(k, 0.0);
                T* R_1_dat = R_1.data();
                T* z_buf_dat = z_buf.data();
                T* R_dat = R.data();
                
                // || R_qp3[k:, :] || / || R_cqrcp[k:, :] ||
                // This will have k - 2 data points
                for(int i = 1; i < n; ++i) {
                    for(int j = 0; j < n; ++j) {
                        blas::copy(i, &z_buf_dat[0], 1, &R_1_dat[n * j], 1);
                        blas::copy(i, &z_buf_dat[0], 1, &R_dat[k * j], 1);
                    }

                    T norm_geqp3 = lapack::lange(Norm::Fro, n, n, R_1.data(), n);
                    T norm_cholqrcp = lapack::lange(Norm::Fro, k, n, R.data(), k);

                    file << norm_geqp3 / norm_cholqrcp << "\n";
                }
                break;
            }
            case 2: {
                // For SVD
                std::vector<T> s(n, 0.0);
                std::vector<T> U(m * n, 0.0);
                std::vector<T> VT(n * n, 0.0);

                // Deterministic SVD
                lapack::gesdd(Job::SomeVec, m, n, A_2.data(), m, s.data(), U.data(), m, VT.data(), n);

                // Diagonal  buffers
                std::vector<T> r(n, 0.0);
                std::vector<T> r_1(n, 0.0);

                RandLAPACK::util::extract_diag(k, n, k, R, r);
                RandLAPACK::util::extract_diag(n, n, n, R_1, r_1);

                // Clear the file if it exists
                std::ofstream ofs;
                ofs.open(path + "r_s_ratio_m_"    + std::to_string(m) 
                                + "_n_"           + std::to_string(n) 
                                + "_k_"           + std::to_string(k) 
                                + "_d_"           + std::to_string(d) 
                                + "_log10(tol)_"  + std::to_string(long(log10(tol)))
                                + "_mat_type_"    + std::to_string(std::get<0>(mat_type))
                                + "_cond_"        + std::to_string(long(std::get<1>(mat_type)))
                                + "_nnz_"         + std::to_string(nnz)
                                + "_OMP_threads_" + std::to_string(36) 
                                + ".dat", std::ofstream::out | std::ofstream::trunc);
                ofs.close();

                // Open a new file
                std::fstream file(path + "r_s_ratio_m_"    + std::to_string(m) 
                                        + "_n_"           + std::to_string(n) 
                                        + "_k_"           + std::to_string(k) 
                                        + "_d_"           + std::to_string(d) 
                                        + "_log10(tol)_"  + std::to_string(long(log10(tol)))
                                        + "_mat_type_"    + std::to_string(std::get<0>(mat_type))
                                        + "_cond_"        + std::to_string(long(std::get<1>(mat_type)))
                                        + "_nnz_"         + std::to_string(nnz)
                                        + "_OMP_threads_" + std::to_string(36) 
                                        + ".dat", std::fstream::app);

                for(int i = 0; i < n; ++i) {
                    file << std::abs(r[i] / s[i]) << "  " << std::abs(r_1[i] / s[i]) << "\n";
                }
                break;
            }
        }

        printf("\n/-----------------------------------------CholQRCP ACCURACY BENCHMARK STOP-----------------------------------------/\n\n");
    }



int main() {
    // Run with env OMP_NUM_THREADS=36 numactl --interleave all ./filename
    // Large condition number may not work for a small matrix
    test_CholQRCP1_approx_qual<double>(131072, 2000, 2000, 10000, 4, std::pow(std::numeric_limits<double>::epsilon(), 0.5265), std::make_tuple(0, 1e10, false), 1, 1, "../testing/RandLAPACK-Testing/test_benchmark/QR/accuracy/raw_data/");
    //test_CholQRCP1_approx_qual<double>(131072, 2000, 2000, 2000, 1, std::pow(std::numeric_limits<double>::epsilon(), 0.5265), std::make_tuple(0, 1e10, false), 1, 2, "../../../testing/RandLAPACK-Testing/test_benchmark/QR/accuracy/raw_data/");
    return 0;
}
