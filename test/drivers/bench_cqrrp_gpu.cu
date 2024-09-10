#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>
#include <chrono>

// Use cuda kernels.
#ifndef USE_CUDA
#define USE_CUDA
#include "RandLAPACK/drivers/rl_cqrrp_gpu.hh"

class BenchCQRRP : public ::testing::TestWithParam<int64_t>
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct CQRRPBenchData {
        int64_t row;
        int64_t col;
        int64_t rank;
        
        std::vector<T> A;
        T* A_sk;
        // Buffers for the GPU data
        T* A_device;
        T* A_sk_device;
        T* tau_device;
        int64_t* J_device;

        T* R_device;
        T* D_device;

        CQRRPBenchData(int64_t m, int64_t n) :
        A(m * n, 0.0)
        {
            row = m;
            col = n;
            cudaMalloc(&A_device,    m * n * sizeof(T));
            cudaMalloc(&tau_device,  n *     sizeof(T));
            cudaMalloc(&J_device,    n *     sizeof(int64_t));
            cudaMalloc(&R_device,    n * n * sizeof(T));
            cudaMalloc(&D_device,    n *     sizeof(T));
        }

        ~CQRRPBenchData() {
            cudaFree(A_device);
            cudaFree(tau_device);
            cudaFree(J_device);
            cudaFree(R_device);
            cudaFree(D_device);
        }
    };

    template <typename T, typename RNG>
    static void data_regen(
                            RandLAPACK::gen::mat_gen_info<T> m_info, 
                            CQRRPBenchData<T> &all_data, 
                            RandBLAS::RNGState<RNG> &state) {

        auto state_const = state;
        auto m = m_info.rows;
        auto n = m_info.cols;

        RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state_const);
        cudaMemset(all_data.J_device, 0.0, n);
        cudaMemset(all_data.tau_device, 0.0, n);
    }

    template <typename T, typename RNG>
    static void bench_CQRRP(
        bool profile_runtime,
        bool run_qrf,
	bool cqrrp_use_qrf,
        RandLAPACK::gen::mat_gen_info<T> m_info,
        T tol,
        int64_t block_size,
        CQRRPBenchData<T> &all_data,
        RandBLAS::RNGState<RNG> state,
        std::string output_filename_breakdown,
        std::string output_filename_speed) {

	T d_factor = 1.0;
        auto m = all_data.row;
        auto n = all_data.col;
        auto state_const = state;
        int64_t d = d_factor * block_size;

        // Skethcing in an sampling regime
        cudaMalloc(&all_data.A_sk_device, d * n * sizeof(T));
        all_data.A_sk  = ( T * ) calloc( d * n, sizeof( T ) );
        T* S           = ( T * ) calloc( d * m, sizeof( T ) );
        RandBLAS::DenseDist D(d, m);
        RandBLAS::fill_dense(D, S, state_const).second;
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, S, d, all_data.A.data(), m, 0.0, all_data.A_sk, d);
        free(S);
        cudaMemcpy(all_data.A_sk_device, all_data.A_sk, d * n * sizeof(double), cudaMemcpyHostToDevice);
	
        RandLAPACK::CQRRP_blocked_GPU<double, r123::Philox4x32> CQRRP_GPU(profile_runtime, tol, block_size);
        CQRRP_GPU.use_qrf = cqrrp_use_qrf;
	    auto start = std::chrono::steady_clock::now();
        CQRRP_GPU.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);
	    auto stop = std::chrono::steady_clock::now();
	    auto diff_icqrrp = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        auto rank = CQRRP_GPU.rank;
        data_regen(m_info, all_data, state);
        cudaFree(all_data.A_sk_device);
        free(all_data.A_sk);

        if(profile_runtime) {
            std::ofstream file(output_filename_breakdown, std::ios::app);
            std::copy(CQRRP_GPU.times.data(), CQRRP_GPU.times.data() + 17, std::ostream_iterator<T>(file, ", "));
            file << "\n";
        } 
        long diff_qrf = 0;
        if (run_qrf) {
            lapack::Queue lapack_queue(0);
            using lapack::device_info_int;
            device_info_int* d_info = blas::device_malloc< device_info_int >( 1, lapack_queue );
            char* d_work_geqrf;
            char* h_work_geqrf;
            size_t d_size_geqrf, h_size_geqrf;
            
            auto start_qrf = std::chrono::steady_clock::now();
            lapack::geqrf_work_size_bytes(m, n, all_data.A_device, m, &d_size_geqrf, &h_size_geqrf, lapack_queue);
            d_work_geqrf = blas::device_malloc< char >( d_size_geqrf, lapack_queue );
            std::vector<char> h_work_geqrf_vector( h_size_geqrf );
            h_work_geqrf = h_work_geqrf_vector.data();
            lapack::geqrf(m, n, all_data.A_device, m, all_data.tau_device, d_work_geqrf, d_size_geqrf, h_work_geqrf, h_size_geqrf, d_info, lapack_queue);
            lapack_queue.sync();
            auto stop_qrf  = std::chrono::steady_clock::now();
	        diff_qrf  = std::chrono::duration_cast<std::chrono::milliseconds>(stop_qrf  - start_qrf).count();
            printf(" QRF TIME (MS) = %ld\n", diff_qrf);
        }

	    printf("  BLOCK SIZE = %ld TIME (MS) = %ld\n", block_size, diff_icqrrp);
        std::ofstream file(output_filename_speed, std::ios::app);
        file << m << "  " << n << "  " << block_size << "  " << diff_icqrrp << "  " << diff_qrf << "\n";
    }

    template <typename T, typename RNG>
    static void bench_CholQR(
        RandLAPACK::gen::mat_gen_info<T> m_info,
        int64_t numcols,
        CQRRPBenchData<T> &all_data,
        RandBLAS::RNGState<RNG> state,
        std::string output_filename) {

        auto m = all_data.row;
        auto n = numcols;
        auto state_const = state;

        // Initialize GPU stuff
        lapack::Queue lapack_queue(0);
        cudaStream_t strm = lapack_queue.stream();
        using lapack::device_info_int;
        device_info_int* d_info = blas::device_malloc< device_info_int >( 1, lapack_queue );
        char* d_work_geqrf;
        char* h_work_geqrf;
        size_t d_size_geqrf, h_size_geqrf;

        // CholQR part
        auto start_cholqr = std::chrono::steady_clock::now();
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T) 1.0, all_data.A_device, m, (T) 0.0, all_data.R_device, n, lapack_queue);
        lapack::potrf(Uplo::Upper,  n, all_data.R_device, n, d_info, lapack_queue);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.R_device, n, all_data.A_device, m, lapack_queue);
        lapack_queue.sync();
        auto stop_cholqr  = std::chrono::steady_clock::now();
        auto diff_cholqr  = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cholqr  - start_cholqr).count();
        
        auto start_orhr_col = std::chrono::steady_clock::now();
        // ORHR_COL part
        RandLAPACK::cuda_kernels::orhr_col_gpu(strm, m, n, all_data.A_device, m, all_data.tau_device, all_data.D_device);  
        RandLAPACK::cuda_kernels::R_cholqr_signs_gpu(strm, n, n, all_data.R_device, all_data.D_device);
        cudaStreamSynchronize(strm);
        auto stop_orhr_col  = std::chrono::steady_clock::now();
        auto diff_orhr_col  = std::chrono::duration_cast<std::chrono::milliseconds>(stop_orhr_col  - start_orhr_col).count();

        // Mandatory data re-generation
        data_regen(m_info, all_data, state);

        // QRF part
        auto start_qrf = std::chrono::steady_clock::now();
        lapack::geqrf_work_size_bytes(m, n, all_data.A_device, m, &d_size_geqrf, &h_size_geqrf, lapack_queue);
        d_work_geqrf = blas::device_malloc< char >( d_size_geqrf, lapack_queue );
        std::vector<char> h_work_geqrf_vector( h_size_geqrf );
        h_work_geqrf = h_work_geqrf_vector.data();
        lapack::geqrf(m, n, all_data.A_device, m, all_data.tau_device, d_work_geqrf, d_size_geqrf, h_work_geqrf, h_size_geqrf, d_info, lapack_queue);
        lapack_queue.sync();
        auto stop_qrf  = std::chrono::steady_clock::now();
        auto diff_qrf  = std::chrono::duration_cast<std::chrono::milliseconds>(stop_qrf  - start_qrf).count();
        printf(" CholQR TIME (MS)   = %ld\n", diff_cholqr);
        printf(" ORHR_COL TIME (MS) = %ld\n", diff_orhr_col);
        printf(" QRF TIME (MS)      = %ld\n", diff_qrf);

        std::ofstream file(output_filename, std::ios::app);
        file << m << "  " << n << "  " << diff_cholqr << "  " << diff_orhr_col << "  " << diff_qrf << "\n";
    }

};
TEST_P(BenchCQRRP, CQRRP_GPU_benchmark_16k) {
    int64_t m            = std::pow(2, 14);
    int64_t n            = std::pow(2, 14);
    int64_t b_sz         = GetParam();
    double tol           = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state           = RandBLAS::RNGState();
    bool profile_runtime = true;
    bool run_qrf         = false;
    bool cqrrp_uses_qrf  = true;
    if(b_sz == 120 || b_sz == 128) {
        run_qrf = true;
    }

    CQRRPBenchData<double> all_data(m, n);
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    cudaMemcpy(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);


    std::string file1 = "ICQRRP_GPU_runtime_breakdown_innerQRF_"              
	                                              + std::to_string(cqrrp_uses_qrf) 
	                            +  "_rows_"       + std::to_string(m)
                                    +  "_cols_"       + std::to_string(n)
                                    +  "_d_factor_1.0.dat";

    std::string file2 = "ICQRRP_GPU_speed_innerQRF_"
                                                      + std::to_string(cqrrp_uses_qrf)
	                            + "_rows_"        + std::to_string(m)
                                    + "_cols_"        + std::to_string(n)
                                    + "_d_factor_1.0.dat";

    bench_CQRRP(profile_runtime, run_qrf, cqrrp_uses_qrf,  m_info, tol, b_sz, all_data, state, file1, file2);
}

INSTANTIATE_TEST_SUITE_P(
    CQRRP_GPU_16k_benchmarks,
    BenchCQRRP,
    ::testing::Values(32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512)
);

TEST_F(BenchCQRRP, Bench_CholQR) {
    int64_t m       = std::pow(2, 14);
    int64_t n_start = 288;
    int64_t n_stop  = std::pow(2, 14);
    auto state      = RandBLAS::RNGState();

    CQRRPBenchData<double> all_data(m, n_stop);
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_stop, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    cudaMemcpy(all_data.A_device, all_data.A.data(), m * n_stop * sizeof(double), cudaMemcpyHostToDevice);


    std::string file = "CholQR_GPU_speed_rows_"      + std::to_string(m)
                                    + "_cols_start_" + std::to_string(n_start)
                                    + "_cols_stop_"  + std::to_string(n_stop)
                                    + ".dat";

    for(int i = n_start; i <= n_stop; i += n_start)
        bench_CholQR(m_info, i, all_data, state, file);
}
#endif
