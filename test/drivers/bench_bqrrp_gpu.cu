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
#include <numeric>

// Use cuda kernels.
#ifndef USE_CUDA
#define USE_CUDA
#include "RandLAPACK/drivers/rl_bqrrp_gpu.hh"

using GPUSubroutines = RandLAPACK::BQRRPGPUSubroutines;

class BenchBQRRP : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct BQRRPBenchData {
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

        BQRRPBenchData(int64_t m, int64_t n) :
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

        ~BQRRPBenchData() {
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
                            BQRRPBenchData<T> &all_data, 
                            RandBLAS::RNGState<RNG> &state) {

        auto state_const = state;
        auto m = m_info.rows;
        auto n = m_info.cols;

        RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state_const);
        cudaMemset(all_data.J_device, 0.0, n);
        cudaMemset(all_data.tau_device, 0.0, n);
    }

    template <typename T, typename RNG>
    static void bench_BQRRP(
        bool profile_runtime,
        bool run_qrf,
        RandLAPACK::gen::mat_gen_info<T> m_info,
        int64_t m,
        int64_t n,
        int64_t block_size,
        BQRRPBenchData<T> &all_data,
        RandBLAS::RNGState<RNG> state,
        std::string* output_filename_breakdown_QRF,
        std::string* output_filename_breakdown_CholQR,
        std::string* output_filename_speed) {

	    T d_factor = 1.0;
        auto state_const = state;
        int64_t d = d_factor * block_size;

        // BQRRP with QRF
        // Skethcing in an sampling regime
        cudaMalloc(&all_data.A_sk_device, d * n * sizeof(T));
        all_data.A_sk = new T[d * n]();
        T* S          = new T[d * m]();

        RandBLAS::DenseDist D(d, m);
        RandBLAS::fill_dense(D, S, state_const);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, S, d, all_data.A.data(), m, 0.0, all_data.A_sk, d);
        cudaMemcpy(all_data.A_sk_device, all_data.A_sk, d * n * sizeof(double), cudaMemcpyHostToDevice);
        RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU_QRF(profile_runtime, block_size);
        BQRRP_GPU_QRF.qr_tall = GPUSubroutines::QRTall::geqrf;
	    auto start_bqrrp_qrf = std::chrono::steady_clock::now();
        BQRRP_GPU_QRF.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);
        auto stop_bqrrp_qrf = std::chrono::steady_clock::now();
	    auto diff_bqrrp_qrf = std::chrono::duration_cast<std::chrono::microseconds>(stop_bqrrp_qrf - start_bqrrp_qrf).count();
        data_regen(m_info, all_data, state);
        cudaFree(all_data.A_sk_device);
        delete[] all_data.A_sk;

        if(profile_runtime) {
            std::ofstream file(*output_filename_breakdown_QRF, std::ios::app);
            std::copy(BQRRP_GPU_QRF.times.data(), BQRRP_GPU_QRF.times.data() + 15, std::ostream_iterator<T>(file, ", "));
            file << "\n";
        } 

        // BQRRP with CholQR
        // Skethcing in an sampling regime
        cudaMalloc(&all_data.A_sk_device, d * n * sizeof(T));
        all_data.A_sk = new T[d * n]();
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, S, d, all_data.A.data(), m, 0.0, all_data.A_sk, d);
        delete[] S;
        cudaMemcpy(all_data.A_sk_device, all_data.A_sk, d * n * sizeof(double), cudaMemcpyHostToDevice);
        RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU_CholQR(profile_runtime, block_size);
        BQRRP_GPU_CholQR.qr_tall = GPUSubroutines::QRTall::cholqr;
	    auto start_bqrrp_cholqr = std::chrono::steady_clock::now();
        BQRRP_GPU_CholQR.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);
	    auto stop_bqrrp_cholqr = std::chrono::steady_clock::now();
	    auto diff_bqrrp_cholqr = std::chrono::duration_cast<std::chrono::microseconds>(stop_bqrrp_cholqr - start_bqrrp_cholqr).count();
        data_regen(m_info, all_data, state);
        cudaFree(all_data.A_sk_device);
        delete[] all_data.A_sk;

        if(profile_runtime) {
            std::ofstream file(*output_filename_breakdown_CholQR, std::ios::app);
            std::copy(BQRRP_GPU_CholQR.times.data(), BQRRP_GPU_CholQR.times.data() + 15, std::ostream_iterator<T>(file, ", "));
            file << "\n";
        } 

        // Optional QRF
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
	        diff_qrf  = std::chrono::duration_cast<std::chrono::microseconds>(stop_qrf  - start_qrf).count();
            printf(" QRF TIME (MS) = %ld\n", diff_qrf);
        }

	    printf("  BLOCK SIZE = %ld BQRRP+QRF TIME (MS) = %ld BQRRP+CholQR TIME (MS) = %ld\n", block_size, diff_bqrrp_qrf, diff_bqrrp_cholqr);
        std::ofstream file(*output_filename_speed, std::ios::app);
        file << diff_bqrrp_qrf << "  " << diff_bqrrp_cholqr << "  " << diff_qrf << "\n";
        cudaError_t ierr = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before bench_bqrrp returned. " << cudaGetErrorString(ierr))
        	abort();
    	}
    }

    // Not using this right now. But there's no harm in keeping it around.
    template <typename T, typename RNG>
    static void bench_CholQR(
        RandLAPACK::gen::mat_gen_info<T> m_info,
        int64_t numrows,
        BQRRPBenchData<T> &all_data,
        RandBLAS::RNGState<RNG> state,
        std::string output_filename) {

        auto m = numrows;
        auto n = all_data.col;
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
        auto diff_cholqr  = std::chrono::duration_cast<std::chrono::microseconds>(stop_cholqr  - start_cholqr).count();
        
        auto start_orhr_col = std::chrono::steady_clock::now();
        // ORHR_COL part
        RandLAPACK::cuda_kernels::orhr_col_gpu(strm, m, n, all_data.A_device, m, all_data.tau_device, all_data.D_device);  
        RandLAPACK::cuda_kernels::R_cholqr_signs_gpu(strm, n, n, all_data.R_device, all_data.D_device);
        cudaStreamSynchronize(strm);
        auto stop_orhr_col  = std::chrono::steady_clock::now();
        auto diff_orhr_col  = std::chrono::duration_cast<std::chrono::microseconds>(stop_orhr_col  - start_orhr_col).count();

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
        auto diff_qrf  = std::chrono::duration_cast<std::chrono::microseconds>(stop_qrf  - start_qrf).count();
        printf(" CholQR TIME (MS)   = %ld\n", diff_cholqr);
        printf(" ORHR_COL TIME (MS) = %ld\n", diff_orhr_col);
        printf(" QRF TIME (MS)      = %ld\n", diff_qrf);

        std::ofstream file(output_filename, std::ios::app);
        file << diff_cholqr << "  " << diff_orhr_col << "  " << diff_qrf << "\n";

        cudaError_t ierr = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before bench_CholQR returned. " << cudaGetErrorString(ierr))
        	abort();
    	}    
    }

    static void setup_bqrrp_speed_comparisons_block_size(
        int64_t m,
        int64_t n,
        std::vector<int64_t> b_sz 
    ){
        // Get a string representation of the block size vector
        std::string b_sz_string = std::accumulate(b_sz.begin(), b_sz.end(), std::string(), 
                                    [](const std::string& a, int b) {
                                        return a.empty() ? std::to_string(b) : a + "," + std::to_string(b);
                                    });

        auto state           = RandBLAS::RNGState();
        bool profile_runtime = true;
        bool run_qrf         = true;

        BQRRPBenchData<double> all_data(m, n);
        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
        cudaMemcpy(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

        std::string* file_name_1 = new std::string(RandLAPACK::util::getCurrentDate<double>() + "BQRRP_GPU_runtime_breakdown_qrf_"       
                            + "_num_info_lines_" + std::to_string(7) +
                            ".txt");

        std::string* file_name_2 = new std::string(RandLAPACK::util::getCurrentDate<double>() + "BQRRP_GPU_runtime_breakdown_cholqr_"  
                            + "_num_info_lines_" + std::to_string(7) +
                              ".txt");

        std::string* file_name_3 = new std::string(RandLAPACK::util::getCurrentDate<double>() + "BQRRP_GPU_speed_comparisons_block_size"  
                            + "_num_info_lines_" + std::to_string(7) +
                              ".txt");

        std::ofstream file1(*file_name_1, std::ios::out | std::ios::app);
        std::ofstream file2(*file_name_2, std::ios::out | std::ios::app);
        std::ofstream file3(*file_name_3, std::ios::out | std::ios::app);

        file1 << "Description: Results from the BQRRP GPU runtime breakdown benchmark, recording the time it takes to perform every subroutine in BQRRP."
                "\nFile format: 15 data columns, each corresponding to a given BQRRP subroutine: preallocation_t_dur, qrcp_main_t_dur, copy_A_sk_t_dur, qrcp_piv_t_dur, copy_A_t_dur, piv_A_t_dur, copy_J_t_dur, updating_J_t_dur, preconditioning_t_dur, qr_tall_t_dur, q_reconstruction_t_dur, apply_transq_t_dur, sample_update_t_dur, t_rest, total_t_dur"
                "               rows correspond to BQRRP runs with block sizes varying in a way unique for a particular run."
                "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
                "\nInput type:"       + std::to_string(m_info.m_type) +
                "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
                "\nAdditional parameters: Tall QR subroutine cholqr BQRRP block sizes: " + b_sz_string +
                "\n";
        file1.flush();

        file2 << "Description: Results from the BQRRP GPU runtime breakdown benchmark, recording the time it takes to perform every subroutine in BQRRP."
                "\nFile format: 15 data columns, each corresponding to a given BQRRP subroutine: preallocation_t_dur, qrcp_main_t_dur, copy_A_sk_t_dur, qrcp_piv_t_dur, copy_A_t_dur, piv_A_t_dur, copy_J_t_dur, updating_J_t_dur, preconditioning_t_dur, qr_tall_t_dur, q_reconstruction_t_dur, apply_transq_t_dur, sample_update_t_dur, t_rest, total_t_dur"
                "               rows correspond to BQRRP runs with block sizes varying in a way unique for a particular run."
                "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
                "\nInput type:"       + std::to_string(m_info.m_type) +
                "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
                "\nAdditional parameters: Tall QR subroutine geqrf BQRRP block sizes: " + b_sz_string +
                "\n";
        file2.flush();

        file3 << "Description: Results from the BQRRP GPU speed comparison benchmark, recording the time it takes to perform BQRRP and alternative QR and QRCP factorizations."
                "\nFile format: 3 columns, containing time for each algorithm: BQRRP+CholQR, BQRRP+QRF, QRF;"
                "               rows correspond to BQRRP runs with block sizes varying in powers of 2 or multiples of 10"
                "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
                "\nInput type:"       + std::to_string(m_info.m_type) +
                "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
                "\nAdditional parameters: BQRRP block sizes: " + b_sz_string +
                "\n";
        file3.flush();

        for(size_t i = 0; i < b_sz.size(); ++i) {
            bench_BQRRP(profile_runtime, run_qrf, m_info, m, n, b_sz[i], all_data, state, file_name_1, file_name_2, file_name_3);
            run_qrf = false;
        }
    }

    static void setup_bqrrp_speed_comparisons_mat_size(
        std::vector<int64_t> m_sz,
        int64_t b_sz 
    ){
        // Get a string representation of the block size vector
        std::string m_sz_string = std::accumulate(m_sz.begin(), m_sz.end(), std::string(), 
                                    [](const std::string& a, int b) {
                                        return a.empty() ? std::to_string(b) : a + "," + std::to_string(b);
                                    });

        auto state           = RandBLAS::RNGState();
        bool profile_runtime = false;
        bool run_qrf         = true;

        int64_t m_max = *std::max_element(m_sz.begin(), m_sz.end());
        BQRRPBenchData<double> all_data(m_max, m_max);
        RandLAPACK::gen::mat_gen_info<double> m_info(m_max, m_max, RandLAPACK::gen::gaussian);
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
        cudaMemcpy(all_data.A_device, all_data.A.data(), m_max * m_max * sizeof(double), cudaMemcpyHostToDevice);

        std::string* file_name = new std::string(RandLAPACK::util::getCurrentDate<double>() + "BQRRP_GPU_speed_comparisons_mat_size"  
                            + "_num_info_lines_" + std::to_string(7) +
                              ".txt");

        std::ofstream file(*file_name, std::ios::out | std::ios::app);
        file << "Description: Results from the BQRRP GPU speed comparison benchmark, recording the time it takes to perform BQRRP and alternative QR and QRCP factorizations."
                "\nFile format: 7 columns, containing time for each algorithm: BQRRP+CholQR, BQRRP+QRF, QRF;"
                "               rows correspond to BQRRP runs with varying mat sizes, with numruns repititions of each mat size."
                "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
                "\nInput type:"       + std::to_string(m_info.m_type) +
                "\nInput size:"       + " dim start: " + m_sz_string +
                "\nAdditional parameters: BQRRP block size: " + std::to_string(b_sz) + 
                "\n";
        file.flush();

        for(size_t i = 0; i < m_sz.size(); ++i) {
            bench_BQRRP(profile_runtime, run_qrf, m_info, m_sz[i], m_sz[i], m_sz[i]/32, all_data, state, nullptr, nullptr, file_name);
            run_qrf = false;
        }
    }
};

TEST_F(BenchBQRRP, BQRRP_GPU_block_sizes_powers_of_two) {
    int64_t m                 = std::pow(2, 15);
    int64_t n                 = std::pow(2, 15);
    std::vector<int64_t> b_sz = {32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048};
    setup_bqrrp_speed_comparisons_block_size(m, n, b_sz);
}

TEST_F(BenchBQRRP, BQRRP_GPU_block_sizes_multiples_of_ten) {
    int64_t m                 = 32000;
    int64_t n                 = 32000;
    std::vector<int64_t> b_sz = {50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1000, 1125, 1250, 1375, 1500, 1625, 1750, 1875, 2000};
    setup_bqrrp_speed_comparisons_block_size(m, n, b_sz);
}

TEST_F(BenchBQRRP, BQRRP_GPU_mat_sizes_powers_of_two) {
    std::vector<int64_t> m_sz = {512, 1024, 2048, 4096, 8192, 32768};
    int64_t b_sz              = 0;
    setup_bqrrp_speed_comparisons_mat_size(m_sz, b_sz);
}

TEST_F(BenchBQRRP, BQRRP_GPU_mat_sizes_multiples_of_ten) {
    std::vector<int64_t> m_sz = {512, 1000, 2000, 4000, 8000, 32000};
    int64_t b_sz              = 0;
    setup_bqrrp_speed_comparisons_mat_size(m_sz, b_sz);
}
#endif
