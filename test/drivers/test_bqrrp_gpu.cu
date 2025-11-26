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
#include "RandLAPACK/drivers/rl_bqrrp_gpu.hh"

using GPUSubroutines = RandLAPACK::BQRRPGPUSubroutines;

class TestBQRRP : public ::testing::TestWithParam<int64_t>
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct BQRRPTestData {
        int64_t row;
        int64_t col;
        int64_t rank;
        
        std::vector<T> A;
        std::vector<T> A_cpu;
        std::vector<T> A_sk;
        std::vector<T> Q;
        std::vector<T> Q_cpu;
        std::vector<T> R;
        std::vector<T> R_cpu;
        std::vector<T> R_full;
        std::vector<T> tau;
        std::vector<T> tau_cpu;
        std::vector<int64_t> J;
        std::vector<int64_t> J_cpu;
        std::vector<T> A_cpy1;
        std::vector<T> A_cpy2;
        std::vector<T> I_ref;
        
        // Buffers for the GPU data
        T* A_device;
        T* A_sk_device;
        T* tau_device;
        int64_t* J_device;

        BQRRPTestData(int64_t m, int64_t n, int64_t k, int64_t d) :
        A(m * n, 0.0),
        A_cpu(m * n, 0.0),
        A_sk(d * n, 0.0),
        Q(m * n, 0.0),
        Q_cpu(m * n, 0.0),
        R_full(m * n, 0.0),
        tau(n, 0.0),
        tau_cpu(n, 0.0),
        J(n, 0),
        J_cpu(n, 0),
        A_cpy1(m * n, 0.0),
        A_cpy2(m * n, 0.0),
        I_ref(k * k, 0.0) 
        {
            row = m;
            col = n;
            rank = k;
            cudaMalloc(&A_device,    m * n * sizeof(T));
            cudaMalloc(&A_sk_device, d * n * sizeof(T));
            cudaMalloc(&tau_device,  n *     sizeof(T));
            cudaMalloc(&J_device,    n *     sizeof(int64_t));
        }

        ~BQRRPTestData() {
            cudaFree(A_device);
            cudaFree(A_sk_device);
            cudaFree(tau_device);
            cudaFree(J_device);
        }
    };

    template <typename T, typename RNG>
    static void norm__sektch_and_copy_computational_helper(T &norm_A, int64_t d, BQRRPTestData<T> &all_data, RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto state_const = state;

        // Skethcing in an sampling regime
        T* S = new T[d * m]();
        RandBLAS::DenseDist D(d, m);
        RandBLAS::fill_dense(D, S, state_const);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, S, d, all_data.A.data(), m, 0.0, all_data.A_sk.data(), d);
        delete[] S;
        cudaMemcpy(all_data.A_sk_device, all_data.A_sk.data(), d * n * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpu.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }

    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, BQRRPTestData<T> &all_data, T atol) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        RandLAPACK::util::upsize(k * k, all_data.I_ref);
        RandLAPACK::util::eye(k, k, all_data.I_ref);

        T* A_dat         = all_data.A_cpy1.data();
        T const* A_cpy_dat = all_data.A_cpy2.data();
        T const* Q_dat   = all_data.Q.data();
        T const* R_dat   = all_data.R.data();
        T* I_ref_dat     = all_data.I_ref.data();

        // Check orthogonality of Q
        // Q' * Q  - I = 0
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, I_ref_dat, k);
        T norm_0 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);

        // A - QR
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, R_dat, k, -1.0, A_dat, m);
        
        // Implementing max col norm metric
        T max_col_norm = 0.0;
        T col_norm = 0.0;
        int max_idx = 0;
        for(int i = 0; i < n; ++i) {
            col_norm = blas::nrm2(m, &A_dat[m * i], 1);
            if(max_col_norm < col_norm) {
                max_col_norm = col_norm;
                max_idx = i;
            }
        }
        T col_norm_A = blas::nrm2(n, &A_cpy_dat[m * max_idx], 1);
        T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, m);
        
        std::cout << "REL NORM OF AP - QR:    " << std::scientific << std::setw(14) << norm_AQR / norm_A << "\n";
        std::cout << "MAX COL NORM METRIC:    " << std::scientific << std::setw(14) << max_col_norm / col_norm_A << "\n";
        std::cout << "FRO NORM OF (Q'Q - I):  " << std::scientific << std::setw(14) << norm_0 / std::sqrt((T) n) << "\n\n";

        ASSERT_NEAR(norm_AQR / norm_A,         0.0, atol);
        ASSERT_NEAR(max_col_norm / col_norm_A, 0.0, atol);
        ASSERT_NEAR(norm_0 / std::sqrt((T) n), 0.0, atol);
    }

    /// General test for BQRRP:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_BQRRP_general(
        int64_t d, 
        T norm_A,
        BQRRPTestData<T> &all_data,
        alg_type &BQRRP_GPU) {

        auto m = all_data.row;
        auto n = all_data.col;
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);

        BQRRP_GPU.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);

        if(BQRRP_GPU.rank == 0) {
            cudaMemcpy(all_data.A.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost);
            for(int i = 0; i < m * n; ++i) {
                ASSERT_NEAR(all_data.A[i], 0.0, atol);
            }
        } else {
            all_data.rank = BQRRP_GPU.rank;
            std::cout << "RANK AS RETURNED BY BQRRP GPU " << std::setw(4) << all_data.rank << "\n";
            
            cudaMemcpy(all_data.R_full.data(), all_data.A_device,   m * n * sizeof(T),   cudaMemcpyDeviceToHost);
            cudaMemcpy(all_data.Q.data(),      all_data.A_device,   m * n * sizeof(T),   cudaMemcpyDeviceToHost);
            cudaMemcpy(all_data.tau.data(),    all_data.tau_device, n * sizeof(T),       cudaMemcpyDeviceToHost);
            cudaMemcpy(all_data.J.data(),      all_data.J_device,   n * sizeof(int64_t), cudaMemcpyDeviceToHost);

            lapack::ungqr(m, n, n, all_data.Q.data(), m, all_data.tau.data());
            RandLAPACK::util::upsize(all_data.rank * n, all_data.R);
            lapack::lacpy(MatrixType::Upper, all_data.rank, n, all_data.R_full.data(), m, all_data.R.data(), all_data.rank);

            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);

            error_check(norm_A, all_data, atol);
        }

        cudaError_t ierr = cudaGetLastError();
        if (ierr != cudaSuccess)
        {
                RandLAPACK_CUDA_ERROR("Error before test_BQRRP_general returned. " << cudaGetErrorString(ierr))
                abort();
        }
    }

    template <typename T, typename RNG, typename alg_gpu, typename alg_cpu>
    static void test_BQRRP_compare_with_CPU(
        int64_t d, 
        BQRRPTestData<T> &all_data,
        alg_gpu &BQRRP_GPU,
        alg_cpu &BQRRP_CPU,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        BQRRP_GPU.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);
        BQRRP_CPU.call(m, n, all_data.A_cpu.data(), m, (T) (d / BQRRP_CPU.block_size) , all_data.tau_cpu.data(), all_data.J_cpu.data(), state);
        
        cudaMemcpy(all_data.R_full.data(), all_data.A_device,   m * n * sizeof(T),   cudaMemcpyDeviceToHost);
        cudaMemcpy(all_data.tau.data(),    all_data.tau_device, n * sizeof(T),       cudaMemcpyDeviceToHost);
        cudaMemcpy(all_data.J.data(),      all_data.J_device,   n * sizeof(int64_t), cudaMemcpyDeviceToHost);

        for(int i = 0; i < n; ++i) {
            all_data.tau[i] -= all_data.tau_cpu[i];
            all_data.J[i] -= all_data.J_cpu[i];

            for(int j = 0; j <= i; ++j) {
                all_data.A_cpu[i * m + j] -= all_data.R_full[i * m + j];
            }
        }
        RandLAPACK::util::get_U(n, n, all_data.A_cpu.data(), m);

        T col_nrm_J   = blas::nrm2(n, all_data.J.data(), 1);
        T col_nrm_tau = blas::nrm2(n, all_data.tau.data(), 1);
        T norm_R_diff = lapack::lange(Norm::Fro, n, n, all_data.A_cpu.data(), m);

        T atol1 = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        T atol2 = std::pow(std::numeric_limits<T>::epsilon(), 0.60);
        ASSERT_NEAR(col_nrm_J,   0.0, atol1);
        ASSERT_NEAR(col_nrm_tau, 0.0, atol1);
        ASSERT_NEAR(norm_R_diff, 0.0, atol2);
    
    	cudaError_t ierr = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before test_BQRRP_compare_with_CPU returned. " << cudaGetErrorString(ierr))
        	abort();
    	}
    }
};
#if !defined(__APPLE__)
// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_GPU_070824) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    int64_t d = d_factor * b_sz;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();
    bool profile_runtime = true;

    BQRRPTestData<double> all_data(m, n, k, d);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(profile_runtime, b_sz);
    BQRRP_GPU.qr_tall = GPUSubroutines::QRTall::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
    test_BQRRP_general<double, RandLAPACK::BQRRP_GPU<double, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_GPU_qrf) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    int64_t d = d_factor * b_sz;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();
    bool profile_runtime = true;

    BQRRPTestData<double> all_data(m, n, k, d);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(profile_runtime, b_sz);
    BQRRP_GPU.qr_tall = GPUSubroutines::QRTall::geqrf;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
    test_BQRRP_general<double, RandLAPACK::BQRRP_GPU<double, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_GPU_vectors) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    int64_t d = d_factor * b_sz;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k, d);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP_CPU(false, b_sz);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
    test_BQRRP_compare_with_CPU(d, all_data, BQRRP_GPU, BQRRP_CPU, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_GPU_near_zero_input) {
    int64_t m = 1000;
    int64_t n = 1000;
    int64_t k = 1000;
    double d_factor = 1;
    int64_t b_sz = 100;
    int64_t d = d_factor * b_sz;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k, d);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
    BQRRP_GPU.qr_tall = GPUSubroutines::QRTall::cholqr;

    std::fill(&(all_data.A.data())[0], &(all_data.A.data())[m * n], 0.0);
    all_data.A[1000*200 + 1] = 1;

    norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
    test_BQRRP_general<double, RandLAPACK::BQRRP_GPU<double, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
}

TEST_F(TestBQRRP, BQRRP_GPU_zero_input) {
    int64_t m = 1000;//5000;
    int64_t n = 1000;//2000;
    int64_t k = 1000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 100;//500;
    int64_t d = d_factor * b_sz;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k, d);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
    BQRRP_GPU.qr_tall = GPUSubroutines::QRTall::cholqr;

    std::fill(&(all_data.A.data())[0], &(all_data.A.data())[m * n], 0.0);

    norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
    test_BQRRP_general<double, RandLAPACK::BQRRP_GPU<double, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
}

TEST_F(TestBQRRP, GEQRF_GPU_ATTEMPT_TO_CATCH_INEFFICIENCY_ON_H100) {
    int64_t m   = 16384;
    int64_t k   = 1024;
    int64_t lda = 16384;
    auto state = RandBLAS::RNGState();

    double* A   = new double[m * m]();
    double* tau = new double[m]();
    double* A_device;
    double* tau_device;
    cudaMalloc(&A_device, m * m * sizeof(double));
    cudaMalloc(&tau_device, m * sizeof(double));

    RandBLAS::DenseDist D1(m, m);
    RandBLAS::DenseDist D2(1, m);
    state = RandBLAS::fill_dense(D1, A, state);
    state = RandBLAS::fill_dense(D2, tau, state);
    cudaMemcpy(A_device, A, m * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tau_device, tau, m * sizeof(double), cudaMemcpyHostToDevice);

    // All things necessary to launch ORMQR from cusolver
    lapack::Queue lapack_queue(0);
    cudaStream_t strm = lapack_queue.stream();
    using lapack::device_info_int;
    device_info_int* d_info = blas::device_malloc< device_info_int >( 1, lapack_queue );
    char* d_work_geqrf_opt;
    char* h_work_geqrf_opt = nullptr;
    size_t d_size_geqrf_opt, h_size_geqrf_opt;
    cudaStreamSynchronize(strm);

        lapack::geqrf_work_size_bytes(m, k, A_device, lda, &d_size_geqrf_opt, &h_size_geqrf_opt, lapack_queue);
        d_work_geqrf_opt = blas::device_malloc< char >( d_size_geqrf_opt, lapack_queue );

    int64_t curr_sz = 0;
    for(int iter = 0; iter < 2; ++iter) {
        //lapack::geqrf_work_size_bytes(m, k, A_device, lda, &d_size_geqrf_opt, &h_size_geqrf_opt, lapack_queue);
        //d_work_geqrf_opt = blas::device_malloc< char >( d_size_geqrf_opt, lapack_queue );
        lapack::geqrf(m, k, A_device, lda, &tau_device[curr_sz], d_work_geqrf_opt, d_size_geqrf_opt, h_work_geqrf_opt, h_size_geqrf_opt, d_info, lapack_queue);
        m       -= k;
        curr_sz += k;
        A_device = &A_device[k * lda + k];
        cudaStreamSynchronize(strm);
    }

    free(A);
    free(tau);
    cudaFree(A_device);
    cudaFree(tau_device);
    blas::device_free(d_work_geqrf_opt, lapack_queue);
}
#endif
#endif
