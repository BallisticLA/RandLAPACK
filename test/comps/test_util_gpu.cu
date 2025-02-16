#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <chrono>
#include <numeric>
#include <random>
#include <gtest/gtest.h>

// Use cuda kernels.
#ifndef USE_CUDA
#define USE_CUDA

#include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"

using namespace std::chrono;


class TestUtil_GPU : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct ColSwpTestData {
        int64_t row;
        int64_t col;
        std::vector<T> A;
        std::vector<T> A_cpy;
        std::vector<T> B;
        std::vector<T> C;
        std::vector<T> A_host_buffer;
        std::vector<int64_t> J;
        std::vector<int64_t> J_host_buffer;
        T* A_device;
        T* B_device;
        int64_t* J_device;
        int64_t* buf_device;
        std::vector<T> Ident;
        std::vector<T> tau;

        ColSwpTestData(int64_t m, int64_t n) :
        A(m * n, 0.0),
        B(m * n, 0.0),
        C(m * n, 0.0),
        A_host_buffer(m * n, 0.0),
        J_host_buffer(n, 0.0),
        J(n, 0.0),
        tau(n, 0.0),
        Ident(n * n, 0.0),
        A_cpy(m * n, 0.0)
        {
            row = m;
            col = n;
            cudaMalloc(&A_device,   m * n * sizeof(T));
            cudaMalloc(&B_device,   m * n * sizeof(T));
            cudaMalloc(&J_device,   n * sizeof(int64_t));
            cudaMalloc(&buf_device, n * sizeof(int64_t));
            cudaDeviceSynchronize();
        }
        
        ~ColSwpTestData() {
            cudaFree(A_device);
            cudaFree(B_device);
            cudaFree(J_device);
            cudaFree(buf_device);
        }
    };

    template <typename T>
    struct ColSwpVecTestData {
        int64_t length_J;
        int64_t length_idx;
        std::vector<int64_t> J;
        std::vector<int64_t> idx;
        std::vector<int64_t> J_host_buffer;
        int64_t* J_device;
        int64_t* idx_device;

        ColSwpVecTestData(int64_t m, int64_t k) :
        J_host_buffer(m, 0.0),
        J(m, 0.0),
        idx(k, 0.0)
        {
            length_J   = m;
            length_idx = k;
            cudaMalloc(&J_device,   m * sizeof(int64_t));
            cudaMalloc(&idx_device, k * sizeof(int64_t));
            cudaDeviceSynchronize();
        }

        ~ColSwpVecTestData() {
            cudaFree(J_device);
            cudaFree(idx_device);
        }
    };

    template <typename T>
    struct TranspTestData {
        int64_t row;
        int64_t col;
        std::vector<T> A;
        std::vector<T> A_T;
        std::vector<T> A_T_buffer;
        T* A_device;
        T* A_device_T;

        TranspTestData(int64_t m, int64_t n) :
        A(m * n, 0.0),
        A_T(n * m, 0.0),
        A_T_buffer(n * m, 0.0)
        {
            row = m;
            col = n;
            cudaMalloc(&A_device,   m * n * sizeof(T));
            cudaMalloc(&A_device_T, n * m * sizeof(T));
        }

        ~TranspTestData() {
            cudaFree(A_device);
            cudaFree(A_device_T);
        }
    };

    template <typename T>
    struct GerTestData {
        int64_t row;
        int64_t col;
        std::vector<T> A;
        std::vector<T> A_buffer;
        std::vector<T> y;
        std::vector<T> x;
        T* A_device;
        T* y_device;
        T* x_device;

        GerTestData(int64_t m, int64_t n) :
        A(m * n, 0.0),
        A_buffer(n * m, 0.0),
        y(n * n, 0.0),
        x(m, 0.0)
        {
            row = m;
            col = n;
            cudaMalloc(&A_device, m * n * sizeof(T));
            cudaMalloc(&y_device, n * n * sizeof(T));
            cudaMalloc(&x_device, m * sizeof(T));
        }

        ~GerTestData() {
            cudaFree(A_device);
            cudaFree(y_device);
            cudaFree(x_device);
        }
    };

    template <typename T>
    static void 
    col_swp_gpu(ColSwpTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        cudaStream_t strm = cudaStreamPerThread;

        cudaMemcpyAsync(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice, strm);
        cudaMemcpyAsync(all_data.J_device, all_data.J.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice, strm);
        cudaStreamSynchronize(strm);


        cudaError_t ierr = cudaGetLastError();
        if (ierr != cudaSuccess)
        {
            RandLAPACK_CUDA_ERROR("GPU ERROR. " << cudaGetErrorString(ierr))
            abort();
        }
        printf("Passed the general error check\n");

        RandLAPACK::cuda_kernels::col_swap_gpu(strm, m, n, n, all_data.A_device, m, all_data.J_device);
        cudaMemcpyAsync(all_data.A_host_buffer.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost, strm);
        cudaMemcpyAsync(all_data.J_host_buffer.data(), all_data.J_device, n * sizeof(int64_t), cudaMemcpyDeviceToHost, strm);
        RandLAPACK::util::col_swap(m, n, n, all_data.A.data(), m, all_data.J);
        cudaStreamSynchronize(strm);

        for(int i = 0; i < m*n; ++i)
            all_data.A[i] -= all_data.A_host_buffer[i];

        T norm_test = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
        printf("\nNorm diff GPU CPU: %e\n", norm_test);
        EXPECT_NEAR(norm_test, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    	ierr = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before test returned. " << cudaGetErrorString(ierr))
        	abort();
    	}
    }

    template <typename T>
    static void 
    col_swp_submatrix_gpu(
        int64_t offset,
        int64_t col_offset,
        int64_t m_submat,
        int64_t n_submat,
        int64_t k_submat,
        ColSwpTestData<T> &all_data) {

        auto m   = all_data.row;
        auto n   = all_data.col;
        auto lda = m;

        cudaStream_t strm = cudaStreamPerThread;
        cudaMemcpyAsync(all_data.A_device, all_data.A.data(), m * n * sizeof(T), cudaMemcpyHostToDevice, strm);
        cudaMemcpyAsync(all_data.J_device, all_data.J.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice, strm);
        T* A_device = all_data.A_device;
        T* A_device_submat = all_data.A_device + (col_offset * m + offset);
        cudaStreamSynchronize(strm);
        RandLAPACK::cuda_kernels::col_swap_gpu(strm, m_submat, n_submat, k_submat, A_device_submat, lda, all_data.J_device);
        cudaMemcpyAsync(all_data.A_host_buffer.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost, strm);
        cudaMemcpyAsync(all_data.J_host_buffer.data(), all_data.J_device, n * sizeof(int64_t), cudaMemcpyDeviceToHost, strm);

        T* A = all_data.A.data();
        T* A_submat = all_data.A.data() + (col_offset * m + offset);
        RandLAPACK::util::col_swap(m_submat, n_submat, k_submat, A_submat, lda, all_data.J);

        cudaStreamSynchronize(strm);

        for(int i = 0; i < m*n; ++i)
            all_data.A[i] -= all_data.A_host_buffer[i];

        T norm_test = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
        printf("\nNorm diff GPU CPU: %e\n", norm_test);
        EXPECT_NEAR(norm_test, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    
    	cudaError_t ierr  = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before test returned. " << cudaGetErrorString(ierr))
        	abort();
    	}
    }

    template <typename T>
    static void 
    col_swp_subvector_gpu(
        int64_t offset,
        ColSwpVecTestData<T> &all_data) {

        auto m   = all_data.length_J;
        auto k   = all_data.length_idx;

        cudaStream_t strm = cudaStreamPerThread;
        cudaMemcpyAsync(all_data.J_device, all_data.J.data(),     m * sizeof(int64_t), cudaMemcpyHostToDevice, strm);
        cudaMemcpyAsync(all_data.idx_device, all_data.idx.data(), k * sizeof(int64_t), cudaMemcpyHostToDevice, strm);
        int64_t* J_device_subvec = all_data.J_device + offset;
        cudaStreamSynchronize(strm);
        RandLAPACK::cuda_kernels::col_swap_gpu<T>(strm, m, k, J_device_subvec, all_data.idx_device);
        cudaMemcpyAsync(all_data.J_host_buffer.data(), all_data.J_device, m * sizeof(int64_t), cudaMemcpyDeviceToHost, strm);

        int64_t* J_subvec = all_data.J.data() + offset;
        std::vector<int64_t> buf;
        RandLAPACK::util::col_swap<T>(m, k, J_subvec, all_data.idx);
        cudaStreamSynchronize(strm);

        for(int i = 0; i < m; ++i){
            all_data.J[i] -= all_data.J_host_buffer[i];
        }

        T norm_test = blas::nrm2(m, all_data.J.data(), 1);
        printf("\nNorm diff GPU CPU: %e\n", norm_test);
        EXPECT_NEAR(norm_test, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75));
   	 
    	cudaError_t ierr = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before test returned. " << cudaGetErrorString(ierr))
        	abort();
    	}
    }


    template <typename T>
    static void 
    qp3_swp_gpu(ColSwpTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        cudaStream_t strm = cudaStreamPerThread;
    
        cudaMemcpy(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
        // Perform Pivoted QR
        lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
        // Swap columns in A's copy
        cudaMemcpy(all_data.J_device, all_data.J.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        RandLAPACK::cuda_kernels::col_swap_gpu(strm, m, n, n, all_data.A_device, m, all_data.J_device);
        cudaMemcpy(all_data.A_host_buffer.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost);

        // Create an identity and store Q in it.
        RandLAPACK::util::eye(m, n, all_data.Ident.data());
        lapack::ormqr(Side::Left, Op::NoTrans, m, n, n, all_data.A.data(), m,  all_data.tau.data(),  all_data.Ident.data(), m);
        // Q * R -> Identity space
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.A.data(), m, all_data.Ident.data(), m);

        // A_piv - A_cpy
        for(int i = 0; i < m * n; ++i)
            all_data.A_host_buffer[i] -= all_data.Ident[i];

        T norm = lapack::lange(Norm::Fro, m, n, all_data.A_host_buffer.data(), m);
        printf("||A_piv - QR||_F:  %e\n", norm);
        EXPECT_NEAR(norm, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    	
    	cudaError_t ierr = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before test returned. " << cudaGetErrorString(ierr))
        	abort();
    	}
    }

    template <typename T>
    static void 
    transp_gpu(TranspTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        cudaStream_t strm = cudaStreamPerThread;
    
        cudaMemcpyAsync(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice, strm);
        cudaStreamSynchronize(strm);
        RandLAPACK::cuda_kernels::transposition_gpu(strm, m, n, all_data.A_device, m, all_data.A_device_T, n, 0);
        cudaMemcpyAsync(all_data.A_T_buffer.data(), all_data.A_device_T, n * m * sizeof(T), cudaMemcpyDeviceToHost, strm);
        RandLAPACK::util::transposition(m, n, all_data.A.data(), m, all_data.A_T.data(), n, 0);
        cudaStreamSynchronize(strm);

        // A_piv - A_cpy
        for(int i = 0; i < m * n; ++i)
            all_data.A_T[i] -= all_data.A_T_buffer[i];

        T norm = lapack::lange(Norm::Fro, n, m, all_data.A_T.data(), n);
        printf("||A_T_host - A_T_device||_F:  %e\n", norm);
        EXPECT_NEAR(norm, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    	
    	cudaError_t ierr = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before test returned. " << cudaGetErrorString(ierr))
        	abort();
    	}
    }

    template <typename T>
    static void 
    ger_gpu(
        T alpha,
        GerTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        cudaStream_t strm = cudaStreamPerThread;
        cudaMemcpyAsync(all_data.A_device, all_data.A.data(), m * n * sizeof(T), cudaMemcpyHostToDevice, strm);
        cudaStreamSynchronize(strm);
        cudaMemcpyAsync(all_data.y_device, all_data.y.data(), n * n * sizeof(T), cudaMemcpyHostToDevice, strm);
        cudaMemcpyAsync(all_data.x_device, all_data.x.data(), m * sizeof(T),     cudaMemcpyHostToDevice, strm);
        RandLAPACK::cuda_kernels::ger_gpu(strm, m, n, alpha, all_data.x_device, 1, all_data.y_device, n + 1, all_data.A_device, m);
        cudaMemcpyAsync(all_data.A_buffer.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost, strm);
        // Y has stride of n + 1
        blas::ger(Layout::ColMajor, m, n, alpha, all_data.x.data(), 1, all_data.y.data(), n + 1, all_data.A.data(), m);
        cudaStreamSynchronize(strm);
        
        // A_piv - A_cpy
        for(int i = 0; i < m * n; ++i)
            all_data.A[i] -= all_data.A_buffer[i];

        T norm = lapack::lange(Norm::Fro, m, n, all_data.A.data(), n);
        printf("||A_host - A_device||_F:  %e\n", norm);
        EXPECT_NEAR(norm, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    	
    	cudaError_t ierr = cudaGetLastError();
    	if (ierr != cudaSuccess)
    	{
        	RandLAPACK_CUDA_ERROR("Error before test returned. " << cudaGetErrorString(ierr))
        	abort();
    	}
    }
};


TEST_F(TestUtil_GPU, test_col_swp_gpu_base) {
    
    int64_t m = 1000;
    int64_t n = 1000;
    auto state = RandBLAS::RNGState();
    ColSwpTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state); 
    // Fill and randomly shuffle a vector
    std::iota(all_data.J.begin(), all_data.J.end(), 1);
    std::random_shuffle(all_data.J.begin(), all_data.J.begin() + n);

    col_swp_gpu<double>(all_data);
}

TEST_F(TestUtil_GPU, test_col_swp_large_gpu) {
    
    int64_t m = 512;
    int64_t n = 8000;
    auto state = RandBLAS::RNGState();
    ColSwpTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state); 
    // Fill and randomly shuffle a vector
    std::iota(all_data.J.begin(), all_data.J.end(), 1);
    std::random_shuffle(all_data.J.begin(), all_data.J.begin() + n);
    col_swp_gpu<double>(all_data);
}

TEST_F(TestUtil_GPU, test_col_swp_gpu_submatrix) {
    
    int64_t m = 5000;
    int64_t n = 2800;

    int64_t offset     = 0;
    int64_t col_offset = 2700;
    
    int64_t m_submat = 2700;
    int64_t n_submat = 100;
    int64_t k_submat = 100;

    auto state = RandBLAS::RNGState();
    ColSwpTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state); 
    // Fill and randomly shuffle a vector
    std::iota(all_data.J.begin(), all_data.J.begin() + n_submat, 1);
    std::shuffle(all_data.J.begin(), all_data.J.begin() + n_submat, std::minstd_rand{std::random_device{}()});

    col_swp_submatrix_gpu<double>(offset, col_offset, m_submat, n_submat, k_submat, all_data);
}

TEST_F(TestUtil_GPU, test_col_swp_gpu_subvector) {
    
    int64_t m          = 2800;
    int64_t col_offset = 2700;
    int64_t k_submat   = 100;

    ColSwpVecTestData<double> all_data(m, k_submat);
    // Fill and randomly shuffle a vector
    std::iota(all_data.J.begin(), all_data.J.begin() + m, 1);
    std::iota(all_data.idx.begin(), all_data.idx.begin() + k_submat, 1);
    std::shuffle(all_data.J.begin(), all_data.J.begin() + m, std::minstd_rand{std::random_device{}()});
    std::shuffle(all_data.idx.begin(), all_data.idx.begin() + k_submat, std::minstd_rand{std::random_device{}()});

    col_swp_subvector_gpu<double>(col_offset, all_data);
}


TEST_F(TestUtil_GPU, test_qp3_swp_gpu) {
    
    int64_t m = 4;
    int64_t n = 4;
    auto state = RandBLAS::RNGState();
    ColSwpTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state); 
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);

    qp3_swp_gpu<double>(all_data);
}

TEST_F(TestUtil_GPU, test_transp_gpu) {
    int64_t m = 2048;
    int64_t n = 1024;
    auto state = RandBLAS::RNGState();
    TranspTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state); 

    transp_gpu<double>(all_data);
}

TEST_F(TestUtil_GPU, test_ger_gpu) {
    int64_t m = 2048;
    int64_t n = 1024;
    double alpha = 2.0;
    auto state = RandBLAS::RNGState();
    GerTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state); 
    RandBLAS::DenseDist D1(n, n);
    state = RandBLAS::fill_dense(D1, all_data.y.data(), state);
    RandBLAS::DenseDist D2(1, m);
    state = RandBLAS::fill_dense(D2, all_data.x.data(), state);
    
    ger_gpu(alpha, all_data);
}
#endif
