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

// The GPU drivers and kernels come in with RandLAPACK.hh, under its __CUDACC__ guard.

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
        std::cout << "Passed the general error check\n";

        RandLAPACK::cuda_kernels::col_swap_gpu(strm, m, n, n, all_data.A_device, m, all_data.J_device);
        cudaMemcpyAsync(all_data.A_host_buffer.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost, strm);
        cudaMemcpyAsync(all_data.J_host_buffer.data(), all_data.J_device, n * sizeof(int64_t), cudaMemcpyDeviceToHost, strm);
        RandLAPACK::util::col_swap(m, n, n, all_data.A.data(), m, all_data.J.data());
        cudaStreamSynchronize(strm);

        for(int i = 0; i < m*n; ++i)
            all_data.A[i] -= all_data.A_host_buffer[i];

        T norm_test = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
        std::cout << "\nNorm diff GPU CPU: " << std::scientific << norm_test << "\n";
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
        RandLAPACK::util::col_swap(m_submat, n_submat, k_submat, A_submat, lda, all_data.J.data());

        cudaStreamSynchronize(strm);

        for(int i = 0; i < m*n; ++i)
            all_data.A[i] -= all_data.A_host_buffer[i];

        T norm_test = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
        std::cout << "\nNorm diff GPU CPU: " << std::scientific << norm_test << "\n";
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
        RandLAPACK::util::col_swap(m, k, J_subvec, all_data.idx.data());
        cudaStreamSynchronize(strm);

        for(int i = 0; i < m; ++i){
            all_data.J[i] -= all_data.J_host_buffer[i];
        }

        T norm_test = blas::nrm2(m, all_data.J.data(), 1);
        std::cout << "\nNorm diff GPU CPU: " << std::scientific << norm_test << "\n";
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
        std::cout << "||A_piv - QR||_F:  " << std::scientific << norm << "\n";
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
        std::cout << "||A_T_host - A_T_device||_F:  " << std::scientific << norm << "\n";
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
        std::cout << "||A_host - A_device||_F:  " << std::scientific << norm << "\n";
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

// In upper-triangle mode, copy_mat_gpu copies the upper-trapezoidal part of the m-by-n source,
// rows j <= min(i, m - 1) of column i; with m < n, rows m:n of the destination stay untouched.
TEST_F(TestUtil_GPU, test_copy_mat_gpu_upper_trapezoid) {
    int64_t m   = 5;
    int64_t n   = 9;
    int64_t lda = 11;
    std::vector<double> src(lda * n), dst(lda * n, -1.0);
    std::iota(src.begin(), src.end(), 1.0);

    double *src_device, *dst_device;
    cudaMalloc(&src_device, lda * n * sizeof(double));
    cudaMalloc(&dst_device, lda * n * sizeof(double));
    cudaMemcpy(src_device, src.data(), lda * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device, dst.data(), lda * n * sizeof(double), cudaMemcpyHostToDevice);

    cudaStream_t strm = cudaStreamPerThread;
    RandLAPACK::cuda_kernels::copy_mat_gpu(strm, m, n, src_device, lda, dst_device, lda, true);
    cudaStreamSynchronize(strm);
    cudaMemcpy(dst.data(), dst_device, lda * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(src_device);
    cudaFree(dst_device);

    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < lda; ++j)
            EXPECT_EQ(dst[lda * i + j], (j <= i && j < m) ? src[lda * i + j] : -1.0) << "row " << j << ", column " << i;

    cudaError_t ierr = cudaGetLastError();
    EXPECT_EQ(ierr, cudaSuccess) << cudaGetErrorString(ierr);
}

// col_swap_gpu's in-place overloads, col_swap_gpu_kernel_sequential for a matrix and vec_ell_swap_gpu
// for a vector, apply the swaps one at a time and keep the remaining pivots in an array that every
// thread reads. Each trial applies a random permutation J and its inverse K alternately, then J once
// more, back to back with no host synchronization in between, and compares the result with
// util::col_swap's. Most trials use random disjoint adjacent transpositions: each costs a grid sync
// and the quickest pivot update, which leaves the widest window for a race on that array.
TEST_F(TestUtil_GPU, test_col_swp_gpu_sequential_stress) {
    int64_t m = 1000;
    int64_t n = 1000;
    int num_trials = 100;
    cudaStream_t strm = cudaStreamPerThread;
    std::mt19937_64 gen(2027);

    std::vector<double> A(m * n), A_ref(m * n), A_out(m * n);
    std::iota(A.begin(), A.end(), 0.0);
    std::vector<int64_t> v(n), v_ref(n), v_out(n), J(n), K(n), J_cpy(n);
    std::iota(v.begin(), v.end(), 1);
    double* A_device;
    int64_t *v_device, *J_device, *K_device;
    cudaMalloc(&A_device, m * n * sizeof(double));
    cudaMalloc(&v_device, n * sizeof(int64_t));
    cudaMalloc(&J_device, n * sizeof(int64_t));
    cudaMalloc(&K_device, n * sizeof(int64_t));

    int wrong_matrix = 0;
    int wrong_vector = 0;
    for (int t = 0; t < num_trials; ++t) {
        // Every tenth trial is a uniform random permutation, whose long pivot searches make it slow.
        bool shuffle = t % 10 == 0;
        int round_trips = shuffle ? 1 : 10;
        std::iota(J.begin(), J.end(), 1);
        if (shuffle) {
            std::shuffle(J.begin(), J.end(), gen);
        } else {
            for (int64_t i = 0; i + 1 < n; ++i)
                if (gen() % 10 != 0) {
                    std::swap(J[i], J[i + 1]);
                    ++i;
                }
        }
        // Column i of the result is column J[i] of the input, so K, with K[J[i]] = i, undoes J.
        for (int64_t i = 0; i < n; ++i)
            K[J[i] - 1] = i + 1;
        cudaMemcpy(A_device, A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(v_device, v.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(J_device, J.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(K_device, K.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);

        for (int r = 0; r < round_trips; ++r) {
            RandLAPACK::cuda_kernels::col_swap_gpu(strm, m, n, n, A_device, m, J_device);
            RandLAPACK::cuda_kernels::col_swap_gpu(strm, m, n, n, A_device, m, K_device);
        }
        RandLAPACK::cuda_kernels::col_swap_gpu(strm, m, n, n, A_device, m, J_device);
        for (int r = 0; r < round_trips; ++r) {
            RandLAPACK::cuda_kernels::col_swap_gpu<double>(strm, n, n, v_device, J_device);
            RandLAPACK::cuda_kernels::col_swap_gpu<double>(strm, n, n, v_device, K_device);
        }
        RandLAPACK::cuda_kernels::col_swap_gpu<double>(strm, n, n, v_device, J_device);
        cudaStreamSynchronize(strm);
        cudaMemcpy(A_out.data(), A_device, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(v_out.data(), v_device, n * sizeof(int64_t), cudaMemcpyDeviceToHost);

        A_ref = A;
        J_cpy = J;
        RandLAPACK::util::col_swap(m, n, n, A_ref.data(), m, J_cpy.data());
        v_ref = v;
        J_cpy = J;
        RandLAPACK::util::col_swap(n, n, v_ref.data(), J_cpy.data());
        wrong_matrix += !(A_out == A_ref);
        wrong_vector += !(v_out == v_ref);
    }
    cudaFree(A_device);
    cudaFree(v_device);
    cudaFree(J_device);
    cudaFree(K_device);

    std::cout << "WRONG RESULTS IN " << num_trials << " TRIALS:  MATRIX " << wrong_matrix << ",  VECTOR " << wrong_vector << "\n";
    EXPECT_EQ(wrong_matrix, 0);
    EXPECT_EQ(wrong_vector, 0);
    cudaError_t ierr = cudaGetLastError();
    EXPECT_EQ(ierr, cudaSuccess) << cudaGetErrorString(ierr);
}

// The in-place matrix overload is a cooperative launch, so its grid must fit on the device at once.
// This A asks for more blocks, one per 128 rows, than the device can hold, so the grid is capped by
// the occupancy of the kernel that col_swap_gpu launches.
TEST_F(TestUtil_GPU, test_col_swp_gpu_tall) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int64_t m = (int64_t) 128 * prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor;
    int64_t n = 4;
    ColSwpTestData<double> all_data(m, n);
    std::iota(all_data.A.begin(), all_data.A.end(), 0.0);
    all_data.J = {3, 1, 4, 2};

    col_swp_gpu<double>(all_data);
}
