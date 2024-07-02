#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <chrono>
#include <gtest/gtest.h>

// Use cuda kernels.
#ifndef USE_CUDA
#define USE_CUDA

#include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"

using namespace std::chrono;


class TestUtil : public ::testing::Test
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
        }
    };

    template <typename T>
    static void 
    col_swp_gpu(ColSwpTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        cudaStream_t strm = cudaStreamPerThread;

        //char input_name [] = "input";
        //char host_name [] = "host";
        //char device_name [] = "device";
        //RandBLAS::util::print_colmaj(m, n, all_data.A.data(), input_name);

        RandLAPACK::util::col_swap(m, n, n, all_data.A.data(), m, all_data.J);
        RandLAPACK::cuda_kernels::col_swap_gpu(m, n, n, all_data.A_device, m, all_data.J_device, all_data.buf_device, strm);
        cudaMemcpy(all_data.A_host_buffer.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(all_data.J_host_buffer.data(), all_data.J_device, n * sizeof(int64_t), cudaMemcpyDeviceToHost);

        //RandBLAS::util::print_colmaj(m, n, all_data.A_host_buffer.data(), device_name);
        //RandBLAS::util::print_colmaj(m, n, all_data.A.data(), host_name);

        for(int i = 0; i < m*n; ++i)
            all_data.A[i] -= all_data.A_host_buffer[i];

        T norm_test = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
        printf("\nNorm diff GPU CPU: %e\n", norm_test);
        ASSERT_NEAR(norm_test, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

    template <typename T>
    static void 
    qp3_swp_gpu(ColSwpTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        cudaStream_t strm = cudaStreamPerThread;
    
        //char input_name [] = "input";
        //char host_name [] = "host";
        //char device_name [] = "device";
        //RandBLAS::util::print_colmaj(m, n, all_data.A.data(), input_name);

        // Perform Pivoted QR
        lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());

        // Swap columns in A's copy
        cudaMemcpy(all_data.J_device, all_data.J.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
        RandLAPACK::cuda_kernels::col_swap_gpu(m, n, n, all_data.A_device, m, all_data.J_device, all_data.buf_device, strm);
        cudaMemcpy(all_data.A_host_buffer.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost);

       // RandBLAS::util::print_colmaj(m, n, all_data.A_host_buffer.data(), device_name);

        // Create an identity and store Q in it.
        RandLAPACK::util::eye(m, n, all_data.Ident.data());
        lapack::ormqr(Side::Left, Op::NoTrans, m, n, n, all_data.A.data(), m,  all_data.tau.data(),  all_data.Ident.data(), m);

        // Q * R -> Identity space
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.A.data(), m, all_data.Ident.data(), m);

        //RandBLAS::util::print_colmaj(m, n, all_data.Ident.data(), host_name);

        // A_piv - A_cpy
        for(int i = 0; i < m * n; ++i)
            all_data.A_host_buffer[i] -= all_data.Ident[i];

        T norm = lapack::lange(Norm::Fro, m, n, all_data.A_host_buffer.data(), m);
        printf("||A_piv - QR||_F:  %e\n", norm);
        ASSERT_NEAR(norm, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.625));
    }

};

TEST_F(TestUtil, test_col_swp_gpu) {
    
    int64_t m = 129;
    int64_t n = 9;
    auto state = RandBLAS::RNGState();
    ColSwpTestData<double> all_data(m, n);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2025;
    m_info.rank = n;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state); 
    cudaMemcpy(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Fill and randomly shuffle a vector
    std::iota(all_data.J.begin(), all_data.J.end(), 1);
    std::random_shuffle(all_data.J.begin(), all_data.J.begin() + n);
    cudaMemcpy(all_data.J_device, all_data.J.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);

    col_swp_gpu<double>(all_data);
}

TEST_F(TestUtil, test_qp3_swp_gpu) {
    
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
    cudaMemcpy(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

    qp3_swp_gpu<double>(all_data);
}
#endif