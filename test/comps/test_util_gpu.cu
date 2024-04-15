#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <RandBLAS/test_util.hh>

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
        std::vector<T> A_host_buffer;
        std::vector<int64_t> J;
        T* A_device;
        T* J_device;

        ColSwpTestData(int64_t m, int64_t n) :
        A(m * n, 0.0),
        A_host_buffer(m * n, 0.0),
        J(n, 0.0)
        {
            row = m;
            col = n;
            cudaMalloc(&A_device, m * n * sizeof(T));
            cudaMalloc(&J_device, n * sizeof(int64_t));
        }
    };

    template <typename T>
    static void 
    test_col_swp_gpu(ColSwpTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        cudaStream_t strm = cudaStreamPerThread;

        char host_name [] = "host";
        char device_name [] = "device";
        RandBLAS::util::print_colmaj(m, n, all_data.A.data(), host_name);

        RandLAPACK::util::col_swap(m, n, n, all_data.A.data(), m, all_data.J);
        RandLAPACK::cuda_kernels::col_swap_gpu(m, n, n, all_data.A_device, m, all_data.J_device, strm);
        cudaMemcpy(all_data.A_host_buffer.data(), all_data.A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost);

        RandBLAS::util::print_colmaj(m, n, all_data.A_host_buffer.data(), device_name);
        RandBLAS::util::print_colmaj(m, n, all_data.A.data(), host_name);

        for(int i = 0; i < m*n; ++i)
            all_data.A[i] -= all_data.A_host_buffer[i];

        T norm_test = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
        printf("Norm diff GPU CPU: %e\n", norm_test);
        ASSERT_NEAR(norm_test, 0.0, std::pow(std::numeric_limits<T>::epsilon(), 0.75));
    }

};

TEST_F(TestUtil, test_col_swp_gpu) {
    
    int64_t m = 5;
    int64_t n = 5;
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

    test_col_swp_gpu<double>(all_data);
}
#endif