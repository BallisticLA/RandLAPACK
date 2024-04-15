#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

#ifndef USE_CUDA
#define USE_CUDA
#include "RandLAPACK/drivers/rl_cqrrpt_gpu.hh"

class TestCQRRPT : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct CQRRPTTestData {
        int64_t row;
        int64_t col;
        int64_t rank; // has to be modifiable
        std::vector<T> A;
        std::vector<T> R;
        std::vector<int64_t> J;
        std::vector<T> A_cpy1;
        std::vector<T> A_cpy2;
        std::vector<T> I_ref;

        CQRRPTTestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0), 
        R(n * n, 0.0),
        J(n, 0),  
        A_cpy1(m * n, 0.0),
        A_cpy2(m * n, 0.0),
        I_ref(k * k, 0.0) 
        {
            row = m;
            col = n;
            rank = k;
        }
    };

    template <typename T, typename RNG>
    static void norm_and_copy_computational_helper(T &norm_A, CQRRPTTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, CQRRPTTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        RandLAPACK::util::upsize(k * k, all_data.I_ref);
        RandLAPACK::util::eye(k, k, all_data.I_ref);

        T* A_dat         = all_data.A_cpy1.data();
        T const* A_cpy_dat = all_data.A_cpy2.data();
        T const* Q_dat   = all_data.A.data();
        T const* R_dat   = all_data.R.data();
        T* I_ref_dat     = all_data.I_ref.data();

        // Check orthogonality of Q
        // Q' * Q  - I = 0
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, I_ref_dat, k);
        T norm_0 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);

        // A - QR
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, R_dat, n, -1.0, A_dat, m);
        
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
        
        printf("REL NORM OF AP - QR:    %15e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC:    %15e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF (Q'Q - I)/sqrt(n): %2e\n\n", norm_0 / std::sqrt((T) n));

        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_NEAR(norm_AQR / norm_A,         0.0, atol);
        ASSERT_NEAR(max_col_norm / col_norm_A, 0.0, atol);
        ASSERT_NEAR(norm_0 / std::sqrt((T) n), 0.0, atol);
    }

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRPT_general(
        T d_factor, 
        T norm_A,
        CQRRPTTestData<T> &all_data,
        alg_type &CQRRPT_GPU,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRPT_GPU.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state);

        all_data.rank = CQRRPT_GPU.rank;
        printf("RANK AS RETURNED BY CQRRPT %ld\n", all_data.rank);

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);

        error_check(norm_A, all_data); 
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, CQRRPT_GPU_full_rank_no_hqrrp) {
    int64_t m = 1000;
    int64_t n = 200;
    int64_t k = 200;
    double d_factor = 2.0;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.num_threads = 4;
    CQRRPT_GPU.no_hqrrp = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32, RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32>>(d_factor, norm_A, all_data, CQRRPT_GPU, state);
}

TEST_F(TestCQRRPT, something) {
    // Simple cuda gemm
    int64_t m = 10;
    int64_t n = 5;
    auto state = RandBLAS::RNGState();
    std::vector<double> A (m * n, 0.0);
    std::vector<double> B (n * n, 0.0);
    std::vector<double> C (m * n, 0.0);
    std::vector<double> C_received (m * n, 0.0);

    RandBLAS::DenseDist D1(m, n);
    RandBLAS::DenseDist D2(n, n);
    state = RandBLAS::fill_dense(D1, A.data(), state).second;
    state = RandBLAS::fill_dense(D2, B.data(), state).second;

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A.data(), m, B.data(), n, 1.0, C.data(), m);
    char name1 [] = "C host";
    RandBLAS::util::print_colmaj(m, n, C.data(), name1);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);

    blas::Queue blas_queue(0);
    double* A_device;
    double* B_device;
    double* C_device;
    cudaMalloc(&A_device, m * n * sizeof(double));
    cudaMalloc(&B_device, n * n * sizeof(double));
    cudaMalloc(&C_device, m * n * sizeof(double));

    cudaMemcpy(A_device, A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_device, m, B_device, n, 1.0, C_device, m, blas_queue);
    //blas_queue.sync();

    cudaMemcpy(C_received.data(), C_device,  m * n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    char name2 [] = "C received device";
    RandBLAS::util::print_colmaj(m, n, C_received.data(), name2);
}
/*
TEST_F(TestCQRRPT, something1) {
    // Matrix dimensions
    int m = 3; // Number of rows
    int n = 2; // Number of columns

    // Host matrices
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> B = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    // Host workspace and info
    std::vector<float> work;
    int lwork;

    int *devInfo = nullptr;

    // CUDA device pointers
    float *d_A = nullptr;
    float *d_B = nullptr;

    // CUDA handles
    cusolverDnHandle_t cusolverH = nullptr;

    // Create cusolver handle
    cusolverDnCreate(&cusolverH);

    // Allocate CUDA memory for matrices and workspace
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_B, m * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_A, A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), m * sizeof(float), cudaMemcpyHostToDevice);

    // Compute optimal workspace size
    cusolverDnSormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, 1, n, d_A, m, nullptr, d_B, m, nullptr);

    // Allocate workspace
    work.resize(lwork);
    cudaMalloc((void**)&work[0], lwork * sizeof(float));

    // Perform QR factorization and solve the least squares problem
    cusolverDnSormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, 1, n, d_A, m, nullptr, d_B, m, nullptr, lwork, devInfo);
}
*/
#endif
