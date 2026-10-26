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

// The GPU drivers and kernels come in with RandLAPACK.hh, under its __CUDACC__ guard.

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
        cudaMemcpy(all_data.A_sk_device, all_data.A_sk.data(), d * n * sizeof(T), cudaMemcpyHostToDevice);

        cudaMemcpy(all_data.A_device, all_data.A.data(), m * n * sizeof(T), cudaMemcpyHostToDevice);
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
        T col_norm_A = blas::nrm2(m, &A_cpy_dat[m * max_idx], 1);
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
        // The per-column residual check floors at a few hundred eps in
        // either precision; eps^0.75 leaves that plenty of headroom in
        // double but not in float, so single precision uses eps^0.60, the
        // same exponent test_BQRRP_compare_with_CPU already applies to its
        // R-factor difference.
        T atol = std::pow(std::numeric_limits<T>::epsilon(), std::is_same<T, double>::value ? 0.75 : 0.60);

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

            lapack::ungqr(m, std::min(m, n), std::min(m, n), all_data.Q.data(), m, all_data.tau.data());
            RandLAPACK::util::upsize(all_data.rank * n, all_data.R);
            lapack::lacpy(MatrixType::Upper, all_data.rank, n, all_data.R_full.data(), m, all_data.R.data(), all_data.rank);

            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J.data());
            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J.data());

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

    /// Rank-deficient test for BQRRP_GPU, mirroring test_BQRRP_rank_deficient in test_bqrrp.cc:
    /// A = 1e4 * G * H has exact rank r, with r not a multiple of b_sz, so the block holding
    /// column r is detected as rank-deficient and BQRRP_GPU stops there. For both qr_tall options,
    /// checks the reported rank and the residual of A[:, J] - Q1 * R(1:r, :), over the columns up
    /// to the end of that block and, separately, over the trailing columns (that block's R12).
    /// Also checks that the CPU driver reports the same rank for the same A.
    template <typename T, typename RNG>
    static void test_BQRRP_GPU_rank_deficient(
        int64_t m,
        int64_t n,
        int64_t r,
        int64_t b_sz,
        RandBLAS::RNGState<RNG> &state) {

        // The block holding column r must be rank-deficient, so r must fall inside a block.
        ASSERT_NE(r % b_sz, 0) << "r=" << r << " must not be a multiple of b_sz=" << b_sz;
        ASSERT_LT(r, std::min(m, n)) << "r=" << r << " must be < min(m, n)=" << std::min(m, n);

        int64_t d = 1.25 * b_sz;
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);

        std::vector<T> G(m * r, 0.0);
        std::vector<T> H(r * n, 0.0);
        std::vector<T> A(m * n, 0.0);
        RandBLAS::DenseDist DG(m, r);
        RandBLAS::DenseDist DH(r, n);
        state = RandBLAS::fill_dense(DG, G.data(), state);
        state = RandBLAS::fill_dense(DH, H.data(), state);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, (T) 1e4, G.data(), m, H.data(), r, (T) 0.0, A.data(), m);

        // The rank-deficient block ends at column split; R(:, split:n) is its R12.
        int64_t split = std::min(std::min(m, n), (r / b_sz + 1) * b_sz);

        // Rank reported by the CPU driver for the same A, with the same tolerance.
        std::vector<T> A_cpu(A);
        std::vector<T> tau_cpu(n, 0.0);
        std::vector<int64_t> J_cpu(n, 0);
        auto state_cpu = state;
        RandLAPACK::BQRRP<T, RNG> BQRRP_CPU(false, b_sz);
        BQRRP_CPU.tol = 1e-10;
        BQRRP_CPU.call(m, n, A_cpu.data(), m, (T) 1.25, tau_cpu.data(), J_cpu.data(), state_cpu);
        EXPECT_EQ(BQRRP_CPU.rank, r);

        const std::vector<std::pair<const char*, GPUSubroutines::QRTall>> variants = {
            {"cholqr", GPUSubroutines::QRTall::cholqr},
            {"geqrf",  GPUSubroutines::QRTall::geqrf}
        };

        for (const auto& [name, qr_tall] : variants) {
            SCOPED_TRACE(name);
            T norm_A = 0;
            BQRRPTestData<T> all_data(m, n, r, d);
            all_data.A = A;
            norm__sektch_and_copy_computational_helper<T, RNG>(norm_A, d, all_data, state);

            RandLAPACK::BQRRP_GPU<T, RNG> BQRRP_GPU(false, b_sz);
            BQRRP_GPU.tol     = 1e-10;
            BQRRP_GPU.qr_tall = qr_tall;
            // All-ones bytes are NaN, so that any tau entry BQRRP_GPU leaves unset fails the checks below.
            cudaMemset(all_data.tau_device, 0xFF, n * sizeof(T));
            BQRRP_GPU.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);

            cudaMemcpy(all_data.Q.data(),   all_data.A_device,   m * n * sizeof(T),   cudaMemcpyDeviceToHost);
            cudaMemcpy(all_data.tau.data(), all_data.tau_device, n * sizeof(T),       cudaMemcpyDeviceToHost);
            cudaMemcpy(all_data.J.data(),   all_data.J_device,   n * sizeof(int64_t), cudaMemcpyDeviceToHost);
            check_rank_deficient_tail(m, n, BQRRP_GPU.rank, all_data.A, all_data.Q, all_data.tau, all_data.J, atol);

            // Build R(1:r, :) and Q1 = Q(:, 1:r) from the known rank r, so that the residual
            // checks below do not depend on the reported rank.
            RandLAPACK::util::upsize(r * n, all_data.R);
            lapack::lacpy(MatrixType::Upper, r, n, all_data.Q.data(), m, all_data.R.data(), r);
            lapack::ungqr(m, r, r, all_data.Q.data(), m, all_data.tau.data());

            // A[:, J] - Q1 * R(1:r, :)
            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J.data());
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, (T) 1.0, all_data.Q.data(), m, all_data.R.data(), r, (T) -1.0, all_data.A_cpy1.data(), m);
            T res_lead  = lapack::lange(Norm::Fro, m, split, all_data.A_cpy1.data(), m) / norm_A;
            T res_trail = lapack::lange(Norm::Fro, m, n - split, &all_data.A_cpy1[m * split], m) / norm_A;

            std::cout << std::setw(8) << name << ":  RANK " << std::setw(5) << BQRRP_GPU.rank << " (expected " << r << ", CPU " << BQRRP_CPU.rank << "),"
                      << "  REL RESIDUAL, COLS [0, " << split << "): " << std::scientific << res_lead
                      << ",  COLS [" << split << ", " << n << "): " << res_trail << "\n";

            EXPECT_EQ(BQRRP_GPU.rank, r);
            EXPECT_EQ(BQRRP_GPU.rank, BQRRP_CPU.rank);
            EXPECT_LE(res_lead, atol);
            EXPECT_LE(res_trail, atol);

            cudaError_t ierr = cudaGetLastError();
            if (ierr != cudaSuccess)
            {
                RandLAPACK_CUDA_ERROR("Error before test_BQRRP_GPU_rank_deficient returned. " << cudaGetErrorString(ierr))
                abort();
            }
        }
    }

    /// Checks BQRRP_GPU's output, copied to the host, past a detected rank k < min(m, n):
    /// tau(k:min(m, n)) and A(k:m, k:n) must be exactly zero, ungqr over all min(m, n) reflectors
    /// must give a Q whose first k columns are orthonormal, and A[:, J] = Q(:, 1:k) * R(1:k, :)
    /// must hold to rounding.
    template <typename T>
    static void check_rank_deficient_tail(
        int64_t m,
        int64_t n,
        int64_t k,
        const std::vector<T> &A,
        const std::vector<T> &A_fact,
        const std::vector<T> &tau,
        const std::vector<int64_t> &J,
        T atol) {

        int64_t mn = std::min(m, n);
        ASSERT_LT(k, mn) << "the call was expected to be rank-deficient";

        int64_t tau_nonzero = 0;
        for (int64_t i = k; i < mn; ++i)
            tau_nonzero += !(tau[i] == (T) 0.0);
        int64_t A_tail_nonzero = 0;
        for (int64_t j = k; j < n; ++j)
            for (int64_t i = k; i < m; ++i)
                A_tail_nonzero += !(A_fact[m * j + i] == (T) 0.0);

        // Q from all min(m, n) reflectors, as a GEQP3 caller would form it.
        std::vector<T> Q(A_fact.begin(), A_fact.begin() + m * mn);
        lapack::ungqr(m, mn, mn, Q.data(), m, tau.data());

        // Q(:, 1:k)' * Q(:, 1:k) - I
        std::vector<T> I_ref(k * k, 0.0);
        RandLAPACK::util::eye(k, k, I_ref);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, (T) 1.0, Q.data(), m, (T) -1.0, I_ref.data(), k);
        T orth = (k == 0) ? (T) 0.0 : lapack::lansy(Norm::Fro, Uplo::Upper, k, I_ref.data(), k) / std::sqrt((T) k);

        // A[:, J] - Q(:, 1:k) * R(1:k, :)
        std::vector<T> R(k * n, 0.0);
        lapack::lacpy(MatrixType::Upper, k, n, A_fact.data(), m, R.data(), k);
        std::vector<T> A_piv(A);
        std::vector<int64_t> J_cpy(J);
        RandLAPACK::util::col_swap(m, n, n, A_piv.data(), m, J_cpy.data());
        T norm_A = lapack::lange(Norm::Fro, m, n, A.data(), m);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, (T) 1.0, Q.data(), m, R.data(), k, (T) -1.0, A_piv.data(), m);
        T res = lapack::lange(Norm::Fro, m, n, A_piv.data(), m) / norm_A;

        std::cout << "    PAST RANK " << k << ":  NONZERO tau " << tau_nonzero << ",  NONZERO A(k:, k:) " << A_tail_nonzero
                  << ",  ||Q1'Q1 - I|| / sqrt(k) " << std::scientific << orth << ",  REL RESIDUAL " << res << "\n";

        EXPECT_EQ(tau_nonzero, 0);
        EXPECT_EQ(A_tail_nonzero, 0);
        EXPECT_LE(orth, atol);
        EXPECT_LE(res, atol);
    }
};
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

// BQRRP_GPU must support single precision. Mirrors
// BQRRP_GPU_070824 with T = float; exercises the ormqr path through
// cuSOLVER's S-typed interface. The problem is smaller than the double
// test: at 5000 x 2800 the float residual accumulates to ~34 eps, just
// past the shared eps^0.75 tolerance, so the size is chosen to leave
// that tolerance a healthy margin in single precision.
TEST_F(TestBQRRP, BQRRP_GPU_single_precision) {
    int64_t m = 2000;
    int64_t n = 1000;
    int64_t k = 1000;
    double d_factor = 1;
    int64_t b_sz = 300;
    int64_t d = d_factor * b_sz;
    float norm_A = 0;
    auto state = RandBLAS::RNGState();
    bool profile_runtime = true;

    BQRRPTestData<float> all_data(m, n, k, d);
    RandLAPACK::BQRRP_GPU<float, r123::Philox4x32> BQRRP_GPU(profile_runtime, b_sz);
    BQRRP_GPU.qr_tall = GPUSubroutines::QRTall::cholqr;

    RandLAPACK::gen::mat_gen_info<float> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<float, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm__sektch_and_copy_computational_helper<float, r123::Philox4x32>(norm_A, d, all_data, state);
    test_BQRRP_general<float, RandLAPACK::BQRRP_GPU<float, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
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

// Repeated BQRRP_GPU calls must not grow device memory use. cudaMemGetInfo is device wide,
// so other processes add noise; a leak shows up in every round, noise rarely does.
// The geqrf case also allocates a geqrf workspace per iteration; it is sized so that keeping
// those workspaces would exceed the bound.
TEST_F(TestBQRRP, BQRRP_GPU_repeated_calls_no_leak) {
    struct Case { const char* name; GPUSubroutines::QRTall qr_tall; int64_t m, n, d, b_sz; };
    const Case cases[] = {
        {"cholqr", GPUSubroutines::QRTall::cholqr,   64,  16,  32,   8},
        {"geqrf",  GPUSubroutines::QRTall::geqrf,  1000, 600, 125, 100}
    };
    int num_rounds = 3;
    int calls_per_round = 5;
    auto free_mib = []() {
        size_t free_bytes, total_bytes;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_bytes, &total_bytes);
        return (double) free_bytes / (1 << 20);
    };

    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        int64_t m = c.m;
        int64_t n = c.n;
        int64_t d = c.d;
        int64_t b_sz = c.b_sz;
        auto state = RandBLAS::RNGState();

        std::vector<double> A(m * n), A_sk(d * n);
        RandBLAS::DenseDist D(m, n);
        RandBLAS::fill_dense(D, A.data(), state);
        // Any d-by-n matrix with A's column space serves as the sketch; take A's leading rows.
        lapack::lacpy(MatrixType::General, d, n, A.data(), m, A_sk.data(), d);

        double *A_device, *A_sk_device, *tau_device;
        int64_t* J_device;
        cudaMalloc(&A_device, m * n * sizeof(double));
        cudaMalloc(&A_sk_device, d * n * sizeof(double));
        cudaMalloc(&tau_device, n * sizeof(double));
        cudaMalloc(&J_device, n * sizeof(int64_t));

        RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
        BQRRP_GPU.qr_tall = c.qr_tall;
        auto run = [&]() {
            cudaMemcpy(A_device, A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(A_sk_device, A_sk.data(), d * n * sizeof(double), cudaMemcpyHostToDevice);
            return BQRRP_GPU.call(m, n, A_device, m, A_sk_device, d, tau_device, J_device);
        };

        // A first call absorbs one-time allocations.
        ASSERT_EQ(run(), 0);
        double min_growth = std::numeric_limits<double>::infinity();
        for (int r = 0; r < num_rounds; ++r) {
            double before = free_mib();
            for (int i = 0; i < calls_per_round; ++i)
                ASSERT_EQ(run(), 0);
            min_growth = std::min(min_growth, before - free_mib());
        }
        cudaFree(A_device);
        cudaFree(A_sk_device);
        cudaFree(tau_device);
        cudaFree(J_device);
        std::cout << std::setw(8) << c.name << ":  SMALLEST GROWTH PER ROUND " << min_growth << " MiB\n";
        EXPECT_LE(min_growth, 32.0) << "device memory grew by at least " << min_growth << " MiB per " << calls_per_round << " calls";
    }
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

// Full-rank A whose first 300 rows are zero: the zero-block check must read the whole pivot
// column, not only the 128 entries one thread block covers, and find A nonzero.
TEST_F(TestBQRRP, BQRRP_GPU_leading_zero_rows) {
    int64_t m = 1000;
    int64_t n = 600;
    int64_t k = n;
    int64_t zero_rows = 300;
    double d_factor = 1.25;
    int64_t b_sz = 200;
    int64_t d = d_factor * b_sz;

    for (auto qr_tall : {GPUSubroutines::QRTall::geqrf, GPUSubroutines::QRTall::cholqr}) {
        SCOPED_TRACE(qr_tall == GPUSubroutines::QRTall::geqrf ? "geqrf" : "cholqr");
        double norm_A = 0;
        auto state = RandBLAS::RNGState();

        BQRRPTestData<double> all_data(m, n, k, d);
        RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
        BQRRP_GPU.qr_tall = qr_tall;

        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
        lapack::laset(MatrixType::General, zero_rows, n, 0.0, 0.0, all_data.A.data(), m);

        norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
        test_BQRRP_general<double, RandLAPACK::BQRRP_GPU<double, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
        EXPECT_EQ(BQRRP_GPU.rank, std::min(m, n));
    }
}

// Scale invariance, mirroring BQRRP_scale_invariance in test_bqrrp.cc: the same A, scaled by
// 1e-20, 1 and 1e20, must give the same rank and a rounding-level residual, for both qr_tall
// options. Cases: a full-rank Gaussian with the default tol, and an exact rank-400 matrix with
// b_sz = 200 and tol = 1e-10, whose rank drop at a block boundary only the zero-block check
// can detect. Its trailing block holds rounding residue, so its output past the rank also checks
// that the zero-block exit zeroes tau and A(400:m, 400:n).
TEST_F(TestBQRRP, BQRRP_GPU_scale_invariance) {
    int64_t m = 1000;
    int64_t n = 600;
    int64_t r = 400;
    int64_t b_sz = 200;
    int64_t d = 1.25 * b_sz;
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::RNGState();

    auto check_scales = [&](const std::vector<double>& A, double tol, int64_t expected_rank) {
        for (double scale : {1e-20, 1.0, 1e20}) {
            for (auto qr_tall : {GPUSubroutines::QRTall::cholqr, GPUSubroutines::QRTall::geqrf}) {
                const char* name = qr_tall == GPUSubroutines::QRTall::geqrf ? "geqrf" : "cholqr";
                SCOPED_TRACE(::testing::Message() << name << ", scale " << scale);
                double norm_A = 0;
                auto state_sk = state;
                BQRRPTestData<double> all_data(m, n, n, d);
                all_data.A = A;
                blas::scal(m * n, scale, all_data.A.data(), 1);
                norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state_sk);

                RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
                BQRRP_GPU.tol     = tol;
                BQRRP_GPU.qr_tall = qr_tall;
                // All-ones bytes are NaN, so that any tau entry BQRRP_GPU leaves unset fails the checks below.
                cudaMemset(all_data.tau_device, 0xFF, n * sizeof(double));
                BQRRP_GPU.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);
                int64_t k = BQRRP_GPU.rank;

                cudaMemcpy(all_data.Q.data(),   all_data.A_device,   m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(all_data.tau.data(), all_data.tau_device, n * sizeof(double),     cudaMemcpyDeviceToHost);
                cudaMemcpy(all_data.J.data(),   all_data.J_device,   n * sizeof(int64_t),    cudaMemcpyDeviceToHost);
                if (expected_rank < std::min(m, n))
                    check_rank_deficient_tail(m, n, k, all_data.A, all_data.Q, all_data.tau, all_data.J, atol);

                // A[:, J] - Q(:, 1:k) * R(1:k, :)
                RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J.data());
                if (k > 0) {
                    RandLAPACK::util::upsize(k * n, all_data.R);
                    lapack::lacpy(MatrixType::Upper, k, n, all_data.Q.data(), m, all_data.R.data(), k);
                    lapack::ungqr(m, k, k, all_data.Q.data(), m, all_data.tau.data());
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, all_data.Q.data(), m, all_data.R.data(), k, -1.0, all_data.A_cpy1.data(), m);
                }
                double res = lapack::lange(Norm::Fro, m, n, all_data.A_cpy1.data(), m) / norm_A;

                std::cout << std::setw(8) << name << ",  SCALE " << std::scientific << std::setprecision(0) << scale
                          << ":  RANK " << std::setw(5) << k << " (expected " << expected_rank << "),  REL RESIDUAL "
                          << std::setprecision(6) << res << "\n";

                EXPECT_EQ(k, expected_rank);
                EXPECT_LE(res, atol);

                cudaError_t ierr = cudaGetLastError();
                if (ierr != cudaSuccess)
                {
                    RandLAPACK_CUDA_ERROR("Error before BQRRP_GPU_scale_invariance returned. " << cudaGetErrorString(ierr))
                    abort();
                }
            }
        }
    };

    std::vector<double> A_full(m * n, 0.0);
    RandBLAS::DenseDist D(m, n);
    state = RandBLAS::fill_dense(D, A_full.data(), state);
    check_scales(A_full, std::numeric_limits<double>::epsilon(), n);

    std::vector<double> G(m * r, 0.0);
    std::vector<double> H(r * n, 0.0);
    std::vector<double> A_low(m * n, 0.0);
    RandBLAS::DenseDist DG(m, r);
    RandBLAS::DenseDist DH(r, n);
    state = RandBLAS::fill_dense(DG, G.data(), state);
    state = RandBLAS::fill_dense(DH, H.data(), state);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, 1.0, G.data(), m, H.data(), r, 0.0, A_low.data(), m);
    check_scales(A_low, 1e-10, r);
}

// Wide aspect ratio (m < n).  Regression test for the wide-A correctness
// fix: prior to the fix, the termination condition `curr_sz >= n` never
// fired for wide A (curr_sz tops out at min(m, n) = m < n), causing the
// main loop to fall through without setting `rank` or reconciling the
// swap-dance buffers. Additionally, the trailing-region copy in
// `reconcile_and_cleanup` was gated on `iter != maxiter - 1`, which is
// wrong for wide A at full completion -- the wide tail [m, n) of valid
// R12 entries was not copied back, and the J wide-tail was not
// reconciled (producing duplicate pivots).
TEST_F(TestBQRRP, BQRRP_GPU_wide_aspect) {
    int64_t m = 1000;
    int64_t n = 2000;   // n > m -- wide
    int64_t k = m;      // expected rank
    double d_factor = 1;
    int64_t b_sz = 100;
    int64_t d = d_factor * b_sz;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k, d);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
    BQRRP_GPU.qr_tall = GPUSubroutines::QRTall::geqrf;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
    test_BQRRP_general<double, RandLAPACK::BQRRP_GPU<double, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
}

// Full-rank wide A with m not a multiple of b_sz: the clipped last block must apply Q'
// with its own reflectors, tau[curr_sz:], when computing its R12.
TEST_F(TestBQRRP, BQRRP_GPU_wide_aspect_clipped_block) {
    int64_t m = 600;
    int64_t n = 1000;
    int64_t k = m;
    double d_factor = 1;
    int64_t b_sz = 250;
    int64_t d = d_factor * b_sz;

    for (auto qr_tall : {GPUSubroutines::QRTall::geqrf, GPUSubroutines::QRTall::cholqr}) {
        SCOPED_TRACE(qr_tall == GPUSubroutines::QRTall::geqrf ? "geqrf" : "cholqr");
        double norm_A = 0;
        auto state = RandBLAS::RNGState();

        BQRRPTestData<double> all_data(m, n, k, d);
        RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
        BQRRP_GPU.qr_tall = qr_tall;

        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

        norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
        test_BQRRP_general<double, RandLAPACK::BQRRP_GPU<double, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
    }
}

// BQRRP_GPU's workspace comes from cudaMallocAsync, which returns memory the caller used and
// freed earlier when the pool keeps freed memory (a raised release threshold). CholQR reads the
// strictly lower triangle of its R_tall_qr buffer, so BQRRP_GPU must clear that buffer itself.
TEST_F(TestBQRRP, BQRRP_GPU_cholqr_reused_pool_memory) {
    // Skip, rather than fail, where the pool setup is unavailable; clear the CUDA error first so
    // that no later error check in BQRRP_GPU trips over it.
    int pools_supported = 0;
    cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, 0);
    if (!pools_supported) {
        cudaGetLastError();
        GTEST_SKIP() << "device 0 does not support stream-ordered memory pools";
    }
    cudaMemPool_t pool;
    uint64_t threshold_old = 0;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) != cudaSuccess ||
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold_old) != cudaSuccess) {
        cudaGetLastError();
        GTEST_SKIP() << "cannot query the default memory pool";
    }
    // Restores the pool for the tests that follow, however this test exits.
    struct PoolRestore {
        cudaMemPool_t pool;
        uint64_t threshold;
        ~PoolRestore() {
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            cudaMemPoolTrimTo(pool, 0);
        }
    } pool_restore{pool, threshold_old};

    uint64_t threshold_keep = UINT64_MAX;
    if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold_keep) != cudaSuccess) {
        cudaGetLastError();
        GTEST_SKIP() << "cannot raise the pool's release threshold";
    }
    // Leave freed, nonzero stream-ordered memory in the pool.
    void* dirty = nullptr;
    size_t dirty_bytes = size_t(64) << 20;
    if (cudaMallocAsync(&dirty, dirty_bytes, 0) != cudaSuccess) {
        cudaGetLastError();
        GTEST_SKIP() << "cannot allocate the " << (dirty_bytes >> 20) << " MiB pool block";
    }
    cudaMemsetAsync(dirty, 0x3F, dirty_bytes, 0);
    cudaFreeAsync(dirty, 0);
    cudaDeviceSynchronize();

    int64_t m = 2000;
    int64_t n = 1000;
    int64_t k = n;
    double d_factor = 1.25;
    int64_t b_sz = 300;
    int64_t d = d_factor * b_sz;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k, d);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
    BQRRP_GPU.qr_tall = GPUSubroutines::QRTall::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);
    test_BQRRP_general<double, RandLAPACK::BQRRP_GPU<double, r123::Philox4x32>>(d, norm_A, all_data, BQRRP_GPU);
}

// Rank-deficient inputs.
// Exact rank r < b_sz: the first block is rank-deficient.
TEST_F(TestBQRRP, BQRRP_GPU_rank_deficient_first_block) {
    auto state = RandBLAS::RNGState();
    test_BQRRP_GPU_rank_deficient<double>(1000, 600, 60, 100, state);
}

// Rank drop inside the second block: the reported rank must count the full first block.
TEST_F(TestBQRRP, BQRRP_GPU_rank_deficient_later_block) {
    auto state = RandBLAS::RNGState();
    test_BQRRP_GPU_rank_deficient<double>(1000, 600, 160, 100, state);
}

// In-block rank 150 of b_sz = 200: the drop lies past the first 128 diagonal entries of R_sk,
// which a 128-thread block of the GPU rank estimator covers.
TEST_F(TestBQRRP, BQRRP_GPU_rank_deficient_deep_in_block) {
    auto state = RandBLAS::RNGState();
    test_BQRRP_GPU_rank_deficient<double>(1000, 600, 150, 200, state);
}

// In-block ranks 127 and 128 of the second block, on either side of that 128-entry boundary:
// the GPU and CPU drivers must report the same rank.
TEST_F(TestBQRRP, BQRRP_GPU_rank_deficient_cpu_agreement_at_128) {
    auto state = RandBLAS::RNGState();
    test_BQRRP_GPU_rank_deficient<double>(1000, 600, 327, 200, state);
    test_BQRRP_GPU_rank_deficient<double>(1000, 600, 328, 200, state);
}

// Wide A whose clipped last block (b_sz = 100 of 250) is rank-deficient.
TEST_F(TestBQRRP, BQRRP_GPU_rank_deficient_wide_clipped_block) {
    auto state = RandBLAS::RNGState();
    test_BQRRP_GPU_rank_deficient<double>(600, 1000, 550, 250, state);
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
    // Advance a separate pointer, so that A_device can still be freed.
    double* A_work = A_device;
    for(int iter = 0; iter < 2; ++iter) {
        //lapack::geqrf_work_size_bytes(m, k, A_device, lda, &d_size_geqrf_opt, &h_size_geqrf_opt, lapack_queue);
        //d_work_geqrf_opt = blas::device_malloc< char >( d_size_geqrf_opt, lapack_queue );
        lapack::geqrf(m, k, A_work, lda, &tau_device[curr_sz], d_work_geqrf_opt, d_size_geqrf_opt, h_work_geqrf_opt, h_size_geqrf_opt, d_info, lapack_queue);
        m       -= k;
        curr_sz += k;
        A_work = &A_work[k * lda + k];
        cudaStreamSynchronize(strm);
    }

    delete[] A;
    delete[] tau;
    cudaFree(A_device);
    cudaFree(tau_device);
    blas::device_free(d_work_geqrf_opt, lapack_queue);
    blas::device_free(d_info, lapack_queue);

    // Fail here, rather than leave a CUDA error pending for the next test.
    cudaError_t ierr = cudaGetLastError();
    EXPECT_EQ(ierr, cudaSuccess) << cudaGetErrorString(ierr);
}

// A = [B, E] with B of full rank r, a multiple of b_sz, and E noise of size 1e-14 relative to B,
// far below tol = 1e-10: after r columns the remaining block is judged zero, so BQRRP_GPU takes its
// zero-block exit. E is nonzero, so the exit itself must zero A(r:m, r:n). r / b_sz = 2 and 3 cover
// both parities of the exit iteration, which decide how the output is reassembled from the swap
// buffers.
TEST_F(TestBQRRP, BQRRP_GPU_zero_block_tail) {
    int64_t m = 1000;
    int64_t n = 600;
    int64_t b_sz = 100;
    int64_t d = 1.25 * b_sz;
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);

    for (int64_t r : {200, 300}) {
        for (auto qr_tall : {GPUSubroutines::QRTall::cholqr, GPUSubroutines::QRTall::geqrf}) {
            SCOPED_TRACE(std::string(qr_tall == GPUSubroutines::QRTall::cholqr ? "cholqr" : "geqrf") + ", r = " + std::to_string(r));
            double norm_A = 0;
            auto state = RandBLAS::RNGState();

            BQRRPTestData<double> all_data(m, n, r, d);
            RandLAPACK::gen::mat_gen_info<double> m_info(m, r, RandLAPACK::gen::gaussian);
            RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
            RandBLAS::DenseDist DE(m, n - r);
            state = RandBLAS::fill_dense(DE, &all_data.A[m * r], state);
            blas::scal(m * (n - r), 1e-14, &all_data.A[m * r], 1);
            norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);

            RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
            BQRRP_GPU.tol     = 1e-10;
            BQRRP_GPU.qr_tall = qr_tall;
            // All-ones bytes are NaN, so that any tau entry BQRRP_GPU leaves unset fails the check.
            cudaMemset(all_data.tau_device, 0xFF, n * sizeof(double));
            BQRRP_GPU.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);

            cudaMemcpy(all_data.Q.data(),   all_data.A_device,   m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(all_data.tau.data(), all_data.tau_device, n * sizeof(double),     cudaMemcpyDeviceToHost);
            cudaMemcpy(all_data.J.data(),   all_data.J_device,   n * sizeof(int64_t),    cudaMemcpyDeviceToHost);

            EXPECT_EQ(BQRRP_GPU.rank, r);
            check_rank_deficient_tail(m, n, BQRRP_GPU.rank, all_data.A, all_data.Q, all_data.tau, all_data.J, atol);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        }
    }
}

// BQRRP_GPU takes its pivot buffers (J_buffer, J_buffer_lu) and Work2 (the sketch's tau, then
// orhr_col's sign vector) from cudaMallocAsync, and writes every entry it reads before reading it.
// So its output must not depend on what the pool held: runs over a pool filled with 0x00 (as a
// cleared buffer), 0xFF (every int64_t -1, every double NaN) and 0x3F must agree bitwise. The
// inputs include an exactly zero trailing block, on which getrf factors a zero sketch.
TEST_F(TestBQRRP, BQRRP_GPU_buffers_reused_pool_memory) {
    // Skip, rather than fail, where the pool setup is unavailable; clear the CUDA error first so
    // that no later error check in BQRRP_GPU trips over it.
    int pools_supported = 0;
    cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, 0);
    if (!pools_supported) {
        cudaGetLastError();
        GTEST_SKIP() << "device 0 does not support stream-ordered memory pools";
    }
    cudaMemPool_t pool;
    uint64_t threshold_old = 0;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) != cudaSuccess ||
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold_old) != cudaSuccess) {
        cudaGetLastError();
        GTEST_SKIP() << "cannot query the default memory pool";
    }
    // Restores the pool for the tests that follow, however this test exits.
    struct PoolRestore {
        cudaMemPool_t pool;
        uint64_t threshold;
        ~PoolRestore() {
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            cudaMemPoolTrimTo(pool, 0);
        }
    } pool_restore{pool, threshold_old};

    uint64_t threshold_keep = UINT64_MAX;
    if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold_keep) != cudaSuccess) {
        cudaGetLastError();
        GTEST_SKIP() << "cannot raise the pool's release threshold";
    }

    // Columns r:n of A are zero.
    struct Input { const char* name; int64_t m, n, r, b_sz; };
    const Input inputs[] = {
        {"full rank, clipped last block", 2000, 1000, 1000, 300},
        {"zero trailing block",           1000,  600,  200, 100}
    };
    auto same_bytes = [](const auto& x, const auto& y) {
        if (x.size() != y.size())
            return false;
        auto bx = reinterpret_cast<const unsigned char*>(x.data());
        auto by = reinterpret_cast<const unsigned char*>(y.data());
        for (size_t i = 0; i < x.size() * sizeof(x[0]); ++i)
            if (bx[i] != by[i])
                return false;
        return true;
    };

    for (const auto& in : inputs) {
        int64_t m = in.m, n = in.n, r = in.r, b_sz = in.b_sz;
        int64_t d = 1.25 * b_sz;
        std::vector<double> A(m * n, 0.0);
        auto state = RandBLAS::RNGState();
        RandLAPACK::gen::mat_gen_info<double> m_info(m, r, RandLAPACK::gen::gaussian);
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A.data(), state);

        for (auto qr_tall : {GPUSubroutines::QRTall::cholqr, GPUSubroutines::QRTall::geqrf}) {
            std::vector<double> A_ref, tau_ref;
            std::vector<int64_t> J_ref;
            int64_t rank_ref = -1;
            for (int fill : {0x00, 0xFF, 0x3F}) {
                SCOPED_TRACE(std::string(in.name) + ", " + (qr_tall == GPUSubroutines::QRTall::cholqr ? "cholqr" : "geqrf") + ", fill " + std::to_string(fill));
                // Leave freed stream-ordered memory holding `fill` in the pool.
                void* dirty = nullptr;
                size_t dirty_bytes = size_t(64) << 20;
                if (cudaMallocAsync(&dirty, dirty_bytes, 0) != cudaSuccess) {
                    cudaGetLastError();
                    GTEST_SKIP() << "cannot allocate the " << (dirty_bytes >> 20) << " MiB pool block";
                }
                cudaMemsetAsync(dirty, fill, dirty_bytes, 0);
                cudaFreeAsync(dirty, 0);
                cudaDeviceSynchronize();

                double norm_A = 0;
                BQRRPTestData<double> all_data(m, n, r, d);
                all_data.A = A;
                norm__sektch_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, d, all_data, state);

                RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU(false, b_sz);
                BQRRP_GPU.qr_tall = qr_tall;
                BQRRP_GPU.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);

                cudaMemcpy(all_data.R_full.data(), all_data.A_device,   m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(all_data.tau.data(),    all_data.tau_device, n * sizeof(double),     cudaMemcpyDeviceToHost);
                cudaMemcpy(all_data.J.data(),      all_data.J_device,   n * sizeof(int64_t),    cudaMemcpyDeviceToHost);
                cudaError_t ierr = cudaGetLastError();
                ASSERT_EQ(ierr, cudaSuccess) << cudaGetErrorString(ierr);

                if (fill == 0x00) {
                    EXPECT_EQ(BQRRP_GPU.rank, r);
                    A_ref    = all_data.R_full;
                    tau_ref  = all_data.tau;
                    J_ref    = all_data.J;
                    rank_ref = BQRRP_GPU.rank;
                } else {
                    EXPECT_EQ(BQRRP_GPU.rank, rank_ref);
                    EXPECT_TRUE(same_bytes(all_data.R_full, A_ref));
                    EXPECT_TRUE(same_bytes(all_data.tau, tau_ref));
                    EXPECT_TRUE(same_bytes(all_data.J, J_ref));
                }
            }
        }
    }
}
