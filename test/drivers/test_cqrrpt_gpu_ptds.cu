#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <cuda_runtime.h>

#include <RandBLAS.hh>
#include <gtest/gtest.h>

// This file is compiled with --default-stream=per-thread. The default stream then does not
// synchronize with other streams, so CQRRPT_GPU's copies are ordered against its device work
// only if they run on its own queue.
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "test_cqrrpt_gpu_ptds.cu must be compiled with --default-stream=per-thread"
#endif

// Repeats CQRRPT_GPU on full-rank and rank-deficient inputs, and checks every factorization.
TEST(TestCQRRPTPerThreadStream, CQRRPT_GPU_copies_ordered_on_its_queue) {
    struct Input { int64_t m, n, rank; };
    double tol  = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    int num_reps = 5;
    int num_calls = 0, num_bad = 0;
    for (auto in : {Input{1000, 200, 200}, Input{3000, 300, 300}, Input{2000, 50, 40}}) {
        int64_t m = in.m, n = in.n;
        std::vector<double> A(m * n, 0.0);
        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
        m_info.cond_num = 2;
        m_info.rank = in.rank;
        m_info.exponent = 2.0;
        auto gen_state = RandBLAS::RNGState();
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A.data(), gen_state);
        double norm_A = lapack::lange(Norm::Fro, m, n, A.data(), m);

        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.no_hqrrp = 1;
        for (int rep = 0; rep < num_reps; ++rep) {
            std::vector<double> Q(A), R(n * n, 0.0), AP(A);
            std::vector<int64_t> J(n, 0);
            auto state = RandBLAS::RNGState();
            ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
            int64_t r = CQRRPT_GPU.rank;

            bool finite = true;
            for (int64_t i = 0; i < m * r; ++i)
                finite = finite && std::isfinite(Q[i]);
            for (int64_t i = 0; i < n * n; ++i)
                finite = finite && std::isfinite(R[i]);
            double err_qr = -1, err_orth = -1;
            if (finite && r == in.rank) {
                // ||A[:, J] - QR||_F / ||A||_F and ||Q'Q - I||_F / sqrt(r)
                RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, -1.0, Q.data(), m, R.data(), n, 1.0, AP.data(), m);
                err_qr = lapack::lange(Norm::Fro, m, n, AP.data(), m) / norm_A;
                std::vector<double> G(r * r, 0.0);
                RandLAPACK::util::eye(r, r, G.data());
                blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, r, m, 1.0, Q.data(), m, -1.0, G.data(), r);
                err_orth = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, r, G.data(), r) / std::sqrt((double) r);
            }
            bool ok = finite && r == in.rank && err_qr <= atol && err_orth <= atol;
            EXPECT_TRUE(ok) << m << " x " << n << ", call " << rep << ": rank " << r << " (expected " << in.rank
                            << "), finite " << finite << ", ||AP - QR||/||A|| = " << err_qr << ", ||Q'Q - I||/sqrt(r) = " << err_orth;
            num_bad += !ok;
            ++num_calls;
        }
    }
    ASSERT_EQ(num_bad, 0) << num_bad << " of " << num_calls << " calls gave a wrong factorization";
}
