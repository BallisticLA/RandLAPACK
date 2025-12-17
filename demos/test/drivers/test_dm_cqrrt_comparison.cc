// Direct comparison between dense CQRRT and CQRRT_linops using identical inputs
#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>

#include "../../functions/drivers/dm_cqrrt_linops.hh"

using namespace RandLAPACK_demos;

class TestCQRRTComparison : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// Compare dense CQRRT vs CQRRT_linops with IDENTICAL inputs
TEST_F(TestCQRRTComparison, SameInputs_Dense_vs_Linops) {
    int64_t m = 10;
    int64_t n = 5;
    double d_factor = 2.0;
    int64_t d = static_cast<int64_t>(d_factor * n);

    // ========== CREATE IDENTICAL INPUT MATRIX ==========
    std::vector<double> A_dense(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<r123::Philox4x32> state(0);
    RandBLAS::fill_dense(D, A_dense.data(), state);

    // Make copies for both algorithms
    std::vector<double> A_for_dense = A_dense;  // Will be modified in-place
    std::vector<double> A_for_linops = A_dense; // Won't be modified

    std::cerr << "\n========== ORIGINAL MATRIX A (" << m << " x " << n << ") ==========" << std::endl;
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            std::cerr << std::setw(12) << A_dense[i + j * m] << " ";
        }
        std::cerr << std::endl;
    }

    // ========== RUN DENSE CQRRT ==========
    std::vector<double> R_dense(n * n, 0.0);
    RandLAPACK::CQRRT<double, r123::Philox4x32> CQRRT_dense(false, std::pow(std::numeric_limits<double>::epsilon(), 0.85));

    // Use SAME RNG state for sketching
    RandBLAS::RNGState<r123::Philox4x32> state_dense(1);
    CQRRT_dense.call(m, n, A_for_dense.data(), m, R_dense.data(), n, d_factor, state_dense);

    std::cerr << "\n========== DENSE CQRRT RESULTS ==========" << std::endl;
    std::cerr << "Q (first 5 rows):" << std::endl;
    for (int64_t i = 0; i < std::min(5L, m); ++i) {
        for (int64_t j = 0; j < n; ++j) {
            std::cerr << std::setw(12) << A_for_dense[i + j * m] << " ";
        }
        std::cerr << std::endl;
    }

    std::cerr << "\nR_dense:" << std::endl;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            std::cerr << std::setw(12) << R_dense[i + j * n] << " ";
        }
        std::cerr << std::endl;
    }

    // Verify A = Q*R for dense
    std::vector<double> QR_dense(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, A_for_dense.data(), m, R_dense.data(), n, 0.0, QR_dense.data(), m);

    double norm_dense = 0.0;
    for (int64_t i = 0; i < m * n; ++i) {
        double diff = A_dense[i] - QR_dense[i];
        norm_dense += diff * diff;
    }
    norm_dense = std::sqrt(norm_dense);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    std::cerr << "\nDENSE: ||A - QR|| / ||A|| = " << norm_dense / norm_A << std::endl;

    // ========== RUN CQRRT_LINOPS WITH SAME MATRIX ==========
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_for_linops.data(), m, Layout::ColMajor);

    std::vector<double> R_linops(n * n, 0.0);
    CQRRT_linops<double, r123::Philox4x32> CQRRT_lo(false, std::pow(std::numeric_limits<double>::epsilon(), 0.85), true);

    // Use SAME RNG state for sketching
    RandBLAS::RNGState<r123::Philox4x32> state_linops(1);
    CQRRT_lo.call(A_linop, R_linops.data(), n, d_factor, state_linops);

    std::cerr << "\n========== CQRRT_LINOPS RESULTS ==========" << std::endl;
    std::cerr << "Q (first 5 rows):" << std::endl;
    for (int64_t i = 0; i < std::min(5L, m); ++i) {
        for (int64_t j = 0; j < n; ++j) {
            std::cerr << std::setw(12) << CQRRT_lo.Q[i + j * m] << " ";
        }
        std::cerr << std::endl;
    }

    std::cerr << "\nR_linops:" << std::endl;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            std::cerr << std::setw(12) << R_linops[i + j * n] << " ";
        }
        std::cerr << std::endl;
    }

    // Verify A = Q*R for linops
    std::vector<double> QR_linops(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CQRRT_lo.Q, m, R_linops.data(), n, 0.0, QR_linops.data(), m);

    double norm_linops = 0.0;
    for (int64_t i = 0; i < m * n; ++i) {
        double diff = A_dense[i] - QR_linops[i];
        norm_linops += diff * diff;
    }
    norm_linops = std::sqrt(norm_linops);

    std::cerr << "\nLINOPS: ||A - QR|| / ||A|| = " << norm_linops / norm_A << std::endl;

    // ========== COMPARE Q MATRICES ==========
    std::cerr << "\n========== COMPARISON: Q_dense vs Q_linops ==========" << std::endl;
    double max_Q_diff = 0.0;
    for (int64_t i = 0; i < m * n; ++i) {
        double diff = std::abs(A_for_dense[i] - CQRRT_lo.Q[i]);
        max_Q_diff = std::max(max_Q_diff, diff);
    }
    std::cerr << "Max absolute difference in Q: " << max_Q_diff << std::endl;

    // ========== COMPARE R MATRICES ==========
    std::cerr << "\n========== COMPARISON: R_dense vs R_linops ==========" << std::endl;
    std::cerr << "Element-by-element comparison:" << std::endl;
    double max_R_diff = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            int64_t idx = i + j * n;
            double diff = std::abs(R_dense[idx] - R_linops[idx]);
            max_R_diff = std::max(max_R_diff, diff);
            std::cerr << "R[" << i << "," << j << "]: dense=" << std::setw(12) << R_dense[idx]
                      << ", linops=" << std::setw(12) << R_linops[idx]
                      << ", diff=" << std::setw(12) << diff << std::endl;
        }
    }
    std::cerr << "Max absolute difference in R: " << max_R_diff << std::endl;

    // Both should pass with same tolerance
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    EXPECT_LE(norm_dense, atol * norm_A) << "Dense CQRRT failed";
    EXPECT_LE(norm_linops, atol * norm_A) << "CQRRT_linops failed";
}
