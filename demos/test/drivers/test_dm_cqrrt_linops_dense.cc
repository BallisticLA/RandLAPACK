// Test CQRRT with simple dense linear operators
#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>

#include "../../functions/drivers/dm_cqrrt_linops.hh"

using namespace RandLAPACK_demos;

class TestDmCQRRTLinopsDense : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// Test with a simple dense matrix (wrapped in DenseLinOp)
TEST_F(TestDmCQRRTLinopsDense, CQRRTLinops_dense_simple) {
    int64_t m = 10;
    int64_t n = 5;
    double d_factor = 2.0;

    // Create a simple dense matrix
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<r123::Philox4x32> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);

    // Make a copy for verification
    std::vector<double> A_copy = A_data;

    // Create DenseLinOp
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    // Allocate R
    std::vector<double> R(n * n, 0.0);

    // Run CQRRT
    CQRRT_linops<double, r123::Philox4x32> CQRRT(false, std::pow(std::numeric_limits<double>::epsilon(), 0.85), true);
    state = RandBLAS::RNGState<r123::Philox4x32>(1);
    CQRRT.call(A_linop, R.data(), n, d_factor, state);

    // Check: A_copy = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CQRRT.Q, m, R.data(), n, 0.0, QR.data(), m);

    // Compute ||A - QR||
    for (int64_t i = 0; i < m * n; ++i) {
        QR[i] = A_copy[i] - QR[i];
    }
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_copy.data(), m);

    // Check orthogonality of Q
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CQRRT.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    std::cerr << "\n=== DENSE LINOP TEST ===" << std::endl;
    std::cerr << "REL NORM OF A - QR: " << norm_AQR / norm_A << std::endl;
    std::cerr << "FRO NORM OF (Q'Q - I)/sqrt(n): " << norm_orth / std::sqrt((double) n) << std::endl;

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));
}
