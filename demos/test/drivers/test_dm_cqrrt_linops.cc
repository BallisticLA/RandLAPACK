// Unified test suite for CQRRT_linops driver with linear operators
// Tests CQRRT with both dense and sparse SPD matrices in CholSolverLinOp
// Unified from test_dm_cqrrt_linops_dense.cc and test_dm_cqrrt_linops_sparse.cc

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include "../../functions/drivers/dm_cqrrt_linops.hh"
#include "../../functions/linops_external/dm_cholsolver_linop.hh"
#include "../../functions/misc/dm_util.hh"
#include "../utils/dm_test_utils.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>

using namespace RandLAPACK_demos;

class TestDmCQRRTLinops : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// ============================================================================
// Helper Functions for Test Verification
// ============================================================================

/// Helper: verify CQRRT R-factor by computing Q = A * R^{-1} externally.
/// Returns {factorization_error, orthogonality_error}.
static std::pair<double, double> verify_R_factor(
    const double* A_data, int64_t m, int64_t n, const double* R, int64_t ldr)
{
    // Q = A * R^{-1}: copy A into Q, then solve Q * R = A for Q via TRSM.
    std::vector<double> Q(m * n);
    std::copy(A_data, A_data + m * n, Q.begin());
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, 1.0, R, ldr, Q.data(), m);

    // Factorization error: ||A - Q*R|| / ||A||
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, Q.data(), m, R, ldr, 0.0, QR.data(), m);
    for (int64_t i = 0; i < m * n; ++i)
        QR[i] = A_data[i] - QR[i];
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A   = lapack::lange(Norm::Fro, m, n, A_data, m);

    // Orthogonality error: ||Q^T*Q - I||
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m,
               1.0, Q.data(), m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    return {norm_AQR / norm_A, norm_orth / std::sqrt((double)n)};
}

// ============================================================================
// Dense SPD Tests
// ============================================================================

// Test with simple dense matrix wrapped in DenseLinOp
TEST_F(TestDmCQRRTLinops, dense_matrix) {
    int64_t m = 100;
    int64_t n = 50;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Create random dense matrix
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<r123::Philox4x32> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);

    // Make copy for verification
    std::vector<double> A_copy = A_data;

    // Create DenseLinOp
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    // Run CQRRT with test mode enabled
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double, r123::Philox4x32> CQRRT(false, tol, true);
    state = RandBLAS::RNGState<r123::Philox4x32>(1);
    CQRRT.call(A_linop, R.data(), n, d_factor, state);

    // Verify A = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CQRRT.Q, m, R.data(), n, 0.0, QR.data(), m);

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

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));
}

// Test with composite operator (dense SPD): CholSolver * Sparse
TEST_F(TestDmCQRRTLinops, composite_operator_dense_spd) {
    int64_t n_spd = 50;
    int64_t n_sparse_cols = 20;
    int64_t m = n_spd;
    int64_t n = n_sparse_cols;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate dense SPD matrix
    std::string spd_filename = "/tmp/test_spd_matrix_cqrrt.mtx";
    RandLAPACK_demos::generate_spd_matrix_file<double>(spd_filename, n_spd, 10.0, state);

    // Create CholSolverLinOp
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(spd_filename);

    // Generate sparse matrix B
    double density = 0.2;
    auto B_coo = RandLAPACK::gen::gen_sparse_mat<double>(n_spd, n_sparse_cols, density, state);

    // Convert to CSC for SparseLinOp
    RandBLAS::sparse_data::csc::CSCMatrix<double> B_csc(n_spd, n_sparse_cols);
    RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

    // Create SparseLinOp and CompositeOperator
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> B_sp_linop(n_spd, n_sparse_cols, B_csc);
    RandLAPACK::linops::CompositeOperator A_composite(m, n, A_inv_linop, B_sp_linop);

    // Compute dense representation for verification
    std::vector<double> B_dense(n_spd * n_sparse_cols, 0.0);
    RandLAPACK::util::sparse_to_dense(B_csc, Layout::ColMajor, B_dense.data());

    std::vector<double> A_dense(m * n, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n_spd,
                1.0, B_dense.data(), n_spd, 0.0, A_dense.data(), m);

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 2;
    CQRRT_linops_alg.call(A_composite, R.data(), n, d_factor, state);

    // Verify A = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CQRRT_linops_alg.Q, m, R.data(), n, 0.0, QR.data(), m);

    for (int64_t i = 0; i < m * n; ++i) {
        QR[i] = A_dense[i] - QR[i];
    }
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // Check orthogonality
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CQRRT_linops_alg.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// Test with nested composite operator (dense SPD): CholSolver * (SASO * Gaussian) (small scale)
TEST_F(TestDmCQRRTLinops, nested_composite_operator_dense_spd) {
    int64_t m = 100;
    int64_t k_dim = 50;
    int64_t n = 20;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate dense SPD matrix for CholSolver
    std::string spd_filename = "/tmp/test_cqrrt_nested_small.mtx";
    RandLAPACK_demos::generate_spd_matrix_file<double>(spd_filename, m, 10.0, state);

    // Create CholSolverLinOp
    RandLAPACK_demos::CholSolverLinOp<double> chol_linop(spd_filename);

    // Generate SASO (sparse) matrix: m × k_dim
    double saso_density = 0.2;
    auto saso_coo = RandLAPACK::gen::gen_sparse_mat<double>(m, k_dim, saso_density, state);

    // Convert to CSC for SparseLinOp
    RandBLAS::sparse_data::csc::CSCMatrix<double> saso_csc(m, k_dim);
    RandBLAS::sparse_data::conversions::coo_to_csc(saso_coo, saso_csc);

    // Create SASO SparseLinOp
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> saso_linop(m, k_dim, saso_csc);

    // Generate Gaussian (dense) matrix: k_dim × n
    std::vector<double> gaussian_data(k_dim * n);
    RandBLAS::DenseDist gaussian_dist(k_dim, n);
    RandBLAS::fill_dense(gaussian_dist, gaussian_data.data(), state);

    // Create Gaussian DenseLinOp
    RandLAPACK::linops::DenseLinOp<double> gaussian_linop(k_dim, n, gaussian_data.data(), k_dim, Layout::ColMajor);

    // Create inner composite: SASO * Gaussian
    RandLAPACK::linops::CompositeOperator inner_composite(m, n, saso_linop, gaussian_linop);

    // Create outer nested composite: CholSolver * (SASO * Gaussian)
    RandLAPACK::linops::CompositeOperator nested_composite(m, n, chol_linop, inner_composite);

    // Compute dense representation for verification
    // Step 1: Densify SASO and compute SASO * Gaussian -> intermediate
    std::vector<double> saso_dense(m * k_dim, 0.0);
    RandLAPACK::util::sparse_to_dense(saso_csc, Layout::ColMajor, saso_dense.data());

    std::vector<double> intermediate(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_dim,
               1.0, saso_dense.data(), m, gaussian_data.data(), k_dim,
               0.0, intermediate.data(), m);

    // Step 2: Compute CholSolver * intermediate -> A_dense
    std::vector<double> A_dense(m * n, 0.0);
    chol_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m,
               1.0, intermediate.data(), m, 0.0, A_dense.data(), m);

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 2;
    CQRRT_linops_alg.call(nested_composite, R.data(), n, d_factor, state);

    // Verify A = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CQRRT_linops_alg.Q, m, R.data(), n, 0.0, QR.data(), m);

    for (int64_t i = 0; i < m * n; ++i) {
        QR[i] = A_dense[i] - QR[i];
    }
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // Check orthogonality
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CQRRT_linops_alg.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// Test with nested composite operator (dense SPD, large scale - similar to benchmark)
TEST_F(TestDmCQRRTLinops, nested_composite_operator_dense_spd_large_square) {
    int64_t m = 1138;
    int64_t k_dim = 1138;  // SQUARE intermediate - matches benchmark!
    int64_t n = 100;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate dense SPD matrix for CholSolver
    std::string spd_filename = "/tmp/test_cqrrt_nested_large_spd_k10.mtx";
    RandLAPACK_demos::generate_spd_matrix_file<double>(spd_filename, m, 10.0, state);

    // Create CholSolverLinOp
    RandLAPACK_demos::CholSolverLinOp<double> chol_linop(spd_filename);

    // Generate SASO (sparse) matrix: m × k_dim (SQUARE like benchmark!)
    double saso_density = 0.5;  // Match benchmark density
    auto saso_coo = RandLAPACK::gen::gen_sparse_mat<double>(m, k_dim, saso_density, state);

    // Convert to CSC for SparseLinOp
    RandBLAS::sparse_data::csc::CSCMatrix<double> saso_csc(m, k_dim);
    RandBLAS::sparse_data::conversions::coo_to_csc(saso_coo, saso_csc);

    // Create SASO SparseLinOp
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> saso_linop(m, k_dim, saso_csc);

    // Generate Gaussian (dense) matrix: k_dim × n
    std::vector<double> gaussian_data(k_dim * n);
    RandBLAS::DenseDist gaussian_dist(k_dim, n);
    RandBLAS::fill_dense(gaussian_dist, gaussian_data.data(), state);

    // Create Gaussian DenseLinOp
    RandLAPACK::linops::DenseLinOp<double> gaussian_linop(k_dim, n, gaussian_data.data(), k_dim, Layout::ColMajor);

    // Create inner composite: SASO * Gaussian
    RandLAPACK::linops::CompositeOperator inner_composite(m, n, saso_linop, gaussian_linop);

    // Create outer nested composite: CholSolver * (SASO * Gaussian)
    RandLAPACK::linops::CompositeOperator nested_composite(m, n, chol_linop, inner_composite);

    // Compute dense representation for verification
    // Step 1: Densify SASO and compute SASO * Gaussian -> intermediate
    std::vector<double> saso_dense(m * k_dim, 0.0);
    RandLAPACK::util::sparse_to_dense(saso_csc, Layout::ColMajor, saso_dense.data());

    std::vector<double> intermediate(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_dim,
               1.0, saso_dense.data(), m, gaussian_data.data(), k_dim,
               0.0, intermediate.data(), m);

    // Step 2: Compute CholSolver * intermediate -> A_dense
    std::vector<double> A_dense(m * n, 0.0);
    chol_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m,
               1.0, intermediate.data(), m, 0.0, A_dense.data(), m);

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 2;
    CQRRT_linops_alg.call(nested_composite, R.data(), n, d_factor, state);

    // Verify A = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CQRRT_linops_alg.Q, m, R.data(), n, 0.0, QR.data(), m);

    for (int64_t i = 0; i < m * n; ++i) {
        QR[i] = A_dense[i] - QR[i];
    }
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // Check orthogonality
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CQRRT_linops_alg.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// ============================================================================
// Sparse SPD Tests
// ============================================================================

// Test with composite operator (sparse SPD): CholSolver * Sparse
TEST_F(TestDmCQRRTLinops, composite_operator_sparse_spd) {
    int64_t n_spd = 50;
    int64_t n_sparse_cols = 20;
    int64_t m = n_spd;
    int64_t n = n_sparse_cols;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate sparse SPD matrix
    std::string spd_filename = "/tmp/test_sparse_spd_matrix_cqrrt.mtx";
    test_utils::generate_sparse_spd_matrix(n_spd, 10.0, spd_filename);

    // Create CholSolverLinOp (will use sparse solver)
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(spd_filename);

    // Generate sparse matrix B
    double density = 0.2;
    auto B_coo = RandLAPACK::gen::gen_sparse_mat<double>(n_spd, n_sparse_cols, density, state);

    // Convert to CSC for SparseLinOp
    RandBLAS::sparse_data::csc::CSCMatrix<double> B_csc(n_spd, n_sparse_cols);
    RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

    // Create SparseLinOp and CompositeOperator
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> B_sp_linop(n_spd, n_sparse_cols, B_csc);
    RandLAPACK::linops::CompositeOperator A_composite(m, n, A_inv_linop, B_sp_linop);

    // Compute dense representation for verification
    std::vector<double> B_dense(n_spd * n_sparse_cols, 0.0);
    RandLAPACK::util::sparse_to_dense(B_csc, Layout::ColMajor, B_dense.data());

    std::vector<double> A_dense(m * n, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n_spd,
                1.0, B_dense.data(), n_spd, 0.0, A_dense.data(), m);

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 2;
    CQRRT_linops_alg.call(A_composite, R.data(), n, d_factor, state);

    // Verify A = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CQRRT_linops_alg.Q, m, R.data(), n, 0.0, QR.data(), m);

    for (int64_t i = 0; i < m * n; ++i) {
        QR[i] = A_dense[i] - QR[i];
    }
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // Check orthogonality
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CQRRT_linops_alg.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// Test with nested composite operator (sparse SPD, large scale)
TEST_F(TestDmCQRRTLinops, nested_composite_operator_sparse_spd_large_square) {
    int64_t m = 1138;
    int64_t k_dim = 1138;  // SQUARE intermediate
    int64_t n = 100;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate sparse SPD matrix for CholSolver
    std::string spd_filename = "/tmp/test_cqrrt_nested_large_sparse_spd_k10.mtx";
    test_utils::generate_sparse_spd_matrix(m, 10.0, spd_filename);

    // Create CholSolverLinOp (will use sparse solver)
    RandLAPACK_demos::CholSolverLinOp<double> chol_linop(spd_filename);

    // Generate SASO (sparse) matrix: m × k_dim (SQUARE)
    double saso_density = 0.5;
    auto saso_coo = RandLAPACK::gen::gen_sparse_mat<double>(m, k_dim, saso_density, state);

    // Convert to CSC for SparseLinOp
    RandBLAS::sparse_data::csc::CSCMatrix<double> saso_csc(m, k_dim);
    RandBLAS::sparse_data::conversions::coo_to_csc(saso_coo, saso_csc);

    // Create SASO SparseLinOp
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> saso_linop(m, k_dim, saso_csc);

    // Generate Gaussian (dense) matrix: k_dim × n
    std::vector<double> gaussian_data(k_dim * n);
    RandBLAS::DenseDist gaussian_dist(k_dim, n);
    RandBLAS::fill_dense(gaussian_dist, gaussian_data.data(), state);

    // Create Gaussian DenseLinOp
    RandLAPACK::linops::DenseLinOp<double> gaussian_linop(k_dim, n, gaussian_data.data(), k_dim, Layout::ColMajor);

    // Create inner composite: SASO * Gaussian
    RandLAPACK::linops::CompositeOperator inner_composite(m, n, saso_linop, gaussian_linop);

    // Create outer nested composite: CholSolver * (SASO * Gaussian)
    RandLAPACK::linops::CompositeOperator nested_composite(m, n, chol_linop, inner_composite);

    // Compute dense representation for verification
    // Step 1: Densify SASO and compute SASO * Gaussian -> intermediate
    std::vector<double> saso_dense(m * k_dim, 0.0);
    RandLAPACK::util::sparse_to_dense(saso_csc, Layout::ColMajor, saso_dense.data());

    std::vector<double> intermediate(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_dim,
               1.0, saso_dense.data(), m, gaussian_data.data(), k_dim,
               0.0, intermediate.data(), m);

    // Step 2: Compute CholSolver * intermediate -> A_dense
    std::vector<double> A_dense(m * n, 0.0);
    chol_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m,
               1.0, intermediate.data(), m, 0.0, A_dense.data(), m);

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 2;
    CQRRT_linops_alg.call(nested_composite, R.data(), n, d_factor, state);

    // Verify A = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CQRRT_linops_alg.Q, m, R.data(), n, 0.0, QR.data(), m);

    for (int64_t i = 0; i < m * n; ++i) {
        QR[i] = A_dense[i] - QR[i];
    }
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // Check orthogonality
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CQRRT_linops_alg.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// ============================================================================
// Column-block processing tests
// ============================================================================

// block_size = 10, n = 50 → 5 full blocks, no remainder
TEST_F(TestDmCQRRTLinops, block_processing_even_division) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<r123::Philox4x32> state(42);
    RandBLAS::fill_dense(D, A_data.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double, r123::Philox4x32> alg(false, tol, false);
    alg.block_size = 10;  // 50 / 10 = 5 full blocks
    state = RandBLAS::RNGState<r123::Philox4x32>(1);
    alg.call(A_linop, R.data(), n, d_factor, state);

    auto [fact_err, orth_err] = verify_R_factor(A_data.data(), m, n, R.data(), n);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(fact_err, atol) << "Factorization error too large with block_size=10";
    ASSERT_LE(orth_err, atol) << "Orthogonality error too large with block_size=10";
}

// block_size = 12, n = 50 → 4 blocks of 12 + 1 block of 2 (remainder)
TEST_F(TestDmCQRRTLinops, block_processing_with_remainder) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<r123::Philox4x32> state(42);
    RandBLAS::fill_dense(D, A_data.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double, r123::Philox4x32> alg(false, tol, false);
    alg.block_size = 12;  // 50 / 12 = 4 blocks of 12, remainder block of 2
    state = RandBLAS::RNGState<r123::Philox4x32>(1);
    alg.call(A_linop, R.data(), n, d_factor, state);

    auto [fact_err, orth_err] = verify_R_factor(A_data.data(), m, n, R.data(), n);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(fact_err, atol) << "Factorization error too large with block_size=12";
    ASSERT_LE(orth_err, atol) << "Orthogonality error too large with block_size=12";
}

// block_size = 1 → column-by-column processing (extreme case)
TEST_F(TestDmCQRRTLinops, block_processing_single_column) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<r123::Philox4x32> state(42);
    RandBLAS::fill_dense(D, A_data.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double, r123::Philox4x32> alg(false, tol, false);
    alg.block_size = 1;  // extreme: 50 iterations, 1 column at a time
    state = RandBLAS::RNGState<r123::Philox4x32>(1);
    alg.call(A_linop, R.data(), n, d_factor, state);

    auto [fact_err, orth_err] = verify_R_factor(A_data.data(), m, n, R.data(), n);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(fact_err, atol) << "Factorization error too large with block_size=1";
    ASSERT_LE(orth_err, atol) << "Orthogonality error too large with block_size=1";
}

// Verify block processing matches full materialization (same RNG seed → same sketch)
TEST_F(TestDmCQRRTLinops, block_vs_full_agreement) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<r123::Philox4x32> state(42);
    RandBLAS::fill_dense(D, A_data.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    // Full path (block_size = 0, no blocking)
    std::vector<double> R_full(n * n, 0.0);
    CQRRT_linops<double, r123::Philox4x32> alg_full(false, tol, false);
    state = RandBLAS::RNGState<r123::Philox4x32>(1);
    alg_full.call(A_linop, R_full.data(), n, d_factor, state);

    // Block path (block_size = 10)
    std::vector<double> R_block(n * n, 0.0);
    CQRRT_linops<double, r123::Philox4x32> alg_block(false, tol, false);
    alg_block.block_size = 10;
    state = RandBLAS::RNGState<r123::Philox4x32>(1);  // same seed!
    alg_block.call(A_linop, R_block.data(), n, d_factor, state);

    // Compare R-factors: ||R_full - R_block|| / ||R_full||
    double norm_R = lapack::lange(Norm::Fro, n, n, R_full.data(), n);
    std::vector<double> diff(n * n);
    for (int64_t i = 0; i < n * n; ++i)
        diff[i] = R_full[i] - R_block[i];
    double norm_diff = lapack::lange(Norm::Fro, n, n, diff.data(), n);

    double rel_diff = norm_diff / norm_R;
    double agreement_tol = 1000 * std::numeric_limits<double>::epsilon();
    ASSERT_LE(rel_diff, agreement_tol)
        << "Block and full R-factors differ by " << rel_diff
        << " (expected < " << agreement_tol << ")";
}

// Column-block processing with composite operator
TEST_F(TestDmCQRRTLinops, block_processing_composite_operator) {
    int64_t n_spd = 50;
    int64_t n_sparse_cols = 20;
    int64_t m = n_spd;
    int64_t n = n_sparse_cols;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate SPD matrix
    std::string spd_filename = "/tmp/test_spd_cqrrt_block.mtx";
    RandLAPACK_demos::generate_spd_matrix_file<double>(spd_filename, n_spd, 10.0, state);

    // Create CholSolverLinOp
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(spd_filename);

    // Generate sparse matrix B
    double density = 0.2;
    auto B_coo = RandLAPACK::gen::gen_sparse_mat<double>(n_spd, n_sparse_cols, density, state);
    RandBLAS::sparse_data::csc::CSCMatrix<double> B_csc(n_spd, n_sparse_cols);
    RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> B_sp_linop(n_spd, n_sparse_cols, B_csc);
    RandLAPACK::linops::CompositeOperator A_composite(m, n, A_inv_linop, B_sp_linop);

    // Compute dense representation for verification: A_dense = A^{-1} * B
    std::vector<double> B_dense(n_spd * n_sparse_cols, 0.0);
    RandLAPACK::util::sparse_to_dense(B_csc, Layout::ColMajor, B_dense.data());
    std::vector<double> A_dense(m * n, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n_spd,
                1.0, B_dense.data(), n_spd, 0.0, A_dense.data(), m);

    // Run CQRRT with block processing (block_size = 5, n = 20 → 4 blocks)
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> alg(false, tol, false);
    alg.block_size = 5;
    alg.nnz = 2;
    alg.call(A_composite, R.data(), n, d_factor, state);

    auto [fact_err, orth_err] = verify_R_factor(A_dense.data(), m, n, R.data(), n);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(fact_err, atol) << "Composite factorization error too large with block_size=5";
    ASSERT_LE(orth_err, atol) << "Composite orthogonality error too large with block_size=5";

    std::remove(spd_filename.c_str());
}
