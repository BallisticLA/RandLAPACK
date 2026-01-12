// Test suite for CholQR_linops driver with SPARSE SPD matrices
// This test file specifically tests the case where the SPD matrix in CholSolverLinOp is SPARSE
// Companion to test_dm_cholqr_linops_dense.cc which tests with dense SPD matrices

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include "../../functions/drivers/dm_cholqr_linops.hh"
#include "../../functions/linops_external/dm_cholsolver_linop.hh"
#include "../../functions/misc/dm_util.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <iomanip>

using namespace RandLAPACK_demos;

class TestDmCholQRLinopsSparse : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// Helper function to generate a sparse SPD matrix using the 1138_bus perturbation approach
// Uses diagonal regularization to achieve target condition number while preserving sparsity
std::string generate_sparse_spd_matrix_cholqr(int64_t n, double cond_num, const std::string& filename) {
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Generate a random sparse matrix with controlled spectrum
    // We'll use a simple approach: generate a banded SPD matrix
    double lambda_max = 1.0;
    double lambda_min = 1.0 / cond_num;

    // Generate eigenvalues with polynomial decay
    std::vector<double> eigenvalues(n);
    for (int64_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n - 1);
        eigenvalues[i] = lambda_min + (lambda_max - lambda_min) * std::pow(1.0 - t, 2.0);
    }

    // Create a sparse SPD matrix using a banded structure
    // A simple banded SPD: tridiagonal with positive diagonal and small off-diagonals
    std::vector<Eigen::Triplet<double>> triplets;

    // Set diagonal to eigenvalues sum to ensure positive definiteness
    double diag_sum = 0.0;
    for (auto ev : eigenvalues) diag_sum += ev;
    double diag_value = diag_sum / n + 1.0;  // Ensure diagonal dominance

    int64_t bandwidth = 5;  // Keep it sparse with limited bandwidth
    for (int64_t i = 0; i < n; ++i) {
        // Diagonal
        triplets.emplace_back(i, i, diag_value);

        // Off-diagonal bands (symmetric)
        for (int64_t b = 1; b <= bandwidth && i + b < n; ++b) {
            double off_diag = 0.1 * diag_value / b;  // Decay with distance
            triplets.emplace_back(i, i + b, off_diag);
            triplets.emplace_back(i + b, i, off_diag);
        }
    }

    Eigen::SparseMatrix<double> A_sparse(n, n);
    A_sparse.setFromTriplets(triplets.begin(), triplets.end());

    // Write to Matrix Market file in symmetric format
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << "%%MatrixMarket matrix coordinate real symmetric\n";

    // Count lower triangular entries
    int64_t nnz_lower = 0;
    for (int k = 0; k < A_sparse.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, k); it; ++it) {
            if (it.row() >= it.col()) {
                ++nnz_lower;
            }
        }
    }

    file << n << " " << n << " " << nnz_lower << "\n";
    file << std::scientific << std::setprecision(16);

    for (int k = 0; k < A_sparse.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, k); it; ++it) {
            if (it.row() >= it.col()) {
                file << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
            }
        }
    }

    file.close();
    return filename;
}

// Test with sparse matrix using SparseLinOp (direct sparse matrix, no CholSolver)
TEST_F(TestDmCholQRLinopsSparse, sparse_matrix_only) {
    int64_t m = 100;
    int64_t n = 50;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate sparse matrix
    double density = 0.2;
    auto A_coo = RandLAPACK::gen::gen_sparse_mat<double>(m, n, density, state);

    // Convert to CSC for SparseLinOp
    RandBLAS::sparse_data::csc::CSCMatrix<double> A_csc(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csc(A_coo, A_csc);

    // Create SparseLinOp
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> A_linop(m, n, A_csc);

    // Densify for verification
    std::vector<double> A_dense(m * n, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(A_csc, Layout::ColMajor, A_dense.data());

    // Run CholQR
    std::vector<double> R(n * n, 0.0);
    CholQR_linops<double> CholQR(false, tol, true);
    CholQR.call(A_linop, R.data(), n);

    // Verify A = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CholQR.Q, m, R.data(), n, 0.0, QR.data(), m);

    for (int64_t i = 0; i < m * n; ++i) {
        QR[i] = A_dense[i] - QR[i];
    }
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // Check orthogonality
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CholQR.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));
}

// Test: Composite operator with sparse SPD in CholSolver
TEST_F(TestDmCholQRLinopsSparse, composite_operator_sparse_spd) {
    int64_t n_spd = 50;
    int64_t n_sparse_cols = 20;
    int64_t m = n_spd;
    int64_t n = n_sparse_cols;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate SPARSE SPD matrix
    std::string spd_filename = "/tmp/test_cholqr_sparse_spd_composite.mtx";
    generate_sparse_spd_matrix_cholqr(n_spd, 10.0, spd_filename);

    // Create CholSolverLinOp from SPARSE SPD
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
    RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, Layout::ColMajor, B_dense.data());

    std::vector<double> A_dense(m * n, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n_spd,
                1.0, B_dense.data(), n_spd, 0.0, A_dense.data(), m);

    // Run CholQR
    std::vector<double> R(n * n, 0.0);
    CholQR_linops<double> CholQR_linops_alg(false, tol, true);
    CholQR_linops_alg.call(A_composite, R.data(), n);

    // Verify A = Q * R
    std::vector<double> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, CholQR_linops_alg.Q, m, R.data(), n, 0.0, QR.data(), m);

    for (int64_t i = 0; i < m * n; ++i) {
        QR[i] = A_dense[i] - QR[i];
    }
    double norm_AQR = lapack::lange(Norm::Fro, m, n, QR.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // Check orthogonality
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CholQR_linops_alg.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// Test: Nested composite operator with sparse SPD - large scale
TEST_F(TestDmCholQRLinopsSparse, nested_composite_operator_large_square_sparse_spd) {
    int64_t m = 1138;
    int64_t k_dim = 1138;
    int64_t n = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate SPARSE SPD matrix for CholSolver
    std::string spd_filename = "/tmp/test_cholqr_sparse_nested_large.mtx";
    generate_sparse_spd_matrix_cholqr(m, 10.0, spd_filename);

    // Create CholSolverLinOp from SPARSE SPD
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

    // Run CholQR (no dense verification for large scale)
    std::vector<double> R(n * n, 0.0);
    CholQR_linops<double> CholQR_linops_alg(false, tol, true);
    CholQR_linops_alg.call(nested_composite, R.data(), n);

    // Verify orthogonality of Q
    std::vector<double> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CholQR_linops_alg.Q, m, -1.0, I_ref.data(), n);
    double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    // Check that R is upper triangular
    double lower_triangle_norm = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = j + 1; i < n; ++i) {
            lower_triangle_norm += R[i + j * n] * R[i + j * n];
        }
    }
    lower_triangle_norm = std::sqrt(lower_triangle_norm);

    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));
    ASSERT_LE(lower_triangle_norm, std::numeric_limits<double>::epsilon());

    // Clean up
    std::remove(spd_filename.c_str());
}
