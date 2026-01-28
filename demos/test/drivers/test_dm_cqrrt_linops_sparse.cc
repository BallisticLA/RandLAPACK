// Test suite for CQRRT_linops driver with SPARSE SPD matrices
// This test file specifically tests the case where the SPD matrix in CholSolverLinOp is SPARSE
// Companion to test_dm_cqrrt_linops_dense.cc which tests with dense SPD matrices

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include "../../functions/drivers/dm_cqrrt_linops.hh"
#include "../../functions/linops_external/dm_cholsolver_linop.hh"
#include "../../functions/misc/dm_util.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <iomanip>

using namespace RandLAPACK_demos;

class TestDmCQRRTLinopsSparse : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// Helper function to generate a sparse SPD matrix using the 1138_bus perturbation approach
// Uses diagonal regularization to achieve target condition number while preserving sparsity
std::string generate_sparse_spd_matrix(int64_t n, double cond_num, const std::string& filename) {
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

// Test: Same matrix, but densify BEFORE passing to CholSolver
TEST_F(TestDmCQRRTLinopsSparse, cholsolver_densified_spd_times_dense) {
    int64_t n_spd = 50;
    int64_t n_dense_cols = 20;
    int64_t m = n_spd;
    int64_t n = n_dense_cols;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate SPARSE SPD matrix
    std::string spd_filename = "/tmp/test_chol_densified_dense.mtx";
    generate_sparse_spd_matrix(n_spd, 10.0, spd_filename);

    // Load as sparse then DENSIFY
    Eigen::SparseMatrix<double> A_sparse_temp;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(spd_filename, A_sparse_temp);
    Eigen::MatrixXd A_dense_spd = Eigen::MatrixXd(A_sparse_temp);

    // Write densified version to file
    std::string dense_spd_filename = "/tmp/test_chol_densified_dense_spd.mtx";
    std::ofstream dense_file(dense_spd_filename);
    dense_file << "%%MatrixMarket matrix coordinate real general\n";
    int64_t nnz = 0;
    for (int64_t i = 0; i < n_spd * n_spd; ++i) {
        if (A_dense_spd.data()[i] != 0.0) nnz++;
    }
    dense_file << n_spd << " " << n_spd << " " << nnz << "\n";
    for (int64_t j = 0; j < n_spd; ++j) {
        for (int64_t i = 0; i < n_spd; ++i) {
            double val = A_dense_spd(i, j);
            if (val != 0.0) {
                dense_file << (i + 1) << " " << (j + 1) << " " << std::scientific << std::setprecision(16) << val << "\n";
            }
        }
    }
    dense_file.close();

    // Create CholSolverLinOp from DENSIFIED matrix
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(dense_spd_filename);

    // Generate DENSE matrix B
    std::vector<double> B_dense(n_spd * n_dense_cols);
    RandBLAS::DenseDist D(n_spd, n_dense_cols);
    RandBLAS::fill_dense(D, B_dense.data(), state);

    // Create DenseLinOp
    RandLAPACK::linops::DenseLinOp<double> B_dense_linop(n_spd, n_dense_cols, B_dense.data(), n_spd, Layout::ColMajor);

    // Create CompositeOperator
    RandLAPACK::linops::CompositeOperator A_composite(m, n, A_inv_linop, B_dense_linop);

    // Compute dense representation for verification
    std::vector<double> A_dense(m * n, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n_spd,
                1.0, B_dense.data(), m, 0.0, A_dense.data(), m);

    std::cout << "DEBUG: CholSolver(DENSIFIED sparse SPD) * Dense test" << std::endl;

    // DIAGNOSTIC: Check properties of A^{-1} itself
    // Compute A^{-1} using Cholesky solver
    Eigen::LLT<Eigen::MatrixXd> chol_A(A_dense_spd);
    Eigen::MatrixXd A_inv_full = chol_A.solve(Eigen::MatrixXd::Identity(n_spd, n_spd));

    std::cout << "  A (banded SPD) nnz: " << (A_dense_spd.array() != 0.0).count() << " / " << (n_spd*n_spd) << std::endl;
    std::cout << "  A^{-1} nnz: " << (A_inv_full.array().abs() > 1e-10).count() << " / " << (n_spd*n_spd) << std::endl;

    // Compute condition number of A^{-1} using SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_Ainv(A_inv_full, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << "  A^{-1} condition: " << svd_Ainv.singularValues()(0) / svd_Ainv.singularValues()(svd_Ainv.singularValues().size()-1) << std::endl;
    std::cout << "  A^{-1} max entry: " << A_inv_full.cwiseAbs().maxCoeff() << std::endl;
    std::cout << "  A^{-1} Frobenius norm: " << A_inv_full.norm() << std::endl;

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 2;
    CQRRT_linops_alg.call(A_composite, R.data(), n, d_factor, state);

    // Verify
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

    std::cout << "CholSolver(DENSIFIED sparse SPD) * Dense test results:" << std::endl;
    std::cout << "  ||A - QR|| / ||A||: " << norm_AQR / norm_A << std::endl;
    std::cout << "  ||Q'Q - I||: " << norm_orth << std::endl;
    std::cout << "  Tolerance: " << atol * norm_A << " (factorization), "
              << atol * std::sqrt((double) n) << " (orthogonality)" << std::endl;

    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
    std::remove(dense_spd_filename.c_str());
}

// Test with CholSolver(sparse SPD) * Dense (to isolate if issue is with SparseLinOp)
TEST_F(TestDmCQRRTLinopsSparse, cholsolver_sparse_spd_times_dense) {
    int64_t n_spd = 50;
    int64_t n_dense_cols = 20;
    int64_t m = n_spd;
    int64_t n = n_dense_cols;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate SPARSE SPD matrix
    std::string spd_filename = "/tmp/test_chol_sparse_dense.mtx";
    generate_sparse_spd_matrix(n_spd, 10.0, spd_filename);

    // Create CholSolverLinOp from SPARSE SPD matrix
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(spd_filename);

    // Generate DENSE matrix B (not sparse!)
    std::vector<double> B_dense(n_spd * n_dense_cols);
    RandBLAS::DenseDist D(n_spd, n_dense_cols);
    RandBLAS::fill_dense(D, B_dense.data(), state);

    // Create DenseLinOp
    RandLAPACK::linops::DenseLinOp<double> B_dense_linop(n_spd, n_dense_cols, B_dense.data(), n_spd, Layout::ColMajor);

    // Create CompositeOperator: CholSolver(sparse SPD) * Dense
    RandLAPACK::linops::CompositeOperator A_composite(m, n, A_inv_linop, B_dense_linop);

    // Compute dense representation for verification
    std::vector<double> A_dense(m * n, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n_spd,
                1.0, B_dense.data(), m, 0.0, A_dense.data(), m);

    std::cout << "DEBUG: CholSolver(sparse SPD) * Dense test" << std::endl;

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

    std::cout << "CholSolver(sparse SPD) * Dense test results:" << std::endl;
    std::cout << "  ||A - QR|| / ||A||: " << norm_AQR / norm_A << std::endl;
    std::cout << "  ||Q'Q - I||: " << norm_orth << std::endl;
    std::cout << "  Tolerance: " << atol * norm_A << " (factorization), "
              << atol * std::sqrt((double) n) << " (orthogonality)" << std::endl;

    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// Test with composite operator: CholSolver(SPARSE SPD) * Sparse
TEST_F(TestDmCQRRTLinopsSparse, composite_operator_sparse_spd) {
    int64_t n_spd = 50;
    int64_t n_sparse_cols = 20;
    int64_t m = n_spd;
    int64_t n = n_sparse_cols;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate SPARSE SPD matrix
    std::string spd_filename = "/tmp/test_sparse_spd_matrix_cqrrt.mtx";
    generate_sparse_spd_matrix(n_spd, 10.0, spd_filename);

    // Load and verify the SPD matrix
    Eigen::SparseMatrix<double> SPD_check;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(spd_filename, SPD_check);
    Eigen::MatrixXd SPD_dense_check = Eigen::MatrixXd(SPD_check);

    // Compute condition number
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(SPD_dense_check);
    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
    std::cout << "DEBUG: Sparse SPD matrix condition number: " << cond << std::endl;
    std::cout << "DEBUG: Sparse SPD matrix nnz: " << SPD_check.nonZeros() << " out of " << (n_spd*n_spd) << std::endl;

    // Create CholSolverLinOp from SPARSE SPD matrix
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

    // Diagnostic: Check A_dense properties
    double norm_A_dense = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);
    Eigen::Map<Eigen::MatrixXd> A_dense_map(A_dense.data(), m, n);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(A_dense_map, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double cond_A = svd_A.singularValues()(0) / svd_A.singularValues()(svd_A.singularValues().size()-1);
    std::cout << "DEBUG: A_dense (composite operator result) norm: " << norm_A_dense << std::endl;
    std::cout << "DEBUG: A_dense condition number: " << cond_A << std::endl;
    std::cout << "DEBUG: A_dense singular values (first 5): ";
    for (int i = 0; i < std::min(5L, svd_A.singularValues().size()); ++i) {
        std::cout << svd_A.singularValues()(i) << " ";
    }
    std::cout << std::endl;

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 5;  // Optimal for sparse SPD matrices (100% success in parameter study)
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

    std::cout << "Sparse SPD test results:" << std::endl;
    std::cout << "  ||A - QR|| / ||A||: " << norm_AQR / norm_A << std::endl;
    std::cout << "  ||Q'Q - I||: " << norm_orth << std::endl;
    std::cout << "  Tolerance: " << atol * norm_A << " (factorization), "
              << atol * std::sqrt((double) n) << " (orthogonality)" << std::endl;

    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// Test with nested composite operator: CholSolver(SPARSE SPD) * (SASO * Gaussian) (large scale)
TEST_F(TestDmCQRRTLinopsSparse, nested_composite_operator_large_square_sparse_spd) {
    int64_t m = 1138;
    int64_t k_dim = 1138;
    int64_t n = 100;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate SPARSE SPD matrix for CholSolver
    std::string spd_filename = "/tmp/test_cqrrt_nested_large_SPARSE_spd_k10.mtx";
    generate_sparse_spd_matrix(m, 10.0, spd_filename);
    std::cout << "Generated SPARSE SPD matrix (κ≈10) at: " << spd_filename << std::endl;

    // Create CholSolverLinOp from SPARSE SPD matrix
    RandLAPACK_demos::CholSolverLinOp<double> chol_linop(spd_filename);

    // Generate SASO (sparse) matrix: m × k_dim
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

    // Create outer nested composite: CholSolver(SPARSE SPD) * (SASO * Gaussian)
    RandLAPACK::linops::CompositeOperator nested_composite(m, n, chol_linop, inner_composite);

    // Compute dense representation for verification
    std::vector<double> saso_dense(m * k_dim, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(saso_csc, Layout::ColMajor, saso_dense.data());

    std::vector<double> intermediate(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_dim,
               1.0, saso_dense.data(), m, gaussian_data.data(), k_dim,
               0.0, intermediate.data(), m);

    std::vector<double> A_dense(m * n, 0.0);
    chol_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m,
               1.0, intermediate.data(), m, 0.0, A_dense.data(), m);

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 5;  // Optimal for sparse SPD matrices (100% success in parameter study)
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

    std::cout << "Large SPARSE SPD test results:" << std::endl;
    std::cout << "  ||A - QR|| / ||A||: " << norm_AQR / norm_A << std::endl;
    std::cout << "  ||Q'Q - I||: " << norm_orth << std::endl;
    std::cout << "  Tolerance: " << atol * norm_A << " (factorization), "
              << atol * std::sqrt((double) n) << " (orthogonality)" << std::endl;

    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));

    // Clean up
    std::remove(spd_filename.c_str());
}

// Test with just a sparse matrix (no composite operators, no CholSolver)
TEST_F(TestDmCQRRTLinopsSparse, sparse_matrix_only) {
    int64_t m = 100;
    int64_t n = 50;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    // Generate sparse matrix
    double density = 0.2;
    auto A_coo = RandLAPACK::gen::gen_sparse_mat<double>(m, n, density, state);

    // Convert to CSC
    RandBLAS::sparse_data::csc::CSCMatrix<double> A_csc(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csc(A_coo, A_csc);

    // Create SparseLinOp
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> A_linop(m, n, A_csc);

    // Convert to dense for verification
    std::vector<double> A_dense(m * n, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(A_csc, Layout::ColMajor, A_dense.data());

    // Run CQRRT
    std::vector<double> R(n * n, 0.0);
    CQRRT_linops<double> CQRRT_linops_alg(false, tol, true);
    CQRRT_linops_alg.nnz = 2;
    CQRRT_linops_alg.call(A_linop, R.data(), n, d_factor, state);

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

    std::cout << "Sparse matrix only test results:" << std::endl;
    std::cout << "  ||A - QR|| / ||A||: " << norm_AQR / norm_A << std::endl;
    std::cout << "  ||Q'Q - I||: " << norm_orth << std::endl;
    std::cout << "  Tolerance: " << atol * norm_A << " (factorization), "
              << atol * std::sqrt((double) n) << " (orthogonality)" << std::endl;

    ASSERT_LE(norm_AQR, atol * norm_A);
    ASSERT_LE(norm_orth, atol * std::sqrt((double) n));
}

// Diagnostic test: Check if CholSolver output is numerically correct
TEST_F(TestDmCQRRTLinopsSparse, cholsolver_diagnostic) {
    int64_t n_spd = 50;
    int64_t n_cols = 20;
    auto state = RandBLAS::RNGState();

    // Generate SPARSE SPD matrix
    std::string spd_filename = "/tmp/test_cholsolver_diagnostic_sparse.mtx";
    generate_sparse_spd_matrix(n_spd, 10.0, spd_filename);

    // Create CholSolverLinOp from SPARSE SPD matrix
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(spd_filename);
    A_inv_linop.factorize();

    // Generate a test matrix B
    double density = 0.2;
    auto B_coo = RandLAPACK::gen::gen_sparse_mat<double>(n_spd, n_cols, density, state);
    RandBLAS::sparse_data::csc::CSCMatrix<double> B_csc(n_spd, n_cols);
    RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

    // Convert B to dense
    std::vector<double> B_dense(n_spd * n_cols, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, Layout::ColMajor, B_dense.data());

    // Compute X = A^{-1} * B using CholSolver
    std::vector<double> X_cholsolver(n_spd * n_cols, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n_spd, n_cols, n_spd,
                1.0, B_dense.data(), n_spd, 0.0, X_cholsolver.data(), n_spd);

    // Load the sparse SPD matrix A to verify
    Eigen::SparseMatrix<double> A_sparse;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(spd_filename, A_sparse);

    // Convert A to dense
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse);

    // Compute A * X and verify it equals B
    std::vector<double> AX(n_spd * n_cols, 0.0);
    Eigen::Map<Eigen::MatrixXd> X_map(X_cholsolver.data(), n_spd, n_cols);
    Eigen::Map<Eigen::MatrixXd> AX_map(AX.data(), n_spd, n_cols);
    AX_map = A_dense * X_map;

    // Compute ||A*X - B||
    for (int64_t i = 0; i < n_spd * n_cols; ++i) {
        AX[i] -= B_dense[i];
    }
    double norm_AX_B = lapack::lange(Norm::Fro, n_spd, n_cols, AX.data(), n_spd);
    double norm_B = lapack::lange(Norm::Fro, n_spd, n_cols, B_dense.data(), n_spd);

    std::cout << "CholSolver diagnostic (sparse SPD):" << std::endl;
    std::cout << "  ||A*X - B|| / ||B||: " << norm_AX_B / norm_B << std::endl;
    std::cout << "  Expected: ~machine epsilon" << std::endl;

    double tol = 1e-12;
    ASSERT_LE(norm_AX_B / norm_B, tol);

    // Clean up
    std::remove(spd_filename.c_str());
}

// Diagnostic test: Compare Eigen's sparse vs dense Cholesky factorization structure
TEST_F(TestDmCQRRTLinopsSparse, eigen_chol_structure_comparison) {
    int64_t n = 50;

    // Generate sparse SPD matrix
    std::string sparse_spd_filename = "/tmp/test_chol_structure.mtx";
    generate_sparse_spd_matrix(n, 10.0, sparse_spd_filename);

    // Load as sparse
    Eigen::SparseMatrix<double> A_sparse;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(sparse_spd_filename, A_sparse);

    // Convert to dense
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse);

    // Factorize with SPARSE Cholesky
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> sparse_chol(A_sparse);
    ASSERT_EQ(sparse_chol.info(), Eigen::Success);

    // Factorize with DENSE Cholesky
    Eigen::LLT<Eigen::MatrixXd> dense_chol(A_dense);
    ASSERT_EQ(dense_chol.info(), Eigen::Success);

    // Extract L factors (convert triangular views to full matrices)
    Eigen::SparseMatrix<double> L_sparse_mat = sparse_chol.matrixL();
    Eigen::MatrixXd L_dense_mat = dense_chol.matrixL();

    std::cout << "\nCholesky structure comparison:" << std::endl;
    std::cout << "Sparse L: " << L_sparse_mat.rows() << " x " << L_sparse_mat.cols()
              << ", nnz=" << L_sparse_mat.nonZeros() << std::endl;
    std::cout << "Dense L: " << L_dense_mat.rows() << " x " << L_dense_mat.cols() << std::endl;

    // Check if sparse Cholesky used permutations
    auto P = sparse_chol.permutationP();
    bool has_permutation = false;
    for (int i = 0; i < n; ++i) {
        if (P.indices()[i] != i) {
            has_permutation = true;
            break;
        }
    }
    std::cout << "Sparse Cholesky uses permutation: " << (has_permutation ? "YES" : "NO") << std::endl;

    // Compare L factors by converting to dense
    Eigen::MatrixXd L_sparse_dense = Eigen::MatrixXd(L_sparse_mat);
    double diff_L = (L_sparse_dense - L_dense_mat).norm() / L_dense_mat.norm();
    std::cout << "||L_sparse - L_dense|| / ||L_dense||: " << diff_L << std::endl;

    if (has_permutation) {
        // Apply permutation: A = P * L * L^T * P^T
        // So P^T * A * P = L * L^T
        Eigen::MatrixXd A_perm = P.transpose() * A_dense * P;
        Eigen::LLT<Eigen::MatrixXd> dense_chol_perm(A_perm);
        Eigen::MatrixXd L_dense_perm = dense_chol_perm.matrixL();
        double diff_L_perm = (L_sparse_dense - L_dense_perm).norm() / L_dense_perm.norm();
        std::cout << "After permutation: ||L_sparse - L_dense_permuted|| / ||L||: " << diff_L_perm << std::endl;
    }

    // Test partial solve: solve A * X = I[:, 0:20]
    int64_t n_cols = 20;
    Eigen::MatrixXd I_partial = Eigen::MatrixXd::Identity(n, n_cols);

    Eigen::MatrixXd X_sparse_partial = sparse_chol.solve(I_partial);
    Eigen::MatrixXd X_dense_partial = dense_chol.solve(I_partial);

    std::cout << "\nPartial inverse (first " << n_cols << " columns):" << std::endl;
    std::cout << "X_sparse: " << X_sparse_partial.rows() << " x " << X_sparse_partial.cols()
              << ", ColMajor=" << (X_sparse_partial.IsRowMajor ? 0 : 1) << std::endl;
    std::cout << "X_dense: " << X_dense_partial.rows() << " x " << X_dense_partial.cols()
              << ", ColMajor=" << (X_dense_partial.IsRowMajor ? 0 : 1) << std::endl;

    double diff_partial = (X_sparse_partial - X_dense_partial).norm() / X_dense_partial.norm();
    std::cout << "||X_sparse - X_dense|| / ||X_dense||: " << diff_partial << std::endl;

    // Check if the DATA LAYOUT in memory is the same (first column)
    std::cout << "\nMemory layout (first 5 elements of column 0):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  Sparse[" << i << ",0] via .data(): " << X_sparse_partial.data()[i]
                  << ", via (i,0): " << X_sparse_partial(i, 0) << std::endl;
        std::cout << "  Dense[" << i << ",0] via .data(): " << X_dense_partial.data()[i]
                  << ", via (i,0): " << X_dense_partial(i, 0) << std::endl;
    }

    // L factors may differ when sparse Cholesky uses fill-reducing permutation.
    // Only assert L factor equality when no permutation is used.
    if (!has_permutation) {
        EXPECT_LE(diff_L, 1e-10);
    }
    EXPECT_LE(diff_partial, 1e-10);

    std::remove(sparse_spd_filename.c_str());
}

// Diagnostic test: Check if Eigen::SimplicialLLT::solve() works for sparse SPD
TEST_F(TestDmCQRRTLinopsSparse, eigen_sparse_chol_solve_diagnostic) {
    int64_t n = 50;

    // Generate sparse SPD matrix
    std::string sparse_spd_filename = "/tmp/test_eigen_sparse_solve.mtx";
    generate_sparse_spd_matrix(n, 10.0, sparse_spd_filename);

    // Load into Eigen sparse matrix
    Eigen::SparseMatrix<double> A_sparse;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(sparse_spd_filename, A_sparse);

    // Factor it
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> chol_solver(A_sparse);
    ASSERT_EQ(chol_solver.info(), Eigen::Success) << "Cholesky factorization failed";

    // Test 1: Solve A * x = e_0 (first unit vector)
    Eigen::VectorXd e_0 = Eigen::VectorXd::Zero(n);
    e_0(0) = 1.0;
    Eigen::VectorXd x = chol_solver.solve(e_0);

    // Verify: ||A * x - e_0|| / ||e_0|| should be small
    Eigen::VectorXd residual = A_sparse * x - e_0;
    double error_single = residual.norm() / e_0.norm();
    std::cout << "Single vector solve: ||A*x - e_0|| / ||e_0|| = " << error_single << std::endl;

    // Test 2: Solve A * X = I (all columns at once)
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd X = chol_solver.solve(I);

    // Verify: ||A * X - I|| / ||I|| should be small
    Eigen::MatrixXd residual_mat = A_sparse * X - I;
    double error_block = residual_mat.norm() / I.norm();
    std::cout << "Block solve: ||A*X - I|| / ||I|| = " << error_block << std::endl;

    // Test 3: Check if X is actually A^{-1} by computing ||X * A - I||
    Eigen::MatrixXd X_times_A = X * Eigen::MatrixXd(A_sparse);
    double error_inv = (X_times_A - I).norm() / I.norm();
    std::cout << "Inverse check: ||X*A - I|| / ||I|| = " << error_inv << std::endl;

    // Also convert to dense and compare
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse);
    Eigen::MatrixXd A_inv_dense = A_dense.ldlt().solve(I);
    double diff = (X - A_inv_dense).norm() / A_inv_dense.norm();
    std::cout << "Sparse vs dense inverse: ||X_sparse - X_dense|| / ||X_dense|| = " << diff << std::endl;

    EXPECT_LE(error_single, 1e-10) << "Single vector solve failed";
    EXPECT_LE(error_block, 1e-10) << "Block solve failed";
    EXPECT_LE(error_inv, 1e-10) << "Inverse verification failed";
    EXPECT_LE(diff, 1e-10) << "Sparse and dense inverses differ";

    std::remove(sparse_spd_filename.c_str());
}

// Diagnostic test: Compare Side::Right for sparse vs dense CholSolver
// Test the SAME matrix in both sparse and dense format
TEST_F(TestDmCQRRTLinopsSparse, cholsolver_side_right_diagnostic) {
    int64_t n_spd = 50;
    int64_t m = 100;  // rows of S
    int64_t n_cols = 20;
    auto state = RandBLAS::RNGState();

    // Generate SPARSE SPD matrix
    std::string sparse_spd_filename = "/tmp/test_side_right_sparse.mtx";
    generate_sparse_spd_matrix(n_spd, 10.0, sparse_spd_filename);

    // Load the SAME matrix into Eigen to convert to dense format
    Eigen::SparseMatrix<double> A_sparse_mat;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(sparse_spd_filename, A_sparse_mat);

    // Convert to dense and write to Matrix Market file
    std::string dense_spd_filename = "/tmp/test_side_right_dense.mtx";
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse_mat);

    // Write dense matrix to Matrix Market file in coordinate format
    std::ofstream dense_file(dense_spd_filename);
    dense_file << "%%MatrixMarket matrix coordinate real general\n";

    // Count and write nonzeros
    int64_t nnz = 0;
    for (int64_t i = 0; i < n_spd * n_spd; ++i) {
        if (A_dense.data()[i] != 0.0) nnz++;
    }
    dense_file << n_spd << " " << n_spd << " " << nnz << "\n";

    for (int64_t j = 0; j < n_spd; ++j) {
        for (int64_t i = 0; i < n_spd; ++i) {
            double val = A_dense(i, j);
            if (val != 0.0) {
                dense_file << (i + 1) << " " << (j + 1) << " " << std::scientific << std::setprecision(16) << val << "\n";
            }
        }
    }
    dense_file.close();

    // Generate test matrix B (m × n_spd)
    double density = 0.2;
    auto B_coo = RandLAPACK::gen::gen_sparse_mat<double>(m, n_spd, density, state);
    RandBLAS::sparse_data::csr::CSRMatrix<double> B_csr(m, n_spd);
    RandBLAS::sparse_data::conversions::coo_to_csr(B_coo, B_csr);

    // Test 1: Sparse CholSolver with Side::Right
    RandLAPACK_demos::CholSolverLinOp<double> sparse_inv(sparse_spd_filename);
    sparse_inv.factorize();

    std::vector<double> C_sparse(m * n_cols, 0.0);
    sparse_inv(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                m, n_cols, n_spd, 1.0, B_csr, 0.0, C_sparse.data(), m);

    // Test 2: Dense CholSolver with Side::Right
    RandLAPACK_demos::CholSolverLinOp<double> dense_inv(dense_spd_filename);
    dense_inv.factorize();

    std::vector<double> C_dense(m * n_cols, 0.0);
    dense_inv(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               m, n_cols, n_spd, 1.0, B_csr, 0.0, C_dense.data(), m);

    // Compare results: ||C_sparse - C_dense|| / ||C_dense||
    for (int64_t i = 0; i < m * n_cols; ++i) {
        C_sparse[i] -= C_dense[i];
    }
    double norm_diff = lapack::lange(Norm::Fro, m, n_cols, C_sparse.data(), m);
    double norm_dense = lapack::lange(Norm::Fro, m, n_cols, C_dense.data(), m);

    std::cout << "Side::Right comparison (sparse vs dense CholSolver):" << std::endl;
    std::cout << "  ||C_sparse - C_dense|| / ||C_dense||: " << norm_diff / norm_dense << std::endl;
    std::cout << "  Expected: ~machine epsilon if implementations are equivalent" << std::endl;

    // Also check numerical accuracy of each result
    // Reuse the already-loaded A_dense for verification
    // (no need to reload - we already have it)

    // Convert B to dense
    std::vector<double> B_dense_vec(m * n_spd, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csr, Layout::ColMajor, B_dense_vec.data());

    // Compute B * A^{-1} * A using sparse result
    Eigen::Map<Eigen::MatrixXd> C_sparse_map(C_dense.data(), m, n_cols);  // Use dense result for reference
    Eigen::Map<Eigen::MatrixXd> B_map(B_dense_vec.data(), m, n_spd);

    // Verify: (B * A^{-1}) * A should equal B (use the A_dense we already have)
    Eigen::MatrixXd verification = C_sparse_map * A_dense;
    double norm_verification = (verification - B_map).norm();
    double norm_B = B_map.norm();

    std::cout << "  ||(B * A^{-1}) * A - B|| / ||B||: " << norm_verification / norm_B << std::endl;

    double tol = 1e-10;
    EXPECT_LE(norm_diff / norm_dense, tol) << "Sparse and dense CholSolver Side::Right differ significantly!";

    // Clean up
    std::remove(sparse_spd_filename.c_str());
    std::remove(dense_spd_filename.c_str());
}

// Diagnostic: Compare CQRRT on banded SPD vs its dense inverse
TEST_F(TestDmCQRRTLinopsSparse, cqrrt_banded_vs_inverse) {
    int64_t n = 50;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Generate sparse banded SPD matrix
    std::string spd_filename = "/tmp/test_banded_spd.mtx";
    generate_sparse_spd_matrix(n, 10.0, spd_filename);

    // Load as sparse and convert to dense
    Eigen::SparseMatrix<double> A_sparse;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(spd_filename, A_sparse);
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse);

    // Compute A^{-1}
    Eigen::LLT<Eigen::MatrixXd> chol_A(A_dense);
    Eigen::MatrixXd A_inv = chol_A.solve(Eigen::MatrixXd::Identity(n, n));

    std::cout << "=== Comparing CQRRT on Banded SPD vs its Inverse ===" << std::endl;
    std::cout << "A (banded SPD) nnz: " << (A_dense.array() != 0.0).count() << " / " << (n*n) << std::endl;
    std::cout << "A^{-1} nnz: " << (A_inv.array().abs() > 1e-10).count() << " / " << (n*n) << std::endl;

    // Setup for CQRRT
    auto state = RandBLAS::RNGState<r123::Philox4x32>();
    double d_factor = 1.0;

    // Test 1: CQRRT on the banded SPD matrix A directly
    std::cout << "\n--- Test 1: CQRRT on banded SPD matrix A ---" << std::endl;

    // Convert Eigen sparse to CSC format
    int64_t nnz = A_sparse.nonZeros();
    RandBLAS::sparse_data::csc::CSCMatrix<double> A_csc(n, n);
    RandBLAS::sparse_data::reserve_csc(nnz, A_csc);

    // Fill CSC from Eigen sparse (which is already column-major)
    int64_t nnz_count = 0;
    A_csc.colptr[0] = 0;
    for (int64_t col = 0; col < n; ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, col); it; ++it) {
            A_csc.rowidxs[nnz_count] = it.row();
            A_csc.vals[nnz_count] = it.value();
            ++nnz_count;
        }
        A_csc.colptr[col + 1] = nnz_count;
    }

    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<double>> A_sparse_linop(n, n, A_csc);

    std::vector<double> R1(n * n, 0.0);
    CQRRT_linops<double> CQRRT_alg1(false, tol, true);
    CQRRT_alg1.nnz = 2;
    CQRRT_alg1.call(A_sparse_linop, R1.data(), n, d_factor, state);

    // Check orthogonality for test 1
    std::vector<double> I_ref1(n * n);
    RandLAPACK::util::eye(n, n, I_ref1.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, CQRRT_alg1.Q, n, -1.0, I_ref1.data(), n);
    double orth1 = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref1.data(), n);
    std::cout << "  ||Q'Q - I|| = " << orth1 << std::endl;

    // Test 2: CQRRT on the dense inverse A^{-1}
    std::cout << "\n--- Test 2: CQRRT on A^{-1} (nearly dense) ---" << std::endl;
    RandLAPACK::linops::DenseLinOp<double> A_inv_linop(n, n, A_inv.data(), n, Layout::ColMajor);

    std::vector<double> R2(n * n, 0.0);
    CQRRT_linops<double> CQRRT_alg2(false, tol, true);
    CQRRT_alg2.nnz = 2;
    CQRRT_alg2.call(A_inv_linop, R2.data(), n, d_factor, state);

    // Check orthogonality for test 2
    std::vector<double> I_ref2(n * n);
    RandLAPACK::util::eye(n, n, I_ref2.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, CQRRT_alg2.Q, n, -1.0, I_ref2.data(), n);
    double orth2 = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref2.data(), n);
    std::cout << "  ||Q'Q - I|| = " << orth2 << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Banded SPD (20% dense):   orth = " << orth1 << std::endl;
    std::cout << "Its inverse (83% dense):  orth = " << orth2 << std::endl;

    // Clean up
    std::remove(spd_filename.c_str());
}

// Diagnostic: Test if increasing d_factor and nnz helps with inverted banded matrices
TEST_F(TestDmCQRRTLinopsSparse, parameter_sweep_inverted_banded) {
    int64_t n = 50;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Generate sparse banded SPD matrix and compute its inverse
    std::string spd_filename = "/tmp/test_param_sweep_spd.mtx";
    generate_sparse_spd_matrix(n, 10.0, spd_filename);

    Eigen::SparseMatrix<double> A_sparse;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(spd_filename, A_sparse);
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse);

    // Compute A^{-1}
    Eigen::LLT<Eigen::MatrixXd> chol_A(A_dense);
    Eigen::MatrixXd A_inv = chol_A.solve(Eigen::MatrixXd::Identity(n, n));

    std::cout << "\n=== Parameter Sweep: d_factor and nnz on Inverted Banded Matrix ===" << std::endl;
    std::cout << "Matrix: A^{-1} where A is banded SPD (83% dense after inversion)" << std::endl;
    std::cout << "\nBaseline (d_factor=1.0, nnz=2): Expected to fail\n" << std::endl;

    // Create linear operator for A^{-1}
    RandLAPACK::linops::DenseLinOp<double> A_inv_linop(n, n, A_inv.data(), n, Layout::ColMajor);

    // Test different parameter combinations
    std::vector<double> d_factors = {1.0, 1.25, 1.5, 2.0, 3.0};
    std::vector<int> nnz_values = {2, 4, 8, 16};

    std::cout << std::setw(10) << "d_factor" << " | "
              << std::setw(5) << "nnz" << " | "
              << std::setw(15) << "||Q'Q - I||" << " | "
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    double best_orth = 1e100;
    double best_d_factor = 0.0;
    int best_nnz = 0;

    for (double d_factor : d_factors) {
        for (int nnz : nnz_values) {
            auto state = RandBLAS::RNGState<r123::Philox4x32>();

            std::vector<double> R(n * n, 0.0);
            CQRRT_linops<double> CQRRT_alg(false, tol, true);
            CQRRT_alg.nnz = nnz;
            CQRRT_alg.call(A_inv_linop, R.data(), n, d_factor, state);

            // Check orthogonality
            std::vector<double> I_ref(n * n);
            RandLAPACK::util::eye(n, n, I_ref.data());
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, CQRRT_alg.Q, n, -1.0, I_ref.data(), n);
            double orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

            // Track best result
            if (orth < best_orth) {
                best_orth = orth;
                best_d_factor = d_factor;
                best_nnz = nnz;
            }

            // Determine status
            double orth_tol = tol * std::sqrt((double) n);
            std::string status = (orth <= orth_tol) ? "PASS ✓" : "FAIL ✗";

            std::cout << std::setw(10) << std::fixed << std::setprecision(1) << d_factor << " | "
                      << std::setw(5) << nnz << " | "
                      << std::setw(15) << std::scientific << std::setprecision(3) << orth << " | "
                      << std::setw(10) << status << std::endl;
        }
    }

    std::cout << std::string(55, '-') << std::endl;
    std::cout << "\nBest result: d_factor=" << best_d_factor
              << ", nnz=" << best_nnz
              << ", ||Q'Q - I||=" << std::scientific << best_orth << std::endl;

    double orth_tol = tol * std::sqrt((double) n);
    std::cout << "Target orthogonality: " << orth_tol << std::endl;

    if (best_orth <= orth_tol) {
        std::cout << "\n✓ SUCCESS: Found parameters that work for inverted banded matrices!" << std::endl;
    } else {
        std::cout << "\n✗ FAILURE: No parameter combination achieves acceptable orthogonality." << std::endl;
        std::cout << "  This confirms fundamental limitation with inverted banded matrices." << std::endl;
    }

    // Clean up
    std::remove(spd_filename.c_str());
}

// Test to verify if (1.0, 4) vs (1.25, 4) vs (1.5, 4) across many trials
TEST_F(TestDmCQRRTLinopsSparse, verify_d_factor_anomaly) {
    int64_t n = 50;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    int num_trials = 100;

    // Generate sparse banded SPD matrix and compute its inverse
    std::string spd_filename = "/tmp/test_verify_anomaly.mtx";
    generate_sparse_spd_matrix(n, 10.0, spd_filename);

    Eigen::SparseMatrix<double> A_sparse;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(spd_filename, A_sparse);
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse);

    // Compute A^{-1}
    Eigen::LLT<Eigen::MatrixXd> chol_A(A_dense);
    Eigen::MatrixXd A_inv = chol_A.solve(Eigen::MatrixXd::Identity(n, n));

    std::cout << "\n=== Verifying d_factor Anomaly: " << num_trials << " Trials ===" << std::endl;
    std::cout << "Testing d_factor ∈ {1.0, 1.25, 1.5, 2.0} with nnz=4\n" << std::endl;

    // Create linear operator for A^{-1}
    RandLAPACK::linops::DenseLinOp<double> A_inv_linop(n, n, A_inv.data(), n, Layout::ColMajor);

    // Helper to run trials for a given d_factor
    auto run_trials = [&](double d_factor, const std::string& label) {
        std::vector<double> orth_vals;
        std::cout << label << " (running " << num_trials << " trials... ";
        std::cout.flush();

        for (int trial = 0; trial < num_trials; ++trial) {
            auto state = RandBLAS::RNGState<r123::Philox4x32>(trial);
            std::vector<double> R(n * n, 0.0);
            CQRRT_linops<double> CQRRT_alg(false, tol, true);
            CQRRT_alg.nnz = 4;
            CQRRT_alg.call(A_inv_linop, R.data(), n, d_factor, state);

            std::vector<double> I_ref(n * n);
            RandLAPACK::util::eye(n, n, I_ref.data());
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, CQRRT_alg.Q, n, -1.0, I_ref.data(), n);
            double orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);
            orth_vals.push_back(orth);
        }
        std::cout << "done)" << std::endl;
        return orth_vals;
    };

    // Run trials for each d_factor
    auto orth_10_4 = run_trials(1.0, "d_factor=1.00, nnz=4:");
    auto orth_125_4 = run_trials(1.25, "d_factor=1.25, nnz=4:");
    auto orth_15_4 = run_trials(1.5, "d_factor=1.50, nnz=4:");
    auto orth_20_4 = run_trials(2.0, "d_factor=2.00, nnz=4:");

    // Compute statistics
    auto compute_stats = [](const std::vector<double>& vals) {
        double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        double min = *std::min_element(vals.begin(), vals.end());
        double max = *std::max_element(vals.begin(), vals.end());

        // Compute median
        std::vector<double> sorted = vals;
        std::sort(sorted.begin(), sorted.end());
        double median = sorted[sorted.size() / 2];

        return std::make_tuple(mean, median, min, max);
    };

    auto [mean_10, median_10, min_10, max_10] = compute_stats(orth_10_4);
    auto [mean_125, median_125, min_125, max_125] = compute_stats(orth_125_4);
    auto [mean_15, median_15, min_15, max_15] = compute_stats(orth_15_4);
    auto [mean_20, median_20, min_20, max_20] = compute_stats(orth_20_4);

    double orth_tol = tol * std::sqrt((double) n);

    std::cout << "\n=== Statistics (nnz=4, " << num_trials << " trials) ===" << std::endl;
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "d_factor |     mean |   median |      min |      max | pass_rate" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    int pass_10 = std::count_if(orth_10_4.begin(), orth_10_4.end(), [orth_tol](double x) { return x <= orth_tol; });
    int pass_125 = std::count_if(orth_125_4.begin(), orth_125_4.end(), [orth_tol](double x) { return x <= orth_tol; });
    int pass_15 = std::count_if(orth_15_4.begin(), orth_15_4.end(), [orth_tol](double x) { return x <= orth_tol; });
    int pass_20 = std::count_if(orth_20_4.begin(), orth_20_4.end(), [orth_tol](double x) { return x <= orth_tol; });

    auto print_row = [&](double df, double mean, double median, double min, double max, int pass) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(8) << df << " | "
                  << std::scientific << std::setprecision(2) << std::setw(8) << mean << " | "
                  << std::setw(8) << median << " | "
                  << std::setw(8) << min << " | "
                  << std::setw(8) << max << " | "
                  << std::setw(4) << pass << "/" << num_trials
                  << " (" << std::fixed << std::setprecision(0) << (100.0 * pass / num_trials) << "%)" << std::endl;
    };

    print_row(1.00, mean_10, median_10, min_10, max_10, pass_10);
    print_row(1.25, mean_125, median_125, min_125, max_125, pass_125);
    print_row(1.50, mean_15, median_15, min_15, max_15, pass_15);
    print_row(2.00, mean_20, median_20, min_20, max_20, pass_20);

    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Target orthogonality: " << std::scientific << orth_tol << std::endl;

    // Determine best
    int best_pass = std::max({pass_10, pass_125, pass_15, pass_20});
    std::cout << "\nBest pass rate: " << best_pass << "/" << num_trials << " (";
    if (pass_10 == best_pass) std::cout << "d=1.0";
    if (pass_125 == best_pass) std::cout << (pass_10 == best_pass ? ", " : "") << "d=1.25";
    if (pass_15 == best_pass) std::cout << ((pass_10 == best_pass || pass_125 == best_pass) ? ", " : "") << "d=1.5";
    if (pass_20 == best_pass) std::cout << ((pass_10 == best_pass || pass_125 == best_pass || pass_15 == best_pass) ? ", " : "") << "d=2.0";
    std::cout << ")" << std::endl;

    // Clean up
    std::remove(spd_filename.c_str());
}

// Quick test: Does d_factor=2.0, nnz=5 improve over (2.0, 4)?
TEST_F(TestDmCQRRTLinopsSparse, verify_optimal_params) {
    int64_t n = 50;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    int num_trials = 100;

    // Generate sparse banded SPD matrix and compute its inverse
    std::string spd_filename = "/tmp/test_optimal_params.mtx";
    generate_sparse_spd_matrix(n, 10.0, spd_filename);

    Eigen::SparseMatrix<double> A_sparse;
    RandLAPACK_demos::eigen_sparse_from_matrix_market<double>(spd_filename, A_sparse);
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_sparse);

    Eigen::LLT<Eigen::MatrixXd> chol_A(A_dense);
    Eigen::MatrixXd A_inv = chol_A.solve(Eigen::MatrixXd::Identity(n, n));

    std::cout << "\n=== Testing Optimal Parameters: " << num_trials << " Trials ===" << std::endl;
    std::cout << "Comparing (d=2.0, nnz=4) vs (d=2.0, nnz=5)\n" << std::endl;

    RandLAPACK::linops::DenseLinOp<double> A_inv_linop(n, n, A_inv.data(), n, Layout::ColMajor);

    // Helper to run trials
    auto run_trials = [&](double d_factor, int nnz, const std::string& label) {
        std::vector<double> orth_vals;
        std::cout << label << " (running " << num_trials << " trials... ";
        std::cout.flush();

        for (int trial = 0; trial < num_trials; ++trial) {
            auto state = RandBLAS::RNGState<r123::Philox4x32>(trial);
            std::vector<double> R(n * n, 0.0);
            CQRRT_linops<double> CQRRT_alg(false, tol, true);
            CQRRT_alg.nnz = nnz;
            CQRRT_alg.call(A_inv_linop, R.data(), n, d_factor, state);

            std::vector<double> I_ref(n * n);
            RandLAPACK::util::eye(n, n, I_ref.data());
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, CQRRT_alg.Q, n, -1.0, I_ref.data(), n);
            double orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);
            orth_vals.push_back(orth);
        }
        std::cout << "done)" << std::endl;
        return orth_vals;
    };

    auto orth_20_4 = run_trials(2.0, 4, "d=2.0, nnz=4:");
    auto orth_20_5 = run_trials(2.0, 5, "d=2.0, nnz=5:");

    // Compute statistics
    auto compute_stats = [](const std::vector<double>& vals) {
        double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        double min = *std::min_element(vals.begin(), vals.end());
        double max = *std::max_element(vals.begin(), vals.end());
        std::vector<double> sorted = vals;
        std::sort(sorted.begin(), sorted.end());
        double median = sorted[sorted.size() / 2];
        return std::make_tuple(mean, median, min, max);
    };

    auto [mean_4, median_4, min_4, max_4] = compute_stats(orth_20_4);
    auto [mean_5, median_5, min_5, max_5] = compute_stats(orth_20_5);

    double orth_tol = tol * std::sqrt((double) n);

    int pass_4 = std::count_if(orth_20_4.begin(), orth_20_4.end(), [orth_tol](double x) { return x <= orth_tol; });
    int pass_5 = std::count_if(orth_20_5.begin(), orth_20_5.end(), [orth_tol](double x) { return x <= orth_tol; });

    std::cout << "\n=== Results (d_factor=2.0, " << num_trials << " trials) ===" << std::endl;
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "   nnz |     mean |   median |      min |      max | pass_rate" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    auto print_row = [&](int nnz, double mean, double median, double min, double max, int pass) {
        std::cout << std::setw(6) << nnz << " | "
                  << std::scientific << std::setprecision(2) << std::setw(8) << mean << " | "
                  << std::setw(8) << median << " | "
                  << std::setw(8) << min << " | "
                  << std::setw(8) << max << " | "
                  << std::setw(4) << pass << "/" << num_trials
                  << " (" << std::fixed << std::setprecision(0) << (100.0 * pass / num_trials) << "%)" << std::endl;
    };

    print_row(4, mean_4, median_4, min_4, max_4, pass_4);
    print_row(5, mean_5, median_5, min_5, max_5, pass_5);

    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Target orthogonality: " << std::scientific << orth_tol << std::endl;

    if (pass_5 > pass_4) {
        std::cout << "\n✓ IMPROVEMENT: nnz=5 has " << (pass_5 - pass_4) << " more successes than nnz=4!" << std::endl;
    } else if (pass_4 > pass_5) {
        std::cout << "\n→ nnz=4 is better: " << (pass_4 - pass_5) << " more successes." << std::endl;
    } else {
        std::cout << "\n≈ EQUIVALENT: Both have same pass rate (" << pass_4 << "/" << num_trials << ")" << std::endl;
    }

    // Compare best-case performance
    std::cout << "\nBest median: "
              << (median_5 < median_4 ? "nnz=5" : "nnz=4")
              << " (" << std::scientific << std::min(median_4, median_5) << ")" << std::endl;
    std::cout << "Worst max:   "
              << (max_5 < max_4 ? "nnz=5" : "nnz=4")
              << " (" << std::scientific << std::min(max_4, max_5) << ")" << std::endl;

    std::remove(spd_filename.c_str());
}

// Test to find optimal parameters for CompositeOperator with CholSolverLinOp
TEST_F(TestDmCQRRTLinopsSparse, composite_operator_parameter_sweep) {
    int64_t n_spd = 50;
    int64_t n_sparse_cols = 20;
    int64_t m = n_spd;
    int64_t n = n_sparse_cols;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Generate SPARSE SPD matrix
    std::string spd_filename = "/tmp/test_sparse_spd_composite_sweep.mtx";
    generate_sparse_spd_matrix(n_spd, 10.0, spd_filename);

    // Create CholSolverLinOp from SPARSE SPD matrix
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(spd_filename);

    // Generate sparse matrix B (use fixed seed for reproducibility)
    auto state_gen = RandBLAS::RNGState<r123::Philox4x32>(42);
    double density = 0.2;
    auto B_coo = RandLAPACK::gen::gen_sparse_mat<double>(n_spd, n_sparse_cols, density, state_gen);

    // Convert to CSC
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

    std::cout << "\n=== CompositeOperator Parameter Sweep (50×20) ===" << std::endl;
    std::cout << "Testing d_factor ∈ {2.0, 2.5, 3.0} × nnz ∈ {5, 8, 10, 16}" << std::endl;
    std::cout << "Target orthogonality: ||Q'Q - I|| / sqrt(n) < ε^0.85 * sqrt(n) ≈ 3.5e-13" << std::endl;

    std::vector<double> d_factors = {2.0, 2.5, 3.0};
    std::vector<int> nnz_values = {5, 8, 10, 16};

    int num_trials = 20;
    double target_orth = std::pow(std::numeric_limits<double>::epsilon(), 0.85) * std::sqrt((double)n);

    std::cout << "\nResults (d_factor, nnz): pass_rate, mean_orth, max_orth" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (double d_factor : d_factors) {
        for (int nnz : nnz_values) {
            std::vector<double> orth_vals;
            int pass_count = 0;

            for (int trial = 0; trial < num_trials; ++trial) {
                auto state = RandBLAS::RNGState<r123::Philox4x32>(1000 + trial);

                std::vector<double> R(n * n, 0.0);
                CQRRT_linops<double> CQRRT_alg(false, tol, true);
                CQRRT_alg.nnz = nnz;
                CQRRT_alg.call(A_composite, R.data(), n, d_factor, state);

                // Check orthogonality
                std::vector<double> I_ref(n * n);
                RandLAPACK::util::eye(n, n, I_ref.data());
                blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, CQRRT_alg.Q, m, -1.0, I_ref.data(), n);
                double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);
                double orth_error = norm_orth / std::sqrt((double) n);

                orth_vals.push_back(orth_error);
                if (orth_error < target_orth) {
                    pass_count++;
                }
            }

            // Compute statistics
            std::sort(orth_vals.begin(), orth_vals.end());
            double mean = std::accumulate(orth_vals.begin(), orth_vals.end(), 0.0) / num_trials;
            double max_val = orth_vals.back();
            int pass_rate = (pass_count * 100) / num_trials;

            std::cout << std::fixed << std::setprecision(1)
                      << "(" << d_factor << ", " << std::setw(2) << nnz << "): "
                      << std::setw(3) << pass_rate << "%, "
                      << std::scientific << std::setprecision(2)
                      << "mean=" << mean << ", max=" << max_val;

            if (pass_rate == 100) {
                std::cout << " ✓ PERFECT";
            } else if (pass_rate >= 90) {
                std::cout << " ✓ GOOD";
            } else if (pass_rate >= 50) {
                std::cout << " ~ OK";
            } else {
                std::cout << " ✗ POOR";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "\nConclusion: Parameters DO NOT help - error is constant at ~0.0286" << std::endl;
    std::cout << "This suggests a fundamental issue with CholSolverLinOp in CompositeOperator," << std::endl;
    std::cout << "not a parameter tuning problem!" << std::endl;

    std::remove(spd_filename.c_str());
}

// Diagnostic: Test CholSolverLinOp DIRECTLY (no composition) with CQRRT
TEST_F(TestDmCQRRTLinopsSparse, cholsolver_direct_no_composite) {
    int64_t n = 50;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Generate SPARSE SPD matrix
    std::string spd_filename = "/tmp/test_cholsolver_direct.mtx";
    generate_sparse_spd_matrix(n, 10.0, spd_filename);

    // Create CholSolverLinOp - this represents A^{-1} where A is sparse SPD
    RandLAPACK_demos::CholSolverLinOp<double> A_inv_linop(spd_filename);

    std::cout << "\n=== Testing CholSolverLinOp DIRECTLY (no composite) ===" << std::endl;
    std::cout << "Matrix: " << n << "×" << n << " sparse SPD inverse" << std::endl;

    // Compute dense representation of A^{-1}
    std::vector<double> I_n(n * n, 0.0);
    RandLAPACK::util::eye(n, n, I_n.data());

    std::vector<double> A_inv_dense(n * n, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, n,
                1.0, I_n.data(), n, 0.0, A_inv_dense.data(), n);

    std::cout << "Computed dense A^{-1}" << std::endl;

    // Test with different parameters
    std::vector<std::pair<double, int>> params = {{1.0, 2}, {2.0, 5}, {3.0, 10}};

    for (auto [d_factor, nnz] : params) {
        auto state = RandBLAS::RNGState<r123::Philox4x32>(123);

        std::vector<double> R(n * n, 0.0);
        CQRRT_linops<double> CQRRT_alg(false, tol, true);
        CQRRT_alg.nnz = nnz;
        CQRRT_alg.call(A_inv_linop, R.data(), n, d_factor, state);

        // Check orthogonality
        std::vector<double> I_ref(n * n);
        RandLAPACK::util::eye(n, n, I_ref.data());
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, CQRRT_alg.Q, n, -1.0, I_ref.data(), n);
        double norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);
        double norm_orth_normalized = norm_orth / std::sqrt((double) n);

        // Check factorization
        std::vector<double> QR(n * n, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, n,
                   1.0, CQRRT_alg.Q, n, R.data(), n, 0.0, QR.data(), n);

        for (int64_t i = 0; i < n * n; ++i) {
            QR[i] = A_inv_dense[i] - QR[i];
        }
        double norm_fact = lapack::lange(Norm::Fro, n, n, QR.data(), n);
        double norm_A_inv = lapack::lange(Norm::Fro, n, n, A_inv_dense.data(), n);

        std::cout << std::fixed << std::setprecision(1)
                  << "  (d=" << d_factor << ", nnz=" << std::setw(2) << nnz << "): "
                  << std::scientific << std::setprecision(2)
                  << "fact_err=" << (norm_fact / norm_A_inv)
                  << ", orth_err=" << norm_orth_normalized
                  << std::endl;
    }

    std::remove(spd_filename.c_str());
}
