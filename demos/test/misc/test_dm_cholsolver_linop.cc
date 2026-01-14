// Comprehensive unit tests for CholSolverLinOp
// Tests all combinations of Side, Layout, trans_A, trans_B with both dense and sparse B matrices
//
// The CholSolverLinOp represents A^{-1} where A is a sparse SPD matrix.
// The inverse is computed via sparse Cholesky factorization: A = L * L^T

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>

#include "../../functions/linops_external/dm_cholsolver_linop.hh"
#include "../../functions/misc/dm_util.hh"
#include "../../../RandLAPACK/RandBLAS/test/comparison.hh"
#include "../../../RandLAPACK/misc/rl_util_test_linop.hh"

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;
using namespace RandLAPACK::util::test;

class TestCholSolverLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Helper function to compute A^{-1} in dense form for reference
    // Takes an SPD matrix A and computes its inverse via Cholesky factorization
    template <typename T>
    static void compute_dense_inverse(
        int64_t n,
        const T* A,
        T* A_inv
    ) {
        // Copy A to A_inv (we'll factorize in place)
        lapack::lacpy(lapack::MatrixType::General, n, n, A, n, A_inv, n);

        // Compute Cholesky factorization: A = L * L^T
        int64_t info = lapack::potrf(lapack::Uplo::Lower, n, A_inv, n);
        ASSERT_EQ(info, 0) << "Cholesky factorization failed in reference computation";

        // Compute inverse from Cholesky factor
        info = lapack::potri(lapack::Uplo::Lower, n, A_inv, n);
        ASSERT_EQ(info, 0) << "Inverse computation failed in reference computation";

        // Fill upper triangle (potri only computes lower triangle)
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                A_inv[i + j * n] = A_inv[j + i * n];
            }
        }
    }

    // Unified test function for CholSolverLinOp
    // Handles both Side::Left and Side::Right, both dense and sparse B, both ColMajor and RowMajor
    template <typename T>
    void test_cholsolver_linop(
        Side side,
        bool sparse_B,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density = 0.0  // Only used if sparse_B == true
    ) {
        RNGState state(42);  // Fixed seed for reproducibility
        T alpha = 1.5;
        T beta = 0.5;

        // Generate SPD matrix and save to temporary file
        // The SPD matrix dimension depends on the side and transpose operations
        int64_t spd_dim = (side == Side::Left) ?
                          ((trans_A == Op::NoTrans) ? m : k) :
                          ((trans_A == Op::NoTrans) ? k : n);

        std::string spd_filename = "/tmp/test_cholsolver_linop_" +
                                   std::to_string(spd_dim) + "_" +
                                   std::to_string(rand()) + ".mtx";

        // Generate SPD matrix with moderate condition number
        T cond_num = 10.0;
        RandLAPACK_demos::generate_spd_matrix_file<T>(spd_filename, spd_dim, cond_num, state);

        // Load the SPD matrix for reference computation
        vector<T> A_spd(spd_dim * spd_dim);
        {
            auto A_coo = RandLAPACK_demos::coo_from_matrix_market<T>(spd_filename);
            RandLAPACK::util::sparse_to_dense(A_coo, Layout::ColMajor, A_spd.data());
        }

        // Compute A^{-1} for reference
        vector<T> A_inv(spd_dim * spd_dim);
        compute_dense_inverse(spd_dim, A_spd.data(), A_inv.data());

        // Calculate dimensions using utility function
        auto dims = calculate_dimensions<T>(side, layout, trans_A, trans_B, m, n, k);

        // Create output buffers
        vector<T> C_cholsolver(m * n);
        vector<T> C_reference(m * n);

        // Initialize test buffers using utility function
        initialize_test_buffers(C_cholsolver, C_reference);

        // Create the CholSolverLinOp operator
        RandLAPACK_demos::CholSolverLinOp<T> A_inv_op(spd_filename);

        if (sparse_B) {
            // Generate sparse matrix B using utility function
            auto B_csc = generate_sparse_matrix<T>(dims.rows_B, dims.cols_B, density, state);

            // Compute using CholSolverLinOp with sparse B
            A_inv_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_cholsolver.data(), dims.ldc);

            // Compute reference: densify B and use BLAS GEMM with A^{-1}
            vector<T> B_dense(dims.rows_B * dims.cols_B);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, layout, B_dense.data());

            // Compute reference using utility function
            compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, alpha,
                                   A_inv.data(), spd_dim, B_dense.data(), dims.ldb,
                                   beta, C_reference.data(), dims.ldc);

            // Compare results with relaxed tolerance for sparse operations
            T atol = 500 * std::numeric_limits<T>::epsilon() * spd_dim;
            T rtol = 100 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_cholsolver.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );
        } else {
            // Generate dense matrix B using utility function
            vector<T> B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);

            // Compute using CholSolverLinOp with dense B
            A_inv_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense.data(), dims.ldb, beta, C_cholsolver.data(), dims.ldc);

            // Compute reference using utility function
            compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, alpha,
                                   A_inv.data(), spd_dim, B_dense.data(), dims.ldb,
                                   beta, C_reference.data(), dims.ldc);

            // Compare results with standard tolerance
            T atol = 500 * std::numeric_limits<T>::epsilon() * spd_dim;
            T rtol = 100 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_cholsolver.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );
        }

        // Clean up temporary file
        std::remove(spd_filename.c_str());
    }
};

// ============================================================================
// Side::Left with dense B - ColMajor
// ============================================================================

TEST_F(TestCholSolverLinOp, left_dense_colmajor_notrans_notrans) {
    test_cholsolver_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestCholSolverLinOp, left_dense_colmajor_notrans_trans) {
    test_cholsolver_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 10);
}

TEST_F(TestCholSolverLinOp, left_dense_colmajor_trans_notrans) {
    test_cholsolver_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestCholSolverLinOp, left_dense_colmajor_trans_trans) {
    test_cholsolver_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 10);
}

// ============================================================================
// Side::Left with dense B - RowMajor
// ============================================================================

TEST_F(TestCholSolverLinOp, left_dense_rowmajor_notrans_notrans) {
    test_cholsolver_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestCholSolverLinOp, left_dense_rowmajor_notrans_trans) {
    test_cholsolver_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 10);
}

TEST_F(TestCholSolverLinOp, left_dense_rowmajor_trans_notrans) {
    test_cholsolver_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestCholSolverLinOp, left_dense_rowmajor_trans_trans) {
    test_cholsolver_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 10);
}

// ============================================================================
// Side::Right with dense B - ColMajor
// NOTE: For Side::Right with CholSolverLinOp (square operator), we must have k = n
// ============================================================================

TEST_F(TestCholSolverLinOp, right_dense_colmajor_notrans_notrans) {
    test_cholsolver_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestCholSolverLinOp, right_dense_colmajor_notrans_trans) {
    test_cholsolver_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 8);
}

TEST_F(TestCholSolverLinOp, right_dense_colmajor_trans_notrans) {
    test_cholsolver_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestCholSolverLinOp, right_dense_colmajor_trans_trans) {
    test_cholsolver_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 8);
}

// ============================================================================
// Side::Right with dense B - RowMajor
// NOTE: For Side::Right with CholSolverLinOp (square operator), we must have k = n
// ============================================================================

TEST_F(TestCholSolverLinOp, right_dense_rowmajor_notrans_notrans) {
    test_cholsolver_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestCholSolverLinOp, right_dense_rowmajor_notrans_trans) {
    test_cholsolver_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 8);
}

TEST_F(TestCholSolverLinOp, right_dense_rowmajor_trans_notrans) {
    test_cholsolver_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestCholSolverLinOp, right_dense_rowmajor_trans_trans) {
    test_cholsolver_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 8);
}

// ============================================================================
// Side::Left with sparse B - ColMajor
// ============================================================================

TEST_F(TestCholSolverLinOp, left_sparse_colmajor_notrans_notrans) {
    test_cholsolver_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10, 0.3);
}

TEST_F(TestCholSolverLinOp, left_sparse_colmajor_notrans_trans) {
    test_cholsolver_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 10, 0.3);
}

TEST_F(TestCholSolverLinOp, left_sparse_colmajor_trans_notrans) {
    test_cholsolver_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 10, 0.3);
}

TEST_F(TestCholSolverLinOp, left_sparse_colmajor_trans_trans) {
    test_cholsolver_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 10, 0.3);
}

// ============================================================================
// Side::Left with sparse B - RowMajor
// ============================================================================

TEST_F(TestCholSolverLinOp, left_sparse_rowmajor_notrans_notrans) {
    test_cholsolver_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10, 0.3);
}

TEST_F(TestCholSolverLinOp, left_sparse_rowmajor_notrans_trans) {
    test_cholsolver_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 10, 0.3);
}

TEST_F(TestCholSolverLinOp, left_sparse_rowmajor_trans_notrans) {
    test_cholsolver_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 10, 0.3);
}

TEST_F(TestCholSolverLinOp, left_sparse_rowmajor_trans_trans) {
    test_cholsolver_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 10, 0.3);
}

// ============================================================================
// Side::Right with sparse B - ColMajor
// NOTE: For Side::Right with CholSolverLinOp (square operator), we must have k = n
// ============================================================================

TEST_F(TestCholSolverLinOp, right_sparse_colmajor_notrans_notrans) {
    test_cholsolver_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8, 0.3);
}

TEST_F(TestCholSolverLinOp, right_sparse_colmajor_notrans_trans) {
    test_cholsolver_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 8, 0.3);
}

TEST_F(TestCholSolverLinOp, right_sparse_colmajor_trans_notrans) {
    test_cholsolver_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 8, 0.3);
}

TEST_F(TestCholSolverLinOp, right_sparse_colmajor_trans_trans) {
    test_cholsolver_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 8, 0.3);
}

// ============================================================================
// Side::Right with sparse B - RowMajor
// NOTE: For Side::Right with CholSolverLinOp (square operator), we must have k = n
// ============================================================================

TEST_F(TestCholSolverLinOp, right_sparse_rowmajor_notrans_notrans) {
    test_cholsolver_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8, 0.3);
}

TEST_F(TestCholSolverLinOp, right_sparse_rowmajor_notrans_trans) {
    test_cholsolver_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 8, 0.3);
}

TEST_F(TestCholSolverLinOp, right_sparse_rowmajor_trans_notrans) {
    test_cholsolver_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 8, 0.3);
}

TEST_F(TestCholSolverLinOp, right_sparse_rowmajor_trans_trans) {
    test_cholsolver_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 8, 0.3);
}
