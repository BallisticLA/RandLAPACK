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

#include "../../linops/ext_cholsolver_linop.hh"
#include "../../misc/ext_util.hh"
#include <RandLAPACK/testing/rl_test_utils.hh>
#include <test/comparison.hh>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;
using namespace RandLAPACK::testing;

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

    // Helper function to compute L^{-1} in dense form for reference (half-solve mode)
    // Takes an SPD matrix A, Cholesky factorizes to get L, then computes L^{-1}
    template <typename T>
    static void compute_dense_L_inv(
        int64_t n,
        const T* A,
        T* L_inv
    ) {
        // Copy A (we'll factorize in place to get L)
        vector<T> L(n * n);
        lapack::lacpy(lapack::MatrixType::General, n, n, A, n, L.data(), n);

        // Compute Cholesky factorization: A = L * L^T
        int64_t info = lapack::potrf(lapack::Uplo::Lower, n, L.data(), n);
        ASSERT_EQ(info, 0) << "Cholesky factorization failed in half-solve reference computation";

        // Set L_inv = identity
        std::fill(L_inv, L_inv + n * n, (T)0.0);
        for (int64_t i = 0; i < n; ++i)
            L_inv[i + i * n] = (T)1.0;

        // Solve L * L_inv = I via triangular solve: L_inv := L^{-1}
        blas::trsm(Layout::ColMajor, Side::Left, blas::Uplo::Lower, Op::NoTrans,
                   blas::Diag::NonUnit, n, n, (T)1.0, L.data(), n, L_inv, n);
    }

    // Test function for CholSolverLinOp half_solve mode
    // The reference is L^{-1} (lower triangular) instead of A^{-1}
    template <typename T>
    void test_cholsolver_half_solve(
        Side side,
        Layout layout,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k
    ) {
        RNGState state(42);
        T alpha = 1.5;
        T beta = 0.5;

        // SPD matrix dimension
        int64_t spd_dim = (side == Side::Left) ? m : k;

        std::string spd_filename = "/tmp/test_cholsolver_halfsv_" +
                                   std::to_string(spd_dim) + "_" +
                                   std::to_string(rand()) + ".mtx";

        // Generate SPD matrix with moderate condition number
        T cond_num = 10.0;
        RandLAPACK::testing::generate_spd_matrix_file<T>(spd_filename, spd_dim, cond_num, state);

        // Load the SPD matrix for reference computation
        vector<T> A_spd(spd_dim * spd_dim);
        {
            auto A_coo = RandLAPACK_extras::coo_from_matrix_market<T>(spd_filename);
            RandLAPACK::util::sparse_to_dense(A_coo, Layout::ColMajor, A_spd.data());
        }

        // Compute L^{-1} for reference (instead of A^{-1})
        // L^{-1} is computed in ColMajor. For RowMajor GEMM reference, we need to
        // transpose it because BLAS GEMM with RowMajor interprets the pointer as RowMajor,
        // so ColMajor(L^{-1}) would be read as L^{-T}. Unlike A^{-1} (symmetric),
        // L^{-1} is lower triangular and NOT symmetric.
        vector<T> L_inv(spd_dim * spd_dim);
        compute_dense_L_inv(spd_dim, A_spd.data(), L_inv.data());

        if (layout == Layout::RowMajor) {
            // In-place transpose: ColMajor(L^{-1}) → RowMajor(L^{-1})
            for (int64_t i = 0; i < spd_dim; ++i) {
                for (int64_t j = i + 1; j < spd_dim; ++j) {
                    std::swap(L_inv[i + j * spd_dim], L_inv[j + i * spd_dim]);
                }
            }
        }

        // Calculate dimensions
        auto dims = calculate_dimensions<T>(side, layout, Op::NoTrans, trans_B, m, n, k);

        // Create output buffers
        vector<T> C_halfsv(m * n);
        vector<T> C_reference(m * n);
        initialize_test_buffers(C_halfsv, C_reference);

        // Create the CholSolverLinOp operator with half_solve=true
        RandLAPACK_extras::linops::CholSolverLinOp<T> L_inv_op(spd_filename, /*half_solve=*/true);

        // Generate dense matrix B
        vector<T> B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);

        // Compute using CholSolverLinOp with half_solve=true
        L_inv_op(side, layout, Op::NoTrans, trans_B, m, n, k, alpha, B_dense.data(), dims.ldb, beta, C_halfsv.data(), dims.ldc);

        // Compute reference: GEMM with L^{-1} instead of A^{-1}
        sided_gemm(side, layout, Op::NoTrans, trans_B, m, n, k, alpha,
                               L_inv.data(), spd_dim, B_dense.data(), dims.ldb,
                               beta, C_reference.data(), dims.ldc);

        // Compare results
        T atol = 500 * std::numeric_limits<T>::epsilon() * spd_dim;
        T rtol = 100 * std::numeric_limits<T>::epsilon();
        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_halfsv.data(), m,
            C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        );

        // Clean up temporary file
        std::remove(spd_filename.c_str());
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
        RandLAPACK::testing::generate_spd_matrix_file<T>(spd_filename, spd_dim, cond_num, state);

        // Load the SPD matrix for reference computation
        vector<T> A_spd(spd_dim * spd_dim);
        {
            auto A_coo = RandLAPACK_extras::coo_from_matrix_market<T>(spd_filename);
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
        RandLAPACK_extras::linops::CholSolverLinOp<T> A_inv_op(spd_filename);

        if (sparse_B) {
            // Generate sparse matrix B using utility function
            auto B_csc = RandLAPACK::gen::gen_sparse_coo<T>(dims.rows_B, dims.cols_B, density, state).as_owning_csc();

            // Compute using CholSolverLinOp with sparse B
            A_inv_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_cholsolver.data(), dims.ldc);

            // Compute reference: densify B and use BLAS GEMM with A^{-1}
            vector<T> B_dense(dims.rows_B * dims.cols_B);
            RandLAPACK::util::sparse_to_dense(B_csc, layout, B_dense.data());

            // Compute reference using utility function
            sided_gemm(side, layout, trans_A, trans_B, m, n, k, alpha,
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
            sided_gemm(side, layout, trans_A, trans_B, m, n, k, alpha,
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

// ============================================================================
// Half-solve mode (L^{-1} only): Side::Left, ColMajor
// ============================================================================

TEST_F(TestCholSolverLinOp, halfsv_left_colmajor_notrans) {
    test_cholsolver_half_solve<double>(Side::Left, Layout::ColMajor, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestCholSolverLinOp, halfsv_left_colmajor_trans) {
    test_cholsolver_half_solve<double>(Side::Left, Layout::ColMajor, Op::Trans, 10, 8, 10);
}

// ============================================================================
// Half-solve mode: Side::Left, RowMajor
// ============================================================================

TEST_F(TestCholSolverLinOp, halfsv_left_rowmajor_notrans) {
    test_cholsolver_half_solve<double>(Side::Left, Layout::RowMajor, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestCholSolverLinOp, halfsv_left_rowmajor_trans) {
    test_cholsolver_half_solve<double>(Side::Left, Layout::RowMajor, Op::Trans, 10, 8, 10);
}

// ============================================================================
// Half-solve mode: Side::Right, ColMajor
// ============================================================================

TEST_F(TestCholSolverLinOp, halfsv_right_colmajor_notrans) {
    test_cholsolver_half_solve<double>(Side::Right, Layout::ColMajor, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestCholSolverLinOp, halfsv_right_colmajor_trans) {
    test_cholsolver_half_solve<double>(Side::Right, Layout::ColMajor, Op::Trans, 10, 8, 8);
}

// ============================================================================
// Half-solve mode: Side::Right, RowMajor
// ============================================================================

TEST_F(TestCholSolverLinOp, halfsv_right_rowmajor_notrans) {
    test_cholsolver_half_solve<double>(Side::Right, Layout::RowMajor, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestCholSolverLinOp, halfsv_right_rowmajor_trans) {
    test_cholsolver_half_solve<double>(Side::Right, Layout::RowMajor, Op::Trans, 10, 8, 8);
}
