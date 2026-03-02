// Comprehensive unit tests for LUSolverLinOp
// Tests all combinations of Side, Layout, trans_A, trans_B with both dense and sparse B matrices
//
// The LUSolverLinOp represents A^{-1} where A is a sparse invertible matrix.
// The inverse is computed via sparse LU factorization: P_r * A * P_c^T = L * U
// Unlike CholSolverLinOp, trans_A matters since A is non-symmetric.

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>

#include "../../functions/linops_external/dm_lusolver_linop.hh"
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

class TestLUSolverLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Helper function to compute A^{-1} in dense form for reference
    // Uses LU factorization (getrf/getri) since A is general (non-symmetric)
    template <typename T>
    static void compute_dense_inverse(
        int64_t n,
        const T* A,
        T* A_inv
    ) {
        // Copy A to A_inv (we'll factorize in place)
        lapack::lacpy(lapack::MatrixType::General, n, n, A, n, A_inv, n);

        // Compute LU factorization: A = P * L * U
        vector<int64_t> ipiv(n);
        int64_t info = lapack::getrf(n, n, A_inv, n, ipiv.data());
        ASSERT_EQ(info, 0) << "LU factorization failed in reference computation";

        // Compute inverse from LU factors
        info = lapack::getri(n, A_inv, n, ipiv.data());
        ASSERT_EQ(info, 0) << "Inverse computation failed in reference computation";
    }

    // Helper function to compute L^{-1} * P_r in dense form for reference (half-solve mode)
    // Uses Eigen SparseLU to get the same L and P_r as the operator, then computes L^{-1} * P_r.
    template <typename T>
    static void compute_dense_L_inv_Pr(
        int64_t n,
        const std::string& filename,
        T* L_inv_Pr
    ) {
        // Use Eigen SparseLU (same as the operator) to get matching L and P_r
        Eigen::SparseMatrix<T, Eigen::ColMajor> A_eigen;
        RandLAPACK_demos::eigen_sparse_from_matrix_market<T>(filename, A_eigen);

        Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> lu;
        lu.compute(A_eigen);
        ASSERT_EQ(lu.info(), Eigen::Success) << "Eigen SparseLU failed in half-solve reference";

        // Extract L as dense
        Eigen::SparseMatrix<T> L_sp = lu.matrixL().toSparse();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> L_dense(L_sp);

        // Compute L^{-1} by solving L * X = I
        vector<T> L_inv(n * n, (T)0.0);
        for (int64_t i = 0; i < n; ++i)
            L_inv[i + i * n] = (T)1.0;
        blas::trsm(Layout::ColMajor, Side::Left, blas::Uplo::Lower, Op::NoTrans,
                   blas::Diag::Unit, n, n, (T)1.0, L_dense.data(), n, L_inv.data(), n);

        // Extract P_r permutation
        const auto& pr_indices = lu.rowsPermutation().indices();

        // Eigen convention: P_r.indices()[i] = sigma(i) means P_r * e_i = e_{sigma(i)}.
        // So column j of P_r is e_{sigma(j)} = e_{pr_indices[j]}.
        // Therefore: (L^{-1} * P_r)[:, j] = L^{-1} * e_{pr_indices[j]}
        //                                  = column pr_indices[j] of L^{-1}.
        for (int64_t j = 0; j < n; ++j) {
            int64_t src_col = pr_indices[j];
            blas::copy(n, L_inv.data() + src_col * n, 1, L_inv_Pr + j * n, 1);
        }
    }

    // Test function for LUSolverLinOp half_solve mode
    template <typename T>
    void test_lusolver_half_solve(
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

        // Matrix dimension
        int64_t dim = (side == Side::Left) ? m : k;

        std::string filename = "/tmp/test_lusolver_halfsv_" +
                               std::to_string(dim) + "_" +
                               std::to_string(rand()) + ".mtx";

        // Generate invertible matrix with moderate condition number
        T cond_num = 10.0;
        RandLAPACK_demos::generate_invertible_matrix_file<T>(filename, dim, cond_num, state);

        // Compute L^{-1} * P_r for reference
        vector<T> M_ref(dim * dim);
        compute_dense_L_inv_Pr(dim, filename, M_ref.data());

        if (layout == Layout::RowMajor) {
            // In-place transpose: ColMajor(M) -> RowMajor(M)
            // M is NOT symmetric, so we need actual transpose
            vector<T> M_tmp(dim * dim);
            for (int64_t i = 0; i < dim; ++i) {
                for (int64_t j = 0; j < dim; ++j) {
                    M_tmp[i * dim + j] = M_ref[i + j * dim];
                }
            }
            std::copy(M_tmp.begin(), M_tmp.end(), M_ref.begin());
        }

        // Calculate dimensions
        auto dims = calculate_dimensions<T>(side, layout, Op::NoTrans, trans_B, m, n, k);

        // Create output buffers
        vector<T> C_halfsv(m * n);
        vector<T> C_reference(m * n);
        initialize_test_buffers(C_halfsv, C_reference);

        // Create the LUSolverLinOp operator with half_solve=true
        RandLAPACK_demos::LUSolverLinOp<T> M_op(filename, /*half_solve=*/true);

        // Generate dense matrix B
        vector<T> B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);

        // Compute using LUSolverLinOp with half_solve=true
        M_op(side, layout, Op::NoTrans, trans_B, m, n, k, alpha, B_dense.data(), dims.ldb, beta, C_halfsv.data(), dims.ldc);

        // Compute reference: GEMM with L^{-1} * P_r
        compute_gemm_reference(side, layout, Op::NoTrans, trans_B, m, n, k, alpha,
                               M_ref.data(), dim, B_dense.data(), dims.ldb,
                               beta, C_reference.data(), dims.ldc);

        // Compare results
        T atol = 500 * std::numeric_limits<T>::epsilon() * dim;
        T rtol = 100 * std::numeric_limits<T>::epsilon();
        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_halfsv.data(), m,
            C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        );

        // Clean up temporary file
        std::remove(filename.c_str());
    }

    // Unified test function for LUSolverLinOp full-solve mode
    template <typename T>
    void test_lusolver_linop(
        Side side,
        bool sparse_B,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density = 0.0
    ) {
        RNGState state(42);
        T alpha = 1.5;
        T beta = 0.5;

        // Matrix dimension depends on side and transpose
        int64_t dim = (side == Side::Left) ?
                      ((trans_A == Op::NoTrans) ? m : k) :
                      ((trans_A == Op::NoTrans) ? k : n);

        std::string filename = "/tmp/test_lusolver_linop_" +
                               std::to_string(dim) + "_" +
                               std::to_string(rand()) + ".mtx";

        // Generate invertible matrix with moderate condition number
        T cond_num = 10.0;
        RandLAPACK_demos::generate_invertible_matrix_file<T>(filename, dim, cond_num, state);

        // Load the matrix for reference computation
        vector<T> A_mat(dim * dim);
        {
            auto A_coo = RandLAPACK_demos::coo_from_matrix_market<T>(filename);
            RandLAPACK::util::sparse_to_dense(A_coo, Layout::ColMajor, A_mat.data());
        }

        // Compute A^{-1} for reference
        vector<T> A_inv(dim * dim);
        compute_dense_inverse(dim, A_mat.data(), A_inv.data());

        // For trans_A == Trans, we need A^{-T} = (A^{-1})^T
        if (trans_A == Op::Trans) {
            // In-place transpose of A_inv
            for (int64_t i = 0; i < dim; ++i) {
                for (int64_t j = i + 1; j < dim; ++j) {
                    std::swap(A_inv[i + j * dim], A_inv[j + i * dim]);
                }
            }
        }

        // A_inv is in ColMajor from getrf/getri. Convert to target layout
        // so the reference GEMM interprets it correctly.
        if (layout == Layout::RowMajor) {
            convert_layout_inplace(A_inv, dim, dim, Layout::ColMajor, Layout::RowMajor);
        }

        // Calculate dimensions
        auto dims = calculate_dimensions<T>(side, layout, trans_A, trans_B, m, n, k);

        // Create output buffers
        vector<T> C_lusolver(m * n);
        vector<T> C_reference(m * n);
        initialize_test_buffers(C_lusolver, C_reference);

        // Create the LUSolverLinOp operator
        RandLAPACK_demos::LUSolverLinOp<T> A_inv_op(filename);

        if (sparse_B) {
            auto B_csc = generate_sparse_matrix<T>(dims.rows_B, dims.cols_B, density, state);

            A_inv_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_lusolver.data(), dims.ldc);

            vector<T> B_dense(dims.rows_B * dims.cols_B);
            RandLAPACK::util::sparse_to_dense(B_csc, layout, B_dense.data());

            // For reference GEMM, use trans_A=NoTrans since we already transposed A_inv above
            compute_gemm_reference(side, layout, Op::NoTrans, trans_B, m, n, k, alpha,
                                   A_inv.data(), dim, B_dense.data(), dims.ldb,
                                   beta, C_reference.data(), dims.ldc);

            T atol = 500 * std::numeric_limits<T>::epsilon() * dim;
            T rtol = 100 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_lusolver.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );
        } else {
            vector<T> B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);

            A_inv_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense.data(), dims.ldb, beta, C_lusolver.data(), dims.ldc);

            // For reference GEMM, use trans_A=NoTrans since we already transposed A_inv above
            compute_gemm_reference(side, layout, Op::NoTrans, trans_B, m, n, k, alpha,
                                   A_inv.data(), dim, B_dense.data(), dims.ldb,
                                   beta, C_reference.data(), dims.ldc);

            T atol = 500 * std::numeric_limits<T>::epsilon() * dim;
            T rtol = 100 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_lusolver.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );
        }

        // Clean up temporary file
        std::remove(filename.c_str());
    }
};

// ============================================================================
// Side::Left with dense B - ColMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, left_dense_colmajor_notrans_notrans) {
    test_lusolver_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestLUSolverLinOp, left_dense_colmajor_notrans_trans) {
    test_lusolver_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 10);
}

TEST_F(TestLUSolverLinOp, left_dense_colmajor_trans_notrans) {
    test_lusolver_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestLUSolverLinOp, left_dense_colmajor_trans_trans) {
    test_lusolver_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 10);
}

// ============================================================================
// Side::Left with dense B - RowMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, left_dense_rowmajor_notrans_notrans) {
    test_lusolver_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestLUSolverLinOp, left_dense_rowmajor_notrans_trans) {
    test_lusolver_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 10);
}

TEST_F(TestLUSolverLinOp, left_dense_rowmajor_trans_notrans) {
    test_lusolver_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestLUSolverLinOp, left_dense_rowmajor_trans_trans) {
    test_lusolver_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 10);
}

// ============================================================================
// Side::Right with dense B - ColMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, right_dense_colmajor_notrans_notrans) {
    test_lusolver_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestLUSolverLinOp, right_dense_colmajor_notrans_trans) {
    test_lusolver_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 8);
}

TEST_F(TestLUSolverLinOp, right_dense_colmajor_trans_notrans) {
    test_lusolver_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestLUSolverLinOp, right_dense_colmajor_trans_trans) {
    test_lusolver_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 8);
}

// ============================================================================
// Side::Right with dense B - RowMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, right_dense_rowmajor_notrans_notrans) {
    test_lusolver_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestLUSolverLinOp, right_dense_rowmajor_notrans_trans) {
    test_lusolver_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 8);
}

TEST_F(TestLUSolverLinOp, right_dense_rowmajor_trans_notrans) {
    test_lusolver_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestLUSolverLinOp, right_dense_rowmajor_trans_trans) {
    test_lusolver_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 8);
}

// ============================================================================
// Side::Left with sparse B - ColMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, left_sparse_colmajor_notrans_notrans) {
    test_lusolver_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10, 0.3);
}

TEST_F(TestLUSolverLinOp, left_sparse_colmajor_notrans_trans) {
    test_lusolver_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 10, 0.3);
}

TEST_F(TestLUSolverLinOp, left_sparse_colmajor_trans_notrans) {
    test_lusolver_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 10, 0.3);
}

TEST_F(TestLUSolverLinOp, left_sparse_colmajor_trans_trans) {
    test_lusolver_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 10, 0.3);
}

// ============================================================================
// Side::Left with sparse B - RowMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, left_sparse_rowmajor_notrans_notrans) {
    test_lusolver_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10, 0.3);
}

TEST_F(TestLUSolverLinOp, left_sparse_rowmajor_notrans_trans) {
    test_lusolver_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 10, 0.3);
}

TEST_F(TestLUSolverLinOp, left_sparse_rowmajor_trans_notrans) {
    test_lusolver_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 10, 0.3);
}

TEST_F(TestLUSolverLinOp, left_sparse_rowmajor_trans_trans) {
    test_lusolver_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 10, 0.3);
}

// ============================================================================
// Side::Right with sparse B - ColMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, right_sparse_colmajor_notrans_notrans) {
    test_lusolver_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8, 0.3);
}

TEST_F(TestLUSolverLinOp, right_sparse_colmajor_notrans_trans) {
    test_lusolver_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 8, 0.3);
}

TEST_F(TestLUSolverLinOp, right_sparse_colmajor_trans_notrans) {
    test_lusolver_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 8, 0.3);
}

TEST_F(TestLUSolverLinOp, right_sparse_colmajor_trans_trans) {
    test_lusolver_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 8, 0.3);
}

// ============================================================================
// Side::Right with sparse B - RowMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, right_sparse_rowmajor_notrans_notrans) {
    test_lusolver_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8, 0.3);
}

TEST_F(TestLUSolverLinOp, right_sparse_rowmajor_notrans_trans) {
    test_lusolver_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 8, 0.3);
}

TEST_F(TestLUSolverLinOp, right_sparse_rowmajor_trans_notrans) {
    test_lusolver_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 8, 0.3);
}

TEST_F(TestLUSolverLinOp, right_sparse_rowmajor_trans_trans) {
    test_lusolver_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 8, 0.3);
}

// ============================================================================
// Half-solve mode (L^{-1} P_r only): Side::Left, ColMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, halfsv_left_colmajor_notrans) {
    test_lusolver_half_solve<double>(Side::Left, Layout::ColMajor, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestLUSolverLinOp, halfsv_left_colmajor_trans) {
    test_lusolver_half_solve<double>(Side::Left, Layout::ColMajor, Op::Trans, 10, 8, 10);
}

// ============================================================================
// Half-solve mode: Side::Left, RowMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, halfsv_left_rowmajor_notrans) {
    test_lusolver_half_solve<double>(Side::Left, Layout::RowMajor, Op::NoTrans, 10, 8, 10);
}

TEST_F(TestLUSolverLinOp, halfsv_left_rowmajor_trans) {
    test_lusolver_half_solve<double>(Side::Left, Layout::RowMajor, Op::Trans, 10, 8, 10);
}

// ============================================================================
// Half-solve mode: Side::Right, ColMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, halfsv_right_colmajor_notrans) {
    test_lusolver_half_solve<double>(Side::Right, Layout::ColMajor, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestLUSolverLinOp, halfsv_right_colmajor_trans) {
    test_lusolver_half_solve<double>(Side::Right, Layout::ColMajor, Op::Trans, 10, 8, 8);
}

// ============================================================================
// Half-solve mode: Side::Right, RowMajor
// ============================================================================

TEST_F(TestLUSolverLinOp, halfsv_right_rowmajor_notrans) {
    test_lusolver_half_solve<double>(Side::Right, Layout::RowMajor, Op::NoTrans, 10, 8, 8);
}

TEST_F(TestLUSolverLinOp, halfsv_right_rowmajor_trans) {
    test_lusolver_half_solve<double>(Side::Right, Layout::RowMajor, Op::Trans, 10, 8, 8);
}
