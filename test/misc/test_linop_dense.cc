// Unit tests for DenseLinOp
// Tests basic dense matrix linear operator functionality

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "../RandLAPACK/RandBLAS/test/comparison.hh"

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;

class TestDenseLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Unified test function for DenseLinOp
    // Handles both Side::Left and Side::Right, both dense and sparse B
    template <typename T>
    void test_dense_linop(
        Side side,
        bool sparse_B,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density = 0.0  // Only used if sparse_B == true
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Dimension calculation depends on Side
        int64_t rows_A, cols_A, rows_B, cols_B;
        if (side == Side::Left) {
            // Side::Left: C := alpha * op(A) * op(B) + beta * C
            // A is the operator (m × k), B is the input (k × n)
            auto [ra, ca] = RandBLAS::dims_before_op(m, k, trans_A);
            auto [rb, cb] = RandBLAS::dims_before_op(k, n, trans_B);
            rows_A = ra; cols_A = ca;
            rows_B = rb; cols_B = cb;
        } else {
            // Side::Right: C := alpha * op(B) * op(A) + beta * C
            // A is the operator (k × n), B is the input (m × k)
            auto [ra, ca] = RandBLAS::dims_before_op(k, n, trans_A);
            auto [rb, cb] = RandBLAS::dims_before_op(m, k, trans_B);
            rows_A = ra; cols_A = ca;
            rows_B = rb; cols_B = cb;
        }

        // Generate dense matrix A
        vector<T> A_dense(rows_A * cols_A);
        RandBLAS::DenseDist D_A(rows_A, cols_A);
        RandBLAS::fill_dense(D_A, A_dense.data(), state);

        int64_t lda = rows_A;
        int64_t ldc = m;

        // Create output buffers
        vector<T> C_dense_op(m * n);
        vector<T> C_reference(m * n);

        // Initialize C with random data (to test beta scaling)
        for (auto& c : C_dense_op) c = static_cast<T>(rand()) / RAND_MAX;
        C_reference = C_dense_op;

        // Create the DenseLinOp operator
        RandLAPACK::linops::DenseLinOp<T> A_op(rows_A, cols_A, A_dense.data(), lda, Layout::ColMajor);

        if (sparse_B) {
            // Generate sparse matrix B in CSC format
            auto B_coo = RandLAPACK::gen::gen_sparse_mat<T>(rows_B, cols_B, density, state);
            RandBLAS::sparse_data::csc::CSCMatrix<T> B_csc(rows_B, cols_B);
            RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

            // Compute using DenseLinOp with sparse B
            A_op(side, Layout::ColMajor, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_dense_op.data(), ldc);

            // Compute reference: densify B and use BLAS GEMM
            // NOTE: gen_sparse_mat can generate duplicates, so we must SUM them, not overwrite
            vector<T> B_dense(rows_B * cols_B, 0.0);
            int64_t ldb = rows_B;
            for (int64_t j = 0; j < cols_B; ++j) {
                for (int64_t idx = B_csc.colptr[j]; idx < B_csc.colptr[j+1]; ++idx) {
                    int64_t i = B_csc.rowidxs[idx];
                    B_dense[i + j * ldb] += B_csc.vals[idx];  // SUM duplicates!
                }
            }

            // GEMM call depends on Side
            if (side == Side::Left) {
                blas::gemm(Layout::ColMajor, trans_A, trans_B, m, n, k, alpha,
                           A_dense.data(), lda, B_dense.data(), ldb, beta, C_reference.data(), ldc);
            } else {
                blas::gemm(Layout::ColMajor, trans_B, trans_A, m, n, k, alpha,
                           B_dense.data(), ldb, A_dense.data(), lda, beta, C_reference.data(), ldc);
            }

            // Compare results with relaxed tolerance for sparse operations
            T atol = 100 * std::numeric_limits<T>::epsilon();
            T rtol = 10 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_dense_op.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );
        } else {
            // Generate dense matrix B
            vector<T> B_dense(rows_B * cols_B);
            RandBLAS::DenseDist D_B(rows_B, cols_B);
            RandBLAS::fill_dense(D_B, B_dense.data(), state);
            int64_t ldb = rows_B;

            // Compute using DenseLinOp with dense B
            A_op(side, Layout::ColMajor, trans_A, trans_B, m, n, k, alpha, B_dense.data(), ldb, beta, C_dense_op.data(), ldc);

            // Compute reference using BLAS GEMM
            if (side == Side::Left) {
                blas::gemm(Layout::ColMajor, trans_A, trans_B, m, n, k, alpha,
                           A_dense.data(), lda, B_dense.data(), ldb, beta, C_reference.data(), ldc);
            } else {
                blas::gemm(Layout::ColMajor, trans_B, trans_A, m, n, k, alpha,
                           B_dense.data(), ldb, A_dense.data(), lda, beta, C_reference.data(), ldc);
            }

            // Compare results with standard tolerance
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_dense_op.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__
            );
        }
    }
};

// ============================================================================
// Side::Left with dense B
// ============================================================================

TEST_F(TestDenseLinOp, left_colmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Left, false, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_colmajor_notrans_trans) {
    test_dense_linop<double>(Side::Left, false, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_colmajor_trans_notrans) {
    test_dense_linop<double>(Side::Left, false, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_colmajor_trans_trans) {
    test_dense_linop<double>(Side::Left, false, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// Side::Right with dense B
// ============================================================================

TEST_F(TestDenseLinOp, right_colmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Right, false, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_colmajor_notrans_trans) {
    test_dense_linop<double>(Side::Right, false, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_colmajor_trans_notrans) {
    test_dense_linop<double>(Side::Right, false, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_colmajor_trans_trans) {
    test_dense_linop<double>(Side::Right, false, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// Side::Left with sparse B
// ============================================================================

TEST_F(TestDenseLinOp, left_sparse_colmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Left, true, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_colmajor_notrans_trans) {
    test_dense_linop<double>(Side::Left, true, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_colmajor_trans_notrans) {
    test_dense_linop<double>(Side::Left, true, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_colmajor_trans_trans) {
    test_dense_linop<double>(Side::Left, true, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right with sparse B
// ============================================================================

TEST_F(TestDenseLinOp, right_sparse_colmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Right, true, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_colmajor_notrans_trans) {
    test_dense_linop<double>(Side::Right, true, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_colmajor_trans_notrans) {
    test_dense_linop<double>(Side::Right, true, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_colmajor_trans_trans) {
    test_dense_linop<double>(Side::Right, true, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}
