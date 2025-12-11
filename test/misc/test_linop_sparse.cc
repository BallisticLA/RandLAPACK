// Comprehensive unit tests for SparseLinOp
// Tests all combinations of Side, Layout, trans_A, trans_B with both dense and sparse B matrices

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

class TestSparseLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Unified test function for SparseLinOp
    // Handles both Side::Left and Side::Right, both dense and sparse B, and both ColMajor and RowMajor layouts
    template <typename T>
    void test_sparse_linop(
        Side side,
        bool sparse_B,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density_A,
        T density_B = 0.0  // Only used if sparse_B == true
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Dimension calculation depends on Side
        int64_t rows_A, cols_A, rows_B, cols_B;
        if (side == Side::Left) {
            // Side::Left: C := alpha * op(A_sp) * op(B) + beta * C
            // A is the operator (m × k), B is the input (k × n)
            auto [ra, ca] = RandBLAS::dims_before_op(m, k, trans_A);
            auto [rb, cb] = RandBLAS::dims_before_op(k, n, trans_B);
            rows_A = ra; cols_A = ca;
            rows_B = rb; cols_B = cb;
        } else {
            // Side::Right: C := alpha * op(B) * op(A_sp) + beta * C
            // A is the operator (k × n), B is the input (m × k)
            auto [ra, ca] = RandBLAS::dims_before_op(k, n, trans_A);
            auto [rb, cb] = RandBLAS::dims_before_op(m, k, trans_B);
            rows_A = ra; cols_A = ca;
            rows_B = rb; cols_B = cb;
        }

        // Generate sparse matrix A in CSC format
        auto A_coo = RandLAPACK::gen::gen_sparse_mat<T>(rows_A, cols_A, density_A, state);
        RandBLAS::sparse_data::csc::CSCMatrix<T> A_csc(rows_A, cols_A);
        RandBLAS::sparse_data::conversions::coo_to_csc(A_coo, A_csc);

        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        // Create output buffers
        vector<T> C_sparse_op(m * n);
        vector<T> C_reference(m * n);

        // Initialize C with random data (to test beta scaling)
        for (auto& c : C_sparse_op) c = static_cast<T>(rand()) / RAND_MAX;
        C_reference = C_sparse_op;

        // Create the SparseLinOp operator
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>>
            A_op(rows_A, cols_A, A_csc);

        if (sparse_B) {
            // Generate sparse matrix B in CSC format
            auto B_coo = RandLAPACK::gen::gen_sparse_mat<T>(rows_B, cols_B, density_B, state);
            RandBLAS::sparse_data::csc::CSCMatrix<T> B_csc(rows_B, cols_B);
            RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

            // Compute using SparseLinOp with sparse B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_sparse_op.data(), ldc);

            // Compute reference: densify both matrices and use BLAS GEMM
            // NOTE: gen_sparse_mat can generate duplicates, so we must SUM them, not overwrite
            vector<T> A_dense(rows_A * cols_A, 0.0);
            vector<T> B_dense(rows_B * cols_B, 0.0);
            int64_t lda = (layout == Layout::ColMajor) ? rows_A : cols_A;
            int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

            // Densify A
            for (int64_t j = 0; j < cols_A; ++j) {
                for (int64_t idx = A_csc.colptr[j]; idx < A_csc.colptr[j+1]; ++idx) {
                    int64_t i = A_csc.rowidxs[idx];
                    if (layout == Layout::ColMajor) {
                        A_dense[i + j * lda] += A_csc.vals[idx];  // SUM duplicates!
                    } else {
                        A_dense[j + i * lda] += A_csc.vals[idx];  // RowMajor
                    }
                }
            }

            // Densify B
            for (int64_t j = 0; j < cols_B; ++j) {
                for (int64_t idx = B_csc.colptr[j]; idx < B_csc.colptr[j+1]; ++idx) {
                    int64_t i = B_csc.rowidxs[idx];
                    if (layout == Layout::ColMajor) {
                        B_dense[i + j * ldb] += B_csc.vals[idx];  // SUM duplicates!
                    } else {
                        B_dense[j + i * ldb] += B_csc.vals[idx];  // RowMajor
                    }
                }
            }

            // GEMM call depends on Side
            if (side == Side::Left) {
                blas::gemm(layout, trans_A, trans_B, m, n, k, alpha,
                           A_dense.data(), lda, B_dense.data(), ldb, beta, C_reference.data(), ldc);
            } else {
                blas::gemm(layout, trans_B, trans_A, m, n, k, alpha,
                           B_dense.data(), ldb, A_dense.data(), lda, beta, C_reference.data(), ldc);
            }

            // Compare results with relaxed tolerance for sparse operations
            T atol = 100 * std::numeric_limits<T>::epsilon();
            T rtol = 10 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_sparse_op.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );
        } else {
            // Generate dense matrix B
            // NOTE: RandBLAS::fill_dense always fills in ColMajor format
            vector<T> B_dense(rows_B * cols_B);
            RandBLAS::DenseDist D_B(rows_B, cols_B);
            RandBLAS::fill_dense(D_B, B_dense.data(), state);

            // For RowMajor, we need to transpose the data after generation
            if (layout == Layout::RowMajor) {
                vector<T> B_temp = B_dense;
                for (int64_t i = 0; i < rows_B; ++i) {
                    for (int64_t j = 0; j < cols_B; ++j) {
                        B_dense[j + i * cols_B] = B_temp[i + j * rows_B];
                    }
                }
            }
            int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

            // Compute using SparseLinOp with dense B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense.data(), ldb, beta, C_sparse_op.data(), ldc);

            // Compute reference using dense GEMM
            // Densify A
            vector<T> A_dense(rows_A * cols_A, 0.0);
            int64_t lda = (layout == Layout::ColMajor) ? rows_A : cols_A;
            for (int64_t j = 0; j < cols_A; ++j) {
                for (int64_t idx = A_csc.colptr[j]; idx < A_csc.colptr[j+1]; ++idx) {
                    int64_t i = A_csc.rowidxs[idx];
                    if (layout == Layout::ColMajor) {
                        A_dense[i + j * lda] += A_csc.vals[idx];  // SUM duplicates!
                    } else {
                        A_dense[j + i * lda] += A_csc.vals[idx];  // RowMajor
                    }
                }
            }

            // GEMM call depends on Side
            if (side == Side::Left) {
                blas::gemm(layout, trans_A, trans_B, m, n, k, alpha,
                           A_dense.data(), lda, B_dense.data(), ldb, beta, C_reference.data(), ldc);
            } else {
                blas::gemm(layout, trans_B, trans_A, m, n, k, alpha,
                           B_dense.data(), ldb, A_dense.data(), lda, beta, C_reference.data(), ldc);
            }

            // Compare results with relaxed tolerance for sparse operations
            T atol = 100 * std::numeric_limits<T>::epsilon();
            T rtol = 10 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_sparse_op.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );
        }
    }
};

// ============================================================================
// Side::Left with dense B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_dense_colmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_trans_trans) {
    test_sparse_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Left with dense B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_dense_rowmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_trans_trans) {
    test_sparse_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right with dense B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_dense_colmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_trans_trans) {
    test_sparse_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right with dense B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_dense_rowmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_trans_trans) {
    test_sparse_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Left with sparse B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_sparse_colmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_trans_trans) {
    test_sparse_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Left with sparse B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_sparse_rowmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_trans_trans) {
    test_sparse_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Right with sparse B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_sparse_colmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_trans_trans) {
    test_sparse_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Right with sparse B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_sparse_rowmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_trans_trans) {
    test_sparse_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}
