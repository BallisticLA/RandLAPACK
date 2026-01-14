// Unit tests for CompositeOperator
// Tests composition of linear operators with comprehensive coverage

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "../RandLAPACK/RandBLAS/test/comparison.hh"
#include "../../RandLAPACK/misc/rl_util_test_linop.hh"  // Test utilities, not part of public API

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;
using namespace RandLAPACK::util::test;

// Static assertion to verify CompositeOperator satisfies LinearOperator concept
using DenseOp = RandLAPACK::linops::DenseLinOp<double>;
using SparseOp = RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<double>>;
static_assert(RandLAPACK::linops::LinearOperator<RandLAPACK::linops::CompositeOperator<DenseOp, DenseOp>, double>,
              "CompositeOperator<DenseOp, DenseOp> must satisfy LinearOperator concept");
static_assert(RandLAPACK::linops::LinearOperator<RandLAPACK::linops::CompositeOperator<SparseOp, SparseOp>, double>,
              "CompositeOperator<SparseOp, SparseOp> must satisfy LinearOperator concept");
static_assert(RandLAPACK::linops::LinearOperator<RandLAPACK::linops::CompositeOperator<DenseOp, SparseOp>, double>,
              "CompositeOperator<DenseOp, SparseOp> must satisfy LinearOperator concept");
static_assert(RandLAPACK::linops::LinearOperator<RandLAPACK::linops::CompositeOperator<SparseOp, DenseOp>, double>,
              "CompositeOperator<SparseOp, DenseOp> must satisfy LinearOperator concept");

class TestCompositeLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Unified test function for CompositeOperator
    // Handles all combinations of left/right operators, input matrix types, and layouts
    template <typename T>
    void test_composite_linop(
        bool left_sparse,   // Left operator: true=sparse, false=dense
        bool right_sparse,  // Right operator: true=sparse, false=dense
        bool sparse_B,      // Input matrix B: true=sparse, false=dense
        Side side,          // Multiplication side
        Layout layout,      // Memory layout
        Op trans_A,         // Transpose for composite operator
        Op trans_B,         // Transpose for input matrix B
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t intermediate_dim,  // Dimension where operators join
        T density_left = 0.3,
        T density_right = 0.3,
        T density_B = 0.3
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Dimension calculation
        int64_t rows_left, cols_left, rows_right, cols_right, rows_B, cols_B;

        if (side == Side::Left) {
            // Side::Left: C := alpha * op(left * right) * op(B) + beta * C
            // Composite operator has dimensions after applying trans_A
            auto [rc, cc] = RandBLAS::dims_before_op(m, k, trans_A);

            // Left operator: m × intermediate_dim (after trans)
            // Right operator: intermediate_dim × k (after trans)
            rows_left = m;
            cols_left = intermediate_dim;
            rows_right = intermediate_dim;
            cols_right = k;

            auto [rb, cb] = RandBLAS::dims_before_op(k, n, trans_B);
            rows_B = rb;
            cols_B = cb;
        } else {
            // Side::Right: C := alpha * op(B) * op(left * right) + beta * C
            auto [rc, cc] = RandBLAS::dims_before_op(k, n, trans_A);

            // Left operator: k × intermediate_dim (after trans)
            // Right operator: intermediate_dim × n (after trans)
            rows_left = k;
            cols_left = intermediate_dim;
            rows_right = intermediate_dim;
            cols_right = n;

            auto [rb, cb] = RandBLAS::dims_before_op(m, k, trans_B);
            rows_B = rb;
            cols_B = cb;
        }

        // Generate left operator matrix
        T* left_dense = nullptr;
        if (!left_sparse) {
            left_dense = generate_dense_matrix<T>(rows_left, cols_left, layout, state);
        }

        // Generate right operator matrix
        T* right_dense = nullptr;
        if (!right_sparse) {
            right_dense = generate_dense_matrix<T>(rows_right, cols_right, layout, state);
        }

        // Create linear operator objects
        int64_t lda_left = (layout == Layout::ColMajor) ? rows_left : cols_left;
        int64_t lda_right = (layout == Layout::ColMajor) ? rows_right : cols_right;

        // Create and initialize output buffers with random data for beta testing
        T* C_composite = generate_dense_matrix<T>(m, n, Layout::ColMajor, state);
        T* C_reference = new T[m * n];
        std::copy(C_composite, C_composite + m * n, C_reference);

        // Create operators and composite
        if (left_sparse && right_sparse) {
            // Sparse-Sparse composition
            auto left_csc = generate_sparse_matrix<T>(rows_left, cols_left, density_left, state);
            auto right_csc = generate_sparse_matrix<T>(rows_right, cols_right, density_right, state);
            RandLAPACK::linops::SparseLinOp left_op(rows_left, cols_left, left_csc);
            RandLAPACK::linops::SparseLinOp right_op(rows_right, cols_right, right_csc);
            RandLAPACK::linops::CompositeOperator composite_op(rows_left, cols_right, left_op, right_op);

            test_composite_with_input(composite_op, sparse_B, side, layout, trans_A, trans_B,
                                     m, n, k, alpha, beta, rows_B, cols_B, density_B,
                                     C_composite, C_reference, state);

        } else if (left_sparse && !right_sparse) {
            // Sparse-Dense composition
            auto left_csc = generate_sparse_matrix<T>(rows_left, cols_left, density_left, state);
            RandLAPACK::linops::SparseLinOp left_op(rows_left, cols_left, left_csc);
            RandLAPACK::linops::DenseLinOp right_op(rows_right, cols_right, right_dense, lda_right, layout);
            RandLAPACK::linops::CompositeOperator composite_op(rows_left, cols_right, left_op, right_op);

            test_composite_with_input(composite_op, sparse_B, side, layout, trans_A, trans_B,
                                     m, n, k, alpha, beta, rows_B, cols_B, density_B,
                                     C_composite, C_reference, state);

        } else if (!left_sparse && right_sparse) {
            // Dense-Sparse composition
            auto right_csc = generate_sparse_matrix<T>(rows_right, cols_right, density_right, state);
            RandLAPACK::linops::DenseLinOp left_op(rows_left, cols_left, left_dense, lda_left, layout);
            RandLAPACK::linops::SparseLinOp right_op(rows_right, cols_right, right_csc);
            RandLAPACK::linops::CompositeOperator composite_op(rows_left, cols_right, left_op, right_op);

            test_composite_with_input(composite_op, sparse_B, side, layout, trans_A, trans_B,
                                     m, n, k, alpha, beta, rows_B, cols_B, density_B,
                                     C_composite, C_reference, state);

        } else {
            // Dense-Dense composition
            RandLAPACK::linops::DenseLinOp left_op(rows_left, cols_left, left_dense, lda_left, layout);
            RandLAPACK::linops::DenseLinOp right_op(rows_right, cols_right, right_dense, lda_right, layout);
            RandLAPACK::linops::CompositeOperator composite_op(rows_left, cols_right, left_op, right_op);

            test_composite_with_input(composite_op, sparse_B, side, layout, trans_A, trans_B,
                                     m, n, k, alpha, beta, rows_B, cols_B, density_B,
                                     C_composite, C_reference, state);
        }

        // Compute reference by materializing composite operator
        // First materialize left and right operators as dense matrices
        // Need to regenerate with same seed for consistent results
        RNGState state_ref(0);
        T* left_mat_dense = new T[rows_left * cols_left];
        T* right_mat_dense = new T[rows_right * cols_right];

        if (left_sparse) {
            auto left_csc_ref = generate_sparse_matrix<T>(rows_left, cols_left, density_left, state_ref);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(left_csc_ref, Layout::ColMajor, left_mat_dense);
        } else {
            // Convert to ColMajor if needed
            if (layout == Layout::ColMajor) {
                std::copy(left_dense, left_dense + rows_left * cols_left, left_mat_dense);
            } else {
                for (int64_t i = 0; i < rows_left; ++i) {
                    for (int64_t j = 0; j < cols_left; ++j) {
                        left_mat_dense[i + j * rows_left] = left_dense[j + i * cols_left];
                    }
                }
            }
        }

        if (right_sparse) {
            auto right_csc_ref = generate_sparse_matrix<T>(rows_right, cols_right, density_right, state_ref);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(right_csc_ref, Layout::ColMajor, right_mat_dense);
        } else {
            // Convert to ColMajor if needed
            if (layout == Layout::ColMajor) {
                std::copy(right_dense, right_dense + rows_right * cols_right, right_mat_dense);
            } else {
                for (int64_t i = 0; i < rows_right; ++i) {
                    for (int64_t j = 0; j < cols_right; ++j) {
                        right_mat_dense[i + j * rows_right] = right_dense[j + i * cols_right];
                    }
                }
            }
        }

        // Compute composite = left * right using BLAS gemm (always in ColMajor first)
        T* composite_colmajor = new T[rows_left * cols_right];
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   rows_left, cols_right, cols_left,
                   1.0, left_mat_dense, rows_left,
                   right_mat_dense, rows_right,
                   0.0, composite_colmajor, rows_left);

        // Convert composite to the required layout
        T* composite_dense = new T[rows_left * cols_right];
        if (layout == Layout::ColMajor) {
            std::copy(composite_colmajor, composite_colmajor + rows_left * cols_right, composite_dense);
        } else {  // RowMajor
            for (int64_t i = 0; i < rows_left; ++i) {
                for (int64_t j = 0; j < cols_right; ++j) {
                    composite_dense[j + i * cols_right] = composite_colmajor[i + j * rows_left];
                }
            }
        }

        // Now apply the materialized composite operator to B to get reference
        // Need to handle the same dense/sparse B cases
        // Generate the same B matrix for reference computation (reusing state_ref)
        T* B_dense_ref = nullptr;

        if (sparse_B) {
            // Generate sparse B and densify
            auto B_csc = generate_sparse_matrix<T>(rows_B, cols_B, density_B, state_ref);
            B_dense_ref = new T[rows_B * cols_B];
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, layout, B_dense_ref);
        } else {
            // Generate dense B using utility function
            B_dense_ref = generate_dense_matrix<T>(rows_B, cols_B, layout, state_ref);
        }

        // Compute reference using utility function
        int64_t lda_ref = (layout == Layout::ColMajor) ? rows_left : cols_right;
        int64_t ldb_ref = (layout == Layout::ColMajor) ? rows_B : cols_B;
        int64_t ldc_ref = (layout == Layout::ColMajor) ? m : n;

        compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, alpha,
                               composite_dense, lda_ref, B_dense_ref, ldb_ref,
                               beta, C_reference, ldc_ref);

        // Compare results
        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        int64_t ldc_cmp = (layout == Layout::ColMajor) ? m : n;
        test::comparison::matrices_approx_equal(
            layout, Op::NoTrans, m, n, C_composite, ldc_cmp,
            C_reference, ldc_cmp, __PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        );

        // Clean up
        delete[] left_mat_dense;
        delete[] right_mat_dense;
        delete[] composite_colmajor;
        delete[] composite_dense;
        delete[] B_dense_ref;
        delete[] C_composite;
        delete[] C_reference;
        if (left_dense) delete[] left_dense;
        if (right_dense) delete[] right_dense;
    }

    // Helper to test composite operator with different input types
    template <typename CompositeOp, typename T>
    void test_composite_with_input(
        CompositeOp& composite_op,
        bool sparse_B,
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        T beta,
        int64_t rows_B,
        int64_t cols_B,
        T density_B,
        T* C_composite,
        T* C_reference,
        RNGState<r123::Philox4x32_R<10>>& state
    ) {
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        if (sparse_B) {
            // Generate sparse input matrix B using utility function
            auto B_csc = generate_sparse_matrix<T>(rows_B, cols_B, density_B, state);

            // Apply composite operator
            composite_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_composite, ldc);

        } else {
            // Generate dense input matrix B using utility function
            T* B_dense = generate_dense_matrix<T>(rows_B, cols_B, layout, state);
            int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

            // Apply composite operator
            composite_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense, ldb, beta, C_composite, ldc);

            delete[] B_dense;
        }
    }
};

// ============================================================================
// Dense-Dense composition tests
// ============================================================================

TEST_F(TestCompositeLinOp, dense_dense_left_dense_colmajor) {
    test_composite_linop<double>(false, false, false, Side::Left, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_dense_left_dense_rowmajor) {
    test_composite_linop<double>(false, false, false, Side::Left, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_dense_right_dense_colmajor) {
    test_composite_linop<double>(false, false, false, Side::Right, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_dense_right_dense_rowmajor) {
    test_composite_linop<double>(false, false, false, Side::Right, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

// ============================================================================
// Sparse-Sparse composition tests
// ============================================================================

TEST_F(TestCompositeLinOp, sparse_sparse_left_dense_colmajor) {
    test_composite_linop<double>(true, true, false, Side::Left, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_sparse_left_dense_rowmajor) {
    test_composite_linop<double>(true, true, false, Side::Left, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_sparse_right_dense_colmajor) {
    test_composite_linop<double>(true, true, false, Side::Right, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_sparse_right_dense_rowmajor) {
    test_composite_linop<double>(true, true, false, Side::Right, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

// ============================================================================
// Dense-Sparse composition tests
// ============================================================================

TEST_F(TestCompositeLinOp, dense_sparse_left_dense_colmajor) {
    test_composite_linop<double>(false, true, false, Side::Left, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_sparse_left_dense_rowmajor) {
    test_composite_linop<double>(false, true, false, Side::Left, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_sparse_right_dense_colmajor) {
    test_composite_linop<double>(false, true, false, Side::Right, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_sparse_right_dense_rowmajor) {
    test_composite_linop<double>(false, true, false, Side::Right, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

// ============================================================================
// Sparse-Dense composition tests
// ============================================================================

TEST_F(TestCompositeLinOp, sparse_dense_left_dense_colmajor) {
    test_composite_linop<double>(true, false, false, Side::Left, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_dense_left_dense_rowmajor) {
    test_composite_linop<double>(true, false, false, Side::Left, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_dense_right_dense_colmajor) {
    test_composite_linop<double>(true, false, false, Side::Right, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_dense_right_dense_rowmajor) {
    test_composite_linop<double>(true, false, false, Side::Right, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

// ============================================================================
// Dense-Dense composition with SPARSE B tests
// ============================================================================

TEST_F(TestCompositeLinOp, dense_dense_left_sparse_colmajor) {
    test_composite_linop<double>(false, false, true, Side::Left, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_dense_left_sparse_rowmajor) {
    test_composite_linop<double>(false, false, true, Side::Left, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_dense_right_sparse_colmajor) {
    test_composite_linop<double>(false, false, true, Side::Right, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_dense_right_sparse_rowmajor) {
    test_composite_linop<double>(false, false, true, Side::Right, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

// ============================================================================
// Sparse-Sparse composition with SPARSE B tests
// ============================================================================

TEST_F(TestCompositeLinOp, sparse_sparse_left_sparse_colmajor) {
    test_composite_linop<double>(true, true, true, Side::Left, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_sparse_left_sparse_rowmajor) {
    test_composite_linop<double>(true, true, true, Side::Left, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_sparse_right_sparse_colmajor) {
    test_composite_linop<double>(true, true, true, Side::Right, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_sparse_right_sparse_rowmajor) {
    test_composite_linop<double>(true, true, true, Side::Right, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

// ============================================================================
// Dense-Sparse composition with SPARSE B tests
// ============================================================================

TEST_F(TestCompositeLinOp, dense_sparse_left_sparse_colmajor) {
    test_composite_linop<double>(false, true, true, Side::Left, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_sparse_left_sparse_rowmajor) {
    test_composite_linop<double>(false, true, true, Side::Left, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_sparse_right_sparse_colmajor) {
    test_composite_linop<double>(false, true, true, Side::Right, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, dense_sparse_right_sparse_rowmajor) {
    test_composite_linop<double>(false, true, true, Side::Right, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

// ============================================================================
// Sparse-Dense composition with SPARSE B tests
// ============================================================================

TEST_F(TestCompositeLinOp, sparse_dense_left_sparse_colmajor) {
    test_composite_linop<double>(true, false, true, Side::Left, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_dense_left_sparse_rowmajor) {
    test_composite_linop<double>(true, false, true, Side::Left, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_dense_right_sparse_colmajor) {
    test_composite_linop<double>(true, false, true, Side::Right, Layout::ColMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}

TEST_F(TestCompositeLinOp, sparse_dense_right_sparse_rowmajor) {
    test_composite_linop<double>(true, false, true, Side::Right, Layout::RowMajor,
                                 Op::NoTrans, Op::NoTrans, 10, 8, 12, 6);
}