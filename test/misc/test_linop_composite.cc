// Unit tests for CompositeOperator
// Tests composition of linear operators with comprehensive coverage

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
        vector<T> left_dense(rows_left * cols_left);
        RandBLAS::sparse_data::csc::CSCMatrix<T> left_csc(rows_left, cols_left);

        if (left_sparse) {
            auto left_coo = RandLAPACK::gen::gen_sparse_mat<T>(rows_left, cols_left, density_left, state);
            RandBLAS::sparse_data::conversions::coo_to_csc(left_coo, left_csc);
        } else {
            RandBLAS::DenseDist D_left(rows_left, cols_left);
            RandBLAS::fill_dense(D_left, left_dense.data(), state);

            if (layout == Layout::RowMajor) {
                vector<T> temp = left_dense;
                for (int64_t i = 0; i < rows_left; ++i) {
                    for (int64_t j = 0; j < cols_left; ++j) {
                        left_dense[j + i * cols_left] = temp[i + j * rows_left];
                    }
                }
            }
        }

        // Generate right operator matrix
        vector<T> right_dense(rows_right * cols_right);
        RandBLAS::sparse_data::csc::CSCMatrix<T> right_csc(rows_right, cols_right);

        if (right_sparse) {
            auto right_coo = RandLAPACK::gen::gen_sparse_mat<T>(rows_right, cols_right, density_right, state);
            RandBLAS::sparse_data::conversions::coo_to_csc(right_coo, right_csc);
        } else {
            RandBLAS::DenseDist D_right(rows_right, cols_right);
            RandBLAS::fill_dense(D_right, right_dense.data(), state);

            if (layout == Layout::RowMajor) {
                vector<T> temp = right_dense;
                for (int64_t i = 0; i < rows_right; ++i) {
                    for (int64_t j = 0; j < cols_right; ++j) {
                        right_dense[j + i * cols_right] = temp[i + j * rows_right];
                    }
                }
            }
        }

        // Create linear operator objects
        int64_t lda_left = (layout == Layout::ColMajor) ? rows_left : cols_left;
        int64_t lda_right = (layout == Layout::ColMajor) ? rows_right : cols_right;

        // Create output buffers
        vector<T> C_composite(m * n);
        vector<T> C_reference(m * n);

        for (auto& c : C_composite) c = static_cast<T>(rand()) / RAND_MAX;
        C_reference = C_composite;

        // Create operators and composite
        if (left_sparse && right_sparse) {
            // Sparse-Sparse composition
            RandLAPACK::linops::SparseLinOp left_op(rows_left, cols_left, left_csc);
            RandLAPACK::linops::SparseLinOp right_op(rows_right, cols_right, right_csc);
            RandLAPACK::linops::CompositeOperator composite_op(rows_left, cols_right, left_op, right_op);

            test_composite_with_input(composite_op, sparse_B, side, layout, trans_A, trans_B,
                                     m, n, k, alpha, beta, rows_B, cols_B, density_B,
                                     C_composite, C_reference, state);

        } else if (left_sparse && !right_sparse) {
            // Sparse-Dense composition
            RandLAPACK::linops::SparseLinOp left_op(rows_left, cols_left, left_csc);
            RandLAPACK::linops::DenseLinOp right_op(rows_right, cols_right, right_dense.data(), lda_right, layout);
            RandLAPACK::linops::CompositeOperator composite_op(rows_left, cols_right, left_op, right_op);

            test_composite_with_input(composite_op, sparse_B, side, layout, trans_A, trans_B,
                                     m, n, k, alpha, beta, rows_B, cols_B, density_B,
                                     C_composite, C_reference, state);

        } else if (!left_sparse && right_sparse) {
            // Dense-Sparse composition
            RandLAPACK::linops::DenseLinOp left_op(rows_left, cols_left, left_dense.data(), lda_left, layout);
            RandLAPACK::linops::SparseLinOp right_op(rows_right, cols_right, right_csc);
            RandLAPACK::linops::CompositeOperator composite_op(rows_left, cols_right, left_op, right_op);

            test_composite_with_input(composite_op, sparse_B, side, layout, trans_A, trans_B,
                                     m, n, k, alpha, beta, rows_B, cols_B, density_B,
                                     C_composite, C_reference, state);

        } else {
            // Dense-Dense composition
            RandLAPACK::linops::DenseLinOp left_op(rows_left, cols_left, left_dense.data(), lda_left, layout);
            RandLAPACK::linops::DenseLinOp right_op(rows_right, cols_right, right_dense.data(), lda_right, layout);
            RandLAPACK::linops::CompositeOperator composite_op(rows_left, cols_right, left_op, right_op);

            test_composite_with_input(composite_op, sparse_B, side, layout, trans_A, trans_B,
                                     m, n, k, alpha, beta, rows_B, cols_B, density_B,
                                     C_composite, C_reference, state);
        }

        // Compute reference by materializing composite operator
        // First materialize left and right operators as dense matrices
        vector<T> left_mat_dense(rows_left * cols_left);
        vector<T> right_mat_dense(rows_right * cols_right);

        if (left_sparse) {
            RandLAPACK::util::sparse_to_dense_summing_duplicates(left_csc, Layout::ColMajor, left_mat_dense.data());
        } else {
            // Convert to ColMajor if needed
            if (layout == Layout::ColMajor) {
                left_mat_dense = left_dense;
            } else {
                for (int64_t i = 0; i < rows_left; ++i) {
                    for (int64_t j = 0; j < cols_left; ++j) {
                        left_mat_dense[i + j * rows_left] = left_dense[j + i * cols_left];
                    }
                }
            }
        }

        if (right_sparse) {
            RandLAPACK::util::sparse_to_dense_summing_duplicates(right_csc, Layout::ColMajor, right_mat_dense.data());
        } else {
            // Convert to ColMajor if needed
            if (layout == Layout::ColMajor) {
                right_mat_dense = right_dense;
            } else {
                for (int64_t i = 0; i < rows_right; ++i) {
                    for (int64_t j = 0; j < cols_right; ++j) {
                        right_mat_dense[i + j * rows_right] = right_dense[j + i * cols_right];
                    }
                }
            }
        }

        // Compute composite = left * right using BLAS gemm (always in ColMajor first)
        vector<T> composite_colmajor(rows_left * cols_right);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   rows_left, cols_right, cols_left,
                   1.0, left_mat_dense.data(), rows_left,
                   right_mat_dense.data(), rows_right,
                   0.0, composite_colmajor.data(), rows_left);

        // Convert composite to the required layout
        vector<T> composite_dense(rows_left * cols_right);
        if (layout == Layout::ColMajor) {
            composite_dense = composite_colmajor;
        } else {  // RowMajor
            for (int64_t i = 0; i < rows_left; ++i) {
                for (int64_t j = 0; j < cols_right; ++j) {
                    composite_dense[j + i * cols_right] = composite_colmajor[i + j * rows_left];
                }
            }
        }

        // Now apply the materialized composite operator to B to get reference
        // Need to handle the same dense/sparse B cases
        if (sparse_B) {
            // Generate the same sparse B
            RNGState state_ref(0);
            auto B_coo = RandLAPACK::gen::gen_sparse_mat<T>(rows_B, cols_B, density_B, state_ref);
            RandBLAS::sparse_data::csc::CSCMatrix<T> B_csc(rows_B, cols_B);
            RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

            // Convert sparse B to dense for reference computation
            vector<T> B_dense_ref(rows_B * cols_B);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, layout, B_dense_ref.data());

            // Apply composite operator: C := alpha * op(composite) * op(B) + beta * C
            // Leading dimensions based on layout
            int64_t lda_ref = (layout == Layout::ColMajor) ? rows_left : cols_right;
            int64_t ldb_ref = (layout == Layout::ColMajor) ? rows_B : cols_B;
            int64_t ldc_ref = (layout == Layout::ColMajor) ? m : n;

            if (side == Side::Left) {
                blas::gemm(layout, trans_A, trans_B,
                           m, n, k,
                           alpha, composite_dense.data(), lda_ref,
                           B_dense_ref.data(), ldb_ref,
                           beta, C_reference.data(), ldc_ref);
            } else {
                blas::gemm(layout, trans_B, trans_A,
                           m, n, k,
                           alpha, B_dense_ref.data(), ldb_ref,
                           composite_dense.data(), lda_ref,
                           beta, C_reference.data(), ldc_ref);
            }
        } else {
            // Generate the same dense B
            RNGState state_ref(0);
            vector<T> B_dense_ref(rows_B * cols_B);
            RandBLAS::DenseDist D_B(rows_B, cols_B);
            RandBLAS::fill_dense(D_B, B_dense_ref.data(), state_ref);

            // B_dense_ref is in ColMajor, convert if layout is RowMajor
            if (layout == Layout::RowMajor) {
                vector<T> temp = B_dense_ref;
                for (int64_t i = 0; i < rows_B; ++i) {
                    for (int64_t j = 0; j < cols_B; ++j) {
                        B_dense_ref[j + i * cols_B] = temp[i + j * rows_B];
                    }
                }
            }

            // Compute leading dimensions based on layout
            int64_t lda_ref = (layout == Layout::ColMajor) ? rows_left : cols_right;
            int64_t ldb_ref = (layout == Layout::ColMajor) ? rows_B : cols_B;
            int64_t ldc_ref = (layout == Layout::ColMajor) ? m : n;

            // Apply composite operator using the correct layout
            if (side == Side::Left) {
                blas::gemm(layout, trans_A, trans_B,
                           m, n, k,
                           alpha, composite_dense.data(), lda_ref,
                           B_dense_ref.data(), ldb_ref,
                           beta, C_reference.data(), ldc_ref);
            } else {
                blas::gemm(layout, trans_B, trans_A,
                           m, n, k,
                           alpha, B_dense_ref.data(), ldb_ref,
                           composite_dense.data(), lda_ref,
                           beta, C_reference.data(), ldc_ref);
            }
        }

        // Compare results
        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        int64_t ldc_cmp = (layout == Layout::ColMajor) ? m : n;
        test::comparison::matrices_approx_equal(
            layout, Op::NoTrans, m, n, C_composite.data(), ldc_cmp,
            C_reference.data(), ldc_cmp, __PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        );
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
        vector<T>& C_composite,
        vector<T>& C_reference,
        RNGState<r123::Philox4x32_R<10>>& state
    ) {
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        if (sparse_B) {
            // Generate sparse input matrix B
            auto B_coo = RandLAPACK::gen::gen_sparse_mat<T>(rows_B, cols_B, density_B, state);
            RandBLAS::sparse_data::csc::CSCMatrix<T> B_csc(rows_B, cols_B);
            RandBLAS::sparse_data::conversions::coo_to_csc(B_coo, B_csc);

            // Apply composite operator
            composite_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_composite.data(), ldc);

        } else {
            // Generate dense input matrix B
            vector<T> B_dense(rows_B * cols_B);
            RandBLAS::DenseDist D_B(rows_B, cols_B);
            RandBLAS::fill_dense(D_B, B_dense.data(), state);

            if (layout == Layout::RowMajor) {
                vector<T> temp = B_dense;
                for (int64_t i = 0; i < rows_B; ++i) {
                    for (int64_t j = 0; j < cols_B; ++j) {
                        B_dense[j + i * cols_B] = temp[i + j * rows_B];
                    }
                }
            }
            int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

            // Apply composite operator
            composite_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense.data(), ldb, beta, C_composite.data(), ldc);
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

// TODO: Add sparse B input tests for all combinations above
