// Unit tests for block view functionality:
//   - DenseLinOp::row_block, col_block, submatrix, materialize_cols
//   - CSRRowBlockView / csr_row_block / csr_split_row_blocks
//   - CSCColBlockView / csc_col_block / csc_split_col_blocks
//   - CSRColBlock / csr_col_block / csr_split_col_blocks (cross-direction)
//   - CSCRowBlock / csc_row_block / csc_split_row_blocks (cross-direction)
//   - SparseLinOp::row_block, col_block, submatrix (dispatch to free functions), materialize_cols
//   - CompositeOperator::row_block, col_block, submatrix (delegation to operands)

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include <RandBLAS/testing/comparison.hh>
#include "../../RandLAPACK/testing/rl_test_utils.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;
using namespace RandLAPACK::testing;

// ============================================================================
// DenseLinOp Block View Tests
// ============================================================================

class TestDenseBlockViews : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Test DenseLinOp::row_block by comparing A_view * B against
    /// the corresponding rows of A * B.
    template <typename T>
    void test_row_block(Layout layout, int64_t m, int64_t n, int64_t k,
                        int64_t row_start, int64_t row_count) {
        RNGState state(42);
        T alpha = 1.0;
        T beta = 0.0;

        // Generate A (m x n) and B (n x k)
        vector<T> A = generate_dense_matrix<T>(m, n, layout, state);
        vector<T> B = generate_dense_matrix<T>(n, k, layout, state);

        int64_t lda = (layout == Layout::ColMajor) ? m : n;
        int64_t ldb = (layout == Layout::ColMajor) ? n : k;

        // Full product: C_full = A * B  (m x k)
        int64_t ldc_full = (layout == Layout::ColMajor) ? m : k;
        vector<T> C_full(m * k, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m, k, n,
                   alpha, A.data(), lda, B.data(), ldb,
                   beta, C_full.data(), ldc_full);

        // Create DenseLinOp and extract row_block view
        RandLAPACK::linops::DenseLinOp<T> A_op(m, n, A.data(), lda, layout);
        auto A_view = A_op.row_block(row_start, row_count);

        // View product: C_block = A_view * B  (row_count x k)
        int64_t ldc_block = (layout == Layout::ColMajor) ? row_count : k;
        vector<T> C_block(row_count * k, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, row_count, k, n,
                   alpha, A_view.A_buff, A_view.lda, B.data(), ldb,
                   beta, C_block.data(), ldc_block);

        // Extract expected rows from C_full
        vector<T> C_expected(row_count * k, 0.0);
        for (int64_t i = 0; i < row_count; ++i) {
            for (int64_t j = 0; j < k; ++j) {
                int64_t idx_full = (layout == Layout::ColMajor)
                    ? (row_start + i) + j * ldc_full
                    : (row_start + i) * ldc_full + j;
                int64_t idx_block = (layout == Layout::ColMajor)
                    ? i + j * ldc_block
                    : i * ldc_block + j;
                C_expected[idx_block] = C_full[idx_full];
            }
        }

        // The two gemms may block each n-term dot product differently: error ~ n*eps*|C|.
        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, k,
            C_block.data(), row_count,
            C_expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        ), "");
    }

    /// Test DenseLinOp::col_block by comparing B * A_view against
    /// the corresponding columns of B * A.
    template <typename T>
    void test_col_block(Layout layout, int64_t m, int64_t n, int64_t k,
                        int64_t col_start, int64_t col_count) {
        RNGState state(42);
        T alpha = 1.0;
        T beta = 0.0;

        // Generate A (m x n) and B (k x m)
        vector<T> A = generate_dense_matrix<T>(m, n, layout, state);
        vector<T> B = generate_dense_matrix<T>(k, m, layout, state);

        int64_t lda = (layout == Layout::ColMajor) ? m : n;
        int64_t ldb = (layout == Layout::ColMajor) ? k : m;

        // Full product: C_full = B * A  (k x n)
        int64_t ldc_full = (layout == Layout::ColMajor) ? k : n;
        vector<T> C_full(k * n, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, k, n, m,
                   alpha, B.data(), ldb, A.data(), lda,
                   beta, C_full.data(), ldc_full);

        // Create DenseLinOp and extract col_block view
        RandLAPACK::linops::DenseLinOp<T> A_op(m, n, A.data(), lda, layout);
        auto A_view = A_op.col_block(col_start, col_count);

        // View product: C_block = B * A_view  (k x col_count)
        int64_t ldc_block = (layout == Layout::ColMajor) ? k : col_count;
        vector<T> C_block(k * col_count, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, k, col_count, m,
                   alpha, B.data(), ldb, A_view.A_buff, A_view.lda,
                   beta, C_block.data(), ldc_block);

        // Extract expected columns from C_full
        vector<T> C_expected(k * col_count, 0.0);
        for (int64_t i = 0; i < k; ++i) {
            for (int64_t j = 0; j < col_count; ++j) {
                int64_t idx_full = (layout == Layout::ColMajor)
                    ? i + (col_start + j) * ldc_full
                    : i * ldc_full + (col_start + j);
                int64_t idx_block = (layout == Layout::ColMajor)
                    ? i + j * ldc_block
                    : i * ldc_block + j;
                C_expected[idx_block] = C_full[idx_full];
            }
        }

        // The two gemms may block each m-term dot product differently: error ~ m*eps*|C|.
        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, k, col_count,
            C_block.data(), k,
            C_expected.data(), k,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        ), "");
    }

    /// Test DenseLinOp::submatrix by verifying that it extracts the correct
    /// block and can be used in BLAS gemm.
    template <typename T>
    void test_submatrix(Layout layout, int64_t m, int64_t n,
                        int64_t row_start, int64_t col_start,
                        int64_t row_count, int64_t col_count, int64_t k) {
        RNGState state(42);
        T alpha = 1.0;
        T beta = 0.0;

        // Generate A (m x n) and B (col_count x k)
        vector<T> A = generate_dense_matrix<T>(m, n, layout, state);
        vector<T> B = generate_dense_matrix<T>(col_count, k, layout, state);

        int64_t lda = (layout == Layout::ColMajor) ? m : n;
        int64_t ldb = (layout == Layout::ColMajor) ? col_count : k;

        // Create DenseLinOp and extract submatrix view
        RandLAPACK::linops::DenseLinOp<T> A_op(m, n, A.data(), lda, layout);
        auto A_sub = A_op.submatrix(row_start, col_start, row_count, col_count);

        // Product: C_block = A_sub * B  (row_count x k)
        int64_t ldc = (layout == Layout::ColMajor) ? row_count : k;
        vector<T> C_block(row_count * k, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, row_count, k, col_count,
                   alpha, A_sub.A_buff, A_sub.lda, B.data(), ldb,
                   beta, C_block.data(), ldc);

        // Compute expected: extract the submatrix into a contiguous buffer, then gemm
        int64_t ld_sub = (layout == Layout::ColMajor) ? row_count : col_count;
        vector<T> A_sub_dense(row_count * col_count);
        for (int64_t i = 0; i < row_count; ++i) {
            for (int64_t j = 0; j < col_count; ++j) {
                int64_t idx_A = (layout == Layout::ColMajor)
                    ? (row_start + i) + (col_start + j) * lda
                    : (row_start + i) * lda + (col_start + j);
                int64_t idx_sub = (layout == Layout::ColMajor)
                    ? i + j * ld_sub
                    : i * ld_sub + j;
                A_sub_dense[idx_sub] = A[idx_A];
            }
        }
        vector<T> C_expected(row_count * k, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, row_count, k, col_count,
                   alpha, A_sub_dense.data(), ld_sub, B.data(), ldb,
                   beta, C_expected.data(), ldc);

        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, k,
            C_block.data(), row_count,
            C_expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__
        ), "");
    }

    /// Test DenseLinOp::materialize_cols (and materialize, which calls it), with padded lda and
    /// ldb, at several column ranges. On the finite, normal-valued data used here it equals the
    /// product with identity columns, entry for entry. With -0.0, Inf and NaN stored it reproduces every entry as
    /// stored, which a product with identity columns need not. The padding rows of buf and its
    /// columns past col_count stay untouched.
    template <typename T>
    void test_materialize_cols(Layout layout, int64_t m, int64_t n) {
        const bool col = (layout == Layout::ColMajor);
        const int64_t lda = col ? m + 3 : n + 2;
        const int64_t ldb = m + 2;
        const T pad = (T)-42;
        vector<T> A(col ? lda * n : m * lda);
        RandBLAS::fill_dense(DenseDist((int64_t)A.size(), 1), A.data(), RNGState(7));
        auto at = [&](int64_t i, int64_t j) { return col ? i + j * lda : i * lda + j; };
        RandLAPACK::linops::DenseLinOp<T> A_mut(m, n, A.data(), lda, layout);
        const auto& A_op = A_mut;

        {   // Finite data: the copy equals A times the identity.
            vector<T> eye(n * n, (T)0), prod(m * n), buf(ldb * n, pad);
            for (int64_t j = 0; j < n; ++j) eye[j + j * n] = (T)1;
            const int64_t ldp = col ? m : n;
            A_mut(layout, Op::NoTrans, Op::NoTrans, m, n, n, (T)1, eye.data(), n, (T)0, prod.data(), ldp);
            A_op.materialize_cols(0, n, buf.data(), ldb);
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i < m; ++i)
                    ASSERT_EQ(buf[i + j * ldb], prod[col ? i + j * ldp : i * ldp + j]) << i << "," << j;
        }

        A[at(0, 1)]         = -(T)0;
        A[at(2, 3)]         = std::numeric_limits<T>::infinity();
        A[at(m - 1, n - 1)] = std::numeric_limits<T>::quiet_NaN();
        auto same = [](T x, T y) {
            return (std::isnan(x) && std::isnan(y)) || (x == y && std::signbit(x) == std::signbit(y));
        };
        for (auto [c0, cnt] : {std::pair<int64_t, int64_t>{0, n}, {2, 3}, {n - 2, 2}, {4, 0}, {n, 0}}) {
            vector<T> buf(ldb * n, pad);
            A_op.materialize_cols(c0, cnt, buf.data(), ldb);
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i < ldb; ++i) {
                    const T got = buf[i + j * ldb];
                    if (j < cnt && i < m)
                        ASSERT_TRUE(same(got, A[at(i, c0 + j)])) << "cols " << c0 << "+" << cnt << " at " << i << "," << j;
                    else
                        ASSERT_EQ(got, pad) << "cols " << c0 << "+" << cnt << " wrote " << i << "," << j;
                }
        }

        vector<T> full(ldb * n, pad);
        RandLAPACK::materialize(A_mut, m, n, full.data(), ldb);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < ldb; ++i)
                ASSERT_TRUE(i < m ? same(full[i + j * ldb], A[at(i, j)]) : full[i + j * ldb] == pad) << i << "," << j;
    }
};

static_assert(RandLAPACK::linops::HasMaterializeCols<RandLAPACK::linops::DenseLinOp<double>>);
static_assert(RandLAPACK::linops::HasMaterializeCols<const RandLAPACK::linops::DenseLinOp<float>>);

// DenseLinOp::row_block tests
TEST_F(TestDenseBlockViews, row_block_colmajor) {
    test_row_block<double>(Layout::ColMajor, 20, 15, 10, 5, 8);
}

TEST_F(TestDenseBlockViews, row_block_rowmajor) {
    test_row_block<double>(Layout::RowMajor, 20, 15, 10, 5, 8);
}

TEST_F(TestDenseBlockViews, row_block_colmajor_first_rows) {
    test_row_block<double>(Layout::ColMajor, 20, 15, 10, 0, 6);
}

TEST_F(TestDenseBlockViews, row_block_colmajor_last_rows) {
    test_row_block<double>(Layout::ColMajor, 20, 15, 10, 14, 6);
}

// DenseLinOp::col_block tests
TEST_F(TestDenseBlockViews, col_block_colmajor) {
    test_col_block<double>(Layout::ColMajor, 20, 15, 10, 4, 7);
}

TEST_F(TestDenseBlockViews, col_block_rowmajor) {
    test_col_block<double>(Layout::RowMajor, 20, 15, 10, 4, 7);
}

TEST_F(TestDenseBlockViews, col_block_colmajor_first_cols) {
    test_col_block<double>(Layout::ColMajor, 20, 15, 10, 0, 5);
}

TEST_F(TestDenseBlockViews, col_block_colmajor_last_cols) {
    test_col_block<double>(Layout::ColMajor, 20, 15, 10, 10, 5);
}

// DenseLinOp::submatrix tests
TEST_F(TestDenseBlockViews, submatrix_colmajor) {
    test_submatrix<double>(Layout::ColMajor, 20, 18, 3, 4, 8, 7, 5);
}

TEST_F(TestDenseBlockViews, submatrix_rowmajor) {
    test_submatrix<double>(Layout::RowMajor, 20, 18, 3, 4, 8, 7, 5);
}

TEST_F(TestDenseBlockViews, submatrix_colmajor_corner) {
    test_submatrix<double>(Layout::ColMajor, 20, 18, 0, 0, 10, 9, 5);
}

// DenseLinOp::materialize_cols tests
TEST_F(TestDenseBlockViews, materialize_cols_colmajor) {
    test_materialize_cols<double>(Layout::ColMajor, 11, 7);
}

TEST_F(TestDenseBlockViews, materialize_cols_rowmajor) {
    test_materialize_cols<double>(Layout::RowMajor, 11, 7);
}

TEST_F(TestDenseBlockViews, materialize_cols_rowmajor_float) {
    test_materialize_cols<float>(Layout::RowMajor, 11, 7);
}

TEST_F(TestDenseBlockViews, materialize_cols_invalid_arguments) {
    const int64_t m = 6, n = 4;
    vector<double> A(m * n, 1.0), buf(m * n, -42.0);
    const RandLAPACK::linops::DenseLinOp<double> A_col(m, n, A.data(), m, Layout::ColMajor);
    const RandLAPACK::linops::DenseLinOp<double> A_row(m, n, A.data(), n, Layout::RowMajor);
    for (const auto* op : {&A_col, &A_row}) {
        EXPECT_THROW(op->materialize_cols(-1, 2, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op->materialize_cols(0, -1, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op->materialize_cols(n + 1, 0, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op->materialize_cols(n - 1, 2, buf.data(), m), RandLAPACK::Error);
        // ldb is checked against n_rows in both layouts (buf is column-major), m - 1 >= n here.
        EXPECT_THROW(op->materialize_cols(0, 1, buf.data(), m - 1), RandLAPACK::Error);
        // The range check cannot overflow: start + count is never formed.
        const int64_t big = std::numeric_limits<int64_t>::max();
        EXPECT_THROW(op->materialize_cols(big, 1, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op->materialize_cols(1, big, buf.data(), m), RandLAPACK::Error);
    }
    for (double v : buf) ASSERT_EQ(v, -42.0);   // every failed call stopped before the copy
}

TEST_F(TestDenseBlockViews, materialize_cols_zero_rows) {
    // An operator with no rows needs no storage: a valid range reads and writes nothing.
    const int64_t n = 4;
    const RandLAPACK::linops::DenseLinOp<double> E_col(0, n, nullptr, 1, Layout::ColMajor);
    const RandLAPACK::linops::DenseLinOp<double> E_row(0, n, nullptr, n, Layout::RowMajor);
    for (const auto* op : {&E_col, &E_row}) {
        op->materialize_cols(1, 2, nullptr, 0);
        op->materialize_cols(0, n, nullptr, 0);
        EXPECT_THROW(op->materialize_cols(n - 1, 2, nullptr, 0), RandLAPACK::Error);
    }
}


// ============================================================================
// CSR Row-Block View Tests
// ============================================================================

class TestCSRRowBlockViews : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Generate a random sparse matrix in CSR format.
    template <typename T>
    RandBLAS::sparse_data::CSRMatrix<T> generate_csr_matrix(
        int64_t rows, int64_t cols, T density, RNGState<>& state
    ) {
        auto coo = RandLAPACK::gen::gen_sparse_coo<T>(rows, cols, density, state);
        RandBLAS::sparse_data::CSRMatrix<T> csr(rows, cols);
        RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
        return csr;
    }

    /// Test that csr_row_block produces a view with correct structure:
    /// rebased rowptr starting from 0, correct nnz, correct dimensions.
    template <typename T>
    void test_csr_row_block_structure(int64_t m, int64_t n, T density,
                                      int64_t row_start, int64_t row_count) {
        RNGState state(42);
        auto A = generate_csr_matrix<T>(m, n, density, state);
        auto view = RandLAPACK::linops::csr_row_block(A, row_start, row_count);

        // Check dimensions
        ASSERT_EQ(view.n_rows, row_count);
        ASSERT_EQ(view.n_cols, n);

        // Check rowptr is rebased: starts from 0
        ASSERT_EQ(view.rowptr[0], 0);
        ASSERT_EQ(view.rowptr[row_count], view.nnz);

        // Check nnz matches parent
        int64_t expected_nnz = A.rowptr[row_start + row_count] - A.rowptr[row_start];
        ASSERT_EQ(view.nnz, expected_nnz);

        // Check rowptr is monotonically non-decreasing
        for (int64_t i = 0; i < row_count; ++i) {
            ASSERT_LE(view.rowptr[i], view.rowptr[i + 1]);
        }

        // Check rowptr differences match parent
        for (int64_t i = 0; i < row_count; ++i) {
            int64_t parent_nnz_row = A.rowptr[row_start + i + 1] - A.rowptr[row_start + i];
            int64_t view_nnz_row = view.rowptr[i + 1] - view.rowptr[i];
            ASSERT_EQ(view_nnz_row, parent_nnz_row);
        }
    }

    /// Test that SpMM on a CSR row-block matches the corresponding rows
    /// of the full SpMM result.
    template <typename T>
    void test_csr_row_block_spmm(int64_t m, int64_t n, int64_t k, T density,
                                  int64_t row_start, int64_t row_count) {
        RNGState state(42);
        auto A = generate_csr_matrix<T>(m, n, density, state);

        // Dense B (n x k)
        vector<T> B = generate_dense_matrix<T>(n, k, Layout::ColMajor, state);

        // Full SpMM: C_full = A * B  (m x k, ColMajor)
        // Densify A, then gemm
        vector<T> A_dense(m * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, A_dense.data());
        vector<T> C_full(m * k, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n,
                   1.0, A_dense.data(), m, B.data(), n,
                   0.0, C_full.data(), m);

        // Row-block SpMM: C_block = A_view * B  (row_count x k)
        auto view = RandLAPACK::linops::csr_row_block(A, row_start, row_count);
        auto A_block = view.as_csr();

        // Densify block and gemm
        vector<T> A_block_dense(row_count * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A_block, Layout::ColMajor, A_block_dense.data());
        vector<T> C_block(row_count * k, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, row_count, k, n,
                   1.0, A_block_dense.data(), row_count, B.data(), n,
                   0.0, C_block.data(), row_count);

        // Extract expected rows from C_full (ColMajor: rows are contiguous within each column)
        vector<T> C_expected(row_count * k, 0.0);
        for (int64_t j = 0; j < k; ++j) {
            for (int64_t i = 0; i < row_count; ++i) {
                C_expected[i + j * row_count] = C_full[(row_start + i) + j * m];
            }
        }

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, k,
            C_block.data(), row_count,
            C_expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        ), "");
    }

    /// Test csr_split_row_blocks: split and verify all blocks are correct.
    template <typename T>
    void test_csr_split(int64_t m, int64_t n, int64_t k, T density, int64_t num_blocks) {
        RNGState state(42);
        auto A = generate_csr_matrix<T>(m, n, density, state);

        // Dense B (n x k)
        vector<T> B = generate_dense_matrix<T>(n, k, Layout::ColMajor, state);

        // Full SpMM
        vector<T> A_dense(m * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, A_dense.data());
        vector<T> C_full(m * k, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n,
                   1.0, A_dense.data(), m, B.data(), n,
                   0.0, C_full.data(), m);

        auto blocks = RandLAPACK::linops::csr_split_row_blocks(A, num_blocks);
        ASSERT_EQ((int64_t) blocks.size(), num_blocks);

        int64_t block_size = m / num_blocks;

        // Total nnz should sum to A.nnz
        int64_t total_nnz = 0;
        for (auto& blk : blocks)
            total_nnz += blk.nnz;
        ASSERT_EQ(total_nnz, A.nnz);

        // Each block's SpMM should match corresponding rows
        for (int64_t b = 0; b < num_blocks; ++b) {
            auto A_block = blocks[b].as_csr();
            vector<T> A_block_dense(block_size * n, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(A_block, Layout::ColMajor, A_block_dense.data());
            vector<T> C_block(block_size * k, 0.0);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, block_size, k, n,
                       1.0, A_block_dense.data(), block_size, B.data(), n,
                       0.0, C_block.data(), block_size);

            int64_t row_start = b * block_size;
            vector<T> C_expected(block_size * k, 0.0);
            for (int64_t j = 0; j < k; ++j) {
                for (int64_t i = 0; i < block_size; ++i) {
                    C_expected[i + j * block_size] = C_full[(row_start + i) + j * m];
                }
            }

            T atol = 100 * std::numeric_limits<T>::epsilon();
            T rtol = 10 * std::numeric_limits<T>::epsilon();
            EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, block_size, k,
                C_block.data(), block_size,
                C_expected.data(), block_size,
                __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            ), "");
        }
    }
};

// CSR row-block structural tests
TEST_F(TestCSRRowBlockViews, structure_middle_rows) {
    test_csr_row_block_structure<double>(20, 15, 0.3, 5, 8);
}

TEST_F(TestCSRRowBlockViews, structure_first_rows) {
    test_csr_row_block_structure<double>(20, 15, 0.3, 0, 6);
}

TEST_F(TestCSRRowBlockViews, structure_last_rows) {
    test_csr_row_block_structure<double>(20, 15, 0.3, 14, 6);
}

TEST_F(TestCSRRowBlockViews, structure_single_row) {
    test_csr_row_block_structure<double>(20, 15, 0.3, 10, 1);
}

TEST_F(TestCSRRowBlockViews, structure_all_rows) {
    test_csr_row_block_structure<double>(20, 15, 0.3, 0, 20);
}

// CSR row-block SpMM tests
TEST_F(TestCSRRowBlockViews, spmm_middle_rows) {
    test_csr_row_block_spmm<double>(20, 15, 10, 0.3, 5, 8);
}

TEST_F(TestCSRRowBlockViews, spmm_first_rows) {
    test_csr_row_block_spmm<double>(20, 15, 10, 0.3, 0, 6);
}

TEST_F(TestCSRRowBlockViews, spmm_last_rows) {
    test_csr_row_block_spmm<double>(20, 15, 10, 0.3, 14, 6);
}

// CSR split tests
TEST_F(TestCSRRowBlockViews, split_4_blocks) {
    test_csr_split<double>(20, 15, 10, 0.3, 4);
}

TEST_F(TestCSRRowBlockViews, split_2_blocks) {
    test_csr_split<double>(20, 15, 10, 0.3, 2);
}

TEST_F(TestCSRRowBlockViews, split_5_blocks) {
    test_csr_split<double>(20, 15, 10, 0.3, 5);
}


// ============================================================================
// CSC Column-Block View Tests
// ============================================================================

class TestCSCColBlockViews : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Test that csc_col_block produces a view with correct structure.
    template <typename T>
    void test_csc_col_block_structure(int64_t m, int64_t n, T density,
                                      int64_t col_start, int64_t col_count) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();

        auto view = RandLAPACK::linops::csc_col_block(A, col_start, col_count);

        // Check dimensions
        ASSERT_EQ(view.n_rows, m);
        ASSERT_EQ(view.n_cols, col_count);

        // Check colptr is rebased: starts from 0
        ASSERT_EQ(view.colptr[0], 0);
        ASSERT_EQ(view.colptr[col_count], view.nnz);

        // Check nnz matches parent
        int64_t expected_nnz = A.colptr[col_start + col_count] - A.colptr[col_start];
        ASSERT_EQ(view.nnz, expected_nnz);

        // Check colptr is monotonically non-decreasing
        for (int64_t j = 0; j < col_count; ++j) {
            ASSERT_LE(view.colptr[j], view.colptr[j + 1]);
        }

        // Check colptr differences match parent
        for (int64_t j = 0; j < col_count; ++j) {
            int64_t parent_nnz_col = A.colptr[col_start + j + 1] - A.colptr[col_start + j];
            int64_t view_nnz_col = view.colptr[j + 1] - view.colptr[j];
            ASSERT_EQ(view_nnz_col, parent_nnz_col);
        }
    }

    /// Test that SpMM on a CSC col-block matches the corresponding columns
    /// of the full SpMM result. We compute B * A and B * A_view.
    template <typename T>
    void test_csc_col_block_spmm(int64_t m, int64_t n, int64_t k, T density,
                                  int64_t col_start, int64_t col_count) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();

        // Dense B (k x m)
        vector<T> B = generate_dense_matrix<T>(k, m, Layout::ColMajor, state);

        // Full product: C_full = B * A  (k x n, ColMajor)
        vector<T> A_dense(m * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, A_dense.data());
        vector<T> C_full(k * n, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, n, m,
                   1.0, B.data(), k, A_dense.data(), m,
                   0.0, C_full.data(), k);

        // Col-block product: C_block = B * A_view  (k x col_count)
        auto view = RandLAPACK::linops::csc_col_block(A, col_start, col_count);
        auto A_block = view.as_csc();
        vector<T> A_block_dense(m * col_count, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A_block, Layout::ColMajor, A_block_dense.data());
        vector<T> C_block(k * col_count, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, col_count, m,
                   1.0, B.data(), k, A_block_dense.data(), m,
                   0.0, C_block.data(), k);

        // Extract expected columns from C_full
        vector<T> C_expected(k * col_count, 0.0);
        for (int64_t j = 0; j < col_count; ++j) {
            for (int64_t i = 0; i < k; ++i) {
                C_expected[i + j * k] = C_full[i + (col_start + j) * k];
            }
        }

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, k, col_count,
            C_block.data(), k,
            C_expected.data(), k,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        ), "");
    }

    /// Test csc_split_col_blocks: split and verify all blocks are correct.
    template <typename T>
    void test_csc_split(int64_t m, int64_t n, int64_t k, T density, int64_t num_blocks) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();

        // Dense B (k x m)
        vector<T> B = generate_dense_matrix<T>(k, m, Layout::ColMajor, state);

        // Full product
        vector<T> A_dense(m * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, A_dense.data());
        vector<T> C_full(k * n, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, n, m,
                   1.0, B.data(), k, A_dense.data(), m,
                   0.0, C_full.data(), k);

        auto blocks = RandLAPACK::linops::csc_split_col_blocks(A, num_blocks);
        ASSERT_EQ((int64_t) blocks.size(), num_blocks);

        int64_t block_size = n / num_blocks;

        // Total nnz should sum to A.nnz
        int64_t total_nnz = 0;
        for (auto& blk : blocks)
            total_nnz += blk.nnz;
        ASSERT_EQ(total_nnz, A.nnz);

        // Each block's product should match corresponding columns
        for (int64_t b = 0; b < num_blocks; ++b) {
            auto A_block = blocks[b].as_csc();
            vector<T> A_block_dense(m * block_size, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(A_block, Layout::ColMajor, A_block_dense.data());
            vector<T> C_block(k * block_size, 0.0);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, block_size, m,
                       1.0, B.data(), k, A_block_dense.data(), m,
                       0.0, C_block.data(), k);

            int64_t col_start = b * block_size;
            vector<T> C_expected(k * block_size, 0.0);
            for (int64_t j = 0; j < block_size; ++j) {
                for (int64_t i = 0; i < k; ++i) {
                    C_expected[i + j * k] = C_full[i + (col_start + j) * k];
                }
            }

            T atol = 100 * std::numeric_limits<T>::epsilon();
            T rtol = 10 * std::numeric_limits<T>::epsilon();
            EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, k, block_size,
                C_block.data(), k,
                C_expected.data(), k,
                __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            ), "");
        }
    }
};

// CSC col-block structural tests
TEST_F(TestCSCColBlockViews, structure_middle_cols) {
    test_csc_col_block_structure<double>(20, 15, 0.3, 4, 7);
}

TEST_F(TestCSCColBlockViews, structure_first_cols) {
    test_csc_col_block_structure<double>(20, 15, 0.3, 0, 5);
}

TEST_F(TestCSCColBlockViews, structure_last_cols) {
    test_csc_col_block_structure<double>(20, 15, 0.3, 10, 5);
}

TEST_F(TestCSCColBlockViews, structure_single_col) {
    test_csc_col_block_structure<double>(20, 15, 0.3, 7, 1);
}

TEST_F(TestCSCColBlockViews, structure_all_cols) {
    test_csc_col_block_structure<double>(20, 15, 0.3, 0, 15);
}

// CSC col-block SpMM tests
TEST_F(TestCSCColBlockViews, spmm_middle_cols) {
    test_csc_col_block_spmm<double>(20, 15, 10, 0.3, 4, 7);
}

TEST_F(TestCSCColBlockViews, spmm_first_cols) {
    test_csc_col_block_spmm<double>(20, 15, 10, 0.3, 0, 5);
}

TEST_F(TestCSCColBlockViews, spmm_last_cols) {
    test_csc_col_block_spmm<double>(20, 15, 10, 0.3, 10, 5);
}

// CSC split tests
TEST_F(TestCSCColBlockViews, split_3_blocks) {
    test_csc_split<double>(20, 15, 10, 0.3, 3);
}

TEST_F(TestCSCColBlockViews, split_5_blocks) {
    test_csc_split<double>(20, 15, 10, 0.3, 5);
}


// ============================================================================
// CSR Column-Block Tests (cross-direction extraction)
// ============================================================================

class TestCSRColBlocks : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Generate a random sparse matrix in CSR format.
    template <typename T>
    RandBLAS::sparse_data::CSRMatrix<T> generate_csr_matrix(
        int64_t rows, int64_t cols, T density, RNGState<>& state
    ) {
        auto coo = RandLAPACK::gen::gen_sparse_coo<T>(rows, cols, density, state);
        RandBLAS::sparse_data::CSRMatrix<T> csr(rows, cols);
        RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
        return csr;
    }

    /// Test that csr_col_block produces a block with correct structure:
    /// rowptr starting from 0, correct nnz, correct dimensions, rebased colidxs.
    template <typename T>
    void test_csr_col_block_structure(int64_t m, int64_t n, T density,
                                      int64_t col_start, int64_t col_count) {
        RNGState state(42);
        auto A = generate_csr_matrix<T>(m, n, density, state);
        auto block = RandLAPACK::linops::csr_col_block(A, col_start, col_count);

        // Check dimensions
        ASSERT_EQ(block.n_rows, m);
        ASSERT_EQ(block.n_cols, col_count);

        // Check rowptr starts from 0 and ends at nnz
        ASSERT_EQ(block.rowptr[0], 0);
        ASSERT_EQ(block.rowptr[m], block.nnz);

        // Check rowptr is monotonically non-decreasing
        for (int64_t i = 0; i < m; ++i) {
            ASSERT_LE(block.rowptr[i], block.rowptr[i + 1]);
        }

        // Check all column indices are in [0, col_count)
        for (int64_t k = 0; k < block.nnz; ++k) {
            ASSERT_GE(block.colidxs[k], 0);
            ASSERT_LT(block.colidxs[k], col_count);
        }

        // Check nnz: count entries in parent with col in [col_start, col_start+col_count)
        int64_t expected_nnz = 0;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t kk = A.rowptr[i]; kk < A.rowptr[i + 1]; ++kk) {
                if (A.colidxs[kk] >= col_start && A.colidxs[kk] < col_start + col_count)
                    expected_nnz++;
            }
        }
        ASSERT_EQ(block.nnz, expected_nnz);
    }

    /// Test that SpMM on a CSR col-block matches the corresponding columns
    /// of the full SpMM result. We compute A * B_full and A_block * B_cols.
    template <typename T>
    void test_csr_col_block_spmm(int64_t m, int64_t n, int64_t k, T density,
                                  int64_t col_start, int64_t col_count) {
        RNGState state(42);
        auto A = generate_csr_matrix<T>(m, n, density, state);

        // Dense B (n x k)
        vector<T> B = generate_dense_matrix<T>(n, k, Layout::ColMajor, state);

        // Full SpMM: C_full = A * B  (m x k, ColMajor)
        vector<T> A_dense(m * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, A_dense.data());
        vector<T> C_full(m * k, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n,
                   1.0, A_dense.data(), m, B.data(), n,
                   0.0, C_full.data(), m);

        // Extract col-block from A
        auto block = RandLAPACK::linops::csr_col_block(A, col_start, col_count);
        auto A_block = block.as_csr();

        // Densify block (m x col_count) and multiply by corresponding rows of B
        // A_block has columns [0, col_count) corresponding to original columns [col_start, col_start+col_count)
        // So A_block * B[col_start:col_start+col_count, :] should equal the contribution
        // of those columns to C_full.
        // Instead, let's directly compare: densify A_block, compute A_block * B_sub
        vector<T> A_block_dense(m * col_count, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A_block, Layout::ColMajor, A_block_dense.data());

        // B_sub = rows [col_start, col_start+col_count) of B = col_count x k submatrix
        // In ColMajor, row i of B is at B[i + j*n] for column j
        vector<T> B_sub(col_count * k, 0.0);
        for (int64_t j = 0; j < k; ++j) {
            for (int64_t i = 0; i < col_count; ++i) {
                B_sub[i + j * col_count] = B[(col_start + i) + j * n];
            }
        }

        // C_block = A_block * B_sub  (m x k)
        vector<T> C_block(m * k, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, col_count,
                   1.0, A_block_dense.data(), m, B_sub.data(), col_count,
                   0.0, C_block.data(), m);

        // Expected: compute A[:,col_start:col_start+col_count] * B[col_start:col_start+col_count,:]
        // from the dense A
        vector<T> A_cols(m * col_count, 0.0);
        for (int64_t j = 0; j < col_count; ++j) {
            for (int64_t i = 0; i < m; ++i) {
                A_cols[i + j * m] = A_dense[i + (col_start + j) * m];
            }
        }
        vector<T> C_expected(m * k, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, col_count,
                   1.0, A_cols.data(), m, B_sub.data(), col_count,
                   0.0, C_expected.data(), m);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, k,
            C_block.data(), m,
            C_expected.data(), m,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        ), "");
    }

    /// Test csr_split_col_blocks: split and verify all blocks are correct.
    template <typename T>
    void test_csr_col_split(int64_t m, int64_t n, T density, int64_t num_blocks) {
        RNGState state(42);
        auto A = generate_csr_matrix<T>(m, n, density, state);

        // Densify full A
        vector<T> A_dense(m * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, A_dense.data());

        auto blocks = RandLAPACK::linops::csr_split_col_blocks(A, num_blocks);
        ASSERT_EQ((int64_t) blocks.size(), num_blocks);

        int64_t block_size = n / num_blocks;

        // Total nnz should sum to A.nnz
        int64_t total_nnz = 0;
        for (auto& blk : blocks)
            total_nnz += blk.nnz;
        ASSERT_EQ(total_nnz, A.nnz);

        // Each block's dense form should match the corresponding columns of A_dense
        for (int64_t b = 0; b < num_blocks; ++b) {
            auto A_block = blocks[b].as_csr();
            vector<T> A_block_dense(m * block_size, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(A_block, Layout::ColMajor, A_block_dense.data());

            int64_t col_start = b * block_size;
            for (int64_t j = 0; j < block_size; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    ASSERT_NEAR(A_block_dense[i + j * m],
                                A_dense[i + (col_start + j) * m],
                                100 * std::numeric_limits<T>::epsilon())
                        << "Block " << b << ", element (" << i << ", " << j << ")";
                }
            }
        }
    }
};

// CSR col-block structural tests
TEST_F(TestCSRColBlocks, structure_middle_cols) {
    test_csr_col_block_structure<double>(20, 15, 0.3, 4, 7);
}

TEST_F(TestCSRColBlocks, structure_first_cols) {
    test_csr_col_block_structure<double>(20, 15, 0.3, 0, 5);
}

TEST_F(TestCSRColBlocks, structure_last_cols) {
    test_csr_col_block_structure<double>(20, 15, 0.3, 10, 5);
}

TEST_F(TestCSRColBlocks, structure_single_col) {
    test_csr_col_block_structure<double>(20, 15, 0.3, 7, 1);
}

TEST_F(TestCSRColBlocks, structure_all_cols) {
    test_csr_col_block_structure<double>(20, 15, 0.3, 0, 15);
}

// CSR col-block SpMM tests
TEST_F(TestCSRColBlocks, spmm_middle_cols) {
    test_csr_col_block_spmm<double>(20, 15, 10, 0.3, 4, 7);
}

TEST_F(TestCSRColBlocks, spmm_first_cols) {
    test_csr_col_block_spmm<double>(20, 15, 10, 0.3, 0, 5);
}

TEST_F(TestCSRColBlocks, spmm_last_cols) {
    test_csr_col_block_spmm<double>(20, 15, 10, 0.3, 10, 5);
}

// CSR col-block split tests
TEST_F(TestCSRColBlocks, split_3_blocks) {
    test_csr_col_split<double>(20, 15, 0.3, 3);
}

TEST_F(TestCSRColBlocks, split_5_blocks) {
    test_csr_col_split<double>(20, 15, 0.3, 5);
}


// ============================================================================
// CSC Row-Block Tests (cross-direction extraction)
// ============================================================================

class TestCSCRowBlocks : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Test that csc_row_block produces a block with correct structure:
    /// colptr starting from 0, correct nnz, correct dimensions, rebased rowidxs.
    template <typename T>
    void test_csc_row_block_structure(int64_t m, int64_t n, T density,
                                      int64_t row_start, int64_t row_count) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();

        auto block = RandLAPACK::linops::csc_row_block(A, row_start, row_count);

        // Check dimensions
        ASSERT_EQ(block.n_rows, row_count);
        ASSERT_EQ(block.n_cols, n);

        // Check colptr starts from 0 and ends at nnz
        ASSERT_EQ(block.colptr[0], 0);
        ASSERT_EQ(block.colptr[n], block.nnz);

        // Check colptr is monotonically non-decreasing
        for (int64_t j = 0; j < n; ++j) {
            ASSERT_LE(block.colptr[j], block.colptr[j + 1]);
        }

        // Check all row indices are in [0, row_count)
        for (int64_t k = 0; k < block.nnz; ++k) {
            ASSERT_GE(block.rowidxs[k], 0);
            ASSERT_LT(block.rowidxs[k], row_count);
        }

        // Check nnz: count entries in parent with row in [row_start, row_start+row_count)
        int64_t expected_nnz = 0;
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t kk = A.colptr[j]; kk < A.colptr[j + 1]; ++kk) {
                if (A.rowidxs[kk] >= row_start && A.rowidxs[kk] < row_start + row_count)
                    expected_nnz++;
            }
        }
        ASSERT_EQ(block.nnz, expected_nnz);
    }

    /// Test that the dense form of a CSC row-block matches the corresponding
    /// rows of the full dense matrix.
    template <typename T>
    void test_csc_row_block_dense(int64_t m, int64_t n, T density,
                                   int64_t row_start, int64_t row_count) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();

        // Densify full A
        vector<T> A_dense(m * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, A_dense.data());

        // Extract row-block
        auto block = RandLAPACK::linops::csc_row_block(A, row_start, row_count);
        auto A_block = block.as_csc();

        // Densify block (row_count x n)
        vector<T> A_block_dense(row_count * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A_block, Layout::ColMajor, A_block_dense.data());

        // Compare against corresponding rows of full dense matrix
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < row_count; ++i) {
                ASSERT_NEAR(A_block_dense[i + j * row_count],
                            A_dense[(row_start + i) + j * m],
                            100 * std::numeric_limits<T>::epsilon())
                    << "element (" << i << ", " << j << ")";
            }
        }
    }

    /// Test csc_split_row_blocks: split and verify all blocks are correct.
    template <typename T>
    void test_csc_row_split(int64_t m, int64_t n, T density, int64_t num_blocks) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();

        // Densify full A
        vector<T> A_dense(m * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, A_dense.data());

        auto blocks = RandLAPACK::linops::csc_split_row_blocks(A, num_blocks);
        ASSERT_EQ((int64_t) blocks.size(), num_blocks);

        int64_t block_size = m / num_blocks;

        // Total nnz should sum to A.nnz
        int64_t total_nnz = 0;
        for (auto& blk : blocks)
            total_nnz += blk.nnz;
        ASSERT_EQ(total_nnz, A.nnz);

        // Each block's dense form should match the corresponding rows of A_dense
        for (int64_t b = 0; b < num_blocks; ++b) {
            auto A_block = blocks[b].as_csc();
            vector<T> A_block_dense(block_size * n, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(A_block, Layout::ColMajor, A_block_dense.data());

            int64_t row_start = b * block_size;
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < block_size; ++i) {
                    ASSERT_NEAR(A_block_dense[i + j * block_size],
                                A_dense[(row_start + i) + j * m],
                                100 * std::numeric_limits<T>::epsilon())
                        << "Block " << b << ", element (" << i << ", " << j << ")";
                }
            }
        }
    }
};

// CSC row-block structural tests
TEST_F(TestCSCRowBlocks, structure_middle_rows) {
    test_csc_row_block_structure<double>(20, 15, 0.3, 5, 8);
}

TEST_F(TestCSCRowBlocks, structure_first_rows) {
    test_csc_row_block_structure<double>(20, 15, 0.3, 0, 6);
}

TEST_F(TestCSCRowBlocks, structure_last_rows) {
    test_csc_row_block_structure<double>(20, 15, 0.3, 14, 6);
}

TEST_F(TestCSCRowBlocks, structure_single_row) {
    test_csc_row_block_structure<double>(20, 15, 0.3, 10, 1);
}

TEST_F(TestCSCRowBlocks, structure_all_rows) {
    test_csc_row_block_structure<double>(20, 15, 0.3, 0, 20);
}

// CSC row-block dense comparison tests
TEST_F(TestCSCRowBlocks, dense_middle_rows) {
    test_csc_row_block_dense<double>(20, 15, 0.3, 5, 8);
}

TEST_F(TestCSCRowBlocks, dense_first_rows) {
    test_csc_row_block_dense<double>(20, 15, 0.3, 0, 6);
}

TEST_F(TestCSCRowBlocks, dense_last_rows) {
    test_csc_row_block_dense<double>(20, 15, 0.3, 14, 6);
}

// CSC row-block split tests
TEST_F(TestCSCRowBlocks, split_4_blocks) {
    test_csc_row_split<double>(20, 15, 0.3, 4);
}

TEST_F(TestCSCRowBlocks, split_2_blocks) {
    test_csc_row_split<double>(20, 15, 0.3, 2);
}

TEST_F(TestCSCRowBlocks, split_5_blocks) {
    test_csc_row_split<double>(20, 15, 0.3, 5);
}


// ============================================================================
// SparseLinOp Block Method Tests (dispatch to free functions)
// ============================================================================

class TestSparseLinOpBlocks : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Generate a random sparse matrix in CSR format.
    template <typename T>
    RandBLAS::sparse_data::CSRMatrix<T> generate_csr(
        int64_t rows, int64_t cols, T density, RNGState<>& state
    ) {
        auto coo = RandLAPACK::gen::gen_sparse_coo<T>(rows, cols, density, state);
        RandBLAS::sparse_data::CSRMatrix<T> csr(rows, cols);
        RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
        return csr;
    }

    /// Densify a sparse matrix into a ColMajor dense array.
    template <typename T, typename SpMat>
    vector<T> to_dense(SpMat& A, int64_t rows, int64_t cols) {
        vector<T> D(rows * cols, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A, Layout::ColMajor, D.data());
        return D;
    }

    /// Extract a submatrix from a ColMajor dense array.
    template <typename T>
    vector<T> extract_dense_submat(const T* A, int64_t lda,
                                    int64_t row_start, int64_t col_start,
                                    int64_t row_count, int64_t col_count) {
        vector<T> sub(row_count * col_count, 0.0);
        for (int64_t j = 0; j < col_count; ++j)
            for (int64_t i = 0; i < row_count; ++i)
                sub[i + j * row_count] = A[(row_start + i) + (col_start + j) * lda];
        return sub;
    }

    // ---------------------------------------------------------------
    // CSR SparseLinOp block tests
    // ---------------------------------------------------------------

    /// Test SparseLinOp<CSR>::row_block by comparing densified result
    /// against expected rows of the full dense matrix.
    template <typename T>
    void test_csr_linop_row_block(int64_t m, int64_t n, T density,
                                   int64_t row_start, int64_t row_count) {
        RNGState state(42);
        auto A = generate_csr<T>(m, n, density, state);
        auto A_dense = to_dense<T>(A, m, n);

        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSRMatrix<T>> op(m, n, A);
        auto blk_op = op.row_block(row_start, row_count);

        ASSERT_EQ(blk_op.n_rows, row_count);
        ASSERT_EQ(blk_op.n_cols, n);

        auto blk_dense = to_dense<T>(blk_op.A_sp, row_count, n);
        auto expected = extract_dense_submat<T>(A_dense.data(), m, row_start, 0, row_count, n);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, n,
            blk_dense.data(), row_count,
            expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    /// Test SparseLinOp<CSR>::col_block by comparing densified result
    /// against expected columns of the full dense matrix.
    template <typename T>
    void test_csr_linop_col_block(int64_t m, int64_t n, T density,
                                   int64_t col_start, int64_t col_count) {
        RNGState state(42);
        auto A = generate_csr<T>(m, n, density, state);
        auto A_dense = to_dense<T>(A, m, n);

        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSRMatrix<T>> op(m, n, A);
        auto blk_op = op.col_block(col_start, col_count);

        ASSERT_EQ(blk_op.n_rows, m);
        ASSERT_EQ(blk_op.n_cols, col_count);

        auto blk_dense = to_dense<T>(blk_op.A_sp, m, col_count);
        auto expected = extract_dense_submat<T>(A_dense.data(), m, 0, col_start, m, col_count);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, col_count,
            blk_dense.data(), m,
            expected.data(), m,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    /// Test SparseLinOp<CSR>::submatrix by comparing densified result
    /// against expected dense submatrix.
    template <typename T>
    void test_csr_linop_submatrix(int64_t m, int64_t n, T density,
                                   int64_t row_start, int64_t col_start,
                                   int64_t row_count, int64_t col_count) {
        RNGState state(42);
        auto A = generate_csr<T>(m, n, density, state);
        auto A_dense = to_dense<T>(A, m, n);

        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSRMatrix<T>> op(m, n, A);
        auto blk_op = op.submatrix(row_start, col_start, row_count, col_count);

        ASSERT_EQ(blk_op.n_rows, row_count);
        ASSERT_EQ(blk_op.n_cols, col_count);

        auto blk_dense = to_dense<T>(blk_op.A_sp, row_count, col_count);
        auto expected = extract_dense_submat<T>(A_dense.data(), m, row_start, col_start, row_count, col_count);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, col_count,
            blk_dense.data(), row_count,
            expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    // ---------------------------------------------------------------
    // CSC SparseLinOp block tests
    // ---------------------------------------------------------------

    /// Test SparseLinOp<CSC>::row_block (cross-direction for CSC).
    template <typename T>
    void test_csc_linop_row_block(int64_t m, int64_t n, T density,
                                   int64_t row_start, int64_t row_count) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();
        auto A_dense = to_dense<T>(A, m, n);

        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>> op(m, n, A);
        auto blk_op = op.row_block(row_start, row_count);

        ASSERT_EQ(blk_op.n_rows, row_count);
        ASSERT_EQ(blk_op.n_cols, n);

        auto blk_dense = to_dense<T>(blk_op.A_sp, row_count, n);
        auto expected = extract_dense_submat<T>(A_dense.data(), m, row_start, 0, row_count, n);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, n,
            blk_dense.data(), row_count,
            expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    /// Test SparseLinOp<CSC>::col_block (natural direction for CSC).
    template <typename T>
    void test_csc_linop_col_block(int64_t m, int64_t n, T density,
                                   int64_t col_start, int64_t col_count) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();
        auto A_dense = to_dense<T>(A, m, n);

        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>> op(m, n, A);
        auto blk_op = op.col_block(col_start, col_count);

        ASSERT_EQ(blk_op.n_rows, m);
        ASSERT_EQ(blk_op.n_cols, col_count);

        auto blk_dense = to_dense<T>(blk_op.A_sp, m, col_count);
        auto expected = extract_dense_submat<T>(A_dense.data(), m, 0, col_start, m, col_count);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, col_count,
            blk_dense.data(), m,
            expected.data(), m,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    /// Test SparseLinOp<CSC>::submatrix.
    template <typename T>
    void test_csc_linop_submatrix(int64_t m, int64_t n, T density,
                                   int64_t row_start, int64_t col_start,
                                   int64_t row_count, int64_t col_count) {
        RNGState state(42);
        auto A = RandLAPACK::gen::gen_sparse_coo<T>(m, n, density, state).as_owning_csc();
        auto A_dense = to_dense<T>(A, m, n);

        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>> op(m, n, A);
        auto blk_op = op.submatrix(row_start, col_start, row_count, col_count);

        ASSERT_EQ(blk_op.n_rows, row_count);
        ASSERT_EQ(blk_op.n_cols, col_count);

        auto blk_dense = to_dense<T>(blk_op.A_sp, row_count, col_count);
        auto expected = extract_dense_submat<T>(A_dense.data(), m, row_start, col_start, row_count, col_count);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, col_count,
            blk_dense.data(), row_count,
            expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }
};

// CSR SparseLinOp row_block tests
TEST_F(TestSparseLinOpBlocks, csr_row_block_middle) {
    test_csr_linop_row_block<double>(20, 15, 0.3, 5, 8);
}

TEST_F(TestSparseLinOpBlocks, csr_row_block_first) {
    test_csr_linop_row_block<double>(20, 15, 0.3, 0, 6);
}

TEST_F(TestSparseLinOpBlocks, csr_row_block_last) {
    test_csr_linop_row_block<double>(20, 15, 0.3, 14, 6);
}

// CSR SparseLinOp col_block tests
TEST_F(TestSparseLinOpBlocks, csr_col_block_middle) {
    test_csr_linop_col_block<double>(20, 15, 0.3, 4, 7);
}

TEST_F(TestSparseLinOpBlocks, csr_col_block_first) {
    test_csr_linop_col_block<double>(20, 15, 0.3, 0, 5);
}

TEST_F(TestSparseLinOpBlocks, csr_col_block_last) {
    test_csr_linop_col_block<double>(20, 15, 0.3, 10, 5);
}

// CSR SparseLinOp submatrix tests
TEST_F(TestSparseLinOpBlocks, csr_submatrix_middle) {
    test_csr_linop_submatrix<double>(20, 15, 0.3, 3, 4, 10, 7);
}

TEST_F(TestSparseLinOpBlocks, csr_submatrix_corner) {
    test_csr_linop_submatrix<double>(20, 15, 0.3, 0, 0, 8, 6);
}

TEST_F(TestSparseLinOpBlocks, csr_submatrix_end) {
    test_csr_linop_submatrix<double>(20, 15, 0.3, 12, 9, 8, 6);
}

// CSC SparseLinOp row_block tests (cross-direction)
TEST_F(TestSparseLinOpBlocks, csc_row_block_middle) {
    test_csc_linop_row_block<double>(20, 15, 0.3, 5, 8);
}

TEST_F(TestSparseLinOpBlocks, csc_row_block_first) {
    test_csc_linop_row_block<double>(20, 15, 0.3, 0, 6);
}

TEST_F(TestSparseLinOpBlocks, csc_row_block_last) {
    test_csc_linop_row_block<double>(20, 15, 0.3, 14, 6);
}

// CSC SparseLinOp col_block tests (natural direction)
TEST_F(TestSparseLinOpBlocks, csc_col_block_middle) {
    test_csc_linop_col_block<double>(20, 15, 0.3, 4, 7);
}

TEST_F(TestSparseLinOpBlocks, csc_col_block_first) {
    test_csc_linop_col_block<double>(20, 15, 0.3, 0, 5);
}

TEST_F(TestSparseLinOpBlocks, csc_col_block_last) {
    test_csc_linop_col_block<double>(20, 15, 0.3, 10, 5);
}

// CSC SparseLinOp submatrix tests
TEST_F(TestSparseLinOpBlocks, csc_submatrix_middle) {
    test_csc_linop_submatrix<double>(20, 15, 0.3, 3, 4, 10, 7);
}

TEST_F(TestSparseLinOpBlocks, csc_submatrix_corner) {
    test_csc_linop_submatrix<double>(20, 15, 0.3, 0, 0, 8, 6);
}

TEST_F(TestSparseLinOpBlocks, csc_submatrix_end) {
    test_csc_linop_submatrix<double>(20, 15, 0.3, 12, 9, 8, 6);
}


// ============================================================================
// SparseLinOp::materialize_cols Tests
// ============================================================================

namespace {

using RandBLAS::sparse_data::CSRMatrix;
using RandBLAS::sparse_data::CSCMatrix;
using RandBLAS::sparse_data::COOMatrix;
using RandBLAS::sparse_data::IndexBase;

struct SpEntry { int64_t i, j; double v; };

// 7 x 6 with row 5 and column 3 empty, three entries at (1, 2) and two at (6, 1), listed out of
// row and column order. The values are dyadic, so every sum is exact in any order.
const vector<SpEntry> kSpEntries = {
    {0, 5, 1.5}, {0, 0, -2.0}, {1, 2, 0.25}, {1, 4, 3.0}, {1, 2, 0.5}, {3, 0, 1.0},
    {1, 2, -1.0}, {4, 5, 2.5}, {6, 1, -0.75}, {6, 5, 4.0}, {3, 4, -3.5}, {4, 1, 1.25},
    {6, 1, 0.125}, {2, 4, -0.5}, {6, 0, 0.375}};

// The CSR, CSC and COO forms of an entry list, and its dense column-major form with duplicates
// summed. CSR keeps each row's entries, and CSC each column's, in list order, so they are
// unsorted.
template <typename T>
struct SparseForms {
    int64_t m, n, nnz;
    vector<int64_t> rowptr, colidxs, colptr, rowidxs, rows, cols;
    vector<T> csr_vals, csc_vals, coo_vals, dense;
    SparseForms(int64_t m_, int64_t n_, const vector<SpEntry>& e)
        : m(m_), n(n_), nnz((int64_t)e.size()), rowptr(m_ + 1, 0), colidxs(nnz), colptr(n_ + 1, 0),
          rowidxs(nnz), rows(nnz), cols(nnz), csr_vals(nnz), csc_vals(nnz), coo_vals(nnz),
          dense(m_ * n_, (T)0) {
        for (const auto& x : e) { ++rowptr[x.i + 1]; ++colptr[x.j + 1]; }
        for (int64_t i = 0; i < m; ++i) rowptr[i + 1] += rowptr[i];
        for (int64_t j = 0; j < n; ++j) colptr[j + 1] += colptr[j];
        vector<int64_t> r_at(rowptr.begin(), rowptr.end() - 1), c_at(colptr.begin(), colptr.end() - 1);
        for (int64_t k = 0; k < nnz; ++k) {
            const SpEntry& x = e[k];
            colidxs[r_at[x.i]] = x.j;  csr_vals[r_at[x.i]++] = (T)x.v;
            rowidxs[c_at[x.j]] = x.i;  csc_vals[c_at[x.j]++] = (T)x.v;
            rows[k] = x.i;  cols[k] = x.j;  coo_vals[k] = (T)x.v;
            dense[x.i + x.j * m] += (T)x.v;
        }
    }
    CSRMatrix<T> csr() { return CSRMatrix<T>(m, n, nnz, csr_vals.data(), rowptr.data(), colidxs.data()); }
    CSCMatrix<T> csc() { return CSCMatrix<T>(m, n, nnz, csc_vals.data(), rowidxs.data(), colptr.data()); }
    COOMatrix<T> coo() { return COOMatrix<T>(m, n, nnz, coo_vals.data(), rows.data(), cols.data()); }
};

template <typename SpMat>
using SpLinOp = RandLAPACK::linops::SparseLinOp<SpMat>;

// materialize_cols at the start, middle and end, over the empty column and with no columns:
// rows 0..m-1 of the written columns equal D's, and the padding rows and the columns past the
// count keep their fill.
template <typename T, typename SpOp>
void check_sparse_materialize_cols(const SpOp& op, const vector<T>& D, int64_t m, int64_t n) {
    const int64_t ldb = m + 2;
    const T pad = (T)-42;
    for (auto [c0, cnt] : {std::pair<int64_t, int64_t>{0, n}, {0, 2}, {2, 3}, {n - 2, 2}, {3, 1}, {4, 0}, {n, 0}}) {
        vector<T> buf(ldb * n, pad);
        op.materialize_cols(c0, cnt, buf.data(), ldb);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < ldb; ++i) {
                const T want = (j < cnt && i < m) ? D[i + (c0 + j) * m] : pad;
                ASSERT_EQ(buf[i + j * ldb], want) << "cols " << c0 << "+" << cnt << " at " << i << "," << j;
            }
    }
}

// materialize_cols against the product with identity columns [c0, c0 + cnt), as the Gram loop
// forms it, entry for entry: for finite data both are exact.
template <typename T, typename SpOp>
void check_sparse_cols_equal_identity_product(SpOp& op, int64_t m, int64_t n) {
    for (auto [c0, cnt] : {std::pair<int64_t, int64_t>{0, n}, {1, n / 2}, {n - 3, 3}}) {
        vector<T> I_blk(n * cnt, (T)0), prod(m * cnt, (T)7), buf(m * cnt, (T)-42);
        for (int64_t j = 0; j < cnt; ++j) I_blk[(c0 + j) + j * n] = (T)1;
        op(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, cnt, n, (T)1, I_blk.data(), n, (T)0, prod.data(), m);
        op.materialize_cols(c0, cnt, buf.data(), m);
        for (int64_t k = 0; k < m * cnt; ++k)
            ASSERT_EQ(buf[k], prod[k]) << "cols " << c0 << "+" << cnt << " at " << k;
    }
}

template <typename T>
void check_sparse_materialize_cols_formats() {
    SparseForms<T> f(7, 6, kSpEntries);
    auto A_csr = f.csr();
    auto A_csc = f.csc();
    auto A_coo = f.coo();
    SpLinOp<CSRMatrix<T>> op_csr(f.m, f.n, A_csr);
    SpLinOp<CSCMatrix<T>> op_csc(f.m, f.n, A_csc);
    SpLinOp<COOMatrix<T>> op_coo(f.m, f.n, A_coo);
    check_sparse_materialize_cols<T>(op_csr, f.dense, f.m, f.n);
    check_sparse_materialize_cols<T>(op_csc, f.dense, f.m, f.n);
    check_sparse_materialize_cols<T>(op_coo, f.dense, f.m, f.n);
    // MKL's CSC product (mkl_sparse_d_mm, transposed, on the CSR of A^T) lost one of two
    // duplicates stored next to each other in a column in probes, so CSC meets the product
    // only on duplicate-free data.
    check_sparse_cols_equal_identity_product<T>(op_csr, f.m, f.n);
    check_sparse_cols_equal_identity_product<T>(op_coo, f.m, f.n);
}

} // namespace

static_assert(RandLAPACK::linops::HasMaterializeCols<SpLinOp<CSRMatrix<double>>>);
static_assert(RandLAPACK::linops::HasMaterializeCols<const SpLinOp<CSCMatrix<float>>>);
static_assert(RandLAPACK::linops::HasMaterializeCols<SpLinOp<COOMatrix<double>>>);
static_assert(RandLAPACK::linops::HasMaterializeCols<
    RandLAPACK::linops::VStackLinOp<SpLinOp<CSRMatrix<double>>, RandLAPACK::linops::ScaledIdentityLinOp<double>>>);
static_assert(!RandLAPACK::linops::HasMaterializeCols<
    RandLAPACK::linops::CompositeOperator<RandLAPACK::linops::DenseLinOp<double>, SpLinOp<CSRMatrix<double>>>>);

// Every format, with duplicates and unsorted indices: duplicates sum, and the CSR and COO
// columns equal the product exactly.
TEST(TestSparseMaterializeCols, formats_duplicates_and_unsorted_indices) {
    check_sparse_materialize_cols_formats<double>();
}

TEST(TestSparseMaterializeCols, formats_duplicates_and_unsorted_indices_float) {
    check_sparse_materialize_cols_formats<float>();
}

// materialize_cols, and so materialize, reject a one-based matrix, as RandBLAS's sparse products
// do. SparseLinOp's view keeps the zero base until the index-base fix, so the test sets it.
TEST(TestSparseMaterializeCols, one_based_matrix_is_rejected) {
    SparseForms<double> f(7, 6, kSpEntries);
    auto A_csr = f.csr();
    auto A_csc = f.csc();
    auto A_coo = f.coo();
    SpLinOp<CSRMatrix<double>> op_csr(f.m, f.n, A_csr);
    SpLinOp<CSCMatrix<double>> op_csc(f.m, f.n, A_csc);
    SpLinOp<COOMatrix<double>> op_coo(f.m, f.n, A_coo);
    op_csr.A_sp.index_base = op_csc.A_sp.index_base = op_coo.A_sp.index_base = IndexBase::One;
    vector<double> buf(f.m * f.n, -42.0);
    auto check = [&](auto& op) {
        EXPECT_THROW(op.materialize_cols(0, f.n, buf.data(), f.m), RandLAPACK::Error);
        EXPECT_THROW(RandLAPACK::materialize(op, f.m, f.n, buf.data(), f.m), RandLAPACK::Error);
    };
    check(op_csr);
    check(op_csc);
    check(op_coo);
    for (double v : buf) ASSERT_EQ(v, -42.0);
}

// A COO operator smaller than its matrix reads the leading n_rows x n_cols window, as its
// sparse-dense products do; the columns equal that window and the product with identity columns.
TEST(TestSparseMaterializeCols, coo_window_of_a_larger_matrix) {
    SparseForms<double> f(7, 6, kSpEntries);
    const int64_t mw = 5, nw = 5;
    auto A_coo = f.coo();
    SpLinOp<COOMatrix<double>> op(mw, nw, A_coo);
    vector<double> window(mw * nw);
    for (int64_t j = 0; j < nw; ++j)
        for (int64_t i = 0; i < mw; ++i) window[i + j * mw] = f.dense[i + j * f.m];
    check_sparse_materialize_cols<double>(op, window, mw, nw);
    check_sparse_cols_equal_identity_product<double>(op, mw, nw);
}

// On random finite data with no duplicates the columns equal the product with identity columns
// in each format.
TEST(TestSparseMaterializeCols, equals_identity_product_on_random_data) {
    const int64_t m = 40, n = 12;
    RNGState state(31);
    auto A_coo = RandLAPACK::gen::gen_sparse_coo<double>(m, n, 0.3, state);
    auto A_csr = A_coo.as_owning_csr();
    auto A_csc = A_coo.as_owning_csc();
    SpLinOp<COOMatrix<double>> op_coo(m, n, A_coo);
    SpLinOp<CSRMatrix<double>> op_csr(m, n, A_csr);
    SpLinOp<CSCMatrix<double>> op_csc(m, n, A_csc);
    check_sparse_cols_equal_identity_product<double>(op_coo, m, n);
    check_sparse_cols_equal_identity_product<double>(op_csr, m, n);
    check_sparse_cols_equal_identity_product<double>(op_csc, m, n);
}

// Large enough that every loop of the full-range copy is threaded, run on at least 4 OpenMP
// threads: with three stored entries per row (a duplicate pair among them), the CSR scatter
// (3 m entries), the zero fill (m n) and CSC's fill and scatter all pass
// kMaterializeColsParallelMin. The copy equals the dense form, and for CSR and COO the product
// with identity columns, entry for entry.
TEST(TestSparseMaterializeCols, threaded_copy) {
    constexpr int64_t min_par = SpLinOp<CSRMatrix<double>>::kMaterializeColsParallelMin;
    constexpr int64_t m = min_par / 2, n = 10;
    static_assert(3 * m >= min_par && m * n >= min_par);
    #if defined(RandBLAS_HAS_OpenMP)
    struct Threads {   // at least 4 OpenMP threads for this test, restored on exit
        int saved = omp_get_max_threads();
        Threads() { omp_set_num_threads(std::max(saved, 4)); }
        ~Threads() { omp_set_num_threads(saved); }
    } threads;
    #endif
    vector<SpEntry> e;
    for (int64_t i = 0; i < m; ++i) {
        e.push_back({i, (3 * i) % n, -0.25});
        e.push_back({i, i % n, 0.5});
        e.push_back({i, i % n, 0.125 * (double)(i % 7)});
    }
    SparseForms<double> f(m, n, e);
    auto A_csr = f.csr();
    auto A_csc = f.csc();
    auto A_coo = f.coo();
    SpLinOp<CSRMatrix<double>> op_csr(m, n, A_csr);
    SpLinOp<CSCMatrix<double>> op_csc(m, n, A_csc);
    SpLinOp<COOMatrix<double>> op_coo(m, n, A_coo);
    check_sparse_materialize_cols<double>(op_csr, f.dense, m, n);
    check_sparse_materialize_cols<double>(op_csc, f.dense, m, n);
    check_sparse_materialize_cols<double>(op_coo, f.dense, m, n);
    check_sparse_cols_equal_identity_product<double>(op_csr, m, n);
    check_sparse_cols_equal_identity_product<double>(op_coo, m, n);
}

// Inf and NaN land in their own entries only, with the rest of their rows as stored (a product
// with identity columns may turn those into NaN, Inf * 0). A stored -0.0 adds to the zeroed
// entry and reads +0.0, as in the product.
TEST(TestSparseMaterializeCols, inf_nan_and_signed_zero) {
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    const vector<SpEntry> e = {{0, 1, 2.0}, {2, 0, inf}, {2, 3, 5.0}, {4, 3, nan}, {4, 0, -1.5}, {1, 1, -0.0}};
    SparseForms<double> f(5, 4, e);
    auto A_csr = f.csr();
    auto A_csc = f.csc();
    auto A_coo = f.coo();
    SpLinOp<CSRMatrix<double>> op_csr(f.m, f.n, A_csr);
    SpLinOp<CSCMatrix<double>> op_csc(f.m, f.n, A_csc);
    SpLinOp<COOMatrix<double>> op_coo(f.m, f.n, A_coo);
    auto check = [&](const auto& op) {
        vector<double> buf(f.m * f.n, -42.0);
        op.materialize_cols(0, f.n, buf.data(), f.m);
        for (int64_t j = 0; j < f.n; ++j)
            for (int64_t i = 0; i < f.m; ++i) {
                const double got = buf[i + j * f.m];
                if (i == 4 && j == 3) ASSERT_TRUE(std::isnan(got));
                else ASSERT_EQ(got, f.dense[i + j * f.m]) << i << "," << j;
            }
        EXPECT_EQ(buf[2], inf);
        EXPECT_EQ(buf[2 + 3 * f.m], 5.0);
        EXPECT_EQ(buf[4], -1.5);
        EXPECT_FALSE(std::signbit(buf[1 + 1 * f.m]));
    };
    check(op_csr);
    check(op_csc);
    check(op_coo);
}

// materialize goes through materialize_cols: duplicates sum in every format (sparse_to_dense
// keeps only the last one in CSR and CSC), and the padding rows stay untouched.
TEST(TestSparseMaterializeCols, materialize_sums_duplicates) {
    SparseForms<double> f(7, 6, kSpEntries);
    auto A_csr = f.csr();
    auto A_csc = f.csc();
    auto A_coo = f.coo();
    SpLinOp<CSRMatrix<double>> op_csr(f.m, f.n, A_csr);
    SpLinOp<CSCMatrix<double>> op_csc(f.m, f.n, A_csc);
    SpLinOp<COOMatrix<double>> op_coo(f.m, f.n, A_coo);
    const int64_t ldb = f.m + 3;
    auto check = [&](auto& op) {
        vector<double> buf(ldb * f.n, -42.0);
        RandLAPACK::materialize(op, f.m, f.n, buf.data(), ldb);
        for (int64_t j = 0; j < f.n; ++j)
            for (int64_t i = 0; i < ldb; ++i)
                ASSERT_EQ(buf[i + j * ldb], i < f.m ? f.dense[i + j * f.m] : -42.0) << i << "," << j;
    };
    check(op_csr);
    check(op_csc);
    check(op_coo);
    EXPECT_EQ(f.dense[1 + 2 * f.m], -0.25);   // the three entries at (1, 2)
}

TEST(TestSparseMaterializeCols, invalid_arguments) {
    SparseForms<double> f(7, 6, kSpEntries);
    const int64_t m = f.m, n = f.n;
    auto A_csr = f.csr();
    auto A_csc = f.csc();
    auto A_coo = f.coo();
    const SpLinOp<CSRMatrix<double>> op_csr(m, n, A_csr);
    const SpLinOp<CSCMatrix<double>> op_csc(m, n, A_csc);
    const SpLinOp<COOMatrix<double>> op_coo(m, n, A_coo);
    // CSR and CSC operators whose dimensions differ from their matrix's, and a COO operator
    // larger than its matrix.
    const SpLinOp<CSRMatrix<double>> short_csr(m - 1, n, A_csr), narrow_csr(m, n - 1, A_csr);
    const SpLinOp<CSCMatrix<double>> short_csc(m - 1, n, A_csc);
    const SpLinOp<COOMatrix<double>> tall_coo(m + 1, n, A_coo), wide_coo(m, n + 1, A_coo);
    vector<double> buf(m * n, -42.0);
    const int64_t big = std::numeric_limits<int64_t>::max();
    auto check = [&](const auto& op) {
        EXPECT_THROW(op.materialize_cols(-1, 2, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op.materialize_cols(0, -1, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op.materialize_cols(n + 1, 0, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op.materialize_cols(n - 1, 2, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op.materialize_cols(0, 1, buf.data(), m - 1), RandLAPACK::Error);
        // The range check cannot overflow: start + count is never formed.
        EXPECT_THROW(op.materialize_cols(big, 1, buf.data(), m), RandLAPACK::Error);
        EXPECT_THROW(op.materialize_cols(1, big, buf.data(), m), RandLAPACK::Error);
    };
    check(op_csr);
    check(op_csc);
    check(op_coo);
    EXPECT_THROW(short_csr.materialize_cols(0, 1, buf.data(), m), RandLAPACK::Error);
    EXPECT_THROW(narrow_csr.materialize_cols(0, 1, buf.data(), m), RandLAPACK::Error);
    EXPECT_THROW(short_csc.materialize_cols(0, 1, buf.data(), m), RandLAPACK::Error);
    EXPECT_THROW(tall_coo.materialize_cols(0, 1, buf.data(), m + 1), RandLAPACK::Error);
    EXPECT_THROW(wide_coo.materialize_cols(0, 1, buf.data(), m), RandLAPACK::Error);
    for (double v : buf) ASSERT_EQ(v, -42.0);   // every failed call stopped before writing
}

// An operator with no rows writes nothing (buf may be null). A matrix with no stored entries,
// whose index arrays are null, gives zero columns.
TEST(TestSparseMaterializeCols, zero_count_zero_rows_and_no_entries) {
    const int64_t n = 4, m = 5, ldb = m + 1;
    vector<int64_t> rowptr0(1, 0), colptr0(n + 1, 0);
    CSRMatrix<double> E_csr(0, n, 0, nullptr, rowptr0.data(), nullptr);
    CSCMatrix<double> E_csc(0, n, 0, nullptr, nullptr, colptr0.data());
    COOMatrix<double> E_coo(0, n, 0, nullptr, nullptr, nullptr);
    const SpLinOp<CSRMatrix<double>> op_csr(0, n, E_csr);
    const SpLinOp<CSCMatrix<double>> op_csc(0, n, E_csc);
    const SpLinOp<COOMatrix<double>> op_coo(0, n, E_coo);
    auto check_empty = [&](const auto& op) {
        op.materialize_cols(1, 2, nullptr, 0);
        op.materialize_cols(0, n, nullptr, 0);
        EXPECT_THROW(op.materialize_cols(n - 1, 2, nullptr, 0), RandLAPACK::Error);
    };
    check_empty(op_csr);
    check_empty(op_csc);
    check_empty(op_coo);

    CSRMatrix<double> Z_csr(m, n);   // standard constructor: no arrays until reserve
    CSCMatrix<double> Z_csc(m, n);
    COOMatrix<double> Z_coo(m, n);
    const SpLinOp<CSRMatrix<double>> z_csr(m, n, Z_csr);
    const SpLinOp<CSCMatrix<double>> z_csc(m, n, Z_csc);
    const SpLinOp<COOMatrix<double>> z_coo(m, n, Z_coo);
    auto check_zero = [&](const auto& op) {
        vector<double> buf(ldb * n, -42.0);
        op.materialize_cols(1, 3, buf.data(), ldb);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < ldb; ++i)
                ASSERT_EQ(buf[i + j * ldb], (j < 3 && i < m) ? 0.0 : -42.0) << i << "," << j;
    };
    check_zero(z_csr);
    check_zero(z_csc);
    check_zero(z_coo);
}


// ============================================================================
// CompositeOperator Block Method Tests
// ============================================================================

class TestCompositeBlockViews : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    /// Helper: compute full dense product via CompositeOperator's operator(),
    /// then compare against the block-derived operator's result.
    ///
    /// For row_block: block_C = (L*R)[row_start:row_start+row_count, :] * B
    ///   should match corresponding rows of C_full = (L*R) * B.
    ///
    /// For col_block: block_C = B * (L*R)[:, col_start:col_start+col_count]
    ///   We use Side::Right for the full product: C_full = B * (L*R),
    ///   then compare corresponding columns.
    ///
    /// For submatrix: we verify via left multiplication with B of matching width.

    // ---------------------------------------------------------------
    // Dense * Dense composition
    // ---------------------------------------------------------------

    /// Test CompositeOperator<Dense,Dense>::row_block.
    /// Full: C_full = (L * R) * B  (m x k)
    /// Block: C_blk = (L[rows,:] * R) * B  (row_count x k)
    /// Verify C_blk matches rows [row_start, row_start+row_count) of C_full.
    template <typename T>
    void test_dense_dense_row_block(int64_t m, int64_t p, int64_t n, int64_t k,
                                     int64_t row_start, int64_t row_count) {
        RNGState state(42);
        Layout layout = Layout::ColMajor;

        // L (m x p), R (p x n), B (n x k)
        vector<T> L_data = generate_dense_matrix<T>(m, p, layout, state);
        vector<T> R_data = generate_dense_matrix<T>(p, n, layout, state);
        vector<T> B(n * k);
        { RNGState s2(99); DenseDist D(n, k); RandBLAS::fill_dense(D, B.data(), s2); }

        RandLAPACK::linops::DenseLinOp<T> L_op(m, p, L_data.data(), m, layout);
        RandLAPACK::linops::DenseLinOp<T> R_op(p, n, R_data.data(), p, layout);
        RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

        // Full product: C_full = (L*R) * B  (m x k, ColMajor)
        vector<T> C_full(m * k, 0.0);
        comp(layout, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, B.data(), n, (T)0.0, C_full.data(), m);

        // Block product
        auto blk = comp.row_block(row_start, row_count);
        ASSERT_EQ(blk.n_rows, row_count);
        ASSERT_EQ(blk.n_cols, n);

        vector<T> C_blk(row_count * k, 0.0);
        blk(layout, Op::NoTrans, Op::NoTrans, row_count, k, n, (T)1.0, B.data(), n, (T)0.0, C_blk.data(), row_count);

        // Extract expected rows from C_full (ColMajor)
        vector<T> C_expected(row_count * k);
        for (int64_t j = 0; j < k; ++j)
            for (int64_t i = 0; i < row_count; ++i)
                C_expected[i + j * row_count] = C_full[(row_start + i) + j * m];

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, k,
            C_blk.data(), row_count,
            C_expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    /// Test CompositeOperator<Dense,Dense>::col_block.
    /// Full: C_full = B * (L * R)  (k x n) via Side::Right
    /// Block: C_blk = B * (L * R[:,cols])  (k x col_count)
    /// But since col_block changes n_cols, we compute:
    ///   Full left-multiply: C_full = (L*R)^T * B_col  ... actually simpler approach:
    ///   Just compute the full product, then compare columns.
    /// Easier approach: use Side::Left with transpose.
    /// Simplest: directly form (L*R) as dense, extract columns, compare.
    template <typename T>
    void test_dense_dense_col_block(int64_t m, int64_t p, int64_t n, int64_t k,
                                     int64_t col_start, int64_t col_count) {
        RNGState state(42);
        Layout layout = Layout::ColMajor;

        // L (m x p), R (p x n), B (k x m)
        vector<T> L_data = generate_dense_matrix<T>(m, p, layout, state);
        vector<T> R_data = generate_dense_matrix<T>(p, n, layout, state);
        vector<T> B(k * m);
        { RNGState s2(99); DenseDist D(k, m); RandBLAS::fill_dense(D, B.data(), s2); }

        RandLAPACK::linops::DenseLinOp<T> L_op(m, p, L_data.data(), m, layout);
        RandLAPACK::linops::DenseLinOp<T> R_op(p, n, R_data.data(), p, layout);
        RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

        // Full product: C_full = B * (L*R)  (k x n)
        vector<T> C_full(k * n, 0.0);
        comp(Side::Right, layout, Op::NoTrans, Op::NoTrans, k, n, m,
             (T)1.0, B.data(), k, (T)0.0, C_full.data(), k);

        // Block product: C_blk = B * (L * R[:,cols])  (k x col_count)
        auto blk = comp.col_block(col_start, col_count);
        ASSERT_EQ(blk.n_rows, m);
        ASSERT_EQ(blk.n_cols, col_count);

        vector<T> C_blk(k * col_count, 0.0);
        blk(Side::Right, layout, Op::NoTrans, Op::NoTrans, k, col_count, m,
            (T)1.0, B.data(), k, (T)0.0, C_blk.data(), k);

        // Extract expected columns from C_full (ColMajor)
        vector<T> C_expected(k * col_count);
        for (int64_t j = 0; j < col_count; ++j)
            for (int64_t i = 0; i < k; ++i)
                C_expected[i + j * k] = C_full[i + (col_start + j) * k];

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, k, col_count,
            C_blk.data(), k,
            C_expected.data(), k,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    /// Test CompositeOperator<Dense,Dense>::submatrix.
    /// Verify that (L*R)[rows, cols] * B equals the corresponding submatrix
    /// of the full product.
    template <typename T>
    void test_dense_dense_submatrix(int64_t m, int64_t p, int64_t n, int64_t k,
                                     int64_t row_start, int64_t col_start,
                                     int64_t row_count, int64_t col_count) {
        RNGState state(42);
        Layout layout = Layout::ColMajor;

        // L (m x p), R (p x n)
        vector<T> L_data = generate_dense_matrix<T>(m, p, layout, state);
        vector<T> R_data = generate_dense_matrix<T>(p, n, layout, state);
        // B (col_count x k), matching the submatrix column dimension
        vector<T> B(col_count * k);
        { RNGState s2(99); DenseDist D(col_count, k); RandBLAS::fill_dense(D, B.data(), s2); }

        RandLAPACK::linops::DenseLinOp<T> L_op(m, p, L_data.data(), m, layout);
        RandLAPACK::linops::DenseLinOp<T> R_op(p, n, R_data.data(), p, layout);
        RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

        // Block product: C_blk = (L*R)[rows,cols] * B  (row_count x k)
        auto blk = comp.submatrix(row_start, col_start, row_count, col_count);
        ASSERT_EQ(blk.n_rows, row_count);
        ASSERT_EQ(blk.n_cols, col_count);

        vector<T> C_blk(row_count * k, 0.0);
        blk(layout, Op::NoTrans, Op::NoTrans, row_count, k, col_count,
            (T)1.0, B.data(), col_count, (T)0.0, C_blk.data(), row_count);

        // Expected: materialize (L*R) as dense (m x n), extract submatrix, multiply by B
        vector<T> LR(m * n, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m, n, p,
                   (T)1.0, L_data.data(), m, R_data.data(), p,
                   (T)0.0, LR.data(), m);
        // Extract submatrix (row_count x col_count)
        vector<T> LR_sub(row_count * col_count);
        for (int64_t j = 0; j < col_count; ++j)
            for (int64_t i = 0; i < row_count; ++i)
                LR_sub[i + j * row_count] = LR[(row_start + i) + (col_start + j) * m];
        // C_expected = LR_sub * B
        vector<T> C_expected(row_count * k, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, row_count, k, col_count,
                   (T)1.0, LR_sub.data(), row_count, B.data(), col_count,
                   (T)0.0, C_expected.data(), row_count);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, k,
            C_blk.data(), row_count,
            C_expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    // ---------------------------------------------------------------
    // Sparse * Dense composition
    // ---------------------------------------------------------------

    /// Test CompositeOperator<Sparse(CSC),Dense>::row_block.
    template <typename T>
    void test_sparse_dense_row_block(int64_t m, int64_t p, int64_t n, int64_t k,
                                      T density, int64_t row_start, int64_t row_count) {
        RNGState state(42);
        Layout layout = Layout::ColMajor;

        auto L_csc = RandLAPACK::gen::gen_sparse_coo<T>(m, p, density, state).as_owning_csc();
        vector<T> R_data = generate_dense_matrix<T>(p, n, layout, state);
        vector<T> B(n * k);
        { RNGState s2(99); DenseDist D(n, k); RandBLAS::fill_dense(D, B.data(), s2); }

        RandLAPACK::linops::SparseLinOp L_op(m, p, L_csc);
        RandLAPACK::linops::DenseLinOp<T> R_op(p, n, R_data.data(), p, layout);
        RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

        // Full product
        vector<T> C_full(m * k, 0.0);
        comp(layout, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, B.data(), n, (T)0.0, C_full.data(), m);

        // Block product
        auto blk = comp.row_block(row_start, row_count);
        ASSERT_EQ(blk.n_rows, row_count);
        ASSERT_EQ(blk.n_cols, n);

        vector<T> C_blk(row_count * k, 0.0);
        blk(layout, Op::NoTrans, Op::NoTrans, row_count, k, n, (T)1.0, B.data(), n, (T)0.0, C_blk.data(), row_count);

        vector<T> C_expected(row_count * k);
        for (int64_t j = 0; j < k; ++j)
            for (int64_t i = 0; i < row_count; ++i)
                C_expected[i + j * row_count] = C_full[(row_start + i) + j * m];

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, k,
            C_blk.data(), row_count,
            C_expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    // ---------------------------------------------------------------
    // Dense * Sparse composition
    // ---------------------------------------------------------------

    /// Test CompositeOperator<Dense,Sparse(CSC)>::col_block.
    template <typename T>
    void test_dense_sparse_col_block(int64_t m, int64_t p, int64_t n, int64_t k,
                                      T density, int64_t col_start, int64_t col_count) {
        RNGState state(42);
        Layout layout = Layout::ColMajor;

        vector<T> L_data = generate_dense_matrix<T>(m, p, layout, state);
        auto R_csc = RandLAPACK::gen::gen_sparse_coo<T>(p, n, density, state).as_owning_csc();
        vector<T> B(k * m);
        { RNGState s2(99); DenseDist D(k, m); RandBLAS::fill_dense(D, B.data(), s2); }

        RandLAPACK::linops::DenseLinOp<T> L_op(m, p, L_data.data(), m, layout);
        RandLAPACK::linops::SparseLinOp R_op(p, n, R_csc);
        RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

        // Full product: C_full = B * (L*R)  (k x n)
        vector<T> C_full(k * n, 0.0);
        comp(Side::Right, layout, Op::NoTrans, Op::NoTrans, k, n, m,
             (T)1.0, B.data(), k, (T)0.0, C_full.data(), k);

        // Block product: C_blk = B * (L * R[:,cols])  (k x col_count)
        auto blk = comp.col_block(col_start, col_count);
        ASSERT_EQ(blk.n_rows, m);
        ASSERT_EQ(blk.n_cols, col_count);

        vector<T> C_blk(k * col_count, 0.0);
        blk(Side::Right, layout, Op::NoTrans, Op::NoTrans, k, col_count, m,
            (T)1.0, B.data(), k, (T)0.0, C_blk.data(), k);

        vector<T> C_expected(k * col_count);
        for (int64_t j = 0; j < col_count; ++j)
            for (int64_t i = 0; i < k; ++i)
                C_expected[i + j * k] = C_full[i + (col_start + j) * k];

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, k, col_count,
            C_blk.data(), k,
            C_expected.data(), k,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }

    /// Test CompositeOperator<Dense,Sparse(CSC)>::submatrix.
    template <typename T>
    void test_dense_sparse_submatrix(int64_t m, int64_t p, int64_t n, int64_t k,
                                      T density, int64_t row_start, int64_t col_start,
                                      int64_t row_count, int64_t col_count) {
        RNGState state(42);
        Layout layout = Layout::ColMajor;

        vector<T> L_data = generate_dense_matrix<T>(m, p, layout, state);
        auto R_csc = RandLAPACK::gen::gen_sparse_coo<T>(p, n, density, state).as_owning_csc();
        vector<T> B(col_count * k);
        { RNGState s2(99); DenseDist D(col_count, k); RandBLAS::fill_dense(D, B.data(), s2); }

        RandLAPACK::linops::DenseLinOp<T> L_op(m, p, L_data.data(), m, layout);
        RandLAPACK::linops::SparseLinOp R_op(p, n, R_csc);
        RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

        // Block product
        auto blk = comp.submatrix(row_start, col_start, row_count, col_count);
        ASSERT_EQ(blk.n_rows, row_count);
        ASSERT_EQ(blk.n_cols, col_count);

        vector<T> C_blk(row_count * k, 0.0);
        blk(layout, Op::NoTrans, Op::NoTrans, row_count, k, col_count,
            (T)1.0, B.data(), col_count, (T)0.0, C_blk.data(), row_count);

        // Expected: materialize (L*R) as dense, extract submatrix, multiply
        // Densify R_csc
        vector<T> R_dense(p * n, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(R_csc, layout, R_dense.data());
        vector<T> LR(m * n, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m, n, p,
                   (T)1.0, L_data.data(), m, R_dense.data(), p,
                   (T)0.0, LR.data(), m);
        vector<T> LR_sub(row_count * col_count);
        for (int64_t j = 0; j < col_count; ++j)
            for (int64_t i = 0; i < row_count; ++i)
                LR_sub[i + j * row_count] = LR[(row_start + i) + (col_start + j) * m];
        vector<T> C_expected(row_count * k, 0.0);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, row_count, k, col_count,
                   (T)1.0, LR_sub.data(), row_count, B.data(), col_count,
                   (T)0.0, C_expected.data(), row_count);

        T atol = 100 * std::numeric_limits<T>::epsilon();
        T rtol = 10 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, row_count, k,
            C_blk.data(), row_count,
            C_expected.data(), row_count,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol), "");
    }
};

// --- Dense*Dense row_block ---
TEST_F(TestCompositeBlockViews, dense_dense_row_block_middle) {
    test_dense_dense_row_block<double>(20, 12, 15, 8, 5, 8);
}

TEST_F(TestCompositeBlockViews, dense_dense_row_block_first) {
    test_dense_dense_row_block<double>(20, 12, 15, 8, 0, 6);
}

TEST_F(TestCompositeBlockViews, dense_dense_row_block_last) {
    test_dense_dense_row_block<double>(20, 12, 15, 8, 14, 6);
}

// --- Dense*Dense col_block ---
TEST_F(TestCompositeBlockViews, dense_dense_col_block_middle) {
    test_dense_dense_col_block<double>(20, 12, 15, 8, 4, 7);
}

TEST_F(TestCompositeBlockViews, dense_dense_col_block_first) {
    test_dense_dense_col_block<double>(20, 12, 15, 8, 0, 5);
}

TEST_F(TestCompositeBlockViews, dense_dense_col_block_last) {
    test_dense_dense_col_block<double>(20, 12, 15, 8, 10, 5);
}

// --- Dense*Dense submatrix ---
TEST_F(TestCompositeBlockViews, dense_dense_submatrix_middle) {
    test_dense_dense_submatrix<double>(20, 12, 15, 8, 3, 4, 10, 7);
}

TEST_F(TestCompositeBlockViews, dense_dense_submatrix_corner) {
    test_dense_dense_submatrix<double>(20, 12, 15, 8, 0, 0, 8, 6);
}

TEST_F(TestCompositeBlockViews, dense_dense_submatrix_end) {
    test_dense_dense_submatrix<double>(20, 12, 15, 8, 12, 9, 8, 6);
}

// --- Sparse*Dense row_block ---
TEST_F(TestCompositeBlockViews, sparse_dense_row_block_middle) {
    test_sparse_dense_row_block<double>(20, 12, 15, 8, 0.3, 5, 8);
}

TEST_F(TestCompositeBlockViews, sparse_dense_row_block_first) {
    test_sparse_dense_row_block<double>(20, 12, 15, 8, 0.3, 0, 6);
}

TEST_F(TestCompositeBlockViews, sparse_dense_row_block_last) {
    test_sparse_dense_row_block<double>(20, 12, 15, 8, 0.3, 14, 6);
}

// --- Dense*Sparse col_block ---
TEST_F(TestCompositeBlockViews, dense_sparse_col_block_middle) {
    test_dense_sparse_col_block<double>(20, 12, 15, 8, 0.3, 4, 7);
}

TEST_F(TestCompositeBlockViews, dense_sparse_col_block_first) {
    test_dense_sparse_col_block<double>(20, 12, 15, 8, 0.3, 0, 5);
}

TEST_F(TestCompositeBlockViews, dense_sparse_col_block_last) {
    test_dense_sparse_col_block<double>(20, 12, 15, 8, 0.3, 10, 5);
}

// --- Dense*Sparse submatrix ---
TEST_F(TestCompositeBlockViews, dense_sparse_submatrix_middle) {
    test_dense_sparse_submatrix<double>(20, 12, 15, 8, 0.3, 3, 4, 10, 7);
}

TEST_F(TestCompositeBlockViews, dense_sparse_submatrix_corner) {
    test_dense_sparse_submatrix<double>(20, 12, 15, 8, 0.3, 0, 0, 8, 6);
}

TEST_F(TestCompositeBlockViews, dense_sparse_submatrix_end) {
    test_dense_sparse_submatrix<double>(20, 12, 15, 8, 0.3, 12, 9, 8, 6);
}
