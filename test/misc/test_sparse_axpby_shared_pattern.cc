// Tests for RandLAPACK_extras::sparse_axpby_shared_pattern: C := alpha*A + beta*B
// for two CSRMatrix operands that share an identical sparsity pattern.

#include "extras/misc/ext_sparse_axpy.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <vector>

using RandBLAS::sparse_data::CSRMatrix;
using RandBLAS::sparse_data::IndexBase;

namespace {

// 3x3 matrix, nnz = 6: row0 -> cols {0,2}, row1 -> col {1}, row2 -> cols {0,1,2}.
CSRMatrix<double, int64_t> make_csr(const std::vector<double>& vals) {
    CSRMatrix<double, int64_t> M(3, 3);
    M.reserve(6);
    int64_t rowptr[4]  = {0, 2, 3, 6};
    int64_t colidxs[6] = {0, 2, 1, 0, 1, 2};
    std::copy(rowptr, rowptr + 4, M.rowptr);
    std::copy(colidxs, colidxs + 6, M.colidxs);
    std::copy(vals.begin(), vals.end(), M.vals);
    return M;
}

} // namespace

TEST(TestSparseAxpbySharedPattern, values_fused_pattern_copied) {
    auto A = make_csr({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto B = make_csr({10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
    const double alpha = 2.0, beta = 0.5;

    auto C = RandLAPACK_extras::sparse_axpby_shared_pattern(alpha, A, beta, B);

    ASSERT_EQ(C.n_rows, A.n_rows);
    ASSERT_EQ(C.n_cols, A.n_cols);
    ASSERT_EQ(C.nnz, A.nnz);
    ASSERT_EQ(C.index_base, A.index_base);
    for (int64_t i = 0; i <= A.n_rows; ++i) ASSERT_EQ(C.rowptr[i], A.rowptr[i]);
    for (int64_t i = 0; i < A.nnz; ++i)     ASSERT_EQ(C.colidxs[i], A.colidxs[i]);
    for (int64_t i = 0; i < A.nnz; ++i)
        ASSERT_DOUBLE_EQ(C.vals[i], alpha * A.vals[i] + beta * B.vals[i]);
}

TEST(TestSparseAxpbySharedPattern, mismatched_colidxs_throws) {
    auto A = make_csr({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto B = make_csr({10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
    B.colidxs[5] = 0;   // was 2; same nnz/dims, different structural pattern

    ASSERT_THROW(RandLAPACK_extras::sparse_axpby_shared_pattern(1.0, A, 1.0, B), RandBLAS::Error);
}

TEST(TestSparseAxpbySharedPattern, mismatched_dims_throws) {
    auto A = make_csr({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    CSRMatrix<double, int64_t> B(4, 3);   // n_rows differs from A
    B.reserve(6);
    int64_t rowptr[5]  = {0, 2, 3, 6, 6};
    int64_t colidxs[6] = {0, 2, 1, 0, 1, 2};
    std::copy(rowptr, rowptr + 5, B.rowptr);
    std::copy(colidxs, colidxs + 6, B.colidxs);
    std::fill(B.vals, B.vals + 6, 1.0);

    ASSERT_THROW(RandLAPACK_extras::sparse_axpby_shared_pattern(1.0, A, 1.0, B), RandBLAS::Error);
}

TEST(TestSparseAxpbySharedPattern, mismatched_index_base_throws) {
    auto A = make_csr({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto B = make_csr({10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
    B.index_base = IndexBase::One;   // same pattern values, different base flag

    ASSERT_THROW(RandLAPACK_extras::sparse_axpby_shared_pattern(1.0, A, 1.0, B), RandBLAS::Error);
}
