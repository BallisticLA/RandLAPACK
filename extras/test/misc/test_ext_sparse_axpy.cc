// Tests for sparse_axpby_shared_pattern in extras/misc/ext_sparse_axpy.hh.

#include "../../misc/ext_sparse_axpy.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using std::vector;


// Build a small CSR matrix (n_rows × n_cols, nnz given) by deep-copying user-provided
// rowptr/colidxs/vals arrays. The matrix owns its memory and frees on destruction.
template <typename T, typename sint_t>
RandBLAS::sparse_data::CSRMatrix<T, sint_t> make_owned_csr(
    int64_t n_rows, int64_t n_cols, int64_t nnz,
    const sint_t* rowptr_src, const sint_t* colidxs_src, const T* vals_src)
{
    RandBLAS::sparse_data::CSRMatrix<T, sint_t> M(n_rows, n_cols);
    M.reserve(nnz);
    for (int64_t i = 0; i <= n_rows; ++i) M.rowptr[i] = rowptr_src[i];
    for (int64_t i = 0; i < nnz; ++i)     M.colidxs[i] = colidxs_src[i];
    for (int64_t i = 0; i < nnz; ++i)     M.vals[i]    = vals_src[i];
    return M;
}


TEST(TestSparseAxpby, shared_pattern_simple) {
    // 3x3 tridiagonal pattern.
    //   row 0: cols 0, 1
    //   row 1: cols 0, 1, 2
    //   row 2: cols 1, 2
    // nnz = 7
    int rowptr[4]   = {0, 2, 5, 7};
    int colidxs[7]  = {0, 1, 0, 1, 2, 1, 2};
    double A_vals[7] = {1, 2, 3, 4, 5, 6, 7};
    double B_vals[7] = {10, 20, 30, 40, 50, 60, 70};

    auto A = make_owned_csr<double, int>(3, 3, 7, rowptr, colidxs, A_vals);
    auto B = make_owned_csr<double, int>(3, 3, 7, rowptr, colidxs, B_vals);

    // C := 2.0 * A + (-3.0) * B
    auto C = RandLAPACK_extras::sparse_axpby_shared_pattern<double, int>(
        2.0, A, -3.0, B);

    ASSERT_EQ(C.n_rows, 3);
    ASSERT_EQ(C.n_cols, 3);
    ASSERT_EQ(C.nnz, 7);

    for (int i = 0; i <= 3; ++i) EXPECT_EQ(C.rowptr[i],  rowptr[i]);
    for (int i = 0; i < 7; ++i)  EXPECT_EQ(C.colidxs[i], colidxs[i]);

    double expected[7];
    for (int i = 0; i < 7; ++i)
        expected[i] = 2.0 * A_vals[i] + (-3.0) * B_vals[i];

    for (int i = 0; i < 7; ++i)
        EXPECT_NEAR(C.vals[i], expected[i], 1e-14);
}


TEST(TestSparseAxpby, shifted_inverse_pattern) {
    // Mimics the rspec use case: X = K - omega * M with K, M sharing sparsity.
    // Symmetric 4x4 pattern with 10 nonzeros.
    int rowptr[5]    = {0, 3, 6, 8, 10};
    int colidxs[10]  = {0, 1, 2,  0, 1, 3,  0, 2,  1, 3};
    double K_vals[10] = {4, 1, 1,  1, 3, 1,  1, 2,  1, 5};
    double M_vals[10] = {2, 0.5, 0.5,  0.5, 2, 0.5,  0.5, 1,  0.5, 2};

    auto K = make_owned_csr<double, int>(4, 4, 10, rowptr, colidxs, K_vals);
    auto M = make_owned_csr<double, int>(4, 4, 10, rowptr, colidxs, M_vals);

    double omega = 0.7;
    auto X = RandLAPACK_extras::sparse_axpby_shared_pattern<double, int>(
        1.0, K, -omega, M);

    for (int i = 0; i < 10; ++i) {
        double expected = K_vals[i] - omega * M_vals[i];
        EXPECT_NEAR(X.vals[i], expected, 1e-14);
    }
    ASSERT_EQ(X.nnz, 10);
    for (int i = 0; i <= 4; ++i) EXPECT_EQ(X.rowptr[i],  rowptr[i]);
    for (int i = 0; i < 10; ++i) EXPECT_EQ(X.colidxs[i], colidxs[i]);
}


TEST(TestSparseAxpby, float_type) {
    int rowptr[3]   = {0, 1, 2};
    int colidxs[2]  = {0, 1};
    float A_vals[2] = {1.5f, 2.5f};
    float B_vals[2] = {0.5f, 1.0f};

    auto A = make_owned_csr<float, int>(2, 2, 2, rowptr, colidxs, A_vals);
    auto B = make_owned_csr<float, int>(2, 2, 2, rowptr, colidxs, B_vals);

    auto C = RandLAPACK_extras::sparse_axpby_shared_pattern<float, int>(
        2.0f, A, 4.0f, B);

    EXPECT_NEAR(C.vals[0], 2.0f * 1.5f + 4.0f * 0.5f, 1e-6f);
    EXPECT_NEAR(C.vals[1], 2.0f * 2.5f + 4.0f * 1.0f, 1e-6f);
}
