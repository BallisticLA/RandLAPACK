#include "rl_sym_linops.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
namespace sparse = RandBLAS::sparse_data;
using blas::Layout;
using RNG = r123::Philox4x32;

template <class SpMat>
class TestSparseSymLinOp : public ::testing::Test {
protected:
    using T = typename SpMat::scalar_t;
    static constexpr int64_t n = 5, k = 2;
    // Both triangles, off-diagonal couplings, and an empty last row/column.
    std::vector<T> A = {4, 1, 0, 0, 0, 1, -2, 3, 0, 0, 0, 3, 5, -1, 0,
                       0, 0, -1, 2, 0, 0, 0, 0, 0, 0};

    SpMat matrix() {
        sparse::COOMatrix<T> coo(n, n);
        sparse::coo::dense_to_coo(Layout::ColMajor, A.data(), T(0), coo);
        if constexpr (std::is_same_v<SpMat, sparse::COOMatrix<T>>) return coo;
        else if constexpr (std::is_same_v<SpMat, sparse::CSRMatrix<T>>) return coo.as_owning_csr();
        else return coo.as_owning_csc();
    }

    static int64_t index(Layout layout, int64_t i, int64_t j, int64_t ld) {
        return layout == Layout::ColMajor ? i + j * ld : i * ld + j;
    }

    // Scalar summation is independent of the sparse multiplication kernels.
    void check_product(Layout layout, T alpha, const T* B, int64_t ldb, T beta,
                       const std::vector<T>& before, const std::vector<T>& C, int64_t ldc) {
        std::vector<T> expected = before;
        for (int64_t j = 0; j < k; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                T sum = 0;
                for (int64_t p = 0; p < n; ++p) sum += A[i + p * n] * B[index(layout, p, j, ldb)];
                const auto pos = index(layout, i, j, ldc);
                expected[pos] = alpha * sum + (beta == T(0) ? T(0) : beta * before[pos]);
            }
        }
        for (size_t pos = 0; pos < C.size(); ++pos) {
            ASSERT_TRUE(std::isfinite(C[pos])) << "buffer entry " << pos;
            EXPECT_NEAR(C[pos], expected[pos], 64 * std::numeric_limits<T>::epsilon()
                        * std::max(T(1), std::abs(expected[pos]))) << "buffer entry " << pos;
        }
    }
};

using SparseTypes = ::testing::Types<sparse::COOMatrix<float>, sparse::CSRMatrix<float>,
                                    sparse::CSCMatrix<float>, sparse::COOMatrix<double>,
                                    sparse::CSRMatrix<double>, sparse::CSCMatrix<double>>;
TYPED_TEST_SUITE(TestSparseSymLinOp, SparseTypes);

TYPED_TEST(TestSparseSymLinOp, DenseProductsLayoutsStridesAndScaling) {
    using T = typename TestFixture::T;
    constexpr auto n = TestFixture::n, k = TestFixture::k;
    auto A = this->matrix();
    RandLAPACK::linops::SparseSymLinOp<T, TypeParam> op(A);
    static_assert(RandLAPACK::linops::SymmetricLinearOperator<decltype(op)>);
    EXPECT_EQ(op.dim, n);
    EXPECT_EQ(op.m, n);
    for (auto layout : {Layout::ColMajor, Layout::RowMajor}) {
        const int64_t minor = layout == Layout::ColMajor ? k : n;
        const int64_t ldb = (layout == Layout::ColMajor ? n : k) + 1, ldc = ldb + 1;
        for (T alpha : {T(-1.25), T(0)}) {
            for (T beta : {T(0), T(-0.5), T(1)}) {
                SCOPED_TRACE(::testing::Message() << "layout=" << char(layout)
                                                 << " alpha=" << alpha << " beta=" << beta);
                std::vector<T> B(ldb * minor, T(91)), C(ldc * minor, T(117));
                for (int64_t j = 0; j < k; ++j) {
                    for (int64_t i = 0; i < n; ++i) {
                        B[this->index(layout, i, j, ldb)] = T(i - 2 * j) / T(4);
                        C[this->index(layout, i, j, ldc)] = beta == T(0)
                            ? std::numeric_limits<T>::quiet_NaN() : T(2 * i + j + 1);
                    }
                }
                const auto before = C, original_B = B;
                op(layout, k, alpha, B.data(), ldb, beta, C.data(), ldc);
                this->check_product(layout, alpha, B.data(), ldb, beta, before, C, ldc);
                EXPECT_EQ(B, original_B);
            }
        }
    }
}

TYPED_TEST(TestSparseSymLinOp, DenseSketchesBothLayouts) {
    using T = typename TestFixture::T;
    constexpr auto n = TestFixture::n, k = TestFixture::k;
    auto A = this->matrix();
    RandLAPACK::linops::SparseSymLinOp<T, TypeParam> op(A);
    for (auto axis : {RandBLAS::Axis::Long, RandBLAS::Axis::Short}) {
        for (bool prefill : {false, true}) {
            RandBLAS::RNGState<RNG> state(17);
            RandBLAS::DenseDist distribution(n, k, RandBLAS::ScalarDist::Gaussian, axis);
            RandBLAS::DenseSkOp<T, RNG> S(distribution, state);
            if (prefill) RandBLAS::fill_dense(S);
            const auto layout = S.layout;
            const int64_t ldc = distribution.dim_major + 2;
            std::vector<T> C(ldc * distribution.dim_minor, T(3));
            const auto before = C;
            op(layout, k, T(0.75), S, T(-0.25), C.data(), ldc);
            ASSERT_NE(S.buff, nullptr);
            this->check_product(layout, T(0.75), S.buff, distribution.dim_major,
                                T(-0.25), before, C, ldc);
        }
    }
}

TYPED_TEST(TestSparseSymLinOp, SparseSketchBackendContract) {
    using T = typename TestFixture::T;
    constexpr auto n = TestFixture::n, k = TestFixture::k;
    auto A = this->matrix();
    RandLAPACK::linops::SparseSymLinOp<T, TypeParam> op(A);
    for (bool prefill : {false, true}) {
        RandBLAS::RNGState<RNG> state(29);
        RandBLAS::SparseDist distribution(n, k, 1);
        RandBLAS::SparseSkOp<T, RNG> S(distribution, state);
        if (prefill) RandBLAS::fill_sparse(S);
        const int64_t ldc = n + 2;
        std::vector<T> C(ldc * k, T(3));
        const auto before = C;
#if defined(RandBLAS_HAS_MKL)
        op(Layout::ColMajor, k, T(-1.5), S, T(0.25), C.data(), ldc);
        ASSERT_GT(S.nnz, 0);
        auto coo = RandBLAS::coo_view_of_skop(S);
        std::vector<T> dense_S(n * k, T(0));
        for (int64_t p = 0; p < coo.nnz; ++p)
            dense_S[coo.rows[p] + coo.cols[p] * n] += coo.vals[p];
        this->check_product(Layout::ColMajor, T(-1.5), dense_S.data(), n,
                            T(0.25), before, C, ldc);
#else
        EXPECT_THROW(op(Layout::ColMajor, k, T(1), S, T(0), C.data(), ldc), RandLAPACK::Error);
        EXPECT_EQ(C, before);
#endif
    }
}

TYPED_TEST(TestSparseSymLinOp, ElementAccessAndInputContracts) {
    using T = typename TestFixture::T;
    constexpr auto n = TestFixture::n, k = TestFixture::k;
    auto A = this->matrix();
    RandLAPACK::linops::SparseSymLinOp<T, TypeParam> op(A);
    if constexpr (std::is_same_v<TypeParam, sparse::CSCMatrix<T>>) {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i) EXPECT_EQ(op(i, j), this->A[i + j * n]);
    } else {
        EXPECT_THROW(op(0, 0), RandLAPACK::Error);
    }
    std::vector<T> B(n * k, T(1)), C(n * k, T(0));
    for (auto layout : {Layout::ColMajor, Layout::RowMajor}) {
        const int64_t ld = layout == Layout::ColMajor ? n : k;
        EXPECT_THROW(op(layout, k, T(1), B.data(), ld - 1, T(0), C.data(), ld), RandLAPACK::Error);
        EXPECT_THROW(op(layout, k, T(1), B.data(), ld, T(0), C.data(), ld - 1), RandLAPACK::Error);
    }
    TypeParam rectangular(n, n + 1);
    EXPECT_THROW((RandLAPACK::linops::SparseSymLinOp<T, TypeParam>(rectangular)), RandLAPACK::Error);
    sparse::reindex_inplace(A, sparse::IndexBase::One);
    EXPECT_THROW((RandLAPACK::linops::SparseSymLinOp<T, TypeParam>(A)), RandLAPACK::Error);
}
} // namespace
