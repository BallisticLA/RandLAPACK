// Include the public helper first to verify its own dependencies.
#include "RandLAPACK/linops/rl_svd_residual.hh"
#include "RandLAPACK/linops/rl_dense_linop.hh"
#include "RandLAPACK/linops/rl_sparse_linop.hh"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

using namespace RandLAPACK::linops;

template <typename T>
class SvdResidualTest : public ::testing::Test {
protected:
    static constexpr int64_t m = 4, n = 3, k = 3;
    std::array<T, m * n> matrix = {8, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0};
    std::array<T, m * k> u = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::array<T, n * k> v = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::array<T, k> sigma = {8, 4, 2};
    T tolerance = T(64) * std::numeric_limits<T>::epsilon();

    DenseLinOp<T> dense() {
        return DenseLinOp<T>(m, n, matrix.data(), m, blas::Layout::ColMajor);
    }

    void perturb() {
        matrix[4] = T(0.5);
        matrix[11] = T(-0.75);
        u[1] = T(0.125);
        u[7] = T(-0.25);
        v[2] = T(0.25);
        v[3] = T(-0.125);
        sigma[2] = T(1.5);
    }

    struct Reference {
        T normalized;
        T one_sided;
        T absolute;
        std::array<T, k> per_triplet;
    };

    // Evaluate both equations with scalar loops and long-double accumulators.
    // This reference does not call BLAS or the residual helpers.
    Reference reference() const {
        long double normalized_sq = 0, one_sided_sq = 0, absolute_sq = 0;
        Reference result{};
        for (int64_t j = 0; j < k; ++j) {
            long double left_sq = 0, right_sq = 0;
            for (int64_t row = 0; row < m; ++row) {
                long double residual = -static_cast<long double>(sigma[j]) * u[row + m * j];
                for (int64_t col = 0; col < n; ++col)
                    residual += static_cast<long double>(matrix[row + m * col]) * v[col + n * j];
                left_sq += residual * residual;
            }
            for (int64_t col = 0; col < n; ++col) {
                long double residual = -static_cast<long double>(sigma[j]) * v[col + n * j];
                for (int64_t row = 0; row < m; ++row)
                    residual += static_cast<long double>(matrix[row + m * col]) * u[row + m * j];
                right_sq += residual * residual;
            }
            long double sigma_sq = static_cast<long double>(sigma[j]) * sigma[j];
            result.per_triplet[j] = static_cast<T>(std::sqrt((left_sq + right_sq) / sigma_sq));
            normalized_sq += (left_sq + right_sq) / sigma_sq;
            one_sided_sq += left_sq / sigma_sq;
            absolute_sq += left_sq + right_sq;
        }
        result.normalized = static_cast<T>(std::sqrt(normalized_sq));
        result.one_sided = static_cast<T>(std::sqrt(one_sided_sq));
        result.absolute = static_cast<T>(std::sqrt(absolute_sq));
        return result;
    }
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SvdResidualTest, ScalarTypes);

TYPED_TEST(SvdResidualTest, ExactDiagonalSvd) {
    using T = TypeParam;
    auto a = this->dense();
    auto u = this->u.data(), v = this->v.data(), sigma = this->sigma.data();
    EXPECT_EQ(svd_residual<T>(a, u, v, sigma, this->k), T(0));
    auto all = svd_residual_all<T>(a, u, v, sigma, this->k);
    EXPECT_EQ(all.two_sided_normalized, T(0));
    EXPECT_EQ(all.one_sided_normalized, T(0));
    EXPECT_EQ(all.two_sided_absolute, T(0));
    std::array<T, SvdResidualTest<T>::k> per_triplet;
    svd_residual_per_triplet<T>(a, u, v, sigma, this->k, per_triplet.data());
    for (T value : per_triplet)
        EXPECT_EQ(value, T(0));
    EXPECT_EQ(svd_triplets_certified<T>(a, u, v, sigma, this->k, T(0)), this->k);
}

TYPED_TEST(SvdResidualTest, PerturbedResidualsMatchIndependentReference) {
    using T = TypeParam;
    this->perturb();
    auto expected = this->reference();
    auto a = this->dense();
    auto u = this->u.data(), v = this->v.data(), sigma = this->sigma.data();
    EXPECT_NEAR(svd_residual<T>(a, u, v, sigma, this->k), expected.normalized, this->tolerance);
    auto all = svd_residual_all<T>(a, u, v, sigma, this->k);
    EXPECT_NEAR(all.two_sided_normalized, expected.normalized, this->tolerance);
    EXPECT_NEAR(all.one_sided_normalized, expected.one_sided, this->tolerance);
    EXPECT_NEAR(all.two_sided_absolute, expected.absolute, this->tolerance);
    std::array<T, SvdResidualTest<T>::k> per_triplet;
    svd_residual_per_triplet<T>(a, u, v, sigma, this->k, per_triplet.data());
    T sum_sq = 0;
    for (int64_t j = 0; j < this->k; ++j) {
        EXPECT_NEAR(per_triplet[j], expected.per_triplet[j], this->tolerance);
        sum_sq += per_triplet[j] * per_triplet[j];
    }
    EXPECT_NEAR(std::sqrt(sum_sq), all.two_sided_normalized, this->tolerance);
    // The independent reference gives residuals on both sides of this threshold.
    int64_t expected_count = 0;
    for (T value : expected.per_triplet)
        expected_count += value <= T(0.4);
    ASSERT_GT(expected_count, 0);
    ASSERT_LT(expected_count, this->k);
    EXPECT_EQ(svd_triplets_certified<T>(a, u, v, sigma, this->k, T(0.4)), expected_count);
}

TYPED_TEST(SvdResidualTest, PositiveSubnormalSingularValue) {
    using T = TypeParam;
    T sigma = std::numeric_limits<T>::min() / T(16);
    ASSERT_GT(sigma, T(0));
    T u = T(1), v = T(1);
    DenseLinOp<T> a(1, 1, &sigma, 1, blas::Layout::ColMajor);
    EXPECT_EQ(svd_residual<T>(a, &u, &v, &sigma, 1), T(0));
    auto all = svd_residual_all<T>(a, &u, &v, &sigma, 1);
    EXPECT_EQ(all.two_sided_normalized, T(0));
    EXPECT_EQ(all.one_sided_normalized, T(0));
    EXPECT_EQ(all.two_sided_absolute, T(0));
    T per_triplet;
    svd_residual_per_triplet<T>(a, &u, &v, &sigma, 1, &per_triplet);
    EXPECT_EQ(per_triplet, T(0));
    v = T(2);
    EXPECT_NEAR(svd_residual<T>(a, &u, &v, &sigma, 1), std::sqrt(T(2)), this->tolerance);
    all = svd_residual_all<T>(a, &u, &v, &sigma, 1);
    EXPECT_NEAR(all.two_sided_normalized, std::sqrt(T(2)), this->tolerance);
    EXPECT_NEAR(all.one_sided_normalized, T(1), this->tolerance);
    svd_residual_per_triplet<T>(a, &u, &v, &sigma, 1, &per_triplet);
    EXPECT_NEAR(per_triplet, std::sqrt(T(2)), this->tolerance);
}

TYPED_TEST(SvdResidualTest, MixedNormalAndSubnormalSingularValues) {
    using T = TypeParam;
    // Keep two ordinary triplets and perturb the subnormal one in a rectangular
    // matrix. Its residual columns contain both zero and nonzero entries.
    const T small = std::numeric_limits<T>::min() / T(16);
    ASSERT_GT(small, T(0));
    this->matrix[10] = small;
    this->sigma[2] = small;
    this->v[8] = T(2);
    auto a = this->dense();
    auto u = this->u.data(), v = this->v.data(), sigma = this->sigma.data();
    EXPECT_NEAR(svd_residual<T>(a, u, v, sigma, this->k), std::sqrt(T(2)), this->tolerance);
    auto all = svd_residual_all<T>(a, u, v, sigma, this->k);
    EXPECT_NEAR(all.two_sided_normalized, std::sqrt(T(2)), this->tolerance);
    EXPECT_NEAR(all.one_sided_normalized, T(1), this->tolerance);
    // Compare on a unit scale so an absolute tolerance cannot hide underflow.
    EXPECT_NEAR(all.two_sided_absolute / small, std::sqrt(T(2)), this->tolerance);
    std::array<T, SvdResidualTest<T>::k> per_triplet;
    svd_residual_per_triplet<T>(a, u, v, sigma, this->k, per_triplet.data());
    EXPECT_EQ(per_triplet[0], T(0));
    EXPECT_EQ(per_triplet[1], T(0));
    EXPECT_NEAR(per_triplet[2], std::sqrt(T(2)), this->tolerance);
    EXPECT_EQ(svd_triplets_certified<T>(a, u, v, sigma, this->k, T(1)), 2);
}

TYPED_TEST(SvdResidualTest, LargeSingularValueNormalizedResidual) {
    using T = TypeParam;
    T sigma = std::numeric_limits<T>::max() * T(0.75);
    T entry = T(0), u = T(1), v = T(1);
    DenseLinOp<T> a(1, 1, &entry, 1, blas::Layout::ColMajor);
    EXPECT_NEAR(svd_residual<T>(a, &u, &v, &sigma, 1), std::sqrt(T(2)), this->tolerance);
    auto all = svd_residual_all<T>(a, &u, &v, &sigma, 1);
    EXPECT_NEAR(all.two_sided_normalized, std::sqrt(T(2)), this->tolerance);
    EXPECT_NEAR(all.one_sided_normalized, T(1), this->tolerance);
    T per_triplet;
    svd_residual_per_triplet<T>(a, &u, &v, &sigma, 1, &per_triplet);
    EXPECT_NEAR(per_triplet, std::sqrt(T(2)), this->tolerance);
    EXPECT_EQ(svd_triplets_certified<T>(a, &u, &v, &sigma, 1, T(2)), 1);
}

TYPED_TEST(SvdResidualTest, NonpositiveSingularValues) {
    using T = TypeParam;
    auto a = this->dense();
    auto u = this->u.data(), v = this->v.data(), sigma = this->sigma.data();
    for (T invalid : {T(0), T(-1)}) {
        this->sigma[2] = invalid;
        EXPECT_EQ(svd_residual<T>(a, u, v, sigma, this->k), std::numeric_limits<T>::infinity());
        auto all = svd_residual_all<T>(a, u, v, sigma, this->k);
        EXPECT_EQ(all.two_sided_normalized, std::numeric_limits<T>::infinity());
        EXPECT_EQ(all.one_sided_normalized, std::numeric_limits<T>::infinity());
        EXPECT_EQ(all.two_sided_absolute, std::numeric_limits<T>::infinity());
        std::array<T, SvdResidualTest<T>::k> per_triplet;
        svd_residual_per_triplet<T>(a, u, v, sigma, this->k, per_triplet.data());
        EXPECT_EQ(per_triplet[0], T(0));
        EXPECT_EQ(per_triplet[1], T(0));
        EXPECT_EQ(per_triplet[2], std::numeric_limits<T>::infinity());
        EXPECT_EQ(svd_triplets_certified<T>(a, u, v, sigma, this->k, T(0)), 2);
    }
    // The per-triplet helper also handles nonpositive values before positive ones.
    this->sigma = {T(0), T(4), T(-1)};
    std::array<T, SvdResidualTest<T>::k> per_triplet;
    svd_residual_per_triplet<T>(a, u, v, sigma, this->k, per_triplet.data());
    EXPECT_EQ(per_triplet[0], std::numeric_limits<T>::infinity());
    EXPECT_EQ(per_triplet[1], T(0));
    EXPECT_EQ(per_triplet[2], std::numeric_limits<T>::infinity());
    EXPECT_EQ(svd_triplets_certified<T>(a, u, v, sigma, this->k, T(0)), 1);
}

TYPED_TEST(SvdResidualTest, EmptyInputDoesNotDereferenceBuffers) {
    using T = TypeParam;
    auto a = this->dense();
    EXPECT_EQ(svd_residual<T>(a, nullptr, nullptr, nullptr, 0), std::numeric_limits<T>::infinity());
    auto all = svd_residual_all<T>(a, nullptr, nullptr, nullptr, 0);
    EXPECT_EQ(all.two_sided_normalized, std::numeric_limits<T>::infinity());
    EXPECT_EQ(all.one_sided_normalized, std::numeric_limits<T>::infinity());
    EXPECT_EQ(all.two_sided_absolute, std::numeric_limits<T>::infinity());
    T sentinel = T(123);
    svd_residual_per_triplet<T>(a, nullptr, nullptr, nullptr, 0, &sentinel);
    EXPECT_EQ(sentinel, T(123));
    EXPECT_EQ(svd_triplets_certified<T>(a, nullptr, nullptr, nullptr, 0, T(0)), 0);
}

TYPED_TEST(SvdResidualTest, DenseSparseEquivalenceAndInputPreservation) {
    using T = TypeParam;
    this->perturb();
    const auto matrix_before = this->matrix, u_before = this->u;
    const auto v_before = this->v;
    const auto sigma_before = this->sigma;
    std::array<T, 5> values = {8, T(0.5), 4, 2, T(-0.75)};
    std::array<int64_t, 5> rows = {0, 0, 1, 2, 3};
    std::array<int64_t, 4> colptr = {0, 1, 3, 5};
    const auto values_before = values;
    const auto rows_before = rows;
    const auto colptr_before = colptr;
    using CSC = RandBLAS::sparse_data::CSCMatrix<T>;
    CSC matrix(this->m, this->n, 5, values.data(), rows.data(), colptr.data());
    SparseLinOp<CSC> sparse(this->m, this->n, matrix);
    auto dense = this->dense();
    auto u = this->u.data(), v = this->v.data(), sigma = this->sigma.data();
    EXPECT_NEAR(svd_residual<T>(dense, u, v, sigma, this->k),
                svd_residual<T>(sparse, u, v, sigma, this->k), this->tolerance);
    auto dense_all = svd_residual_all<T>(dense, u, v, sigma, this->k);
    auto sparse_all = svd_residual_all<T>(sparse, u, v, sigma, this->k);
    EXPECT_NEAR(dense_all.two_sided_normalized, sparse_all.two_sided_normalized, this->tolerance);
    EXPECT_NEAR(dense_all.one_sided_normalized, sparse_all.one_sided_normalized, this->tolerance);
    EXPECT_NEAR(dense_all.two_sided_absolute, sparse_all.two_sided_absolute, this->tolerance);
    std::array<T, SvdResidualTest<T>::k> dense_per_triplet, sparse_per_triplet;
    svd_residual_per_triplet<T>(dense, u, v, sigma, this->k, dense_per_triplet.data());
    svd_residual_per_triplet<T>(sparse, u, v, sigma, this->k, sparse_per_triplet.data());
    for (int64_t j = 0; j < this->k; ++j)
        EXPECT_NEAR(dense_per_triplet[j], sparse_per_triplet[j], this->tolerance);
    EXPECT_EQ(svd_triplets_certified<T>(dense, u, v, sigma, this->k, T(0.4)),
              svd_triplets_certified<T>(sparse, u, v, sigma, this->k, T(0.4)));
    EXPECT_EQ(this->matrix, matrix_before);
    EXPECT_EQ(this->u, u_before);
    EXPECT_EQ(this->v, v_before);
    EXPECT_EQ(this->sigma, sigma_before);
    EXPECT_EQ(values, values_before);
    EXPECT_EQ(rows, rows_before);
    EXPECT_EQ(colptr, colptr_before);
}

TYPED_TEST(SvdResidualTest, ResidualQualificationDoesNotCertifySpectralContent) {
    using T = TypeParam;
    auto a = this->dense();
    // Two identical, unnormalized triplets and a zero-vector pair all satisfy
    // the equations, though sigma=3 is not a singular value of this matrix.
    this->u.fill(T(0));
    this->v.fill(T(0));
    this->u[0] = this->u[this->m] = T(2);
    this->v[0] = this->v[this->n] = T(2);
    this->sigma = {T(8), T(8), T(3)};
    EXPECT_EQ(svd_triplets_certified<T>(a, this->u.data(), this->v.data(),
                                      this->sigma.data(), this->k, T(0)), this->k);
}

} // namespace
