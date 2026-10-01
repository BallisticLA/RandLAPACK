#include "rl_nystrom_evd.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

namespace {
using RNG = r123::Philox4x32;

template <typename T>
struct NystromResult {
    T* U = nullptr;
    T* lambda = nullptr;
    int64_t U_size = 0, lambda_size = 0;
    RandLAPACK::NystromEVD_workspace<T> workspace;
    ~NystromResult() { delete[] U; delete[] lambda; }
};

template <typename T>
class TestNystromEVD : public ::testing::Test {
protected:
    // A Householder similarity gives a non-diagonal PSD matrix with a known spectrum.
    static std::vector<T> matrix(const std::vector<T>& eigenvalues) {
        const int64_t n = eigenvalues.size();
        T norm_squared = 0;
        for (int64_t i = 0; i < n; ++i) norm_squared += T((i + 1) * (i + 1));
        std::vector<T> Q(n * n), A(n * n, T(0));
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i)
                Q[i + j * n] = T(i == j) - T(2 * (i + 1) * (j + 1)) / norm_squared;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i)
                for (int64_t p = 0; p < n; ++p)
                    A[i + j * n] += Q[i + p * n] * eigenvalues[p] * Q[j + p * n];
        return A;
    }

    static void check_basis(const NystromResult<T>& result, int64_t n, int64_t k) {
        for (int64_t j = 0; j < k; ++j) {
            ASSERT_TRUE(std::isfinite(result.lambda[j]));
            EXPECT_GE(result.lambda[j], T(0));
            if (j > 0) EXPECT_GE(result.lambda[j - 1], result.lambda[j]);
            for (int64_t i = 0; i <= j; ++i) {
                T dot = 0;
                for (int64_t p = 0; p < n; ++p) dot += result.U[p + i * n] * result.U[p + j * n];
                EXPECT_NEAR(dot, T(i == j), 100 * n * std::numeric_limits<T>::epsilon());
            }
        }
    }

    static void check_eigenpairs(const std::vector<T>& A, const std::vector<T>& spectrum,
                                const NystromResult<T>& result, int64_t k, T relative_tolerance) {
        const int64_t n = spectrum.size();
        check_basis(result, n, k);
        for (int64_t j = 0; j < k; ++j) {
            EXPECT_NEAR(result.lambda[j], spectrum[j], relative_tolerance * spectrum[0]);
            T residual_squared = 0;
            for (int64_t i = 0; i < n; ++i) {
                T residual = -result.lambda[j] * result.U[i + j * n];
                for (int64_t p = 0; p < n; ++p) residual += A[i + p * n] * result.U[p + j * n];
                residual_squared += residual * residual;
            }
            EXPECT_LE(std::sqrt(residual_squared), relative_tolerance * spectrum[0]);
        }
    }

    static void run(const std::vector<T>& A, int64_t k, int64_t q, int64_t vec_nnz,
                    RandBLAS::RNGState<RNG>& state, NystromResult<T>& result) {
        const int64_t n = static_cast<int64_t>(std::sqrt(A.size()));
        RandLAPACK::linops::ExplicitSymLinOp<T> op(n, blas::Uplo::Upper, A.data(), n, blas::Layout::ColMajor);
        RandLAPACK::NystromEVD<T>(op, k, q, vec_nnz, state, result.U, result.U_size,
                                result.lambda, result.lambda_size, result.workspace);
    }
};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TestNystromEVD, RealTypes);

TYPED_TEST(TestNystromEVD, FullRankRecovery) {
    using T = TypeParam;
    const std::vector<T> spectrum = {8, 4, 2, 1};
    const auto A = this->matrix(spectrum);
    for (int64_t q : {1, 2}) {
        RandBLAS::RNGState<RNG> state(7);
        NystromResult<T> result;
        ASSERT_NO_THROW(this->run(A, 4, q, -1, state, result));
        this->check_eigenpairs(A, spectrum, result, 4, T(5000) * std::numeric_limits<T>::epsilon());
        for (int64_t j = 0; j < 4; ++j) {
            for (int64_t i = 0; i < 4; ++i) {
                T reconstructed = 0;
                for (int64_t p = 0; p < 4; ++p)
                    reconstructed += result.U[i + 4 * p] * result.lambda[p] * result.U[j + 4 * p];
                EXPECT_NEAR(reconstructed, A[i + 4 * j], T(5000) * std::numeric_limits<T>::epsilon() * spectrum[0]);
            }
        }
    }
}

TYPED_TEST(TestNystromEVD, RankDeficientSpectrumDenseAndSparseSketches) {
    using T = TypeParam;
    const std::vector<T> spectrum = {16, 8, 4, 0, 0, 0, 0, 0};
    const auto A = this->matrix(spectrum);
    for (int64_t vec_nnz : {-1, 0, 2}) {
        for (int64_t q : {1, 2}) {
            SCOPED_TRACE(::testing::Message() << "vec_nnz=" << vec_nnz << " q=" << q);
            RandBLAS::RNGState<RNG> state(43);
            NystromResult<T> result;
            ASSERT_NO_THROW(this->run(A, 4, q, vec_nnz, state, result));
            this->check_eigenpairs(A, spectrum, result, 4, T(1000) * std::numeric_limits<T>::epsilon());
        }
    }
}

TYPED_TEST(TestNystromEVD, IterationRecoversLeadingApproximateEigenpairs) {
    using T = TypeParam;
    const std::vector<T> spectrum = {16, 8, 4, T(0.1), T(0.05), T(0.02), T(0.01), T(0.005)};
    const auto A = this->matrix(spectrum);
    for (int64_t vec_nnz : {-1, 2}) {
        RandBLAS::RNGState<RNG> state(43);
        NystromResult<T> result;
        ASSERT_NO_THROW(this->run(A, 3, 3, vec_nnz, state, result));
        this->check_eigenpairs(A, spectrum, result, 3, T(5e-5));
    }
}

TYPED_TEST(TestNystromEVD, ZeroImageReturnsZeroSpectrumAndOrthonormalBasis) {
    using T = TypeParam;
    const std::vector<T> A(8 * 8, T(0));
    for (int64_t vec_nnz : {-1, 1}) {
        for (int64_t q : {1, 2}) {
            RandBLAS::RNGState<RNG> state(19);
            NystromResult<T> result;
            ASSERT_NO_THROW(this->run(A, 3, q, vec_nnz, state, result));
            this->check_basis(result, 8, 3);
            for (int64_t i = 0; i < 3; ++i) EXPECT_EQ(result.lambda[i], T(0));
        }
    }
}

TYPED_TEST(TestNystromEVD, NonzeroPSDWithNullspaceSketchReturnsZeroApproximation) {
    using T = TypeParam;
    RandBLAS::RNGState<RNG> state(31);
    RandBLAS::SparseSkOp<T, RNG> sketch(RandBLAS::SparseDist(2, 1, 1), state);
    RandBLAS::fill_sparse(sketch);
    auto coo = RandBLAS::coo_view_of_skop(sketch);
    T signs[2] = {};
    for (int64_t p = 0; p < coo.nnz; ++p) signs[coo.rows[p]] = coo.vals[p];
    const T correlation = signs[0] * signs[1];
    // Eigenvalues are 2 and 0; the sampled sign vector is exactly in the null space.
    const std::vector<T> A = {1, -correlation, -correlation, 1};
    NystromResult<T> result;
    ASSERT_NO_THROW(this->run(A, 1, 1, 1, state, result));
    this->check_basis(result, 2, 1);
    EXPECT_EQ(result.lambda[0], T(0));
}

TYPED_TEST(TestNystromEVD, SingularFullSizeSketchRejectsFalseRecovery) {
    using T = TypeParam;
    // Select a singular sign sketch by its exact determinant, independently of the recovery.
    for (int64_t seed = 0; seed < 32; ++seed) {
        RandBLAS::RNGState<RNG> state(seed);
        RandBLAS::SparseSkOp<T, RNG> sketch(RandBLAS::SparseDist(2, 2, 2), state);
        RandBLAS::fill_sparse(sketch);
        auto coo = RandBLAS::coo_view_of_skop(sketch);
        T signs[4] = {};
        for (int64_t p = 0; p < coo.nnz; ++p) signs[coo.rows[p] + 2 * coo.cols[p]] = coo.vals[p];
        if (signs[0] * signs[3] != signs[1] * signs[2]) continue;
        const T correlation = signs[0] * signs[1];
        const std::vector<T> A = {1, -correlation, -correlation, 1};
        NystromResult<T> result;
        EXPECT_THROW(this->run(A, 2, 1, 2, state, result), std::runtime_error);
        return;
    }
    FAIL() << "fixture requires a singular 2-by-2 sign sketch";
}
} // namespace
