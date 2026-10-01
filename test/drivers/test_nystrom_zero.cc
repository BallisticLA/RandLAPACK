#include "rl_fun_nystrom_pp.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <cmath>

namespace {
using RNG = r123::Philox4x32;

TEST(NystromZeroImage, ZeroOperatorPreservesFunctionAtZero) {
    constexpr int64_t n = 8, k = 3;
    double diagonal[n] = {};
    double probes[n * n] = {};
    for (int64_t i = 0; i < n; ++i) probes[i + n * i] = std::sqrt(double(n));
    RandLAPACK::linops::DiagSymLinOp<double> op(n, diagonal);
    auto f = [](double x) { return x + 1; };
    auto exact_fa = [](int64_t rows, int64_t cols, const double* input, double* output) {
        std::copy(input, input + rows * cols, output);
    };
    for (int64_t q : {1, 2}) {
        for (int64_t vec_nnz : {-1, 1}) {
            for (bool zero_fill : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "q=" << q << " vec_nnz=" << vec_nnz
                                                 << " zero_fill=" << zero_fill);
                RandLAPACK::FunNystromPP<double> driver;
                driver.vec_nnz = vec_nnz;
                RandBLAS::RNGState<RNG> state(19);
                double t1 = 0, t2 = 0, estimate = 0;
                const std::optional<double> f_zero = zero_fill ? std::optional<double>(1) : std::nullopt;
                ASSERT_NO_THROW(estimate = driver.call(op, exact_fa, f, k, n, q, state,
                                                        probes, t1, t2, f_zero));
                EXPECT_NEAR(estimate, 8.0, 1e-13);
                if (zero_fill) {
                    EXPECT_DOUBLE_EQ(t1, 8.0);
                    EXPECT_NEAR(t2, 0.0, 1e-13);
                }
                for (int64_t j = 0; j < k; ++j) {
                    EXPECT_DOUBLE_EQ(driver.lambda[j], 0.0);
                    for (int64_t i = 0; i < k; ++i)
                        EXPECT_NEAR(blas::dot(n, driver.U + n * i, 1, driver.U + n * j, 1),
                                    i == j ? 1.0 : 0.0, 1e-14);
                }
            }
        }
    }
}

TEST(NystromZeroImage, NullspaceSketchDoesNotSkipResidual) {
    // Reproduce the one-column sign sketch to choose a nonzero PSD matrix
    // with that sketch in its null space. Its eigenvalues are exactly 0, 2.
    RandBLAS::RNGState<RNG> state(31);
    RandBLAS::SparseDist distribution(2, 1, 1);
    RandBLAS::SparseSkOp<double, RNG> sketch(distribution, state);
    RandBLAS::fill_sparse(sketch);
    auto coo = RandBLAS::coo_view_of_skop(sketch);
    double signs[2] = {};
    for (int64_t i = 0; i < coo.nnz; ++i) signs[coo.rows[i]] = coo.vals[i];
    const double correlation = signs[0] * signs[1];
    const double matrix[4] = {1, -correlation, -correlation, 1};
    const double probes[4] = {std::sqrt(2.0), 0, 0, std::sqrt(2.0)};
    RandLAPACK::linops::ExplicitSymLinOp<double> op(2, blas::Uplo::Upper,
                                                  matrix, 2, blas::Layout::ColMajor);
    auto f = [](double x) { return x + 1; };
    auto exact_fa = [&](int64_t, int64_t cols, const double* input, double* output) {
        for (int64_t j = 0; j < cols; ++j) {
            output[2 * j] = 2 * input[2 * j] - correlation * input[2 * j + 1];
            output[2 * j + 1] = -correlation * input[2 * j] + 2 * input[2 * j + 1];
        }
    };
    RandLAPACK::FunNystromPP<double> driver;
    driver.vec_nnz = 1;
    double t1 = 0, t2 = 0, estimate = 0;
    ASSERT_NO_THROW(estimate = driver.call(op, exact_fa, f, 1, 2, 1, state,
                                            probes, t1, t2, std::optional<double>(1)));
    EXPECT_DOUBLE_EQ(driver.lambda[0], 0.0);
    EXPECT_DOUBLE_EQ(t1, 2.0);
    EXPECT_NEAR(t2, 2.0, 1e-13);
    EXPECT_NEAR(estimate, 4.0, 1e-13);
}

TEST(NystromZeroImage, SingularSquareSketchDoesNotClaimFullRecovery) {
    // Find a singular 2-by-2 sign sketch. A nonzero PSD matrix can annihilate
    // both columns, but k == n would cause the driver to skip the residual.
    for (int64_t seed = 0; seed < 32; ++seed) {
        RandBLAS::RNGState<RNG> state(seed);
        RandBLAS::SparseDist distribution(2, 2, 2);
        RandBLAS::SparseSkOp<double, RNG> sketch(distribution, state);
        RandBLAS::fill_sparse(sketch);
        auto coo = RandBLAS::coo_view_of_skop(sketch);
        double signs[4] = {};
        for (int64_t i = 0; i < coo.nnz; ++i)
            signs[coo.rows[i] + 2 * coo.cols[i]] = coo.vals[i];
        if (signs[0] * signs[3] != signs[1] * signs[2]) continue;
        const double correlation = signs[0] * signs[1];
        const double matrix[4] = {1, -correlation, -correlation, 1};
        RandLAPACK::linops::ExplicitSymLinOp<double> op(2, blas::Uplo::Upper,
                                                      matrix, 2, blas::Layout::ColMajor);
        RandLAPACK::FunNystromPP<double> driver;
        auto f = [](double x) { return x; };
        auto unused_oracle = [](int64_t, int64_t, const double*, double*) {};
        double t1 = 0, t2 = 0;
        EXPECT_THROW(driver.call(op, unused_oracle, f, 2, 0, 1, state,
                                 nullptr, t1, t2), std::runtime_error);
        return;
    }
    FAIL() << "fixture requires a singular sign sketch";
}
} // namespace
