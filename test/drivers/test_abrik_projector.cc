#include "RandLAPACK.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <vector>

namespace {
using Vec = std::vector<double>;
using RNG = r123::Philox4x32;
constexpr int n = 64, rank = 24, block = 3;
static_assert(n <= 1000 && rank >= block && n - rank >= block);
static_assert(n == 64); // H_64 / 8 has exactly representable entries.
constexpr double solver_tol = 1e-12;
// A bounded regression threshold for these well-conditioned double fixtures,
// not a stability theorem for arbitrary inputs, dimensions, or QR backends.
constexpr double check_tol = 1000 * n * std::numeric_limits<double>::epsilon();

bool all_finite(const double* data, int count) {
    return data && std::all_of(data, data + count,
                              [](double x) { return std::isfinite(x); });
}

// Column-major scalar arithmetic, independent of the production residual and
// orthogonality helpers. No assumption of wider-than-double arithmetic is needed.
Vec multiply(const double* a, int rows, int inner, const double* b, int cols) {
    Vec result(rows * cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            for (int t = 0; t < inner; ++t)
                result[i + rows*j] += a[i + rows*t] * b[t + inner*j];
    return result;
}

Vec transpose_multiply(const double* a, int rows, int acols,
                       const double* b, int bcols) {
    Vec result(acols * bcols);
    for (int j = 0; j < bcols; ++j)
        for (int i = 0; i < acols; ++i)
            for (int t = 0; t < rows; ++t)
                result[i + acols*j] += a[t + rows*i] * b[t + rows*j];
    return result;
}

double orthogonality_error(const double* q, int rows, int cols) {
    Vec gram = transpose_multiply(q, rows, cols, q, cols);
    double squared = 0;
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < cols; ++i) {
            const double delta = gram[i + cols*j] - (i == j);
            squared += delta * delta;
        }
    return std::sqrt(squared / cols);
}

struct Residual { double left, right; };

std::vector<Residual> triplet_residuals(const Vec& a, const double* u,
                                      const double* v, const double* sigma, int k) {
    std::vector<Residual> result(k);
    for (int j = 0; j < k; ++j) {
        double left_squared = 0, right_squared = 0;
        for (int i = 0; i < n; ++i) {
            double av = 0, atu = 0;
            for (int t = 0; t < n; ++t) {
                av += a[i + n*t] * v[t + n*j];
                atu += a[t + n*i] * u[t + n*j];
            }
            const double left = av - sigma[j] * u[i + n*j];
            const double right = atu - sigma[j] * v[i + n*j];
            left_squared += left * left;
            right_squared += right * right;
        }
        result[j] = {std::sqrt(left_squared) / sigma[j],
                     std::sqrt(right_squared) / sigma[j]};
    }
    return result;
}

double band_error(const Vec& a, const double* x, int nx, const double* y,
                  int ny, const double* band, int ld) {
    Vec ay = multiply(a.data(), n, n, y, ny);
    Vec projected = transpose_multiply(x, n, nx, ay.data(), ny);
    double error_squared = 0, norm_squared = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double expected = projected[i + nx*j];
            const double delta = band[i + ld*j] - expected;
            error_squared += delta * delta;
            norm_squared += expected * expected;
        }
    return std::sqrt(error_squared / norm_squared);
}

void orthonormalize(Vec& a, int rows) {
    Vec tau(block);
    ASSERT_EQ(lapack::geqrf(rows, block, a.data(), rows, tau.data()), 0);
    ASSERT_EQ(lapack::ungqr(rows, block, block, a.data(), rows, tau.data()), 0);
}

void check_start_conditioning(Vec a, int rows) {
    Vec sigma(block);
    ASSERT_EQ(lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec,
                          rows, block, a.data(), rows, sigma.data(),
                          static_cast<double*>(nullptr), 1,
                          static_cast<double*>(nullptr), 1), 0);
    ASSERT_TRUE(all_finite(sigma.data(), block));
    // Both projected starts must have full column rank and condition number < 10.
    ASSERT_GT(sigma.back(), 0.1 * sigma.front());
}

class TestABRIKProjector : public ::testing::TestWithParam<bool> {
protected:
    Vec a, reference, reference_sigma;

    void SetUp() override {
        Vec q(n * rank);
        for (int j = 0; j < rank; ++j)
            for (int i = 0; i < n; ++i) {
                const unsigned col = (13*j + 7) % n; // distinct, nonconsecutive
                q[i + n*j] = GetParam()
                    ? (std::popcount(static_cast<unsigned>(i) & col) % 2 ? -0.125 : 0.125)
                    : (i == j ? 1.0 : 0.0);
            }
        ASSERT_EQ(orthogonality_error(q.data(), n, rank), 0.0);
        Vec p(n * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                for (int t = 0; t < rank; ++t)
                    p[i + n*j] += q[i + n*t] * q[j + n*t];
        ASSERT_EQ(multiply(p.data(), n, n, p.data(), n), p);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                ASSERT_EQ(p[i + n*j], p[j + n*i]);
        a = p;
        for (int i = 0; i < n; ++i) a[i + n*i] += 1.0;
        double squared_norm = 0;
        for (double value : a) squared_norm += value * value;
        ASSERT_EQ(squared_norm, n + 3 * rank);

        // Replay the driver's starting Gaussian block. Exact Krylov reachability
        // is span(P*Omega, (I-P)*Omega), not the full rank-24 leading eigenspace.
        auto state = RandBLAS::RNGState<RNG>(0);
        Vec omega(n * block);
        RandBLAS::DenseDist distribution(n, block);
        RandBLAS::fill_dense(distribution, omega.data(), state);
        Vec upper = transpose_multiply(q.data(), n, rank, omega.data(), block);
        Vec lower = multiply(q.data(), n, rank, upper.data(), block);
        for (int i = 0; i < n * block; ++i) lower[i] = omega[i] - lower[i];
        ASSERT_NO_FATAL_FAILURE(check_start_conditioning(upper, rank));
        ASSERT_NO_FATAL_FAILURE(check_start_conditioning(lower, n));
        ASSERT_NO_FATAL_FAILURE(orthonormalize(upper, rank));
        ASSERT_NO_FATAL_FAILURE(orthonormalize(lower, n));
        reference = multiply(q.data(), n, rank, upper.data(), block);
        reference.insert(reference.end(), lower.begin(), lower.end());
        reference_sigma.assign(2 * block, 1.0);
        std::fill_n(reference_sigma.begin(), block, 2.0);
        ASSERT_LT(orthogonality_error(reference.data(), n, 2 * block), check_tol);
    }
};

TEST_P(TestABRIKProjector, IndependentOracleDetectsCorruption) {
    const int k = 2 * block;
    for (const auto& residual : triplet_residuals(
             a, reference.data(), reference.data(), reference_sigma.data(), k)) {
        EXPECT_LT(residual.left, check_tol);
        EXPECT_LT(residual.right, check_tol);
    }
    // Duplicate within the eigenvalue-2 cluster: residuals alone cannot reject it.
    Vec duplicate = reference;
    std::copy_n(reference.begin(), n, duplicate.begin() + n);
    EXPECT_GT(orthogonality_error(duplicate.data(), n, k), 0.1);
    for (const auto& residual : triplet_residuals(
             a, duplicate.data(), duplicate.data(), reference_sigma.data(), k)) {
        EXPECT_LT(residual.left, check_tol);
        EXPECT_LT(residual.right, check_tol);
    }
    Vec wrong_sigma = reference_sigma;
    wrong_sigma[0] += 0.25;
    const auto wrong = triplet_residuals(
        a, reference.data(), reference.data(), wrong_sigma.data(), k);
    EXPECT_GT(wrong[0].left, 0.1);
    EXPECT_GT(wrong[0].right, 0.1);

    Vec band(k * k);
    for (int j = 0; j < k; ++j) band[j + k*j] = reference_sigma[j];
    EXPECT_LT(band_error(a, reference.data(), k, reference.data(), k, band.data(), k), check_tol);
    band[0] += 0.25;
    EXPECT_GT(band_error(a, reference.data(), k, reference.data(), k, band.data(), k), 0.01);
}

struct BKOutput {
    double *x = nullptr, *y = nullptr, *r = nullptr, *s = nullptr;
    int64_t rows = 0, cols = 0;
    bool odd = false;
    ~BKOutput() { std::free(x); std::free(y); std::free(r); std::free(s); }
};

TEST_P(TestABRIKProjector, BKPreservesBasisAndStoredBand) {
    for (bool cqrrt : {false, true}) for (int budget : {2, 3, 4, 6, 12}) {
        SCOPED_TRACE(::testing::Message() << "cqrrt=" << cqrrt << ", budget=" << budget);
        RandLAPACK::BK<double, RNG> solver(false, false, solver_tol);
        if (cqrrt) solver.qr_exp = RandLAPACK::BKSubroutines::QR_explicit::cqrrt;
        solver.max_krylov_iters = budget;
        auto state = RandBLAS::RNGState<RNG>(0);
        Vec input = a;
        BKOutput out;
        ASSERT_EQ(solver.call(n, n, input.data(), n, block, out.x, out.y,
                              out.r, out.s, out.rows, out.cols, out.odd, state), 0);
        ASSERT_GE(solver.num_krylov_iters, 1);
        ASSERT_LE(solver.num_krylov_iters, budget);
        ASSERT_GE(out.rows, block);
        ASSERT_GE(out.cols, block);
        ASSERT_LE(out.rows, std::min(n, block * (1 + solver.num_krylov_iters / 2)));
        ASSERT_LE(out.cols, std::min(n, block * ((solver.num_krylov_iters + 1) / 2)));
        EXPECT_EQ(out.odd, solver.num_krylov_iters % 2 != 0);
        ASSERT_TRUE(all_finite(out.x, n * out.rows));
        ASSERT_TRUE(all_finite(out.y, n * out.cols));
        const double* band = out.odd ? out.r : out.s;
        const int ld = out.odd ? n : n + block;
        ASSERT_NE(band, nullptr);
        for (int j = 0; j < out.cols; ++j)
            ASSERT_TRUE(all_finite(band + ld*j, out.rows));
        EXPECT_LT(orthogonality_error(out.x, n, out.rows), check_tol);
        EXPECT_LT(orthogonality_error(out.y, n, out.cols), check_tol);
        // Strict as-stored orientation; accepting the transpose would hide a bug.
        EXPECT_LT(band_error(a, out.x, out.rows, out.y, out.cols, band, ld), check_tol);
        // Budget 2 has not completed the reachable subspace, so no triplet
        // residual or global low-rank reconstruction requirement belongs here.
    }
}

struct ABRIKOutput {
    double *u = nullptr, *v = nullptr, *sigma = nullptr;
    ~ABRIKOutput() { delete[] u; delete[] v; delete[] sigma; }
};

TEST_P(TestABRIKProjector, CompletedBudgetsReturnAccurateIndependentTriplets) {
    for (bool cqrrt : {false, true}) for (int budget : {3, 6, 12}) {
        SCOPED_TRACE(::testing::Message() << "cqrrt=" << cqrrt << ", budget=" << budget);
        RandLAPACK::ABRIK<double, RNG> solver(false, false, solver_tol);
        // Pins prune-and-narrow; with refills on, budget 6 ends mid-cycle (see the refill test).
        solver.refill_dead_columns = false;
        if (cqrrt) solver.qr_exp = RandLAPACK::ABRIKSubroutines::QR_explicit::cqrrt;
        solver.max_krylov_iters = budget;
        auto state = RandBLAS::RNGState<RNG>(0);
        Vec input = a;
        ABRIKOutput out;
        ASSERT_EQ(solver.call(n, n, input.data(), n, block,
                              out.u, out.v, out.sigma, state), 0);
        ASSERT_GE(solver.num_krylov_iters, 1);
        ASSERT_LE(solver.num_krylov_iters, budget);
        const auto count = solver.singular_triplets_found;
        ASSERT_GE(count, block);
        ASSERT_LE(count, std::min(n, block * ((solver.num_krylov_iters + 1) / 2)));
        const int k = static_cast<int>(count);
        ASSERT_TRUE(all_finite(out.u, n * k));
        ASSERT_TRUE(all_finite(out.v, n * k));
        ASSERT_TRUE(all_finite(out.sigma, k));
        ASSERT_GT(*std::min_element(out.sigma, out.sigma + k), 0.0);
        EXPECT_TRUE(std::is_sorted(out.sigma, out.sigma + k, std::greater<double>()));
        EXPECT_LT(orthogonality_error(out.u, n, k), check_tol);
        EXPECT_LT(orthogonality_error(out.v, n, k), check_tol);
        const auto residuals = triplet_residuals(a, out.u, out.v, out.sigma, k);
        for (int j = 0; j < k; ++j) {
            SCOPED_TRACE(::testing::Message() << "triplet=" << j);
            EXPECT_LT(residuals[j].left, check_tol);
            EXPECT_LT(residuals[j].right, check_tol);
        }
        for (int j = 0; j < block; ++j) EXPECT_NEAR(out.sigma[j], 2.0, check_tol);
        // Finite precision may expose extra independent, accurate directions.
        // Do not require six outputs, a specific stop, or exact containment in
        // the six-dimensional reference space built from the starting block.
    }
}

// Each refill of a dead block restarts the process from a fresh random block, and on I + P
// every cycle closes a new six-dimensional invariant space in four iterations. Budgets 4, 8
// and 12 end right after a refill whose unprobed columns are excluded from the count.
// Fails without refilling: prune-and-narrow stops at the first dead block with 6 triplets.
TEST_P(TestABRIKProjector, RefillsReachBeyondTheStartingBlock) {
    struct Case { int budget; int64_t expected; };
    // Derived analytically and confirmed by running, for both QR backends and both fixtures.
    // Budgets 5, 6, 9 and 10 end mid-cycle, where the newest triplets have not converged.
    constexpr Case cases[] = {{4, 6}, {7, 12}, {8, 12}, {11, 18}, {12, 18}};
    for (bool cqrrt : {false, true}) for (const Case& c : cases) {
        SCOPED_TRACE(::testing::Message() << "cqrrt=" << cqrrt << ", budget=" << c.budget);
        RandLAPACK::ABRIK<double, RNG> solver(false, false, solver_tol);
        if (cqrrt) solver.qr_exp = RandLAPACK::ABRIKSubroutines::QR_explicit::cqrrt;
        solver.max_krylov_iters = c.budget;
        auto state = RandBLAS::RNGState<RNG>(0);
        Vec input = a;
        ABRIKOutput out;
        ASSERT_EQ(solver.call(n, n, input.data(), n, block,
                              out.u, out.v, out.sigma, state), 0);
        const auto count = solver.singular_triplets_found;
        EXPECT_EQ(count, c.expected);
        if (c.budget >= 7) { EXPECT_GT(count, 2 * block); }
        ASSERT_GE(count, block);
        const int k = static_cast<int>(count);
        ASSERT_TRUE(all_finite(out.u, n * k));
        ASSERT_TRUE(all_finite(out.v, n * k));
        ASSERT_TRUE(all_finite(out.sigma, k));
        EXPECT_TRUE(std::is_sorted(out.sigma, out.sigma + k, std::greater<double>()));
        EXPECT_LT(orthogonality_error(out.u, n, k), check_tol);
        EXPECT_LT(orthogonality_error(out.v, n, k), check_tol);
        const auto residuals = triplet_residuals(a, out.u, out.v, out.sigma, k);
        for (int j = 0; j < k; ++j) {
            SCOPED_TRACE(::testing::Message() << "triplet=" << j);
            EXPECT_LT(residuals[j].left, check_tol);
            EXPECT_LT(residuals[j].right, check_tol);
        }
        for (int j = 0; j < block; ++j) EXPECT_NEAR(out.sigma[j], 2.0, check_tol);
    }
}

INSTANTIATE_TEST_SUITE_P(CoordinateAndHadamard, TestABRIKProjector,
                         ::testing::Bool());
} // namespace
