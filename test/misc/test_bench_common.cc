// Tests for the helpers the least-squares benchmarks share (benchmark/bench_CQRRTO_linops/
// cqrrto_bench_common.hh): the sketched Karlson-Walden backward-error estimate, the
// preconditioned orthogonality and condition check, and the named-argument parser.

#include "cqrrto_bench_common.hh"

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <gtest/gtest.h>

#include "../lsq_test_helpers.hh"
#include "../test_helpers.hh"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

using RandLAPACK::linops::DenseLinOp;
using blas::Layout;
namespace bench = RandLAPACK::bench;

// ============================================================================
// Sketched Karlson-Walden backward error
// ============================================================================

class TestKWBackwardError : public ::testing::Test {
protected:
    using T = double;
    using RNG = r123::Philox4x32;
    int64_t m = 300, n = 20;
    std::vector<T> A, b, x_true, x_ls;

    void SetUp() override {
        std::vector<T> noise(m);
        lsq_test::make_problem<T>(m, n, 501, A, b, x_true);
        lsq_test::fill_random<T>(noise, 503, (T)1);
        for (int64_t i = 0; i < m; ++i) b[i] += noise[i];   // inconsistent
        x_ls = lsq_test::gels_reference(A, b, m, n);
    }

    // b - A x and A^T (b - A x).
    void residuals(const std::vector<T>& x, std::vector<T>& r, std::vector<T>& ATr) const {
        r = b;
        ATr.assign(n, 0);
        blas::gemv(Layout::ColMajor, blas::Op::NoTrans, m, n, (T)-1, A.data(), m, x.data(), 1, (T)1, r.data(), 1);
        blas::gemv(Layout::ColMajor, blas::Op::Trans, m, n, (T)1, A.data(), m, r.data(), 1, (T)0, ATr.data(), 1);
    }

    // The sketched estimate for x: {be_theta_rel, be_inf_rel}.
    std::pair<T, T> sketched(const bench::KWBackwardErrorRef<T>& ref, const std::vector<T>& x) const {
        std::vector<T> r, ATr;
        residuals(x, r, ATr);
        T be_theta, be_inf, theta, res_orth;
        bench::kw_backward_error<T>(ref, ATr.data(), blas::nrm2(m, r.data(), 1), blas::nrm2(n, x.data(), 1),
                                    blas::nrm2(m, b.data(), 1), be_theta, be_inf, theta, res_orth);
        return {be_theta, be_inf};
    }

    // The Karlson-Walden estimate with the exact singular values and vectors of A in place of
    // SA's: {be_theta_rel, be_inf_rel}.
    std::pair<T, T> unsketched(const std::vector<T>& x) const {
        std::vector<T> Ac(A), sigma(n), VT(n * n), r, ATr, z(n);
        lapack::gesdd(lapack::Job::OverwriteVec, m, n, Ac.data(), m, sigma.data(), nullptr, 1, VT.data(), n);
        residuals(x, r, ATr);
        blas::gemv(Layout::ColMajor, blas::Op::NoTrans, n, n, (T)1, VT.data(), n, ATr.data(), 1, (T)0, z.data(), 1);
        auto estimate = [&](T nu_sq, T prefactor) {
            T acc = 0;
            for (int64_t i = 0; i < n; ++i) acc += z[i] * z[i] / (sigma[i] * sigma[i] + nu_sq);
            return prefactor * std::sqrt(acc);
        };
        const T A_fro = lapack::lange(lapack::Norm::Fro, m, n, A.data(), m);
        const T r_norm = blas::nrm2(m, r.data(), 1), x_norm = blas::nrm2(n, x.data(), 1);
        const T theta = A_fro / blas::nrm2(m, b.data(), 1);
        const T den = 1 + theta * theta * x_norm * x_norm;
        return {estimate(theta * theta * r_norm * r_norm / den, theta / std::sqrt(den)) / A_fro,
                estimate(r_norm * r_norm / (x_norm * x_norm), 1 / x_norm) / A_fro};
    }

    // The exact backward error for theta = inf (Walden, Karlson and Sun):
    // min(||r|| / ||x||, sigma_min([A, (||r|| / ||x||) (I - r r^T / ||r||^2)])).
    T exact_be_inf(const std::vector<T>& x) const {
        std::vector<T> r, ATr;
        residuals(x, r, ATr);
        const T r_norm = blas::nrm2(m, r.data(), 1), c = r_norm / blas::nrm2(n, x.data(), 1);
        std::vector<T> M(m * (n + m));
        std::copy(A.begin(), A.end(), M.begin());
        for (int64_t j = 0; j < m; ++j)
            for (int64_t i = 0; i < m; ++i)
                M[i + (n + j) * m] = c * ((i == j ? (T)1 : (T)0) - r[i] * r[j] / (r_norm * r_norm));
        std::vector<T> sv(m);
        lapack::gesdd(lapack::Job::NoVec, m, n + m, M.data(), m, sv.data(), nullptr, 1, nullptr, 1);
        return std::min(c, sv[m - 1]);
    }

    bench::KWBackwardErrorRef<T> reference() {
        DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
        return bench::build_kw_reference<T, RNG>(Aop, m, n, 2 * n, bench::kKWSketchNNZ,
                                                 RandBLAS::RNGState<RNG>(bench::kKWSeed), 7);
    }

    std::vector<T> perturbed_solution() const {
        std::vector<T> x(x_ls), dx(n);
        lsq_test::fill_random<T>(dx, 507, (T)1e-6);
        for (int64_t j = 0; j < n; ++j) x[j] += dx[j];
        return x;
    }
};


// ||A||_F is exact, and the least-squares solution has a backward error at rounding level.
TEST_F(TestKWBackwardError, least_squares_solution_has_rounding_level_backward_error) {
    auto ref = reference();
    EXPECT_NEAR(ref.A_fro, lapack::lange(lapack::Norm::Fro, m, n, A.data(), m), 1e-12 * ref.A_fro);
    auto [be_theta, be_inf] = sketched(ref, x_ls);
    EXPECT_GE(be_theta, 0.0);
    EXPECT_LE(be_theta, 1e-13);
    EXPECT_LE(be_inf, 1e-13);
}


// For a perturbed solution the sketched estimate is within a factor 2 of the estimate with the
// exact singular values and vectors of A.
TEST_F(TestKWBackwardError, sketched_estimate_is_within_a_factor_2_of_the_unsketched_one) {
    auto ref = reference();
    const auto x = perturbed_solution();
    auto [s_theta, s_inf] = sketched(ref, x);
    auto [u_theta, u_inf] = unsketched(x);
    ASSERT_GT(u_theta, 1e-10);
    EXPECT_GE(s_theta / u_theta, 0.5);
    EXPECT_LE(s_theta / u_theta, 2.0);
    EXPECT_GE(s_inf / u_inf, 0.5);
    EXPECT_LE(s_inf / u_inf, 2.0);
}


// Against the exact backward error for theta = inf: the unsketched estimate is within sqrt(2)
// of it (Karlson and Walden), and the sketched one within 2 sqrt(2).
TEST_F(TestKWBackwardError, estimates_bracket_the_exact_backward_error) {
    auto ref = reference();
    const auto x = perturbed_solution();
    const T A_fro = ref.A_fro, eta = exact_be_inf(x);
    ASSERT_GT(eta, 0.0);
    const T u = unsketched(x).second * A_fro / eta, s = sketched(ref, x).second * A_fro / eta;
    EXPECT_GE(u, 1 / std::sqrt(2.0));
    EXPECT_LE(u, std::sqrt(2.0));
    EXPECT_GE(s, 1 / (2 * std::sqrt(2.0)));
    EXPECT_LE(s, 2 * std::sqrt(2.0));
}


// An operator with materialize_cols gives the reference its columns with no product: the same
// ||A||_F and singular values as the product path, whose identity products are exact.
TEST_F(TestKWBackwardError, reference_reads_columns_directly) {
    test_helpers::MaterializeColsProbe<T> probe{m, n, A.data()};
    auto ref_cols = bench::build_kw_reference<T, RNG>(probe, m, n, 2 * n, bench::kKWSketchNNZ,
                                                      RandBLAS::RNGState<RNG>(bench::kKWSeed), 7);
    test_helpers::ConceptOnlyLinOp<T> products{m, n, A.data()};
    auto ref = bench::build_kw_reference<T, RNG>(products, m, n, 2 * n, bench::kKWSketchNNZ,
                                                 RandBLAS::RNGState<RNG>(bench::kKWSeed), 7);
    EXPECT_EQ(probe.product_cols, 0);
    EXPECT_EQ(probe.direct_cols, n);
    const T eps = std::numeric_limits<T>::epsilon();
    EXPECT_NEAR(ref_cols.A_fro, ref.A_fro, 4 * eps * ref.A_fro);
    for (int64_t j = 0; j < n; ++j) EXPECT_NEAR(ref_cols.sigma[j], ref.sigma[j], 10 * eps * ref.sigma[0]) << j;
}


// For n < 4 the sketch's columns have fewer than 8 entries; the reference caps the nonzeros
// at the sketch size and records the value it used.
TEST_F(TestKWBackwardError, small_n_caps_the_sketch_nonzeros) {
    const int64_t m3 = 50, n3 = 3;
    std::vector<T> A3, b3, x3;
    lsq_test::make_problem<T>(m3, n3, 509, A3, b3, x3);
    DenseLinOp<T> Aop(m3, n3, A3.data(), m3, Layout::ColMajor);
    EXPECT_EQ(bench::kw_sketch_nnz(m3, n3), 6);
    auto ref = bench::build_kw_reference<T, RNG>(Aop, m3, n3, 2 * n3, bench::kKWSketchNNZ,
                                                 RandBLAS::RNGState<RNG>(bench::kKWSeed), 0);
    EXPECT_EQ(ref.nnz, 6);
    EXPECT_EQ(ref.d, 6);
    EXPECT_GT(ref.sigma[n3 - 1], 0.0);
}


// m and n must be the operator's dimensions.
TEST_F(TestKWBackwardError, reference_rejects_mismatched_dimensions) {
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    const RandBLAS::RNGState<RNG> state(bench::kKWSeed);
    EXPECT_THROW((bench::build_kw_reference<T, RNG>(Aop, m + 1, n, 2 * n, bench::kKWSketchNNZ, state, 7)),
                 RandLAPACK::Error);
    EXPECT_THROW((bench::build_kw_reference<T, RNG>(Aop, m, n - 1, 2 * n, bench::kKWSketchNNZ, state, 7)),
                 RandLAPACK::Error);
}


// A NaN solution maps to +inf, never to the -1 "undefined" sentinel; x = 0 leaves be_inf
// undefined.
TEST_F(TestKWBackwardError, nan_maps_to_inf_and_zero_x_leaves_be_inf_undefined) {
    auto ref = reference();
    std::vector<T> x_nan(x_ls);
    x_nan[3] = std::numeric_limits<T>::quiet_NaN();
    auto [nan_theta, nan_inf] = sketched(ref, x_nan);
    EXPECT_EQ(nan_theta, std::numeric_limits<T>::infinity());
    EXPECT_EQ(nan_inf, std::numeric_limits<T>::infinity());
    auto [zero_theta, zero_inf] = sketched(ref, std::vector<T>(n, 0));
    EXPECT_GE(zero_theta, 0.0);
    EXPECT_EQ(zero_inf, -1.0);
}

// ============================================================================
// Preconditioned orthogonality and condition
// ============================================================================

// Q = A R^{-1} with R = I and A = U diag(sigma) V^T, U and V random orthonormal, sigma
// log-spaced from 1 to 1/kappa, so cond(Q) = kappa, beyond what the eigenvalues of Q^T Q
// resolve, and ||Q^T Q - I||_F / sqrt(n) = ||sigma^2 - 1|| / sqrt(n).
template <typename T>
static void check_orth_and_cond(T kappa, T cond_tol) {
    const int64_t m = 60, n = 8;
    std::vector<T> S(n * n, (T)0), A(m * n), R(n * n, (T)0);
    T target = 0;
    for (int64_t j = 0; j < n; ++j) {
        const T sigma = std::pow(kappa, -(T)j / (T)(n - 1));
        S[j + j * n] = sigma;
        target += (sigma * sigma - 1) * (sigma * sigma - 1);
        R[j + j * n] = 1;
    }
    RandBLAS::RNGState<> state(601);
    RandLAPACK::gen::gen_singvec(m, n, A.data(), n, S.data(), state);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);

    T cond = 0;
    const T orth_with    = bench::compute_orth_error_explicit<T>(Aop, R.data(), 3, &cond);
    const T orth_without = bench::compute_orth_error_explicit<T>(Aop, R.data(), 3);
    EXPECT_EQ(orth_with, orth_without);
    EXPECT_NEAR(orth_with, std::sqrt(target / (T)n), (T)1000 * std::numeric_limits<T>::epsilon());
    EXPECT_NEAR(cond / kappa, (T)1, cond_tol);
}

TEST(TestOrthErrorExplicit, cond_beyond_the_gram_limit_double) { check_orth_and_cond<double>(1e10, 1e-4); }
TEST(TestOrthErrorExplicit, cond_beyond_the_gram_limit_float)  { check_orth_and_cond<float>(1e5f, 5e-2f); }

// ============================================================================
// Named arguments
// ============================================================================

static bench::BenchArgs args(std::vector<std::string> v) {
    v.insert(v.begin(), "prog");
    std::vector<char*> argv;
    for (auto& s : v) argv.push_back(s.data());
    return bench::BenchArgs((int)argv.size(), argv.data());
}

TEST(TestBenchArgs, values_and_defaults) {
    auto a = args({"--s=abc", "--i=-12", "--d=2.5e-3", "--flag", "--yes=true", "--no=0"});
    EXPECT_EQ(a.str("s", "x"), "abc");
    EXPECT_EQ(a.str("absent", "x"), "x");
    EXPECT_EQ(a.i64("i", 0), -12);
    EXPECT_EQ(a.i32("i", 0), -12);
    EXPECT_EQ(a.i64("absent", 7), 7);
    EXPECT_EQ(a.dbl("d", 0), 2.5e-3);
    EXPECT_TRUE(a.boolean("flag"));
    EXPECT_TRUE(a.boolean("yes"));
    EXPECT_FALSE(a.boolean("no"));
    EXPECT_FALSE(a.boolean("absent"));
    EXPECT_TRUE(a.has("flag"));
    EXPECT_THROW(a.require_str("absent"), std::runtime_error);
}

TEST(TestBenchArgs, malformed_arguments_throw) {
    EXPECT_THROW(args({"--k=1", "--k=2"}), std::runtime_error);   // repeated
    EXPECT_THROW(args({"positional"}), std::runtime_error);       // no leading --
    auto a = args({"--x=", "--flag", "--big=3000000000", "--frac=1.5", "--inf=inf", "--nan=nan",
                   "--junk=12abc", "--b=maybe"});
    EXPECT_THROW(a.str("x", "d"), std::runtime_error);            // present but empty
    EXPECT_THROW(a.i64("x", 0), std::runtime_error);
    EXPECT_THROW(a.str("flag", "d"), std::runtime_error);         // bare flag has no value
    EXPECT_EQ(a.i64("big", 0), 3000000000LL);
    EXPECT_THROW(a.i32("big", 0), std::runtime_error);            // out of int range
    EXPECT_THROW(a.i64("frac", 0), std::runtime_error);           // fractional integer
    EXPECT_THROW(a.dbl("inf", 0), std::runtime_error);            // not finite
    EXPECT_THROW(a.dbl("nan", 0), std::runtime_error);
    EXPECT_THROW(a.dbl("junk", 0), std::runtime_error);           // trailing characters
    EXPECT_THROW(a.boolean("b"), std::runtime_error);
}

TEST(TestBenchArgs, unknown_arguments_are_rejected_by_name) {
    auto a = args({"--known=1", "--typo=2"});
    EXPECT_NO_THROW(a.reject_unknown({"known", "typo"}));
    try {
        a.reject_unknown({"known"});
        FAIL() << "reject_unknown accepted --typo";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("unknown argument --typo"), std::string::npos);
    }
}
