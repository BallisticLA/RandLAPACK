// Paired old-vs-new tests for the panel-blocked triangular kernels in comps/rl_cholqr.hh.
//
// invert_upper_into and trmm_upper_upper_left replace full-width trsm/trmm calls whose
// right-hand side is known to be upper triangular. They skip the provably-zero rows, turning
// n^3 flops into n^3/3 (measured 1.97x at n=8304, b=256, 8 threads).
//
// They are NOT bit-identical to the full-width calls: the arithmetic on the nonzero part is
// the same, but BLAS chooses different internal blocking for a different operand shape, which
// reorders accumulation. These tests therefore pin the two properties that must hold:
//   1. agreement with the full-width result to a tight relative tolerance, and
//   2. the exact structural zeros (the result really is upper triangular).
// A tolerance test is the right instrument here; an equality test would fail for a correct
// implementation, and no test at all would let a genuine indexing bug through.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

namespace {

// Well-conditioned upper triangular: unit-ish diagonal keeps the inverse from blowing up, so a
// tight tolerance stays meaningful rather than being swamped by conditioning.
template <typename T>
std::vector<T> make_upper(int64_t n, uint32_t seed) {
    std::vector<T> U(n * n, T(0));
    RandBLAS::RNGState<> state(seed);
    std::vector<T> raw(n * n);
    RandBLAS::DenseDist D(n, n);
    RandBLAS::fill_dense(D, raw.data(), state);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i <= j; ++i)
            U[i + j * n] = (i == j) ? (T)2.0 + std::abs(raw[i + j * n]) : (T)0.25 * raw[i + j * n];
    return U;
}

template <typename T>
double rel_fro(const std::vector<T>& a, const std::vector<T>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return (den > 0) ? std::sqrt(num / den) : std::sqrt(num);
}

template <typename T>
void assert_strict_lower_zero(const std::vector<T>& M, int64_t n, int64_t ld) {
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            ASSERT_EQ(M[i + j * ld], T(0)) << "entry (" << i << "," << j << ") must be exactly 0";
}

} // namespace

class TestTriangularPanels : public ::testing::Test {};

// invert_upper_into(P) must match a single full-width trsm against the identity.
TEST_F(TestTriangularPanels, invert_upper_matches_full_width_trsm) {
    for (int64_t n : {1, 2, 7, 33, 128, 257}) {
        for (int64_t b : {1, 5, 32, 256}) {
            auto P = make_upper<double>(n, 11);

            std::vector<double> X_ref(n * n, 0.0);
            for (int64_t i = 0; i < n; ++i) X_ref[i + i * n] = 1.0;
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit,
                       n, n, 1.0, P.data(), n, X_ref.data(), n);

            std::vector<double> X_new(n * n, 0.0);
            for (int64_t i = 0; i < n; ++i) X_new[i + i * n] = 1.0;
            RandLAPACK::invert_upper_into<double>(P.data(), n, X_new.data(), n, n, b);

            EXPECT_LE(rel_fro(X_new, X_ref), 1e-12)
                << "n=" << n << " panel=" << b;
            assert_strict_lower_zero(X_new, n, n);
        }
    }
}

// The panelled inverse must actually invert: P * P^{-1} = I.
TEST_F(TestTriangularPanels, invert_upper_is_a_real_inverse) {
    const int64_t n = 96, b = 32;
    auto P = make_upper<double>(n, 5);
    std::vector<double> X(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i) X[i + i * n] = 1.0;
    RandLAPACK::invert_upper_into<double>(P.data(), n, X.data(), n, n, b);

    std::vector<double> prod(n * n, 0.0);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               n, n, n, 1.0, P.data(), n, X.data(), n, 0.0, prod.data(), n);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i)
            ASSERT_NEAR(prod[i + j * n], (i == j) ? 1.0 : 0.0, 1e-10)
                << "P*P^-1 at (" << i << "," << j << ")";
}

// trmm_upper_upper_left must match a single full-width trmm.
TEST_F(TestTriangularPanels, trmm_upper_upper_matches_full_width) {
    for (int64_t n : {1, 2, 7, 33, 128, 257}) {
        for (int64_t b : {1, 5, 32, 256}) {
            auto U = make_upper<double>(n, 3);
            auto B = make_upper<double>(n, 9);   // RHS is upper triangular, as at the call site

            std::vector<double> B_ref = B;
            blas::trmm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit,
                       n, n, 1.0, U.data(), n, B_ref.data(), n);

            std::vector<double> B_new = B;
            RandLAPACK::trmm_upper_upper_left<double>(U.data(), n, B_new.data(), n, n, b);

            EXPECT_LE(rel_fro(B_new, B_ref), 1e-12)
                << "n=" << n << " panel=" << b;
            assert_strict_lower_zero(B_new, n, n);
        }
    }
}

// A panel width at or above n degenerates to the original single full-width call, so that case
// must be bit-identical. This pins the boundary of the claim: only panel < n reorders anything.
TEST_F(TestTriangularPanels, panel_at_least_n_is_bit_identical) {
    const int64_t n = 64;
    auto P = make_upper<double>(n, 17);

    std::vector<double> X_ref(n * n, 0.0), X_new(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i) { X_ref[i + i * n] = 1.0; X_new[i + i * n] = 1.0; }
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
               blas::Op::NoTrans, blas::Diag::NonUnit,
               n, n, 1.0, P.data(), n, X_ref.data(), n);
    RandLAPACK::invert_upper_into<double>(P.data(), n, X_new.data(), n, n, /*b_panel=*/n);

    for (int64_t k = 0; k < n * n; ++k)
        ASSERT_EQ(X_new[k], X_ref[k]) << "flat index " << k;
}
