// Tests for comps/rl_cholqr.hh: the panel-blocked triangular kernels, cholqr_primitive and
// cholqr_iterate. The drivers built on them are tested in test/drivers/test_orth_linop.cc.
//
// The panel kernels hand BLAS different operand shapes than one full-width trsm/trmm, so they
// are checked to a tight tolerance, for exact structural zeros, and for touching only what
// they should.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include "../test_helpers.hh"
#include <vector>
#include <cmath>
#include <limits>

using RandLAPACK::testing::rel_fro_error;
using test_helpers::random_matrix;

namespace {

// Well-conditioned upper triangular: unit-ish diagonal keeps the inverse from blowing up, so a
// tight tolerance stays meaningful rather than being swamped by conditioning.
template <typename T>
std::vector<T> make_upper(int64_t n, uint32_t seed) {
    std::vector<T> U(n * n, T(0));
    auto raw = random_matrix<T>(n, n, seed);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i <= j; ++i)
            U[i + j * n] = (i == j) ? (T)2.0 + std::abs(raw[i + j * n]) : (T)0.25 * raw[i + j * n];
    return U;
}

// a and b agree to within one ulp: results of the same computation in separate buffers, which
// BLAS may process along alignment-dependent paths.
template <typename T>
::testing::AssertionResult within_one_ulp(T a, T b) {
    if (std::abs(a - b) <= std::numeric_limits<T>::epsilon() * std::max(std::abs(a), std::abs(b)))
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << a << " and " << b << " differ by more than one ulp";
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

            std::vector<double> X_new(n * n, -1.0);   // invert_upper_into sets the identity itself
            RandLAPACK::invert_upper_into<double>(P.data(), n, X_new.data(), n, n, b);

            EXPECT_LE(rel_fro_error(X_new.data(), X_ref.data(), n * n), 1e-12)
                << "n=" << n << " panel=" << b;
            assert_strict_lower_zero(X_new, n, n);
        }
    }
}

// The panelled inverse must actually invert: P * P^{-1} = I.
TEST_F(TestTriangularPanels, invert_upper_is_a_real_inverse) {
    const int64_t n = 96, b = 32;
    auto P = make_upper<double>(n, 5);
    std::vector<double> X(n * n);
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

            EXPECT_LE(rel_fro_error(B_new.data(), B_ref.data(), n * n), 1e-12)
                << "n=" << n << " panel=" << b;
            assert_strict_lower_zero(B_new, n, n);
        }
    }
}

// trsm_upper_trans_left_upper_part(U, B) must match a single full-width trsm with U^T. Without
// mirror: on the upper triangle and each panel's diagonal block, with the rows below those blocks
// left as B. With mirror: on the upper triangle, with the strict lower one its exact transpose.
TEST_F(TestTriangularPanels, trsm_upper_trans_left_upper_part_matches_full_width) {
    for (int64_t n : {1, 2, 7, 33, 128, 257}) {
        for (int64_t b : {1, 5, 32, 256}) {
            auto U = make_upper<double>(n, 41);
            auto B = random_matrix<double>(n, n, 43);   // full, as the Gram at the call site

            std::vector<double> B_ref = B;
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::Trans, blas::Diag::NonUnit,
                       n, n, 1.0, U.data(), n, B_ref.data(), n);

            for (bool mirror : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "n " << n << ", panel " << b << ", mirror " << mirror);
                std::vector<double> B_new = B;
                RandLAPACK::trsm_upper_trans_left_upper_part<double>(U.data(), n, B_new.data(), n, n, b, mirror);

                // Compare the solved entries (rows above the end of the column's panel; only the
                // upper triangle with mirror) and check the rest exactly.
                const int64_t bp = std::min(b, n);
                std::vector<double> solved_ref(n * n, 0.0), solved_new(n * n, 0.0);
                for (int64_t j = 0; j < n; ++j) {
                    const int64_t lead = mirror ? j + 1 : std::min(n, (j / bp + 1) * bp);
                    for (int64_t i = 0; i < n; ++i) {
                        if (i < lead) {
                            solved_ref[i + j * n] = B_ref[i + j * n];
                            solved_new[i + j * n] = B_new[i + j * n];
                        } else if (mirror) {
                            ASSERT_EQ(B_new[i + j * n], B_new[j + i * n]) << "(" << i << "," << j << ") must mirror";
                        } else {
                            ASSERT_EQ(B_new[i + j * n], B[i + j * n]) << "(" << i << "," << j << ") must be untouched";
                        }
                    }
                }
                EXPECT_LE(rel_fro_error(solved_new.data(), solved_ref.data(), n * n), 1e-12);
            }
        }
    }
}

// A panel width at or above n is one full-width call: it matches the plain trsm to rounding.
TEST_F(TestTriangularPanels, panel_at_least_n_is_one_full_width_call) {
    const int64_t n = 64;
    auto P = make_upper<double>(n, 17);

    std::vector<double> X_ref(n * n, 0.0), X_new(n * n);
    for (int64_t i = 0; i < n; ++i) X_ref[i + i * n] = 1.0;
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
               blas::Op::NoTrans, blas::Diag::NonUnit,
               n, n, 1.0, P.data(), n, X_ref.data(), n);
    RandLAPACK::invert_upper_into<double>(P.data(), n, X_new.data(), n, n, /*b_panel=*/n);

    for (int64_t k = 0; k < n * n; ++k)
        ASSERT_TRUE(within_one_ulp(X_new[k], X_ref[k])) << "flat index " << k;
}

// Float instantiation, and n = 0 is a no-op.
TEST_F(TestTriangularPanels, float_and_empty) {
    const int64_t n = 40, b = 16;
    auto P = make_upper<float>(n, 23);
    std::vector<float> X_ref(n * n, 0.0f), X_new(n * n);
    for (int64_t i = 0; i < n; ++i) X_ref[i + i * n] = 1.0f;
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
               blas::Op::NoTrans, blas::Diag::NonUnit, n, n, 1.0f, P.data(), n, X_ref.data(), n);
    RandLAPACK::invert_upper_into<float>(P.data(), n, X_new.data(), n, n, b);
    EXPECT_LE(rel_fro_error(X_new.data(), X_ref.data(), n * n), 1e-5);

    auto B = make_upper<float>(n, 29);
    std::vector<float> B_ref = B, B_new = B;
    blas::trmm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
               blas::Op::NoTrans, blas::Diag::NonUnit, n, n, 1.0f, P.data(), n, B_ref.data(), n);
    RandLAPACK::trmm_upper_upper_left<float>(P.data(), n, B_new.data(), n, n, b);
    EXPECT_LE(rel_fro_error(B_new.data(), B_ref.data(), n * n), 1e-5);

    // The upper-part solve agrees with the full-width trsm on the upper triangle.
    B_ref = B; B_new = B;
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
               blas::Op::Trans, blas::Diag::NonUnit, n, n, 1.0f, P.data(), n, B_ref.data(), n);
    RandLAPACK::trsm_upper_trans_left_upper_part<float>(P.data(), n, B_new.data(), n, n, b, false);
    RandLAPACK::util::get_U(n, n, B_ref.data(), n);
    RandLAPACK::util::get_U(n, n, B_new.data(), n);
    EXPECT_LE(rel_fro_error(B_new.data(), B_ref.data(), n * n), 1e-5);

    float dummy = 3.0f;
    RandLAPACK::invert_upper_into<float>(P.data(), 1, &dummy, 1, 0, b);
    RandLAPACK::trmm_upper_upper_left<float>(P.data(), 1, &dummy, 1, 0, b);
    RandLAPACK::trsm_upper_trans_left_upper_part<float>(P.data(), 1, &dummy, 1, 0, b, true);
    EXPECT_EQ(dummy, 3.0f);
}

// Padded, unequal leading dimensions: the kernels must read only the triangle they document
// (P's and U's strict lower triangles hold NaN), write only the n x n block (padding rows hold
// a sentinel), and match the results on clean, unpadded copies.
TEST_F(TestTriangularPanels, padded_leading_dimensions_and_poisoned_lower_triangle) {
    const int64_t n = 37, b = 8, ldp = n + 3, ldx = n + 5;
    const double nan = std::numeric_limits<double>::quiet_NaN(), sentinel = -123.0;
    auto P = make_upper<double>(n, 31);
    std::vector<double> P_pad(ldp * n, sentinel);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i)
            P_pad[i + j * ldp] = (i <= j) ? P[i + j * n] : nan;

    std::vector<double> X_ref(n * n), X(ldx * n, sentinel);
    RandLAPACK::invert_upper_into<double>(P.data(), n, X_ref.data(), n, n, b);
    RandLAPACK::invert_upper_into<double>(P_pad.data(), ldp, X.data(), ldx, n, b);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < n; ++i) ASSERT_TRUE(within_one_ulp(X[i + j * ldx], X_ref[i + j * n])) << i << "," << j;
        for (int64_t i = n; i < ldx; ++i) ASSERT_EQ(X[i + j * ldx], sentinel);
    }

    auto B = make_upper<double>(n, 37);
    std::vector<double> B_ref = B, B_pad(ldx * n, sentinel);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i) B_pad[i + j * ldx] = B[i + j * n];
    RandLAPACK::trmm_upper_upper_left<double>(P.data(), n, B_ref.data(), n, n, b);
    RandLAPACK::trmm_upper_upper_left<double>(P_pad.data(), ldp, B_pad.data(), ldx, n, b);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < n; ++i) ASSERT_TRUE(within_one_ulp(B_pad[i + j * ldx], B_ref[i + j * n])) << i << "," << j;
        for (int64_t i = n; i < ldx; ++i) ASSERT_EQ(B_pad[i + j * ldx], sentinel);
    }

    // The upper-part solve, in both modes: padding untouched, every entry finite, the rows below
    // each panel untouched (mirror off) or the exact transpose (mirror on), and the n x n block
    // equal to the unpadded result up to twice the solve's forward error, about 2 n eps kappa(P)
    // = 7e-14 here (kappa(P) < 4.1); the tolerance is 1e-12.
    auto C = random_matrix<double>(n, n, 39);
    for (bool mirror : {false, true}) {
        std::vector<double> C_ref = C, C_pad(ldx * n, sentinel), C_blk(n * n);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i) C_pad[i + j * ldx] = C[i + j * n];
        RandLAPACK::trsm_upper_trans_left_upper_part<double>(P.data(), n, C_ref.data(), n, n, b, mirror);
        RandLAPACK::trsm_upper_trans_left_upper_part<double>(P_pad.data(), ldp, C_pad.data(), ldx, n, b, mirror);
        for (int64_t j = 0; j < n; ++j) {
            const int64_t lead = std::min(n, (j / b + 1) * b);
            for (int64_t i = 0; i < n; ++i) {
                const double v = C_pad[i + j * ldx];
                ASSERT_TRUE(std::isfinite(v)) << i << "," << j << " mirror " << mirror;
                if (!mirror && i >= lead) { ASSERT_EQ(v, C[i + j * n]) << i << "," << j; }
                if (mirror && i > j)      { ASSERT_EQ(v, C_pad[j + i * ldx]) << i << "," << j; }
                C_blk[i + j * n] = v;
            }
            for (int64_t i = n; i < ldx; ++i) ASSERT_EQ(C_pad[i + j * ldx], sentinel);
        }
        EXPECT_LE(rel_fro_error(C_blk.data(), C_ref.data(), n * n), 1e-12) << "mirror " << mirror;
    }
}

// ============================================================================
// cholqr_primitive and cholqr_iterate
// ============================================================================

namespace {

using DLO = RandLAPACK::linops::DenseLinOp<double>;
using RandLAPACK::CholQRPrecondMethod;

// One cholqr_primitive call with its own scratch; P == nullptr is plain CholQR.
int prim_call(DLO& A, const double* P, double* R, int64_t ldr, CholQRPrecondMethod method,
              int64_t block_size, double shift_factor, const RandLAPACK::CholQROptions<double>& opts,
              RandLAPACK::CholQRShiftRecord<double>* rec = nullptr,
              RandBLAS::RNGState<RandBLAS::DefaultRNG>* state = nullptr, double bqrrp_ratio = 1.0) {
    int64_t m = A.n_rows, n = A.n_cols, b = RandLAPACK::gram_block_width(block_size, n);
    std::vector<double> R_pre(n * n), G(n * n), A_temp(m * b), Z_buf(n * b);
    long t[6] = {0};
    return RandLAPACK::cholqr_primitive<double, DLO>(
        A, P, n, R, ldr, method, block_size, bqrrp_ratio, R_pre.data(), G.data(), A_temp.data(),
        Z_buf.data(), state, t[0], t[1], t[2], t[3], t[4], t[5], /*timing=*/false,
        shift_factor, opts, rec);
}

// A = U diag(s) V^T with s from 1 down to 10^-log10_kappa, then column j scaled by
// 10^(scale_exp * ((5 j) mod n) / n): a shuffled scaling, so column pivoting reorders
// columns in a long cycle.
std::vector<double> hard_matrix(int64_t m, int64_t n, double log10_kappa, double scale_exp, uint32_t seed) {
    std::vector<double> S(n * n, 0.0), A(m * n);
    for (int64_t i = 0; i < n; ++i) S[i + i * n] = std::pow(10.0, -log10_kappa * (double)i / (double)(n - 1));
    RandBLAS::RNGState<> st(seed);
    RandLAPACK::gen::gen_singvec(m, n, A.data(), n, S.data(), st);
    for (int64_t j = 0; j < n; ++j) {
        double d = std::pow(10.0, scale_exp * (double)((5 * j) % n) / (double)n);
        for (int64_t i = 0; i < m; ++i) A[i + j * m] *= d;
    }
    return A;
}

} // namespace

class TestCholQRPrimitive : public ::testing::Test {};

// Every inversion method, both left-factor routes, symmetrized or not, blocked and unblocked,
// BQRRP with its default and a sub-1 block ratio, must turn a one-pass CholQR factor of a
// column-scaled, ill-conditioned matrix into a factor with A R^{-1} orthonormal. The shuffled
// column scaling gives geqp3/bqrrp a nontrivial pivot order, so a wrong permutation direction
// fails.
TEST_F(TestCholQRPrimitive, every_precond_method_refines_a_cholqr_factor) {
    int64_t m = 150, n = 24;
    auto A_data = hard_matrix(m, n, 5.0, 4.0, 41);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> P(n * n, 0.0);
    ASSERT_EQ(prim_call(A, nullptr, P.data(), n, CholQRPrecondMethod::trsm_identity, 0, 0.0, {}), 0);

    for (CholQRPrecondMethod method : {CholQRPrecondMethod::trsm_identity, CholQRPrecondMethod::trtri,
                                       CholQRPrecondMethod::geqp3, CholQRPrecondMethod::bqrrp}) {
        for (int64_t block_size : {int64_t(0), int64_t(5)}) {
            for (bool gemm_left : {false, true}) {
                for (bool symmetrize : {true, false}) {
                    for (double ratio : {1.0, 0.25}) {
                        if (ratio != 1.0 && method != CholQRPrecondMethod::bqrrp) continue;
                        RandBLAS::RNGState<RandBLAS::DefaultRNG> st(43);
                        RandLAPACK::CholQROptions<double> opts;
                        opts.gram_left_gemm  = gemm_left;
                        opts.symmetrize_gram = symmetrize;
                        std::vector<double> R(n * n, 0.0);
                        SCOPED_TRACE(::testing::Message() << "method " << (int)method << ", block " << block_size
                                     << ", gram_left_gemm " << gemm_left << ", symmetrize " << symmetrize
                                     << ", ratio " << ratio);
                        ASSERT_EQ(prim_call(A, P.data(), R.data(), n, method, block_size, 0.0, opts, nullptr, &st, ratio), 0);
                        auto [fact, orth] = RandLAPACK::testing::verify_R_factor(A_data.data(), m, n, R.data(), n);
                        EXPECT_LT(fact, 1e-12);
                        EXPECT_LT(orth, 1e-10);
                    }
                }
            }
        }
    }
}

// Every method must refuse a singular preconditioner (an exact zero diagonal entry of P) and
// leave R untouched.
TEST_F(TestCholQRPrimitive, singular_preconditioner_is_reported) {
    int64_t m = 60, n = 20;
    auto A_data = random_matrix<double>(m, n, 201);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> P(n * n, 0.0);
    RandLAPACK::util::eye(n, n, P.data());
    P[5 + 5 * n] = 0.0;
    for (CholQRPrecondMethod method : {CholQRPrecondMethod::trsm_identity, CholQRPrecondMethod::trtri,
                                       CholQRPrecondMethod::geqp3, CholQRPrecondMethod::bqrrp}) {
        RandBLAS::RNGState<RandBLAS::DefaultRNG> st(1);
        std::vector<double> R(n * n, -5.0);
        EXPECT_EQ(prim_call(A, P.data(), R.data(), n, method, 0, 0.0, {}, nullptr, &st),
                  RandLAPACK::kCholQRSingularPreconditioner) << (int)method;
        for (double v : R) ASSERT_EQ(v, -5.0) << "R must not be written on failure";
    }
}

TEST_F(TestCholQRPrimitive, missing_scratch_or_state_throws) {
    int64_t m = 40, n = 10;
    auto A_data = random_matrix<double>(m, n, 9);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> P(n * n, 0.0), R(n * n, 0.0);
    RandLAPACK::util::eye(n, n, P.data());
    EXPECT_THROW(prim_call(A, P.data(), R.data(), n, CholQRPrecondMethod::bqrrp, 0, 0.0, {}, nullptr, nullptr),
                 RandLAPACK::Error);
    std::vector<double> R_pre(n * n), G(n * n), A_temp(m * n);
    long t[6] = {0};
    EXPECT_THROW((RandLAPACK::cholqr_primitive<double, DLO>(
        A, P.data(), n, R.data(), n, CholQRPrecondMethod::geqp3, 0, 1.0, R_pre.data(), G.data(),
        A_temp.data(), /*Z_buf=*/nullptr, (RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
        t[0], t[1], t[2], t[3], t[4], t[5], false)), RandLAPACK::Error);
    // R_pre may be null only on trsm_identity's default route (left factor by TRSM).
    auto call_no_R_pre = [&](CholQRPrecondMethod method, bool left_gemm) {
        RandLAPACK::CholQROptions<double> opts;
        opts.gram_left_gemm = left_gemm;
        std::vector<double> Z_buf(n * n);
        RandBLAS::RNGState<RandBLAS::DefaultRNG> st(3);
        return RandLAPACK::cholqr_primitive<double, DLO>(
            A, P.data(), n, R.data(), n, method, 0, 1.0, /*R_pre=*/nullptr, G.data(), A_temp.data(),
            Z_buf.data(), &st, t[0], t[1], t[2], t[3], t[4], t[5], false, 0.0, opts);
    };
    EXPECT_EQ(call_no_R_pre(CholQRPrecondMethod::trsm_identity, false), 0);
    EXPECT_THROW(call_no_R_pre(CholQRPrecondMethod::trsm_identity, true), RandLAPACK::Error);
    for (CholQRPrecondMethod method : {CholQRPrecondMethod::trtri, CholQRPrecondMethod::geqp3,
                                       CholQRPrecondMethod::bqrrp})
        EXPECT_THROW(call_no_R_pre(method, false), RandLAPACK::Error) << (int)method;
}

// P may be R itself (ldp == ldr), as cholqr_iterate passes it: the factor equals that of a
// separate copy of P, also with a padded leading dimension and P's strict lower triangle
// poisoned. ldp < n, and P == R with ldp != ldr, throw before R is written.
TEST_F(TestCholQRPrimitive, preconditioner_may_alias_R) {
    int64_t m = 90, n = 17, ldr = n + 3;
    auto A_data = hard_matrix(m, n, 4.0, 2.0, 61);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> P(n * n, 0.0);
    ASSERT_EQ(prim_call(A, nullptr, P.data(), n, CholQRPrecondMethod::trsm_identity, 0, 0.0, {}), 0);
    for (CholQRPrecondMethod method : {CholQRPrecondMethod::trsm_identity, CholQRPrecondMethod::trtri,
                                       CholQRPrecondMethod::geqp3}) {
        for (bool left_gemm : {false, true}) {
            RandLAPACK::CholQROptions<double> opts;
            opts.gram_left_gemm = left_gemm;
            std::vector<double> R_sep(n * n, 0.0), R_alias(ldr * n, std::numeric_limits<double>::quiet_NaN());
            ASSERT_EQ(prim_call(A, P.data(), R_sep.data(), n, method, 5, 0.0, opts), 0);
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i <= j; ++i) R_alias[i + j * ldr] = P[i + j * n];
            std::vector<double> R_pre(n * n), G(n * n), A_temp(m * 5), Z_buf(n * 5);
            long t[6] = {0};
            ASSERT_EQ((RandLAPACK::cholqr_primitive<double, DLO>(
                A, R_alias.data(), ldr, R_alias.data(), ldr, method, 5, 1.0, R_pre.data(), G.data(),
                A_temp.data(), Z_buf.data(), (RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
                t[0], t[1], t[2], t[3], t[4], t[5], false, 0.0, opts)), 0) << (int)method << " " << left_gemm;
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = j + 1; i < n; ++i) ASSERT_EQ(R_alias[i + j * ldr], 0.0) << i << " " << j;
            std::vector<double> R_c(n * n);
            lapack::lacpy(lapack::MatrixType::General, n, n, R_alias.data(), ldr, R_c.data(), n);
            EXPECT_LE(rel_fro_error(R_c.data(), R_sep.data(), n * n), 1e-13) << (int)method << " " << left_gemm;
        }
    }
    std::vector<double> R(n * n, -2.0), G(n * n), A_temp(m * n);
    long t[6] = {0};
    auto call = [&](const double* Pp, int64_t ldp, double* Rp, int64_t ldr_) {
        return RandLAPACK::cholqr_primitive<double, DLO>(
            A, Pp, ldp, Rp, ldr_, CholQRPrecondMethod::trsm_identity, 0, 1.0, nullptr, G.data(),
            A_temp.data(), nullptr, (RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
            t[0], t[1], t[2], t[3], t[4], t[5], false);
    };
    EXPECT_THROW(call(P.data(), n - 1, R.data(), n), RandLAPACK::Error);
    std::vector<double> R_big(ldr * n, -2.0);
    EXPECT_THROW(call(R_big.data(), n, R_big.data(), ldr), RandLAPACK::Error);
    for (double v : R) ASSERT_EQ(v, -2.0);
    for (double v : R_big) ASSERT_EQ(v, -2.0);
}

// With bqrrp a block ratio outside (0, 1] throws before R is written; other methods ignore it.
TEST_F(TestCholQRPrimitive, bqrrp_block_ratio_outside_unit_interval_throws) {
    int64_t m = 40, n = 10;
    auto A_data = random_matrix<double>(m, n, 19);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> P(n * n, 0.0), R(n * n, -4.0);
    RandLAPACK::util::eye(n, n, P.data());
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    for (double ratio : {0.0, -0.5, 1.5, nan, inf}) {
        RandBLAS::RNGState<RandBLAS::DefaultRNG> st(7);
        EXPECT_THROW(prim_call(A, P.data(), R.data(), n, CholQRPrecondMethod::bqrrp, 0, 0.0, {}, nullptr, &st, ratio),
                     RandLAPACK::Error) << "ratio " << ratio;
    }
    for (double v : R) ASSERT_EQ(v, -4.0);
    EXPECT_EQ(prim_call(A, P.data(), R.data(), n, CholQRPrecondMethod::geqp3, 0, 0.0, {}, nullptr, nullptr, nan), 0);
}

// max_retries = 0 on an exactly singular Gram: potrf's positive info, no retry, R untouched.
TEST_F(TestCholQRPrimitive, potrf_exhaustion_returns_positive_info_and_leaves_R_untouched) {
    int64_t m = 60, n = 20;
    auto A_data = random_matrix<double>(m, n, 211);
    for (int64_t i = 0; i < m; ++i) A_data[3 * m + i] = 0.0;
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> R(n * n, -777.0);
    RandLAPACK::CholQRShiftRecord<double> rec;
    rec.n_retries = -7;
    int info = prim_call(A, nullptr, R.data(), n, CholQRPrecondMethod::trsm_identity, 0, 0.0, {}, &rec);
    EXPECT_GT(info, 0);
    EXPECT_EQ(rec.n_retries, 0);
    EXPECT_EQ(rec.applied_shift, 0.0);
    EXPECT_GT(rec.gram_trace, 0.0);
    for (double v : R) EXPECT_EQ(v, -777.0);
}

// After a forced retry, R factors the shifted Gram: R^T R = A^T A + s I with s the reported
// shift.
TEST_F(TestCholQRPrimitive, retried_factor_matches_the_shifted_gram) {
    int64_t m = 80, n = 16;
    auto A_data = random_matrix<double>(m, n, 213);
    for (int64_t i = 0; i < m; ++i) A_data[2 * m + i] = 0.0;
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    RandLAPACK::CholQROptions<double> opts;
    opts.max_retries = -1;
    RandLAPACK::CholQRShiftRecord<double> rec;
    std::vector<double> R(n * n, 0.0);
    ASSERT_EQ(prim_call(A, nullptr, R.data(), n, CholQRPrecondMethod::trsm_identity, 4, 0.0, opts, &rec), 0);
    ASSERT_GE(rec.n_retries, 1);
    ASSERT_GT(rec.applied_shift, 0.0);
    std::vector<double> RtR(n * n, 0.0), G(n * n, 0.0);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, n, n, n, 1.0, R.data(), n, R.data(), n, 0.0, RtR.data(), n);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, n, n, m, 1.0, A_data.data(), m, A_data.data(), m, 0.0, G.data(), n);
    for (int64_t i = 0; i < n; ++i) G[i + i * n] += rec.applied_shift;
    EXPECT_LT(rel_fro_error(RtR.data(), G.data(), n * n), 1e-13);
}

namespace {
// R factor (geqrf) of a Gaussian 16n x n matrix: a generic, well-conditioned upper triangular
// matrix (condition number below 2 for the sizes and seeds tested here).
std::vector<double> gaussian_r_factor(int64_t n, uint32_t seed) {
    const int64_t m = 16 * n;
    auto B = random_matrix<double>(m, n, seed);
    std::vector<double> tau(n), P(n * n, 0.0);
    lapack::geqrf(m, n, B.data(), m, tau.data());
    lapack::lacpy(lapack::MatrixType::Upper, n, n, B.data(), m, P.data(), n);
    return P;
}

// R = R^chol P from the Gram G = A^T A P^{-1} formed as the trsm_identity route forms it (block
// width b), the full-width solve G := P^{-T} G, and potrf.
std::vector<double> full_width_solve_factor(DLO& A, const std::vector<double>& P, int64_t b) {
    int64_t m = A.n_rows, n = A.n_cols;
    std::vector<double> G(n * n), A_temp(m * b), R = P;
    long t[4] = {0};
    RandLAPACK::blocked_preconditioned_gram<double, DLO>(A, nullptr, P.data(), n, G.data(), m, n, b,
                                                         A_temp.data(), nullptr, /*skip_left_factor=*/true,
                                                         t[0], t[1], t[2], t[3], false);
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper, blas::Op::Trans,
               blas::Diag::NonUnit, n, n, 1.0, P.data(), n, G.data(), n);
    EXPECT_EQ(lapack::potrf(lapack::Uplo::Upper, n, G.data(), n), 0);
    RandLAPACK::util::get_U(n, n, G.data(), n);
    blas::trmm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper, blas::Op::NoTrans,
               blas::Diag::NonUnit, n, n, 1.0, G.data(), n, R.data(), n);
    return R;
}
}

// Unsymmetrized, the TRSM route solves only the upper triangle of P^{-T} G, so R must match a
// full-width-solve factor from the same Gram, over panel widths and with or without the mirror.
// 1e-11 is five times 2 n eps kappa(P)^2 kappa(H) for these well-conditioned Gaussian factors.
TEST_F(TestCholQRPrimitive, unsymmetrized_trsm_route_matches_full_width_solve) {
    struct Case { int64_t n; std::vector<int64_t> blocks; };
    for (const Case& c : {Case{1, {0}}, Case{17, {0, 1, 5, 16, 64}}, Case{64, {0, 1, 5, 16, 64}},
                          Case{300, {0}}}) {
        const int64_t n = c.n, m = 16 * n;
        auto A_data = random_matrix<double>(m, n, 71);
        DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
        auto P = gaussian_r_factor(n, 73);
        for (int64_t block_size : c.blocks) {
            auto R_ref = full_width_solve_factor(A, P, RandLAPACK::gram_block_width(block_size, n));
            for (int max_retries : {0, -1}) {
                SCOPED_TRACE(::testing::Message() << "n " << n << ", block " << block_size
                             << ", max_retries " << max_retries);
                RandLAPACK::CholQROptions<double> opts;
                opts.symmetrize_gram = false;
                opts.max_retries     = max_retries;
                RandLAPACK::CholQRShiftRecord<double> rec;
                std::vector<double> R(n * n, 0.0);
                ASSERT_EQ(prim_call(A, P.data(), R.data(), n, CholQRPrecondMethod::trsm_identity, block_size,
                                    0.0, opts, &rec), 0);
                EXPECT_EQ(rec.n_retries, 0);
                EXPECT_LE(rel_fro_error(R.data(), R_ref.data(), n * n), 1e-11);
            }
        }
    }
}

// Unsymmetrized, a retry must restore from the mirror of the solved upper triangle. A zero first
// column of A makes attempt 0 fail at pivot 1; then (R P^{-1})^T (R P^{-1}) = H + s I, s the
// recorded shift, to about (m + n) eps kappa(P)^2 (tolerance 1e-11). Without retries it fails.
TEST_F(TestCholQRPrimitive, unsymmetrized_retry_restores_the_solved_gram) {
    const double eps = std::numeric_limits<double>::epsilon();
    struct Case { int64_t n; std::vector<int64_t> blocks; };
    for (const Case& c : {Case{18, {1, 4, 6, 0}}, Case{300, {0}}}) {
        const int64_t n = c.n, m = 16 * n;
        auto A_data = random_matrix<double>(m, n, 81);
        std::fill(A_data.begin(), A_data.begin() + m, 0.0);
        DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
        auto P = gaussian_r_factor(n, 83);

        // H = W^T W with W = A P^{-1}.
        std::vector<double> W = A_data, H(n * n, 0.0);
        blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans,
                   blas::Diag::NonUnit, m, n, 1.0, P.data(), n, W.data(), m);
        blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, n, n, m, 1.0, W.data(), m,
                   W.data(), m, 0.0, H.data(), n);

        for (int64_t block_size : c.blocks) {
            SCOPED_TRACE(::testing::Message() << "n " << n << ", block " << block_size);
            RandLAPACK::CholQROptions<double> opts;
            opts.symmetrize_gram = false;
            opts.max_retries     = -1;
            RandLAPACK::CholQRShiftRecord<double> rec;
            std::vector<double> R(n * n, 0.0);
            ASSERT_EQ(prim_call(A, P.data(), R.data(), n, CholQRPrecondMethod::trsm_identity, block_size,
                                0.0, opts, &rec), 0);
            EXPECT_EQ(rec.n_retries, 1);
            EXPECT_EQ(rec.applied_shift, eps * rec.gram_trace);

            std::vector<double> X = R, XtX(n * n, 0.0), H_s = H;
            blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans,
                       blas::Diag::NonUnit, n, n, 1.0, P.data(), n, X.data(), n);
            blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, n, n, n, 1.0, X.data(), n,
                       X.data(), n, 0.0, XtX.data(), n);
            for (int64_t i = 0; i < n; ++i) H_s[i + i * n] += rec.applied_shift;
            EXPECT_LT(rel_fro_error(XtX.data(), H_s.data(), n * n), 1e-11);

            opts.max_retries = 0;
            std::vector<double> R0(n * n, -9.0);
            EXPECT_EQ(prim_call(A, P.data(), R0.data(), n, CholQRPrecondMethod::trsm_identity, block_size,
                                0.0, opts, &rec), 1);
            for (double v : R0) ASSERT_EQ(v, -9.0);
        }
    }
}

// Non-finite input: an infinite shift factor and NaN data both stop at attempt 0 with
// kCholQRNonFinite and leave R untouched.
TEST_F(TestCholQRPrimitive, nonfinite_shift_or_data_is_reported_at_attempt_0) {
    int64_t m = 60, n = 20;
    auto A_data = random_matrix<double>(m, n, 231);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    RandLAPACK::CholQROptions<double> opts;
    opts.max_retries = -1;
    RandLAPACK::CholQRShiftRecord<double> rec;
    std::vector<double> R(n * n, -777.0);
    EXPECT_EQ(prim_call(A, nullptr, R.data(), n, CholQRPrecondMethod::trsm_identity, 0,
                        std::numeric_limits<double>::infinity(), opts, &rec), RandLAPACK::kCholQRNonFinite);
    EXPECT_EQ(rec.n_retries, 0);
    EXPECT_EQ(rec.applied_shift, 0.0);

    A_data[7 + 3 * m] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(prim_call(A, nullptr, R.data(), n, CholQRPrecondMethod::trsm_identity, 0, 0.0, opts, &rec),
              RandLAPACK::kCholQRNonFinite);
    EXPECT_EQ(rec.n_retries, 0);
    for (double v : R) EXPECT_EQ(v, -777.0);
}

namespace {
// Dense operator whose Gram picks up a NaN in one off-diagonal entry: A^T applies poison
// entry (0, 1) of C. trace(G) stays finite, so only the retries can bail.
struct NaNOffDiagonalGram {
    using scalar_t = double;
    const int64_t n_rows;
    const int64_t n_cols;
    DLO& A;
    void operator()(blas::Layout layout, blas::Op trans_A, blas::Op trans_B, int64_t m, int64_t n, int64_t k,
                    double alpha, const double* B, int64_t ldb, double beta, double* C, int64_t ldc) {
        A(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        if (trans_A == blas::Op::Trans && n >= 2) C[0 + 1 * ldc] = std::numeric_limits<double>::quiet_NaN();
    }
};

// Unbounded retries (max_retries = -1) on a Gram that no shift can make factorable.
int nan_gram_call(double shift_growth, RandLAPACK::CholQRShiftRecord<double>& rec) {
    int64_t m = 40, n = 6;
    auto A_data = random_matrix<double>(m, n, 233);
    DLO A_dense(m, n, A_data.data(), m, blas::Layout::ColMajor);
    NaNOffDiagonalGram A{m, n, A_dense};
    std::vector<double> R(n * n), G(n * n), A_temp(m * n);
    long t[6] = {0};
    RandLAPACK::CholQROptions<double> opts;
    opts.max_retries  = -1;
    opts.shift_growth = shift_growth;
    return RandLAPACK::cholqr_primitive<double, NaNOffDiagonalGram>(
        A, nullptr, n, R.data(), n, CholQRPrecondMethod::trsm_identity, 0, 1.0, nullptr, G.data(),
        A_temp.data(), nullptr, (RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
        t[0], t[1], t[2], t[3], t[4], t[5], false, 0.0, opts, &rec);
}
}

// With fast shift growth the retries escalate until the shift overflows and the primitive
// bails with kCholQRNonFinite after at least one retry.
TEST_F(TestCholQRPrimitive, nonfinite_shift_on_a_retry_is_reported) {
    RandLAPACK::CholQRShiftRecord<double> rec;
    EXPECT_EQ(nan_gram_call(1e100, rec), RandLAPACK::kCholQRNonFinite);
    EXPECT_GE(rec.n_retries, 1);
    EXPECT_TRUE(std::isfinite(rec.gram_trace));
}

// With slow growth the shift stays finite, so the attempt cap ends the retries: potrf's
// positive info after kMaxShiftAttempts attempts, all but the first of them retries.
TEST_F(TestCholQRPrimitive, attempt_cap_ends_unbounded_retries) {
    RandLAPACK::CholQRShiftRecord<double> rec;
    EXPECT_GT(nan_gram_call(1.01, rec), 0);
    EXPECT_EQ(rec.n_retries, RandLAPACK::kMaxShiftAttempts - 1);
    double expected = std::numeric_limits<double>::epsilon() * rec.gram_trace * std::pow(1.01, rec.n_retries - 1);
    EXPECT_NEAR(rec.applied_shift, expected, 1e-10 * expected);
}

namespace {
// A one-column Gram of 0.75 max(T) is finite, but a shift factor of 1 takes its shifted
// diagonal past max(T): kCholQRNonFinite, R untouched. A shift factor of 0.25 stays finite.
template <typename T>
void check_shifted_diagonal_overflow() {
    const int64_t m = 4, n = 1;
    std::vector<T> A_data(m, T(0)), G(1), A_temp(m);
    A_data[0] = std::sqrt(T(0.75) * std::numeric_limits<T>::max());
    RandLAPACK::linops::DenseLinOp<T> A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    long t[6] = {0};
    auto call = [&](T shift_factor, T* R) {
        return RandLAPACK::cholqr_primitive<T, RandLAPACK::linops::DenseLinOp<T>>(
            A, nullptr, 1, R, 1, CholQRPrecondMethod::trsm_identity, 0, T(1), nullptr, G.data(),
            A_temp.data(), nullptr, (RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
            t[0], t[1], t[2], t[3], t[4], t[5], false, shift_factor);
    };
    T R = T(-7);
    EXPECT_EQ(call(T(1), &R), RandLAPACK::kCholQRNonFinite);
    EXPECT_EQ(R, T(-7));
    EXPECT_EQ(call(T(0.25), &R), 0);
    EXPECT_TRUE(std::isfinite(R));
    EXPECT_GT(R, T(0));
}
}

TEST_F(TestCholQRPrimitive, shifted_diagonal_overflow_is_reported) {
    check_shifted_diagonal_overflow<double>();
    check_shifted_diagonal_overflow<float>();
}

// A negative or NaN shift factor and shift_growth <= 1 are rejected up front with
// kCholQRInvalidShift, and the record is reset rather than left stale.
TEST_F(TestCholQRPrimitive, invalid_shift_options_rejected) {
    int64_t m = 60, n = 20;
    auto A_data = random_matrix<double>(m, n, 3);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQROptions<double> opts;
    RandLAPACK::CholQRShiftRecord<double> rec;
    rec.n_retries = -7; rec.applied_shift = -7.0; rec.gram_trace = -7.0;
    EXPECT_EQ(prim_call(A, nullptr, R.data(), n, CholQRPrecondMethod::trsm_identity, 0, -1.0, opts, &rec),
              RandLAPACK::kCholQRInvalidShift);
    EXPECT_EQ(rec.n_retries, 0);
    EXPECT_EQ(rec.applied_shift, 0.0);
    EXPECT_EQ(rec.gram_trace, 0.0);
    EXPECT_EQ(prim_call(A, nullptr, R.data(), n, CholQRPrecondMethod::trsm_identity, 0,
                        std::numeric_limits<double>::quiet_NaN(), opts, &rec), RandLAPACK::kCholQRInvalidShift);
    opts.shift_growth = 1.0;
    EXPECT_EQ(prim_call(A, nullptr, R.data(), n, CholQRPrecondMethod::trsm_identity, 0, 0.0, opts, &rec),
              RandLAPACK::kCholQRInvalidShift);
}

// Caller bugs (m < n, ldr < n, null R) throw RandLAPACK::Error.
TEST_F(TestCholQRPrimitive, invalid_input_throws_randlapack_error) {
    int64_t m = 60, n = 20;
    auto A_data = random_matrix<double>(m, n, 221);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> R(n * n, 0.0);
    int64_t m_short = 10;
    std::vector<double> A_short(m_short * n, 1.0);
    DLO A_short_op(m_short, n, A_short.data(), m_short, blas::Layout::ColMajor);
    EXPECT_THROW(prim_call(A_short_op, nullptr, R.data(), n, CholQRPrecondMethod::trsm_identity, 0, 0.0, {}), RandLAPACK::Error);
    EXPECT_THROW(prim_call(A, nullptr, R.data(), n - 1, CholQRPrecondMethod::trsm_identity, 0, 0.0, {}), RandLAPACK::Error);
    EXPECT_THROW(prim_call(A, nullptr, nullptr, n, CholQRPrecondMethod::trsm_identity, 0, 0.0, {}), RandLAPACK::Error);
}

class TestCholQRIterate : public ::testing::Test {};

// Pass 1 fails deterministically on an exactly singular Gram with no retries: index 1 and the
// primitive's positive info.
TEST_F(TestCholQRIterate, pass_1_failure_reports_index_and_cause) {
    int64_t m = 60, n = 20;
    auto A_data = random_matrix<double>(m, n, 231);
    for (int64_t i = 0; i < m; ++i) A_data[2 * m + i] = 0.0;
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQRShiftRecord<double> records[2];
    int status = -9;
    EXPECT_EQ((RandLAPACK::cholqr_iterate<double, DLO>(A, R.data(), n, 0, 2, 0.0, 0.0, {}, false,
                                                         nullptr, records, &status)), 1);
    EXPECT_GT(status, 0);
    EXPECT_GT(records[0].gram_trace, 0.0);
    EXPECT_EQ(records[1].gram_trace, 0.0);   // pass 2 never ran
}

// A = [e_1, 0]: pass 1 with a positive shift factor factors diag(1 + s, s); pass 2's Gram is
// diag(1/(1 + s), 0), whose unshifted Cholesky fails at the second pivot. The call returns 2,
// keeps pass 1's factor in R, and leaves pass 3's record at its default.
TEST_F(TestCholQRIterate, pass_2_failure_keeps_the_pass_1_factor) {
    int64_t m = 4, n = 2;
    std::vector<double> A_data(m * n, 0.0);
    A_data[0] = 1.0;
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    const double s = 0.5;
    std::vector<double> R1(n * n, 0.0), R(n * n, 0.0);
    ASSERT_EQ((RandLAPACK::cholqr_iterate<double, DLO>(A, R1.data(), n, 0, 1, s, 0.0, {}, false)), 0);
    RandLAPACK::CholQRShiftRecord<double> records[3];
    int status = -9;
    EXPECT_EQ((RandLAPACK::cholqr_iterate<double, DLO>(A, R.data(), n, 0, 3, s, 0.0, {}, false,
                                                         nullptr, records, &status)), 2);
    EXPECT_EQ(status, 2);   // potrf: second leading minor not positive definite
    for (int64_t i = 0; i < n * n; ++i) EXPECT_EQ(R[i], R1[i]) << i;
    EXPECT_EQ(records[0].applied_shift, s * 1.0);
    EXPECT_GT(records[1].gram_trace, 0.0);
    EXPECT_EQ(records[2].gram_trace, 0.0);
    EXPECT_EQ(records[2].n_retries, 0);
}

// A positive later-pass shift factor f shifts that pass's first attempt by f * trace(G).
TEST_F(TestCholQRIterate, positive_later_pass_shift_factor_is_applied) {
    int64_t m = 60, n = 20;
    auto A_data = random_matrix<double>(m, n, 17);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQRShiftRecord<double> records[2];
    const double f = 1e-8;
    ASSERT_EQ((RandLAPACK::cholqr_iterate<double, DLO>(A, R.data(), n, 0, 2, 0.0, f, {}, false,
                                                         nullptr, records)), 0);
    EXPECT_EQ(records[0].applied_shift, 0.0);
    EXPECT_EQ(records[1].n_retries, 0);
    EXPECT_GT(records[1].gram_trace, 0.0);
    EXPECT_EQ(records[1].applied_shift, f * records[1].gram_trace);
}

// Both shift factors are validated before pass 1 and fail as pass 1 (R untouched); num_iters
// must be >= 1.
TEST_F(TestCholQRIterate, invalid_arguments) {
    int64_t m = 30, n = 6;
    auto A_data = random_matrix<double>(m, n, 5);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> R(n * n, -3.0);
    int status = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ((RandLAPACK::cholqr_iterate<double, DLO>(A, R.data(), n, 0, 2, -1.0, 0.0, {}, false, nullptr, nullptr, &status)), 1);
    EXPECT_EQ(status, RandLAPACK::kCholQRInvalidShift);
    EXPECT_EQ((RandLAPACK::cholqr_iterate<double, DLO>(A, R.data(), n, 0, 2, 0.0, nan, {}, false, nullptr, nullptr, &status)), 1);
    EXPECT_EQ(status, RandLAPACK::kCholQRInvalidShift);
    for (double v : R) EXPECT_EQ(v, -3.0);
    EXPECT_THROW((RandLAPACK::cholqr_iterate<double, DLO>(A, R.data(), n, 0, 0, 0.0, 0.0, {}, false)), RandLAPACK::Error);
}

// BQRRP truncates the rank of a preconditioner whose diagonal spans more than 1/eps, zeroing
// trailing pivots of its triangular factor: the primitive reports a singular preconditioner
// instead of dividing by them. geqp3 keeps every pivot and succeeds.
TEST_F(TestCholQRPrimitive, bqrrp_rank_truncation_is_reported) {
    int64_t m = 50, n = 8;
    auto A_data = random_matrix<double>(m, n, 251);
    DLO A(m, n, A_data.data(), m, blas::Layout::ColMajor);
    std::vector<double> P(n * n, 0.0);
    for (int64_t j = 0; j < n; ++j) {
        P[j + j * n] = std::pow(10.0, -3.0 * (double)j);
        for (int64_t i = 0; i < j; ++i) P[i + j * n] = 1e-3 * P[j + j * n];
    }
    RandBLAS::RNGState<RandBLAS::DefaultRNG> st(3);
    std::vector<double> R(n * n, -5.0);
    EXPECT_EQ(prim_call(A, P.data(), R.data(), n, CholQRPrecondMethod::bqrrp, 0, 0.0, {}, nullptr, &st),
              RandLAPACK::kCholQRSingularPreconditioner);
    for (double v : R) ASSERT_EQ(v, -5.0);
    EXPECT_EQ(prim_call(A, P.data(), R.data(), n, CholQRPrecondMethod::geqp3, 0, 0.0, {}), 0);
}
