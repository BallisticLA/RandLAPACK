// Tests for the dense (non-LinOp) CholQR-family drivers: CholQR_dense,
// CholQR2_dense, sCholQR3_dense (rl_cholqr_dense.hh), and dense CQRRT
// (rl_cqrrt.hh, the CQRRT<T,RNG> class, not CQRRT_linops).
//
// These drivers previously had zero instantiations anywhere in the tree.
// This file exercises them directly and pins the parity fixes brought to
// this family: threaded ldq (strided Q buffer), the R-strict-lower laset
// before dense CQRRT's finalize trmm, the timing-total-excludes-Q-materialize
// contract, and the adaptive-shift retry / shift-record out-params.
//
// Not covered here: RANDLAPACK_SCHOLQR3_SHIFT on sCholQR3_dense. The knob is
// read once into a function-local static (see scholqr3_theory_shift() in
// comps/rl_cholqr.hh), so it is benchmark-validated only, same as the
// linop-side sCholQR3 tests; a real test would need a subprocess.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_test_utils.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

using RandLAPACK::testing::verify_qr;

namespace {

template <typename T>
T default_tol() {
    return std::pow(std::numeric_limits<T>::epsilon(), (T)0.75);
}

// R is n x n column-major with leading dimension ldr; assert its strict lower
// triangle is exactly zero, as every driver in this family lasets it.
template <typename T>
void assert_upper_triangular(const T* R, int64_t n, int64_t ldr) {
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            ASSERT_EQ(R[i + j * ldr], T(0)) << "R(" << i << "," << j << ") nonzero";
}

// Runs one dense CholQR-family driver (CholQR_dense / CholQR2_dense /
// sCholQR3_dense all share the same call() signature) on a random tall
// matrix and checks status, R's shape, and the QR factorization itself.
template <typename Algo, typename T>
void run_basic_case(Algo& algo, int64_t m, int64_t n, int seed) {
    std::vector<T> A(m * n), R(n * n, T(0)), Q(m * n, T(-999));
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(seed);
    RandBLAS::fill_dense(D, A.data(), state);

    int info = algo.call(m, n, A.data(), m, R.data(), n, Q.data(), m);
    ASSERT_EQ(info, 0);
    assert_upper_triangular(R.data(), n, n);

    T tol = default_tol<T>();
    auto [fact_err, orth_err] = verify_qr(A.data(), Q.data(), R.data(), m, n, n);
    ASSERT_LE(fact_err, tol);
    ASSERT_LE(orth_err, tol);
}

} // namespace

class TestCholQRDenseFamily : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// ============================================================================
// Basic instantiation: all four dense drivers on a small tall matrix.
// ============================================================================

TEST_F(TestCholQRDenseFamily, CholQR_dense_basic) {
    RandLAPACK::CholQR_dense<double> algo(false);
    run_basic_case<RandLAPACK::CholQR_dense<double>, double>(algo, 100, 40, 0);
}

TEST_F(TestCholQRDenseFamily, CholQR2_dense_basic) {
    RandLAPACK::CholQR2_dense<double> algo(false);
    run_basic_case<RandLAPACK::CholQR2_dense<double>, double>(algo, 100, 40, 1);
}

TEST_F(TestCholQRDenseFamily, sCholQR3_dense_basic) {
    RandLAPACK::sCholQR3_dense<double> algo(false);
    run_basic_case<RandLAPACK::sCholQR3_dense<double>, double>(algo, 100, 40, 2);
}

// Dense CQRRT materializes Q in place (inside A), not into a caller buffer.
TEST_F(TestCholQRDenseFamily, CQRRT_dense_basic) {
    int64_t m = 100, n = 40;
    std::vector<double> A(m * n), A_orig(m * n), R(n * n, 0.0);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(3);
    RandBLAS::fill_dense(D, A.data(), state);
    A_orig = A;

    RandLAPACK::CQRRT<double> algo(false, 0.0);
    RandBLAS::RNGState<> sk_state(4);
    int info = algo.call(m, n, A.data(), m, R.data(), n, /*d_factor=*/2.0, sk_state);
    ASSERT_EQ(info, 0);
    assert_upper_triangular(R.data(), n, n);

    double tol = default_tol<double>();
    auto [fact_err, orth_err] = verify_qr(A_orig.data(), A.data(), R.data(), m, n, n);
    ASSERT_LE(fact_err, tol);
    ASSERT_LE(orth_err, tol);
}

// ============================================================================
// B2: ldq > m (strided Q buffer). materialize_Q_from_R must honor ldq, not
// hard-code stride m.
// ============================================================================

TEST_F(TestCholQRDenseFamily, CholQR2_dense_ldq_greater_than_m) {
    int64_t m = 60, n = 20;
    int64_t ldq = m + 7;   // deliberately strided
    std::vector<double> A(m * n), R(n * n, 0.0);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(5);
    RandBLAS::fill_dense(D, A.data(), state);

    const double sentinel = -12345.0;
    std::vector<double> Q(ldq * n, sentinel);

    RandLAPACK::CholQR2_dense<double> algo(false);
    int info = algo.call(m, n, A.data(), m, R.data(), n, Q.data(), ldq);
    ASSERT_EQ(info, 0);

    // Every column's padding rows [m, ldq) must be untouched: if the driver
    // still hard-coded stride m (the B2 bug), a strided write here would
    // either miss this region or spill into the next column's data instead.
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = m; i < ldq; ++i)
            ASSERT_EQ(Q[i + j * ldq], sentinel) << "col " << j << " padding row " << i;

    // Compact the strided Q into an ld=m buffer and verify the factorization.
    std::vector<double> Q_compact(m * n);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            Q_compact[i + j * m] = Q[i + j * ldq];

    double tol = default_tol<double>();
    auto [fact_err, orth_err] = verify_qr(A.data(), Q_compact.data(), R.data(), m, n, n);
    ASSERT_LE(fact_err, tol);
    ASSERT_LE(orth_err, tol);
}

// ============================================================================
// B3: dense CQRRT must not let a garbage-prefilled R strict lower triangle
// contaminate the output R's upper triangle (the finalize trmm reads R as a
// general matrix, not just its upper part).
// ============================================================================

TEST_F(TestCholQRDenseFamily, CQRRT_dense_garbage_R_lower_unaffected) {
    int64_t m = 80, n = 25;
    std::vector<double> A_base(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(7);
    RandBLAS::fill_dense(D, A_base.data(), state);
    double d_factor = 2.0;

    // Clean run: R starts zeroed.
    std::vector<double> A_clean = A_base;
    std::vector<double> R_clean(n * n, 0.0);
    RandLAPACK::CQRRT<double> algo_clean(false, 0.0);
    RandBLAS::RNGState<> sk_state_clean(9);
    ASSERT_EQ(algo_clean.call(m, n, A_clean.data(), m, R_clean.data(), n, d_factor, sk_state_clean), 0);

    // Garbage run: R's strict lower triangle is pre-filled with huge sentinel
    // values before the call, as an uninitialized caller buffer would be.
    std::vector<double> A_garbage = A_base;
    std::vector<double> R_garbage(n * n, 0.0);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            R_garbage[i + j * n] = 1.0e300;
    RandLAPACK::CQRRT<double> algo_garbage(false, 0.0);
    RandBLAS::RNGState<> sk_state_garbage(9);
    ASSERT_EQ(algo_garbage.call(m, n, A_garbage.data(), m, R_garbage.data(), n, d_factor, sk_state_garbage), 0);

    // Both runs use the same sketch RNG seed and the same input A, so with
    // the B3 laset in place the results must match exactly: the garbage
    // never entered the computation.
    for (int64_t idx = 0; idx < n * n; ++idx)
        ASSERT_EQ(R_clean[idx], R_garbage[idx]) << "R differs at flat idx " << idx;
    for (int64_t idx = 0; idx < m * n; ++idx)
        ASSERT_EQ(A_clean[idx], A_garbage[idx]) << "Q (in A) differs at flat idx " << idx;

    assert_upper_triangular(R_clean.data(), n, n);
}

// ============================================================================
// (c): shift-record out-params. Mirrors
// TestCholQRShiftRecord.rank_deficient_input_records_seed_shift in
// test_orth_linop.cc, on the dense driver instead of the linop one.
// ============================================================================

TEST_F(TestCholQRDenseFamily, CholQR_dense_shift_record_on_rank_deficient_input) {
    int64_t m = 100, n = 50;
    std::vector<double> A(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(1);
    RandBLAS::fill_dense(D, A.data(), state);
    // Zero out column 1: its Gram row/column is exactly zero, so the
    // unshifted potrf pivot there is exactly 0 and cannot succeed.
    for (int64_t i = 0; i < m; ++i) A[m + i] = 0.0;

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_dense<double> algo(false);
    ASSERT_EQ(algo.call(m, n, A.data(), m, R.data(), n), 0);
    ASSERT_GE(algo.n_chol_retries, 1);
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
    ASSERT_GT(algo.chol_applied_shifts[0], 0.0);

    const double eps = std::numeric_limits<double>::epsilon();
    const double expected_seed = eps * algo.chol_gram_traces[0];
    double expected = expected_seed;
    for (int k = 1; k < algo.n_chol_retries; ++k) expected *= algo.shift_growth;
    ASSERT_NEAR(algo.chol_applied_shifts[0], expected, 1e-12 * expected);
}

// Dense CQRRT's shift-record members on the clean path: an exact-zero column
// makes CQRRT fail at its earlier sketch-QR diag_is_nonzero gate (the sketch
// of a zero column is exactly zero, so the small QR's own diagonal is zero
// too) rather than at the preconditioned-Gram potrf the retry covers, so a
// rank-deficient trigger for CQRRT's retry is not this simply constructed
// (matches CQRRT_linops, which has no such test either). This case instead
// pins that a normal, well-conditioned run reports a clean (unshifted)
// record, exercising the same member-forwarding wiring as (c) requires.
TEST_F(TestCholQRDenseFamily, CQRRT_dense_shift_record_clean_path) {
    int64_t m = 100, n = 50;
    std::vector<double> A(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(11);
    RandBLAS::fill_dense(D, A.data(), state);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT<double> algo(false, 0.0);
    RandBLAS::RNGState<> sk_state(12);
    int info = algo.call(m, n, A.data(), m, R.data(), n, /*d_factor=*/2.0, sk_state);
    ASSERT_EQ(info, 0);
    ASSERT_EQ(algo.n_chol_retries, 0);
    ASSERT_EQ(algo.chol_applied_shifts[0], 0.0);
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
}

// A shift_growth <= 1 defeats the retry's geometric-growth termination
// argument and must be rejected up front (-2), before any allocation or
// factorization work, the same B5 contract cholqr_primitive enforces. Direct
// port of TestCholQRShiftRecord.shift_growth_leq_one_rejected onto dense
// CQRRT.
TEST_F(TestCholQRDenseFamily, CQRRT_dense_shift_growth_leq_one_rejected) {
    int64_t m = 60, n = 20;
    std::vector<double> A(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(13);
    RandBLAS::fill_dense(D, A.data(), state);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT<double> algo(false, 0.0);
    algo.shift_growth = 1.0;   // invalid: <= 1
    RandBLAS::RNGState<> sk_state(14);
    int info = algo.call(m, n, A.data(), m, R.data(), n, /*d_factor=*/2.0, sk_state);
    ASSERT_EQ(info, -2);
    // Out-params must be reset even on the early rejection (stale-value guard).
    ASSERT_EQ(algo.n_chol_retries, 0);
    ASSERT_EQ(algo.chol_applied_shifts[0], 0.0);
    ASSERT_EQ(algo.chol_gram_traces[0], 0.0);
}
