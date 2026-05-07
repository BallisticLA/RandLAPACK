// Unit tests for RandLAPACK::IterRefineLSQ — iterative-refinement LSQ
// using R as a right preconditioner.
//
// Strategy: wrap a small dense tall A as a DenseLinOp, build R from
// lapack::geqrf on a copy of A, then verify the IR-LSQ solution matches
// a closed-form reference:
//   * unregularized cases  → lapack::gels on (A, b)
//   * Tikhonov case        → lapack::posv on (A^T A + lambda^2 I, A^T b)
// across well-conditioned, residualful, regularized, and imperfect-R cases.

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include <random>
#include <vector>


using RandLAPACK::IterRefineLSQ;
using RandLAPACK::linops::DenseLinOp;
using blas::Layout;


template <typename T>
static void fill_random(std::vector<T>& v, uint32_t seed, T scale = 1.0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (auto& x : v) x = scale * dist(rng);
}


// Build R from QR of A_copy (in-place destructive QR), zeroing the strictly
// lower triangle and copying the upper-triangular n × n into R.
template <typename T>
static void build_R_from_A(const T* A, int64_t m, int64_t n, T* R, int64_t ldr) {
    std::vector<T> A_copy(m * n);
    std::copy(A, A + m * n, A_copy.begin());
    std::vector<T> tau(n);
    lapack::geqrf(m, n, A_copy.data(), m, tau.data());
    // Copy upper-triangular n × n into R
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i <= j; ++i) {
            R[i + j * ldr] = A_copy[i + j * m];
        }
        for (int64_t i = j + 1; i < n; ++i) {
            R[i + j * ldr] = (T)0;
        }
    }
}


class TestIterRefineLSQ : public ::testing::Test {};


// Well-conditioned synthetic problem; exact R from QR(A). M should be
// numerically identity, so CG converges in ~1 iteration.
TEST_F(TestIterRefineLSQ, dense_well_conditioned) {
    using T = double;
    int64_t m = 80, n = 12;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 42);
    fill_random(x_true, 99);

    // b = A * x_true (exact RHS, in the column space of A).
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    // Reference: gels destroys both A and b.
    std::vector<T> A_ref(A.begin(), A.end()), b_ref(b.begin(), b.end());
    lapack::gels(blas::Op::NoTrans, m, n, 1, A_ref.data(), m, b_ref.data(), m);
    std::vector<T> x_ref(b_ref.begin(), b_ref.begin() + n);

    // Build R from QR(A) — perfect preconditioner.
    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);

    // Solve via IR-LSQ.
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*tol=*/1e-12, /*max_inner=*/50, /*n_steps=*/2);
    std::vector<T> x_ir(n, 0);
    int status = ir.call(J, R.data(), n, b.data(), m, x_ir.data(), n);
    EXPECT_EQ(status, 0);

    // x_ir should match x_ref (within accumulated rounding error).
    T diff_norm = 0;
    for (int64_t i = 0; i < n; ++i) {
        T d = x_ir[i] - x_ref[i];
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);
    T xref_norm = blas::nrm2(n, x_ref.data(), 1);
    EXPECT_LT(diff_norm / xref_norm, 1e-10);

    // CG should converge fast with a perfect preconditioner.
    EXPECT_LE(ir.inner_iters_per_step.front(), 5);
}


// LS with a real residual (b not exactly in range(A)).
TEST_F(TestIterRefineLSQ, dense_with_residual) {
    using T = double;
    int64_t m = 100, n = 7;

    std::vector<T> A(m * n), b(m), noise(m), x_true(n);
    fill_random(A, 7);
    fill_random(x_true, 17);
    fill_random(noise, 27, 0.05);

    // b = A * x_true + noise
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    for (int64_t i = 0; i < m; ++i) b[i] += noise[i];

    std::vector<T> A_ref(A.begin(), A.end()), b_ref(b.begin(), b.end());
    lapack::gels(blas::Op::NoTrans, m, n, 1, A_ref.data(), m, b_ref.data(), m);
    std::vector<T> x_ref(b_ref.begin(), b_ref.begin() + n);

    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);

    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    IterRefineLSQ<T> ir(1e-12, 50, 2);
    std::vector<T> x_ir(n, 0);
    int status = ir.call(J, R.data(), n, b.data(), m, x_ir.data(), n);
    EXPECT_EQ(status, 0);

    T diff_norm = 0;
    for (int64_t i = 0; i < n; ++i) {
        T d = x_ir[i] - x_ref[i];
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);
    T xref_norm = blas::nrm2(n, x_ref.data(), 1);
    EXPECT_LT(diff_norm / xref_norm, 1e-10);
}


// Tikhonov regularization: solving min ||Jx-b||² + λ²||x||² should match
// the closed-form solution x = (J^T J + λ² I)^{-1} J^T b.
TEST_F(TestIterRefineLSQ, tikhonov_regularization) {
    using T = double;
    int64_t m = 80, n = 12;
    T lambda = 0.3;

    std::vector<T> A(m * n), b(m);
    fill_random(A, 101);
    fill_random(b, 102);

    // Reference: solve (A^T A + λ² I) x = A^T b directly via lapack::posv.
    std::vector<T> Gram(n * n, 0.0), AtB(n, 0.0);
    blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, m, (T)1.0, A.data(), m, (T)0.0, Gram.data(), n);
    for (int64_t i = 0; i < n; ++i) Gram[i + i * n] += lambda * lambda;
    blas::gemv(blas::Layout::ColMajor, blas::Op::Trans, m, n, (T)1.0, A.data(), m,
               b.data(), 1, (T)0.0, AtB.data(), 1);
    std::vector<T> x_ref(AtB);
    lapack::posv(blas::Uplo::Upper, n, 1, Gram.data(), n, x_ref.data(), n);

    // Build R from QR(A) — perfect preconditioner for J^T J alone, plus the
    // Tikhonov shift gets added inside the inner CG.
    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);

    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*tol=*/1e-12, /*max_inner=*/200, /*n_steps=*/2,
                       /*timing=*/false, /*verbose=*/false, /*lambda=*/lambda);
    std::vector<T> x_ir(n, 0);
    int status = ir.call(J, R.data(), n, b.data(), m, x_ir.data(), n);
    EXPECT_EQ(status, 0);

    // Compare: x_ir vs x_ref, both solving the same regularized normal equations.
    T diff_norm = 0;
    for (int64_t i = 0; i < n; ++i) {
        T d = x_ir[i] - x_ref[i];
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);
    T xref_norm = blas::nrm2(n, x_ref.data(), 1);
    EXPECT_LT(diff_norm / xref_norm, 1e-10);
}


// Imperfect R: from QR of a slightly perturbed A. CG should take more iters
// but the final solution must still match gels.
TEST_F(TestIterRefineLSQ, imperfect_preconditioner) {
    using T = double;
    int64_t m = 60, n = 10;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 3);
    fill_random(x_true, 13);

    // b in col space (zero residual).
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    std::vector<T> A_ref(A.begin(), A.end()), b_ref(b.begin(), b.end());
    lapack::gels(blas::Op::NoTrans, m, n, 1, A_ref.data(), m, b_ref.data(), m);
    std::vector<T> x_ref(b_ref.begin(), b_ref.begin() + n);

    // Perturbed A for R-construction (simulates a sketch-based R).
    std::vector<T> A_pert = A, perturb(m * n);
    fill_random(perturb, 99, 0.1);
    for (size_t i = 0; i < A_pert.size(); ++i) A_pert[i] += perturb[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);

    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    IterRefineLSQ<T> ir(1e-12, 100, 2);
    std::vector<T> x_ir(n, 0);
    int status = ir.call(J, R.data(), n, b.data(), m, x_ir.data(), n);
    EXPECT_EQ(status, 0);

    T diff_norm = 0;
    for (int64_t i = 0; i < n; ++i) {
        T d = x_ir[i] - x_ref[i];
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);
    T xref_norm = blas::nrm2(n, x_ref.data(), 1);
    EXPECT_LT(diff_norm / xref_norm, 1e-9);
    // Imperfect R: CG should take more than 1 iter but well under the cap.
    EXPECT_LT(ir.inner_iters_per_step.front(), 100);
}
