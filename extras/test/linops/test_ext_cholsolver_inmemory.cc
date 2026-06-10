// Tests for CholSolverLinOp's in-memory (RandBLAS CSR) constructor.
//
// CholSolverLinOp normally reads an SPD matrix from a Matrix Market file. The
// in-memory constructor instead takes a RandBLAS CSR matrix computed at runtime
// (e.g. X = K - omega*M in the reduced-spectral application). Eigen is confined
// to the factorization; all solves go through RandBLAS sparse TRSM.
//
// These tests build the SPD 1D Laplacian via RandLAPACK::gen::gen_tridiag_csr
// (no Eigen), construct the operator from that CSR, and verify A^{-1} by
// re-applying A and checking the normwise residual. No std::vector, no Eigen:
// matrices are RandBLAS CSR, vectors are raw T*, and the matvec is the RandBLAS
// sparse x dense SparseLinOp.

#include <blas.hh>
#include <RandBLAS.hh>
#include "RandLAPACK.hh"
#include "rl_gen.hh"
#include <gtest/gtest.h>

#include "../../linops/ext_cholsolver_linop.hh"

using blas::Layout;
using blas::Op;
using blas::Side;

template <typename T>
using CSR = RandBLAS::sparse_data::csr::CSRMatrix<T>;


// ||u - ref||_2 / ||ref||_2 on raw pointers (BLAS: diff = ref - u via axpy).
template <typename T>
static T relative_error(const T* u, const T* ref, int64_t n) {
    T* diff = new T[n];
    blas::copy(n, ref, 1, diff, 1);          // diff = ref
    blas::axpy(n, (T)(-1), u, 1, diff, 1);   // diff = ref - u
    T num = blas::nrm2(n, diff, 1);
    T den = blas::nrm2(n, ref, 1);
    delete[] diff;
    return num / den;
}


// Action: solve A X = B through the in-memory inverse operator, then re-apply A
// (as a RandBLAS sparse x dense product) and assert the normwise residual
// ||A X - B|| / ||B|| is at machine-precision level, column by column. A_csr is
// the same matrix used to build A_inv, so SparseLinOp can apply it.
template <typename T>
static void check_inverse_solve(CSR<T>& A_csr,
                                RandLAPACK_extras::linops::CholSolverLinOp<T>& A_inv,
                                const T* B, int64_t n, int64_t n_rhs) {
    T* X = new T[n * n_rhs];
    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, (T)1, B, n, (T)0, X, n);

    RandLAPACK::linops::SparseLinOp<CSR<T>> A_op(n, n, A_csr);
    T* AX = new T[n * n_rhs];
    A_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         n, n_rhs, n, (T)1, X, n, (T)0, AX, n);

    for (int64_t j = 0; j < n_rhs; ++j)
        ASSERT_LE(relative_error(AX + j * n, B + j * n, n), (T)1e-12);

    delete[] X;
    delete[] AX;
}


// Action: verify the alpha/beta accumulation contract
//   C := alpha * A^{-1} * B + beta * C_initial
// against a direct recomputation (alpha=1, beta=0 solve combined by hand).
template <typename T>
static void check_alpha_beta(CSR<T>& A_csr,
                             RandLAPACK_extras::linops::CholSolverLinOp<T>& A_inv,
                             const T* B, int64_t n, int64_t n_rhs, T alpha, T beta) {
    T* C        = new T[n * n_rhs];
    T* C_init   = new T[n * n_rhs];
    for (int64_t i = 0; i < n * n_rhs; ++i) { C[i] = (T)7; C_init[i] = (T)7; }

    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, alpha, B, n, beta, C, n);

    // Reference: Ainv_B = A^{-1} B, then expected = alpha*Ainv_B + beta*C_init.
    T* Ainv_B = new T[n * n_rhs];
    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, (T)1, B, n, (T)0, Ainv_B, n);
    T* expected = new T[n * n_rhs];
    for (int64_t i = 0; i < n * n_rhs; ++i)
        expected[i] = alpha * Ainv_B[i] + beta * C_init[i];

    ASSERT_LE(relative_error(C, expected, n * n_rhs), (T)1e-12);

    delete[] C; delete[] C_init; delete[] Ainv_B; delete[] expected;
}


TEST(TestCholSolverInMemory, spd_tridiag_single_rhs) {
    // 1D Laplacian (d=2, e=-1) is SPD, so Cholesky factors. Build it in CSR via
    // rl_gen (no Eigen) and solve A x = b for a simple RHS with explicit factorize.
    const int64_t n = 8;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);

    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A);
    A_inv.factorize();

    double* b = new double[n];
    for (int64_t i = 0; i < n; ++i) b[i] = (double)(i + 1);

    check_inverse_solve<double>(A, A_inv, b, n, 1);
    delete[] b;
}


TEST(TestCholSolverInMemory, spd_tridiag_multi_rhs_lazy_factorize) {
    // Multiple right-hand sides, lazy factorization (first apply triggers it).
    const int64_t n = 10, n_rhs = 3;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);

    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A);   // no explicit factorize()

    double* B = new double[n * n_rhs];
    for (int64_t i = 0; i < n * n_rhs; ++i) B[i] = std::cos((double)i * 0.5);

    check_inverse_solve<double>(A, A_inv, B, n, n_rhs);
    delete[] B;
}


TEST(TestCholSolverInMemory, spd_tridiag_alpha_beta) {
    // Exercise the C := alpha*A^{-1}*B + beta*C accumulation path.
    const int64_t n = 6, n_rhs = 3;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 4.0, -1.0);

    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A);

    double* B = new double[n * n_rhs];
    for (int64_t i = 0; i < n * n_rhs; ++i) B[i] = std::sin((double)i * 0.3);

    check_alpha_beta<double>(A, A_inv, B, n, n_rhs, /*alpha=*/2.0, /*beta=*/1.5);
    delete[] B;
}
