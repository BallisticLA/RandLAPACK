// Tests for SparseLUSolverLinOp (extras/linops/ext_sparselu_linop.hh).

#include "../../linops/ext_sparselu_linop.hh"

#include <blas.hh>
#include <RandBLAS.hh>
#include <Eigen/Sparse>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;


// Build a small symmetric tridiagonal sparse matrix A of dimension n,
// with chosen diagonal value d and off-diagonal value e.  Eigen ColMajor.
Eigen::SparseMatrix<double> build_tridiag(int n, double d, double e) {
    Eigen::SparseMatrix<double> A(n, n);
    A.reserve(Eigen::VectorXi::Constant(n, 3));
    for (int i = 0; i < n; ++i) {
        A.insert(i, i) = d;
        if (i + 1 < n) {
            A.insert(i, i + 1) = e;
            A.insert(i + 1, i) = e;
        }
    }
    A.makeCompressed();
    return A;
}


// Reference: compute A * v (dense GEMV) for verification.
void apply_A(const Eigen::SparseMatrix<double>& A,
             const vector<double>& v, vector<double>& out)
{
    int n = (int)A.rows();
    out.assign(n, 0.0);
    Eigen::Map<const Eigen::VectorXd> v_map(v.data(), n);
    Eigen::Map<Eigen::VectorXd> out_map(out.data(), n);
    out_map = A * v_map;
}


TEST(TestSparseLUSolverLinOp, spd_tridiag_inverse) {
    // SPD case: d=2, e=-1 makes the 1D Laplacian (SPD).
    int n = 8;
    auto A = build_tridiag(n, 2.0, -1.0);

    RandLAPACK_extras::linops::SparseLUSolverLinOp<double> A_inv(A);
    A_inv.factorize();

    vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = (double)(i + 1);   // pick a known b
    vector<double> x(n, 0.0);

    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, 1.0, b.data(), n, 0.0, x.data(), n);

    // Verify A x ≈ b
    vector<double> Ax(n);
    apply_A(A, x, Ax);
    double err = 0, ref = 0;
    for (int i = 0; i < n; ++i) {
        double d = Ax[i] - b[i];
        err += d * d;
        ref += b[i] * b[i];
    }
    ASSERT_LE(std::sqrt(err / ref), 1e-12);
}


TEST(TestSparseLUSolverLinOp, indefinite_inverse) {
    // Indefinite case: d=-1 makes A = -I + tridiag(e, -1, e) which can be indefinite.
    // More directly: build an explicitly indefinite matrix.
    int n = 6;
    // Diagonal [1, -2, 3, -4, 5, -6] with off-diagonal 0.1
    Eigen::SparseMatrix<double> A(n, n);
    A.reserve(Eigen::VectorXi::Constant(n, 3));
    double diag_vals[6] = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
    for (int i = 0; i < n; ++i) {
        A.insert(i, i) = diag_vals[i];
        if (i + 1 < n) {
            A.insert(i, i + 1) = 0.1;
            A.insert(i + 1, i) = 0.1;
        }
    }
    A.makeCompressed();

    RandLAPACK_extras::linops::SparseLUSolverLinOp<double> A_inv(A);

    vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = std::sin((double)i);
    vector<double> x(n, 0.0);

    // Lazy factorization: don't call factorize() explicitly.
    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, 1.0, b.data(), n, 0.0, x.data(), n);

    vector<double> Ax(n);
    apply_A(A, x, Ax);
    double err = 0, ref = 0;
    for (int i = 0; i < n; ++i) {
        double d = Ax[i] - b[i];
        err += d * d;
        ref += b[i] * b[i];
    }
    ASSERT_LE(std::sqrt(err / ref), 1e-12);
}


TEST(TestSparseLUSolverLinOp, transpose_inverse) {
    // Use a NON-symmetric matrix so A^{-T} != A^{-1}, to actually exercise trans dispatch.
    int n = 5;
    Eigen::SparseMatrix<double> A(n, n);
    A.reserve(Eigen::VectorXi::Constant(n, 3));
    for (int i = 0; i < n; ++i) {
        A.insert(i, i) = (double)(i + 2);
        if (i + 1 < n) A.insert(i, i + 1) = 1.0;       // upper off-diag
        if (i > 0)     A.insert(i, i - 1) = 0.3;       // lower off-diag (different value)
    }
    A.makeCompressed();

    RandLAPACK_extras::linops::SparseLUSolverLinOp<double> A_inv(A);

    vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = (double)(i - 2);
    vector<double> x_trans(n, 0.0);

    A_inv(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
          n, 1, n, 1.0, b.data(), n, 0.0, x_trans.data(), n);

    // Verify A^T * x_trans ≈ b  (i.e., x_trans = A^{-T} * b)
    Eigen::SparseMatrix<double> A_T = A.transpose();
    vector<double> AT_x(n);
    apply_A(A_T, x_trans, AT_x);
    double err = 0, ref = 0;
    for (int i = 0; i < n; ++i) {
        double d = AT_x[i] - b[i];
        err += d * d;
        ref += b[i] * b[i];
    }
    ASSERT_LE(std::sqrt(err / ref), 1e-12);
}


TEST(TestSparseLUSolverLinOp, multi_rhs_with_alpha_beta) {
    int n = 6;
    auto A = build_tridiag(n, 4.0, -1.0);  // SPD
    RandLAPACK_extras::linops::SparseLUSolverLinOp<double> A_inv(A);

    int n_rhs = 3;
    vector<double> B(n * n_rhs);
    for (int i = 0; i < n * n_rhs; ++i) B[i] = std::cos((double)i * 0.5);

    const double alpha = 2.0, beta = 1.5;
    vector<double> C(n * n_rhs, 7.0);   // initial C
    vector<double> C_initial = C;

    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, alpha, B.data(), n, beta, C.data(), n);

    // Reference: C_ref = alpha * A^{-1} * B + beta * C_initial.
    // Compute A^{-1} * B column-by-column with a fresh solver call.
    vector<double> Ainv_B(n * n_rhs, 0.0);
    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, 1.0, B.data(), n, 0.0, Ainv_B.data(), n);

    for (int i = 0; i < n * n_rhs; ++i) {
        double expected = alpha * Ainv_B[i] + beta * C_initial[i];
        ASSERT_NEAR(C[i], expected, 1e-12);
    }
}
