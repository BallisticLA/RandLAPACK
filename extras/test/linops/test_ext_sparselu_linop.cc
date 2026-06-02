// Tests for SparseLUSolverLinOp (extras/linops/ext_sparselu_linop.hh).

#include "../../linops/ext_sparselu_linop.hh"

#include <blas.hh>
#include <Eigen/Sparse>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using SpMat = Eigen::SparseMatrix<double>;


// Symmetric tridiagonal matrix with diagonal d and off-diagonal e.
static SpMat build_tridiag(int n, double d, double e) {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(3 * n);
    for (int i = 0; i < n; ++i) {
        t.emplace_back(i, i, d);
        if (i + 1 < n) {
            t.emplace_back(i, i + 1, e);
            t.emplace_back(i + 1, i, e);
        }
    }
    SpMat A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    return A;
}


// out := A * v.
static void apply_A(const SpMat& A, const vector<double>& v, vector<double>& out) {
    int n = (int)A.rows();
    out.assign(n, 0.0);
    for (int j = 0; j < A.outerSize(); ++j) {
        for (SpMat::InnerIterator it(A, j); it; ++it) {
            out[it.row()] += it.value() * v[it.col()];
        }
    }
}


// Relative 2-norm error ||u - ref|| / ||ref||.
static double rel_err(const vector<double>& u, const vector<double>& ref) {
    int n = (int)u.size();
    vector<double> diff(n);
    for (int i = 0; i < n; ++i) diff[i] = u[i] - ref[i];
    double dn = blas::nrm2(n, diff.data(), 1);
    double rn = blas::nrm2(n, ref.data(), 1);
    return dn / rn;
}


TEST(TestSparseLUSolverLinOp, spd_tridiag_inverse) {
    // SPD case: d=2, e=-1 is the 1D Laplacian.
    int n = 8;
    auto A = build_tridiag(n, 2.0, -1.0);

    RandLAPACK_extras::linops::SparseLUSolverLinOp<double> A_inv(A);
    A_inv.factorize();

    vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = (double)(i + 1);
    vector<double> x(n, 0.0);

    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, 1.0, b.data(), n, 0.0, x.data(), n);

    vector<double> Ax(n);
    apply_A(A, x, Ax);
    ASSERT_LE(rel_err(Ax, b), 1e-12);
}


TEST(TestSparseLUSolverLinOp, indefinite_inverse) {
    // Indefinite: diagonal sign-changing, small off-diagonal.
    int n = 6;
    double d[6] = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
    std::vector<Eigen::Triplet<double>> t;
    for (int i = 0; i < n; ++i) {
        t.emplace_back(i, i, d[i]);
        if (i + 1 < n) {
            t.emplace_back(i, i + 1, 0.1);
            t.emplace_back(i + 1, i, 0.1);
        }
    }
    SpMat A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();

    RandLAPACK_extras::linops::SparseLUSolverLinOp<double> A_inv(A);

    vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = std::sin((double)i);
    vector<double> x(n, 0.0);

    // Lazy factorization — first apply triggers factorize().
    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, 1.0, b.data(), n, 0.0, x.data(), n);

    vector<double> Ax(n);
    apply_A(A, x, Ax);
    ASSERT_LE(rel_err(Ax, b), 1e-12);
}


TEST(TestSparseLUSolverLinOp, transpose_inverse) {
    // Non-symmetric (upper != lower) so A^{-T} != A^{-1}.
    int n = 5;
    std::vector<Eigen::Triplet<double>> t;
    for (int i = 0; i < n; ++i) {
        t.emplace_back(i, i, (double)(i + 2));
        if (i + 1 < n) t.emplace_back(i, i + 1, 1.0);
        if (i > 0)     t.emplace_back(i, i - 1, 0.3);
    }
    SpMat A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();

    RandLAPACK_extras::linops::SparseLUSolverLinOp<double> A_inv(A);

    vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = (double)(i - 2);
    vector<double> x_trans(n, 0.0);

    A_inv(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
          n, 1, n, 1.0, b.data(), n, 0.0, x_trans.data(), n);

    // x_trans = A^{-T} b  ⇒  A^T x_trans = b.
    SpMat A_T = A.transpose();
    vector<double> AT_x(n);
    apply_A(A_T, x_trans, AT_x);
    ASSERT_LE(rel_err(AT_x, b), 1e-12);
}


TEST(TestSparseLUSolverLinOp, multi_rhs_with_alpha_beta) {
    int n = 6;
    auto A = build_tridiag(n, 4.0, -1.0);
    RandLAPACK_extras::linops::SparseLUSolverLinOp<double> A_inv(A);

    int n_rhs = 3;
    vector<double> B(n * n_rhs);
    for (int i = 0; i < n * n_rhs; ++i) B[i] = std::cos((double)i * 0.5);

    const double alpha = 2.0, beta = 1.5;
    vector<double> C(n * n_rhs, 7.0);
    vector<double> C_initial = C;

    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, alpha, B.data(), n, beta, C.data(), n);

    // Reference: alpha * A^{-1} B + beta * C_initial.
    vector<double> Ainv_B(n * n_rhs, 0.0);
    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, 1.0, B.data(), n, 0.0, Ainv_B.data(), n);

    vector<double> expected(n * n_rhs);
    for (int i = 0; i < n * n_rhs; ++i)
        expected[i] = alpha * Ainv_B[i] + beta * C_initial[i];
    ASSERT_LE(rel_err(C, expected), 1e-12);
}
