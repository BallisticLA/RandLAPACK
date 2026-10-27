#pragma once

// Problem builders shared by the least-squares solver tests. Random data is seeded RandBLAS
// output; its Gaussian transform uses libm, so values may differ in the last bits across
// platforms, and the tests' assertions are tolerant of that.

#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <cmath>
#include <cstdint>
#include <vector>

namespace lsq_test {

using RNG = r123::Philox4x32;

/// Fill v with uniform random values on [-scale, scale].
template <typename T>
void fill_random(std::vector<T>& v, uint32_t seed, T scale = 1.0) {
    RandBLAS::RNGState<RNG> state(seed);
    RandBLAS::DenseDist D((int64_t)v.size(), 1, RandBLAS::ScalarDist::Uniform);
    RandBLAS::fill_dense(D, v.data(), state);
    const T s = scale / std::sqrt((T)3);   // RandBLAS's uniform distribution is on [-sqrt(3), sqrt(3)]
    for (auto& x : v) x *= s;
}

/// Gaussian m x n matrix (column-major) with n added to its diagonal, so its columns are far
/// from dependent, a Gaussian x_true, and b = A x_true.
template <typename T>
void make_problem(int64_t m, int64_t n, uint32_t seed,
                  std::vector<T>& A, std::vector<T>& b, std::vector<T>& x_true) {
    A.resize(m * n); b.resize(m); x_true.resize(n);
    RandBLAS::RNGState<RNG> state(seed);
    RandBLAS::DenseDist DA(m, n);
    auto s2 = RandBLAS::fill_dense(DA, A.data(), state);
    RandBLAS::DenseDist DX(n, 1);
    RandBLAS::fill_dense(DX, x_true.data(), s2);
    for (int64_t j = 0; j < n; ++j) A[j + j * m] += (T)n;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
}

/// An ill-conditioned (columns scaled down to 1e-8, so CG's recursive residual flatlines) and
/// inconsistent (the optimal residual is O(1)) problem built on make_problem: only a floor
/// exit or a budget can end a solve to a tight tolerance.
template <typename T>
void floor_problem(int64_t m, int64_t n, std::vector<T>& A, std::vector<T>& b) {
    std::vector<T> x_true;
    make_problem(m, n, 61, A, b, x_true);
    for (int64_t j = 0; j < n; ++j) {
        T s = std::pow((T)10.0, (T)(-8.0 * (double)j / (n - 1)));
        blas::scal(m, s, A.data() + j * m, 1);
    }
    for (int64_t i = 0; i < m; ++i) b[i] += (T)0.1 / (T)(1 + i);
}

/// R factor (n x n, ColMajor, strict lower triangle zero) of the QR of the m x n matrix A.
template <typename T>
std::vector<T> qr_r_factor(const T* A, int64_t m, int64_t n) {
    std::vector<T> Ac(A, A + m * n), tau(n), R(n * n, (T)0);
    lapack::geqrf(m, n, Ac.data(), m, tau.data());
    lapack::lacpy(lapack::MatrixType::Upper, n, n, Ac.data(), m, R.data(), n);
    return R;
}

template <typename T>
struct Problem {
    int64_t m, n;
    std::vector<T> A, b, x_true, R;
};

/// A consistent problem b = A x_true with uniform random A and x_true, and a preconditioner R
/// from the QR of A + pert_scale * E (E uniform random from pert_seed; pert_scale = 0 gives
/// the exact factor, so A R^{-1} is orthonormal).
template <typename T>
Problem<T> consistent_problem(int64_t m, int64_t n, uint32_t seed_A, uint32_t seed_x,
                              uint32_t pert_seed = 0, T pert_scale = (T)0) {
    Problem<T> p{m, n, std::vector<T>(m * n), std::vector<T>(m), std::vector<T>(n), {}};
    fill_random(p.A, seed_A);
    fill_random(p.x_true, seed_x);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, p.A.data(), m, p.x_true.data(), n, (T)0.0, p.b.data(), m);
    if (pert_scale == (T)0) {
        p.R = qr_r_factor(p.A.data(), m, n);
    } else {
        std::vector<T> A_pert(p.A), pert(m * n);
        fill_random(pert, pert_seed, pert_scale);
        for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
        p.R = qr_r_factor(A_pert.data(), m, n);
    }
    return p;
}

/// LAPACK reference least-squares solution (gels destroys its inputs, so it works on copies).
template <typename T>
std::vector<T> gels_reference(const std::vector<T>& A, const std::vector<T>& b, int64_t m, int64_t n) {
    std::vector<T> Ac(A), bc(b);
    lapack::gels(blas::Op::NoTrans, m, n, 1, Ac.data(), m, bc.data(), m);
    return std::vector<T>(bc.begin(), bc.begin() + n);
}

/// ||x - ref|| / ||ref||.
template <typename T>
T rel_err(const std::vector<T>& x, const std::vector<T>& ref) {
    return (T)RandLAPACK::testing::rel_fro_error(x.data(), ref.data(), (int64_t)ref.size());
}

/// An operator whose transposed apply is not the adjoint of its forward apply, (-A)^T in
/// place of A^T: the preconditioned normal-equation operator is then indefinite, so CG breaks
/// down at once.
template <typename T>
struct TwoFacedOp {
    using scalar_t = T;
    RandLAPACK::linops::DenseLinOp<T> fwd;
    RandLAPACK::linops::DenseLinOp<T> adj;
    const int64_t n_rows, n_cols;
    TwoFacedOp(int64_t m, int64_t n, const T* A, const T* negA)
        : fwd(m, n, A, m, blas::Layout::ColMajor), adj(m, n, negA, m, blas::Layout::ColMajor),
          n_rows(m), n_cols(n) {}
    void operator()(blas::Layout layout, blas::Op tA, blas::Op tB, int64_t mm, int64_t nn, int64_t kk,
                    T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc) {
        if (tA == blas::Op::NoTrans) fwd(layout, tA, tB, mm, nn, kk, alpha, B, ldb, beta, C, ldc);
        else                         adj(layout, tA, tB, mm, nn, kk, alpha, B, ldb, beta, C, ldc);
    }
};

} // namespace lsq_test
