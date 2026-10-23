#pragma once

// Test inputs shared by the CPU and GPU CQRRPT tests.

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cqrrpt_test {

/// Builds an m-by-n A that defeats the sketch CQRRPT draws from RNGState(seed):
/// S * A is a padded Kahan matrix (angle theta), while A itself is dominated by
/// orthonormal columns in null(S). Preconditioning by the sketch's R then leaves
/// a badly conditioned A_pre, so CholQR's potrf can fail.
inline std::vector<double> sketch_defeating_input(int64_t m, int64_t n, int64_t nnz, double d_factor, uint32_t seed, double theta) {
    int64_t d = d_factor * n;
    RandBLAS::RNGState<r123::Philox4x32> state(seed);
    RandBLAS::SparseDist DS(d, m, nnz);
    RandBLAS::SparseSkOp<double, r123::Philox4x32> S(DS, state);
    RandBLAS::fill_sparse(S);
    // Full QR of S' (m-by-d): S' = [Q1 Q2] [R1; 0], so S^+ = Q1 R1^{-T} and Q2 spans null(S).
    std::vector<double> I(m * m, 0.0), Sd(d * m, 0.0), Q(m * m, 0.0), tau(d), R1(d * d, 0.0);
    for (int64_t i = 0; i < m; ++i)
        I[i + i * m] = 1.0;
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, m, 1.0, S, 0, 0, I.data(), m, 0.0, Sd.data(), d);
    for (int64_t i = 0; i < d; ++i)
        for (int64_t j = 0; j < m; ++j)
            Q[j + i * m] = Sd[i + j * d];
    lapack::geqrf(m, d, Q.data(), m, tau.data());
    lapack::lacpy(MatrixType::Upper, d, d, Q.data(), m, R1.data(), d);
    lapack::orgqr(m, m, d, Q.data(), m, tau.data());
    // Kahan matrix; the diagonal tilt keeps column pivoting from reordering it.
    std::vector<double> K(d * n, 0.0);
    double sn = std::sin(theta), cs = std::cos(theta);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < j; ++i)
            K[i + j * d] = -cs * std::pow(sn, i);
        K[j + j * d] = std::pow(sn, j) * (1 + 1e-3 * (n - j));
    }
    blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans, Diag::NonUnit, d, n, 1.0, R1.data(), d, K.data(), d);
    std::vector<double> A(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, d, 1.0, Q.data(), m, K.data(), d, 0.0, A.data(), m);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            A[i + j * m] += Q[i + (d + j) * m];
    return A;
}

} // namespace cqrrpt_test
