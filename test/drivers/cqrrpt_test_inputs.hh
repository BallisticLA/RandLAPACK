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
/// The factorizations use plain loops, not threaded BLAS or LAPACK, so A does not
/// depend on the thread count.
inline std::vector<double> sketch_defeating_input(int64_t m, int64_t n, int64_t nnz, double d_factor, uint32_t seed, double theta) {
    int64_t d = d_factor * n;
    RandBLAS::RNGState<r123::Philox4x32> state(seed);
    RandBLAS::SparseDist DS(d, m, nnz);
    RandBLAS::SparseSkOp<double, r123::Philox4x32> S(DS, state);
    RandBLAS::fill_sparse(S);
    // S' (m-by-d). Sketching the identity is exact: each entry of S * I is a single product.
    std::vector<double> I(m * m, 0.0), Sd(d * m, 0.0), V(m * d, 0.0);
    for (int64_t i = 0; i < m; ++i)
        I[i + i * m] = 1.0;
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, m, 1.0, S, 0, 0, I.data(), m, 0.0, Sd.data(), d);
    for (int64_t i = 0; i < d; ++i)
        for (int64_t j = 0; j < m; ++j)
            V[j + i * m] = Sd[i + j * d];
    // Householder QR S' = [Q1 Q2] [R1; 0], with geqrf's conventions, so S^+ = Q1 R1^{-T} and Q2 spans null(S).
    std::vector<double> tau(d, 0.0);
    for (int64_t j = 0; j < d; ++j) {
        double* v = &V[j * m];
        double alpha = v[j], xnorm = 0.0;
        for (int64_t i = j + 1; i < m; ++i)
            xnorm += v[i] * v[i];
        xnorm = std::sqrt(xnorm);
        if (xnorm == 0.0)
            continue;
        double beta = -std::copysign(std::hypot(alpha, xnorm), alpha);
        tau[j] = (beta - alpha) / beta;
        for (int64_t i = j + 1; i < m; ++i)
            v[i] /= alpha - beta;
        v[j] = beta;
        for (int64_t c = j + 1; c < d; ++c) {
            double* x = &V[c * m];
            double w = x[j];
            for (int64_t i = j + 1; i < m; ++i)
                w += v[i] * x[i];
            w *= tau[j];
            x[j] -= w;
            for (int64_t i = j + 1; i < m; ++i)
                x[i] -= w * v[i];
        }
    }
    // The first d + n columns of Q = H_0 ... H_{d-1}.
    int64_t nq = d + n;
    std::vector<double> Q(m * nq, 0.0);
    for (int64_t c = 0; c < nq; ++c) {
        double* x = &Q[c * m];
        x[c] = 1.0;
        for (int64_t j = d - 1; j >= 0; --j) {
            const double* v = &V[j * m];
            double w = x[j];
            for (int64_t i = j + 1; i < m; ++i)
                w += v[i] * x[i];
            w *= tau[j];
            x[j] -= w;
            for (int64_t i = j + 1; i < m; ++i)
                x[i] -= w * v[i];
        }
    }
    // Kahan matrix; the diagonal tilt keeps column pivoting from reordering it.
    std::vector<double> K(d * n, 0.0);
    double sn = std::sin(theta), cs = std::cos(theta);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < j; ++i)
            K[i + j * d] = -cs * std::pow(sn, i);
        K[j + j * d] = std::pow(sn, j) * (1 + 1e-3 * (n - j));
    }
    // K := R1^{-T} K, by forward substitution with the lower triangle R1'.
    for (int64_t c = 0; c < n; ++c)
        for (int64_t i = 0; i < d; ++i) {
            double s = K[i + c * d];
            for (int64_t l = 0; l < i; ++l)
                s -= V[l + i * m] * K[l + c * d];
            K[i + c * d] = s / V[i + i * m];
        }
    // A = Q1 K + Q2[:, 0:n].
    std::vector<double> A(m * n, 0.0);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) {
            double s = 0.0;
            for (int64_t l = 0; l < d; ++l)
                s += Q[i + l * m] * K[l + j * d];
            A[i + j * m] = s + Q[i + (d + j) * m];
        }
    return A;
}

} // namespace cqrrpt_test
