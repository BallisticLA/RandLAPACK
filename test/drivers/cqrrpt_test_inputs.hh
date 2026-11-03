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
    RandBLAS::SparseDist DS(d, m, nnz, RandBLAS::Axis::Short);
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

/// Builds a full-rank m-by-2 A on which CholQR's potrf fails at column 2, exactly and on any BLAS,
/// for the count sketch (nnz = 1, d rows) drawn from RNGState(seed). Columns: 2 s_r0 e_r0 + 2C z
/// and s_r1 e_r1 + C z, with r0, r1 in sketch rows 0 and 1, and z in null(S), from two pairs of
/// rows that share a sketch row, with opposite signs. The sketch is then diag(2, 1), so A_pre =
/// [s_r0 e_r0 + C z, s_r1 e_r1 + C z] exactly; with C = 2^27 and ||z||^2 = 4 its Gram matrix rounds
/// to 2^56 [1 1; 1 1]. Returns an empty vector if the sketch has no such rows.
inline std::vector<double> cholesky_failure_k2_input(int64_t m, int64_t d, uint32_t seed) {
    double C = std::ldexp(1.0, 27);
    // The sketch: each column of S has one entry, +-1, in row bucket[i].
    RandBLAS::RNGState<r123::Philox4x32> sk_state(seed);
    RandBLAS::SparseDist DS(d, m, 1);
    RandBLAS::SparseSkOp<double, r123::Philox4x32> S(DS, sk_state);
    RandBLAS::fill_sparse(S);
    std::vector<double> I(m * m, 0.0), Sd(d * m, 0.0);
    for (int64_t i = 0; i < m; ++i)
        I[i + i * m] = 1.0;
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, m, 1.0, S, 0, 0, I.data(), m, 0.0, Sd.data(), d);
    std::vector<int64_t> bucket(m, -1);
    std::vector<double> sign(m, 0.0);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t b = 0; b < d; ++b)
            if (Sd[b + i * d] != 0.0) {
                bucket[i] = b;
                sign[i] = Sd[b + i * d];
            }

    int64_t r0 = -1, r1 = -1;
    for (int64_t i = 0; i < m; ++i) {
        if (r0 < 0 && bucket[i] == 0)
            r0 = i;
        else if (r1 < 0 && bucket[i] == 1)
            r1 = i;
    }
    if (r0 < 0 || r1 < 0)
        return {};
    std::vector<double> z(m, 0.0);
    std::vector<bool> used(m, false);
    used[r0] = used[r1] = true;
    int pairs = 0;
    for (int64_t i = 0; i < m && pairs < 2; ++i)
        for (int64_t j = i + 1; j < m && !used[i]; ++j)
            if (!used[j] && bucket[j] == bucket[i]) {
                z[i] = sign[i];
                z[j] = -sign[j];
                used[i] = used[j] = true;
                ++pairs;
            }
    if (pairs < 2)
        return {};

    std::vector<double> A(m * 2, 0.0);
    for (int64_t i = 0; i < m; ++i) {
        A[i] = 2 * C * z[i];
        A[i + m] = C * z[i];
    }
    A[r0] = 2 * sign[r0];
    A[r1 + m] = sign[r1];
    return A;
}

/// Builds a full-rank m-by-n A on which CholQR's potrf fails at column q + 1 (1-based), exactly and on any
/// BLAS, for the count sketch (nnz = 1, d >= n rows) drawn from RNGState(seed), with p < q < n. Column j
/// is w_j s_rj e_rj, plus w_j C z for j = p and q, with w_j = 2^(-6 j), r_j a row in sketch row j, z in
/// null(S) from two pairs of rows that share a sketch row, with opposite signs, and C = 2^27. The sketch is
/// then diag(w), whose columns pivoting keeps in order, so A_pre's columns are s_rj e_rj, plus C z for p and
/// q. With ||z||^2 = 4 its Gram matrix is the identity but for the block of p and q, which rounds to
/// 2^56 [1 1; 1 1]. Returns an empty vector if the sketch has no such rows.
/// The rank estimate before CholQR (threshold 2 u^0.95 on the sketch's R) cuts the columns j >= 9, whose
/// w_j = 2^(-6 j) falls below it, so potrf fails only for q <= 8: for any p and q with n <= 9.
inline std::vector<double> cholesky_failure_input(int64_t m, int64_t n, int64_t d, uint32_t seed, int64_t p, int64_t q) {
    double C = std::ldexp(1.0, 27);
    // The sketch: each column of S has one entry, +-1, in row bucket[i].
    RandBLAS::RNGState<r123::Philox4x32> sk_state(seed);
    RandBLAS::SparseDist DS(d, m, 1);
    RandBLAS::SparseSkOp<double, r123::Philox4x32> S(DS, sk_state);
    RandBLAS::fill_sparse(S);
    std::vector<double> I(m * m, 0.0), Sd(d * m, 0.0);
    for (int64_t i = 0; i < m; ++i)
        I[i + i * m] = 1.0;
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, m, 1.0, S, 0, 0, I.data(), m, 0.0, Sd.data(), d);
    std::vector<int64_t> bucket(m, -1);
    std::vector<double> sign(m, 0.0);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t b = 0; b < d; ++b)
            if (Sd[b + i * d] != 0.0) {
                bucket[i] = b;
                sign[i] = Sd[b + i * d];
            }

    std::vector<int64_t> r(n, -1);
    std::vector<bool> used(m, false);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m && r[j] < 0; ++i)
            if (bucket[i] == j)
                r[j] = i;
        if (r[j] < 0)
            return {};
        used[r[j]] = true;
    }
    std::vector<double> z(m, 0.0);
    int pairs = 0;
    for (int64_t i = 0; i < m && pairs < 2; ++i)
        for (int64_t j = i + 1; j < m && !used[i]; ++j)
            if (!used[j] && bucket[j] == bucket[i]) {
                z[i] = sign[i];
                z[j] = -sign[j];
                used[i] = used[j] = true;
                ++pairs;
            }
    if (pairs < 2)
        return {};

    std::vector<double> A(m * n, 0.0);
    for (int64_t j = 0; j < n; ++j) {
        double w = std::ldexp(1.0, -6 * (int) j);
        if (j == p || j == q)
            for (int64_t i = 0; i < m; ++i)
                A[i + j * m] = w * C * z[i];
        A[r[j] + j * m] = w * sign[r[j]];
    }
    return A;
}

/// For d > m the rows of CQRRPT's sparse sketch carry its nonzeros, so a row of A can get no weight. Finds
/// the first seed below 1000 whose sketch, drawn as CQRRPT draws it from RNGState(seed), leaves a row of A
/// unweighted, and returns that row, or -1 if there is none.
inline int64_t unweighted_row(int64_t d, int64_t m, int64_t nnz, uint32_t& seed) {
    for (seed = 0; seed < 1000; ++seed) {
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandBLAS::SparseDist DS(d, m, nnz, RandBLAS::Axis::Short);
        RandBLAS::SparseSkOp<double, r123::Philox4x32> S(DS, state);
        RandBLAS::fill_sparse(S);
        std::vector<int64_t> weight(m, 0);
        for (int64_t e = 0; e < S.nnz; ++e)
            ++weight[S.cols[e]];
        for (int64_t row = 0; row < m; ++row)
            if (weight[row] == 0)
                return row;
    }
    return -1;
}

/// Finds the first seed below 1000 whose sparse sketch of A, drawn as CQRRPT draws it from RNGState(seed),
/// overflows (A itself being finite), and returns it, or -1 if there is none.
inline int64_t overflowing_sketch_seed(int64_t m, int64_t n, int64_t nnz, double d_factor, const std::vector<double>& A) {
    int64_t d = d_factor * n;
    std::vector<double> A_hat(d * n);
    for (uint32_t seed = 0; seed < 1000; ++seed) {
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandBLAS::SparseDist DS(d, m, nnz, RandBLAS::Axis::Short);
        RandBLAS::SparseSkOp<double, r123::Philox4x32> S(DS, state);
        RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, S, 0, 0, A.data(), m, 0.0, A_hat.data(), d);
        for (double x : A_hat)
            if (!std::isfinite(x))
                return seed;
    }
    return -1;
}

} // namespace cqrrpt_test
