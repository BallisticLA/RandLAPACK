#pragma once

#include "rl_concepts.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

namespace RandLAPACK::linops {

/// Computes the per-triplet-normalized SVD residual:
///     sqrt( ||A V diag(Sigma)^{-1} - U||^2_F + ||A' U diag(Sigma)^{-1} - V||^2_F ).
/// Each triplet's residual is divided by its own positive singular value.
/// U is m x k (col-major, ld m), V is n x k (col-major, ld n), and Sigma is
/// length k, sorted in descending order. The scalar type is float or double,
/// and m and n must be positive. The operator must support ColMajor multiplication.
/// Inputs are preserved. An empty set or a nonpositive smallest singular value
/// gives an infinite residual.
///
/// These residuals measure the singular-vector equations. They do not check
/// normalization or linear independence: duplicate triplets and pairs of zero
/// vectors can have zero residual. Callers must check those properties separately
/// when interpreting the residual as evidence for an SVD.
template <typename T, LinearOperator GLO>
T svd_residual(GLO& A, T* U, T* V, T* Sigma, int64_t k) {
    // Sigma is descending, so a positive last entry implies all entries are
    // positive. Check k first to avoid accessing Sigma[-1] for an empty set.
    if (k < 1 || Sigma[k - 1] <= T(0))
        return std::numeric_limits<T>::infinity();

    int64_t m = A.n_rows;
    int64_t n = A.n_cols;

    auto U_cpy = std::make_unique<T[]>(m * k);
    auto V_cpy = std::make_unique<T[]>(n * k);

    // U_cpy = A V - U diag(Sigma), then column i scaled by 1/sigma_i, giving
    // A V diag(Sigma)^{-1} - U. LASCL avoids an overflowing reciprocal.
    lapack::lacpy(MatrixType::General, m, k, U, m, U_cpy.get(), m);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(m, Sigma[i], &U_cpy[m * i], 1);
    A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, V, n, (T)-1.0, U_cpy.get(), m);
    for (int64_t i = 0; i < k; ++i)
        lapack::lascl(MatrixType::General, 0, 0, Sigma[i], T(1), m, 1, &U_cpy[m * i], m);

    // V_cpy = A' U - V diag(Sigma), then column i scaled by 1/sigma_i, giving
    // A' U diag(Sigma)^{-1} - V.
    lapack::lacpy(MatrixType::General, n, k, V, n, V_cpy.get(), n);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(n, Sigma[i], &V_cpy[n * i], 1);
    A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, (T)1.0, U, m, (T)-1.0, V_cpy.get(), n);
    for (int64_t i = 0; i < k; ++i)
        lapack::lascl(MatrixType::General, 0, 0, Sigma[i], T(1), n, 1, &V_cpy[n * i], n);

    T nrm1 = lapack::lange(Norm::Fro, m, k, U_cpy.get(), m);
    T nrm2 = lapack::lange(Norm::Fro, n, k, V_cpy.get(), n);

    return std::hypot(nrm1, nrm2);
}


/// Three aggregate residual metrics, computed using two operator applications.
/// With S = diag(Sigma), the fields are:
///
///   two_sided_normalized  sqrt(||A V S^-1 - U||_F^2 + ||A' U S^-1 - V||_F^2)
///   one_sided_normalized  ||A V S^-1 - U||_F
///   two_sided_absolute    sqrt(||A V - U S||_F^2 + ||A' U - V S||_F^2)
///
/// The one-sided metric omits the transposed equation. The absolute metric weights
/// triplets by their singular values, so a small absolute residual need not imply
/// a small normalized residual for each triplet. These metrics do not check vector
/// normalization or independence. All fields are infinite if k < 1 or the smallest
/// singular value is nonpositive; inputs follow the svd_residual contract.
template <typename T>
struct SvdResidualTriple {
    T two_sided_normalized;
    T one_sided_normalized;
    T two_sided_absolute;
};

template <typename T, LinearOperator GLO>
SvdResidualTriple<T> svd_residual_all(GLO& A, T* U, T* V, T* Sigma, int64_t k) {
    const T inf = std::numeric_limits<T>::infinity();
    if (k < 1 || Sigma[k - 1] <= T(0))
        return SvdResidualTriple<T>{inf, inf, inf};

    int64_t m = A.n_rows;
    int64_t n = A.n_cols;

    auto U_cpy = std::make_unique<T[]>(m * k);
    auto V_cpy = std::make_unique<T[]>(n * k);

    // U_cpy = A V - U diag(Sigma); V_cpy = A' U - V diag(Sigma). Unnormalized.
    lapack::lacpy(MatrixType::General, m, k, U, m, U_cpy.get(), m);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(m, Sigma[i], &U_cpy[m * i], 1);
    A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, V, n, (T)-1.0, U_cpy.get(), m);

    lapack::lacpy(MatrixType::General, n, k, V, n, V_cpy.get(), n);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(n, Sigma[i], &V_cpy[n * i], 1);
    A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, (T)1.0, U, m, (T)-1.0, V_cpy.get(), n);

    // Absolute variant, before any normalization destroys it.
    T abs1 = lapack::lange(Norm::Fro, m, k, U_cpy.get(), m);
    T abs2 = lapack::lange(Norm::Fro, n, k, V_cpy.get(), n);

    // LASCL avoids forming 1/sigma_i, which can overflow for subnormal values.
    for (int64_t i = 0; i < k; ++i) {
        lapack::lascl(MatrixType::General, 0, 0, Sigma[i], T(1), m, 1, &U_cpy[m * i], m);
        lapack::lascl(MatrixType::General, 0, 0, Sigma[i], T(1), n, 1, &V_cpy[n * i], n);
    }
    T nrm1 = lapack::lange(Norm::Fro, m, k, U_cpy.get(), m);
    T nrm2 = lapack::lange(Norm::Fro, n, k, V_cpy.get(), n);

    return SvdResidualTriple<T>{
        std::hypot(nrm1, nrm2),   // two-sided, normalized
        nrm1,                     // one-sided, normalized
        std::hypot(abs1, abs2)    // two-sided, absolute
    };
}

/// The two-sided normalized residual of each triplet:
///
///     res[i] = sqrt(||A v_i - sigma_i u_i||^2 + ||A' u_i - sigma_i v_i||^2) / sigma_i
///
/// Writes k entries to caller-allocated res_out using two operator applications.
/// Each nonpositive sigma_i gives an infinite entry, while the other triplets
/// are measured independently. If k < 1, res_out is left untouched. Storage and
/// preservation of inputs follow svd_residual; Sigma need not be sorted here.
/// The output buffer must not overlap the inputs.
template <typename T, LinearOperator GLO>
void svd_residual_per_triplet(GLO& A, T* U, T* V, T* Sigma, int64_t k, T* res_out) {
    if (k < 1)
        return;

    const T inf = std::numeric_limits<T>::infinity();
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;

    auto U_cpy = std::make_unique<T[]>(m * k);
    auto V_cpy = std::make_unique<T[]>(n * k);

    // U_cpy = A V - U diag(Sigma); V_cpy = A' U - V diag(Sigma). Left unnormalized here:
    // the scaling is per column below, so that a single zero sigma cannot contaminate its
    // neighbours the way an in-place whole-block scal would.
    lapack::lacpy(MatrixType::General, m, k, U, m, U_cpy.get(), m);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(m, Sigma[i], &U_cpy[m * i], 1);
    A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, V, n, (T)-1.0, U_cpy.get(), m);

    lapack::lacpy(MatrixType::General, n, k, V, n, V_cpy.get(), n);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(n, Sigma[i], &V_cpy[n * i], 1);
    A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, (T)1.0, U, m, (T)-1.0, V_cpy.get(), n);

    for (int64_t i = 0; i < k; ++i) {
        if (Sigma[i] <= T(0)) {
            res_out[i] = inf;
            continue;
        }
        T ru = blas::nrm2(m, &U_cpy[m * i], 1);
        T rv = blas::nrm2(n, &V_cpy[n * i], 1);
        // Normalize before combining to avoid overflow in hypot(ru, rv).
        res_out[i] = std::hypot(ru / Sigma[i], rv / Sigma[i]);
    }

}

/// Count triplets whose two-sided normalized residual is at most tol.
/// The tolerance must be finite and nonnegative. Despite the name, this counts
/// only residual-qualified triplets; it does not certify normalization, linear
/// independence, or genuine spectral content. Duplicate triplets and zero-vector
/// pairs with positive Sigma can pass. See svd_residual_per_triplet for inputs.
template <typename T, LinearOperator GLO>
int64_t svd_triplets_certified(GLO& A, T* U, T* V, T* Sigma, int64_t k, T tol) {
    if (k < 1)
        return 0;
    auto res = std::make_unique<T[]>(k);
    svd_residual_per_triplet<T>(A, U, V, Sigma, k, res.get());
    int64_t count = 0;
    for (int64_t i = 0; i < k; ++i)
        if (res[i] <= tol) ++count;
    return count;
}

} // end namespace RandLAPACK::linops
