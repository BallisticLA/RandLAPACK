#pragma once

#include "rl_linops.hh"

#include <cmath>
#include <limits>

namespace RandLAPACK::linops {

/// Computes the per-triplet-normalized SVD residual:
///     sqrt( ||A V diag(Sigma)^{-1} - U||^2_F + ||A' U diag(Sigma)^{-1} - V||^2_F ).
/// Each triplet's residual (A v_i - sigma_i u_i, and A' u_i - sigma_i v_i) is divided
/// by its own sigma_i. Normalizing per triplet, rather than leaving the residual
/// Sigma-scaled or dividing by a single sigma, lets the estimate be driven to machine
/// precision as the triplets converge (a single-sigma or Sigma-scaled residual is
/// dominated by the larger singular values and plateaus above machine precision).
/// U is m x k (col-major, ld m), V is n x k (col-major, ld n), Sigma is length k,
/// sorted in descending order.
template <typename T, LinearOperator GLO>
T svd_residual(GLO& A, T* U, T* V, T* Sigma, int64_t k) {
    // No triplets to measure, or the smallest included singular value is
    // (numerically) zero: the normalized residual is undefined. Report an
    // infinite residual so the caller's adaptive loop treats it as "not
    // converged" rather than dividing by zero or indexing Sigma[-1]. Sigma is
    // descending, so Sigma[k-1] > 0 guarantees every Sigma[i] > 0 below.
    if (k < 1 || Sigma[k - 1] <= T(0))
        return std::numeric_limits<T>::infinity();

    int64_t m = A.n_rows;
    int64_t n = A.n_cols;

    T* U_cpy = new T[m * k]();
    T* V_cpy = new T[n * k]();

    // U_cpy = A V - U diag(Sigma), then column i scaled by 1/sigma_i, giving
    // A V diag(Sigma)^{-1} - U.
    lapack::lacpy(MatrixType::General, m, k, U, m, U_cpy, m);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(m, Sigma[i], &U_cpy[m * i], 1);
    A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, V, n, (T)-1.0, U_cpy, m);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(m, T(1) / Sigma[i], &U_cpy[m * i], 1);

    // V_cpy = A' U - V diag(Sigma), then column i scaled by 1/sigma_i, giving
    // A' U diag(Sigma)^{-1} - V.
    lapack::lacpy(MatrixType::General, n, k, V, n, V_cpy, n);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(n, Sigma[i], &V_cpy[n * i], 1);
    A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, (T)1.0, U, m, (T)-1.0, V_cpy, n);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(n, T(1) / Sigma[i], &V_cpy[n * i], 1);

    T nrm1 = lapack::lange(Norm::Fro, m, k, U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, n, k, V_cpy, n);

    delete[] U_cpy;
    delete[] V_cpy;

    return std::hypot(nrm1, nrm2);
}


/// The three residual metrics used by block Krylov SVD software, computed together.
///
/// They differ along two independent axes: whether the residual is one-sided or
/// two-sided, and whether it is normalized per triplet. Only the two-sided and
/// normalized combination bounds the relative backward error of a full triplet.
///
///   two_sided_normalized  ours: sqrt( ||A V S^-1 - U||_F^2 + ||A' U S^-1 - V||_F^2 )
///   one_sided_normalized  sqrt( ||A V S^-1 - U||_F^2 ) alone; small while its
///                         counterpart is not, so it forfeits the backward-error
///                         interpretation for the pair (u_j, v_j)
///   two_sided_absolute    sqrt( ||A V - U S||_F^2 + ||A' U - V S||_F^2 ), unnormalized;
///                         certifies only ~eps_mach * sigma_1 / sigma_i relative accuracy
///                         for triplet i, and accepts sigma_i <= eps vacuously
///
/// Computed in one pass because all three are the same two operator applications with
/// different scalings; running them separately would triple the matvec cost, which
/// dominates. The unscaled residuals are formed first, the absolute norms taken, then
/// the columns are scaled by 1/sigma_i in place for the normalized variants.
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

    T* U_cpy = new T[m * k]();
    T* V_cpy = new T[n * k]();

    // U_cpy = A V - U diag(Sigma); V_cpy = A' U - V diag(Sigma). Unnormalized.
    lapack::lacpy(MatrixType::General, m, k, U, m, U_cpy, m);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(m, Sigma[i], &U_cpy[m * i], 1);
    A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, V, n, (T)-1.0, U_cpy, m);

    lapack::lacpy(MatrixType::General, n, k, V, n, V_cpy, n);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(n, Sigma[i], &V_cpy[n * i], 1);
    A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, (T)1.0, U, m, (T)-1.0, V_cpy, n);

    // Absolute variant, before any normalization destroys it.
    T abs1 = lapack::lange(Norm::Fro, m, k, U_cpy, m);
    T abs2 = lapack::lange(Norm::Fro, n, k, V_cpy, n);

    // Normalize each column by its own sigma_i, in place.
    for (int64_t i = 0; i < k; ++i) {
        blas::scal(m, T(1) / Sigma[i], &U_cpy[m * i], 1);
        blas::scal(n, T(1) / Sigma[i], &V_cpy[n * i], 1);
    }
    T nrm1 = lapack::lange(Norm::Fro, m, k, U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, n, k, V_cpy, n);

    delete[] U_cpy;
    delete[] V_cpy;

    return SvdResidualTriple<T>{
        std::hypot(nrm1, nrm2),   // ours
        nrm1,                     // one-sided, normalized
        std::hypot(abs1, abs2)    // two-sided, absolute
    };
}

/// The two-sided normalized residual of EACH triplet separately:
///
///     res[i] = sqrt( ||A v_i - sigma_i u_i||^2 + ||A' u_i - sigma_i v_i||^2 ) / sigma_i
///
/// The two helpers above answer "how accurate is this set of triplets, taken together".
/// They cannot answer "how many of the returned triplets are real", because a single
/// Frobenius norm mixes converged triplets with junk and reports one number. That second
/// question is the one that matters for a block Krylov method that may commit basis columns
/// carrying no operator content: such a column comes back as a triplet with sigma near
/// zero, and only a per-triplet test exposes it. A caller can then count how many triplets
/// certify, which is the honest measure of what was delivered.
///
/// Writes k entries to res_out, which the caller allocates. Two operator applications
/// total, the same cost as either aggregate helper.
///
/// A triplet with sigma_i <= 0 is reported as infinite rather than skipped or bailed on.
/// The aggregate helpers return early in that case, which is right for an adaptive loop
/// that just needs "not converged", but wrong here: a returned triplet with no singular
/// value is precisely the failure this function exists to catch, so it must be visible as
/// one bad entry rather than poison the whole array.
template <typename T, LinearOperator GLO>
void svd_residual_per_triplet(GLO& A, T* U, T* V, T* Sigma, int64_t k, T* res_out) {
    if (k < 1)
        return;

    const T inf = std::numeric_limits<T>::infinity();
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;

    T* U_cpy = new T[m * k]();
    T* V_cpy = new T[n * k]();

    // U_cpy = A V - U diag(Sigma); V_cpy = A' U - V diag(Sigma). Left unnormalized here:
    // the scaling is per column below, so that a single zero sigma cannot contaminate its
    // neighbours the way an in-place whole-block scal would.
    lapack::lacpy(MatrixType::General, m, k, U, m, U_cpy, m);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(m, Sigma[i], &U_cpy[m * i], 1);
    A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, V, n, (T)-1.0, U_cpy, m);

    lapack::lacpy(MatrixType::General, n, k, V, n, V_cpy, n);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(n, Sigma[i], &V_cpy[n * i], 1);
    A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, (T)1.0, U, m, (T)-1.0, V_cpy, n);

    for (int64_t i = 0; i < k; ++i) {
        if (Sigma[i] <= T(0)) {
            res_out[i] = inf;
            continue;
        }
        T ru = blas::nrm2(m, &U_cpy[m * i], 1);
        T rv = blas::nrm2(n, &V_cpy[n * i], 1);
        res_out[i] = std::hypot(ru, rv) / Sigma[i];
    }

    delete[] U_cpy;
    delete[] V_cpy;
}

/// How many of the k returned triplets certify at `tol`, under the per-triplet two-sided
/// normalized residual. This is the "delivered real content" count: a fabricated direction
/// cannot pass a two-sided test, so it is a lower bound on genuine spectral content and an
/// upper bound on what the caller may honestly claim.
template <typename T, LinearOperator GLO>
int64_t svd_triplets_certified(GLO& A, T* U, T* V, T* Sigma, int64_t k, T tol) {
    if (k < 1)
        return 0;
    T* res = new T[k]();
    svd_residual_per_triplet<T>(A, U, V, Sigma, k, res);
    int64_t count = 0;
    for (int64_t i = 0; i < k; ++i)
        if (res[i] <= tol) ++count;
    delete[] res;
    return count;
}

} // end namespace RandLAPACK::linops
