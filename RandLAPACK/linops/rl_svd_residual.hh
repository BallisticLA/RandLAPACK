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

} // end namespace RandLAPACK::linops
