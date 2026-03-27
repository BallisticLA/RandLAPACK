#pragma once

#include "rl_linops.hh"

namespace RandLAPACK::linops {

/// Computes the SVD residual:
///     sqrt(||AV - U * diag(Sigma)||^2_F + ||A'U - V * diag(Sigma)||^2_F) / sigma_k.
/// U is m x k (col-major, ld m), V is n x k (col-major, ld n), Sigma is length k.
template <typename T, LinearOperator GLO>
T svd_residual(GLO& A, T* U, T* V, T* Sigma, int64_t k) {
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;

    T* U_cpy = new T[m * k]();
    T* V_cpy = new T[n * k]();

    // U_cpy = U * diag(Sigma)
    lapack::lacpy(MatrixType::General, m, k, U, m, U_cpy, m);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(m, Sigma[i], &U_cpy[m * i], 1);

    // U_cpy = AV - U*diag(Sigma)
    A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, (T)1.0, V, n, (T)-1.0, U_cpy, m);

    // V_cpy = V * diag(Sigma)
    lapack::lacpy(MatrixType::General, n, k, V, n, V_cpy, n);
    for (int64_t i = 0; i < k; ++i)
        blas::scal(n, Sigma[i], &V_cpy[n * i], 1);

    // V_cpy = A'U - V*diag(Sigma)
    A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, (T)1.0, U, m, (T)-1.0, V_cpy, n);

    T nrm1 = lapack::lange(Norm::Fro, m, k, U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, n, k, V_cpy, n);

    delete[] U_cpy;
    delete[] V_cpy;

    return std::hypot(nrm1, nrm2) / Sigma[k - 1];
}

} // end namespace RandLAPACK::linops
