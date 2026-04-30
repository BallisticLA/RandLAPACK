#pragma once

#include "rl_blaspp.hh"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdint>

namespace RandLAPACK::linops {


/// Residual operator (f(A) - f(Â)) for use as a SymmetricLinearOperator.
///
/// Computes C = α*(f(A)*B - f(Â)*B) + β*C via:
///   Z1 = f(A)*B  using a matrix-function oracle (e.g. LanczosFA)
///   Z2 = f(Â)*B = V*(diag(F_vec)*(V^T B))  using two GEMMs
///
/// Primary use: Hutchinson correction in funNyström++ (FunNystromPP driver).
/// Â = V diag(λ) V^T is the Nyström approximation; V (n×k) and F_vec (k,
/// holding f(λ)) are provided by the caller.
///
/// @tparam T      Scalar type.
/// @tparam SLO_t  Type of the operator A.
/// @tparam LFA_t  Type of the matrix-function oracle (e.g. LanczosFA<T,RNG>).
/// @tparam F_t    Callable type T→T.
template <typename T, typename SLO_t, typename LFA_t, typename F_t>
struct ResidualOp {
    using scalar_t = T;
    const int64_t dim;

    SLO_t&  A;
    LFA_t&  matfun_oracle;  // e.g. LanczosFA: computes f(A)*B
    F_t     f;
    int64_t d;              // Lanczos steps (or oracle depth)
    int64_t k;              // Nyström rank
    T*      V;              // n×k Nyström eigenvectors
    T*      F_vec;          // k-vector: f applied to Nyström eigenvalues
    T*      tmp;            // k×s workspace (owned by caller)
    T*      Z1;             // n×s workspace: f(A)*Ω (owned by caller)
    T*      Z2;             // n×s workspace: f(Â)*Ω (owned by caller)

    ResidualOp(int64_t n_, SLO_t& A_, LFA_t& oracle_, F_t f_,
               int64_t d_, int64_t k_, T* V_, T* Fv_, T* tmp_, T* Z1_, T* Z2_)
        : dim(n_), A(A_), matfun_oracle(oracle_), f(f_),
          d(d_), k(k_), V(V_), F_vec(Fv_), tmp(tmp_), Z1(Z1_), Z2(Z2_) {}

    void operator()([[maybe_unused]] Layout layout, int64_t n_vecs, T alpha,
                    T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        // Z1 = f(A)*B via matrix-function oracle
        matfun_oracle.call(A, B, dim, n_vecs, f, d, Z1);

        // Z2 = f(Â)*B = V * (diag(F_vec) * (V^T B))
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                   k, n_vecs, dim, (T)1.0, V, dim, B, ldb, (T)0.0, tmp, k);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t j = 0; j < n_vecs; ++j)
            for (int64_t i = 0; i < k; ++i)
                tmp[j * k + i] *= F_vec[i];
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   dim, n_vecs, k, (T)1.0, V, dim, tmp, k, (T)0.0, Z2, dim);

        // C = alpha*(Z1 - Z2) + beta*C  column-by-column to respect ldc.
        // Z1 and Z2 are always stored flat (stride dim); C may have ldc >= dim.
        // beta==0 uses a fused read of Z1,Z2 to avoid scal(0,C) propagating NaN.
        if (beta == (T)0) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t j = 0; j < n_vecs; ++j)
                for (int64_t i = 0; i < dim; ++i)
                    C[j * ldc + i] = alpha * (Z1[j * dim + i] - Z2[j * dim + i]);
        } else {
            for (int64_t j = 0; j < n_vecs; ++j) {
                blas::scal(dim, beta, C + j * ldc, 1);
                blas::axpy(dim,  alpha, Z1 + j * dim, 1, C + j * ldc, 1);
                blas::axpy(dim, -alpha, Z2 + j * dim, 1, C + j * ldc, 1);
            }
        }
    }
};


} // end namespace RandLAPACK::linops
