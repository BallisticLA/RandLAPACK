#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace RandLAPACK::detail {

// Caller-owned storage for shifted Nyström recovery. Y is m-by-k, G and
// VT_B are k-by-k, and Sigma has length k. All matrices are column-major.
// G and VT_B may share storage: the Cholesky factor is no longer needed
// when the SVD writes its right singular vectors.
template <typename T>
struct NystromRecoveryBuffers {
    T* Y;
    T* G;
    T* Sigma;
    T* VT_B;
    int64_t& clamped_eigenvalues;
};

// Recover k eigenpairs from Y = A*Omega for a PSD matrix A and 1 <= k <= m.
// add_shift(nu) must overwrite Y with Y + nu*Omega. form_gram() must then
// write both triangles of Omega^T*Y into G. These operations let the caller
// retain a sparse sketch without changing the recovery or allocating a dense
// copy. U_out and lambda_out have sizes m*k and k, respectively.
// The Gram-EVD alternative is opt-in; NystromEVD restricts it to double
// precision, while REVD2 keeps the default thin SVD.
//
// Returns the shift used in the recovery. A zero sampled image returns an
// orthonormal basis and zero eigenvalues when k < m. At k == m, Cholesky
// must succeed: a singular square sketch cannot establish full recovery.
template <typename T, typename AddShift, typename FormGram>
T nystrom_recovery(
    int64_t m, int64_t k,
    NystromRecoveryBuffers<T> ws,
    T* U_out, T* lambda_out,
    AddShift&& add_shift, FormGram&& form_gram,
    int64_t vec_nnz = -1,
    const char* caller = "NystromEVD",
    bool use_gram_evd = false
) {
    using namespace blas;

    if (k < m && std::all_of(ws.Y, ws.Y + m * k, [](T value) { return value == (T)0; })) {
        lapack::laset(lapack::MatrixType::General, m, k, (T)0, (T)1, U_out, m);
        std::fill(lambda_out, lambda_out + k, (T)0);
        ws.clamped_eigenvalues = 0;
        return (T)0;
    }

    const T eps_mach = std::numeric_limits<T>::epsilon();
    // This is the shift used by NystromEVD's fixed-rank recovery. In single
    // precision it can dominate the spectral tail for large sparse sketches.
    if constexpr (sizeof(T) < 8) {
        static std::atomic<bool> nu_note_emitted{false};
        if (vec_nnz >= 0 && !nu_note_emitted.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr,
                "NOTE %s: single-precision shift nu ~ n*eps*sqrt(vec_nnz)*||A||_2 "
                "can reach ~1e-3*||A||_2 at n ~ 3000 and clamps the recovered "
                "eigenvalue tail; prefer double precision for spectra spanning "
                "more than a few decades.\n", caller);
        }
    }
    const T nu = std::sqrt((T)m) * eps_mach * blas::nrm2(m * k, ws.Y, 1);

    add_shift(nu);
    form_gram();
    // Each triangle is accumulated independently. Factor their symmetric
    // average so the result does not depend on the selected triangle.
    RandLAPACK::util::symmetrize(k, ws.G, k);

    int64_t chol_status = lapack::potrf(Uplo::Upper, k, ws.G, k);
    if (chol_status != 0) {
        std::string message = std::string(caller) + ": shifted Cholesky failed (potrf status " +
            std::to_string(chol_status) + " at rank k=" + std::to_string(k) +
            ", n=" + std::to_string(m);
        if (vec_nnz >= 0) {
            message += ", vec_nnz=" + std::to_string(vec_nnz) +
                "). At large k the likely cause is an exactly-zero column of the "
                "sparse SASO sketch (probability (1 - vec_nnz/k)^n per column): a "
                "rank-deficient Omega makes the Gram Omega^T(A+nu I)Omega exactly "
                "singular, and the shift adds nu*Omega, not nu*I, so it cannot "
                "help. Raise vec_nnz (or pass vec_nnz = 0 for the ~log(k) auto "
                "policy) and keep the sketch rank k <= n/2.";
        } else {
            message += "). The supplied sketch may be rank deficient.";
        }
        throw std::runtime_error(message);
    }

    // B = (A + nu*I)*Omega*C^{-1}, where C^T*C is the shifted Gram.
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
               m, k, (T)1, ws.G, k, ws.Y, m);
    // [Alg. 2, line 7] [U, Σ, ~] ← svd_econ(B). U_out ← left singular vectors (m×k).
    // Opt-in performance switch, off by default: RANDLAPACK_PERF_NYSEIG=1 replaces the thin SVD of the
    // m×k B by an eigendecomposition of the k×k Gram BᵀB: its eigenvalues are Σ² (all line 8 needs) and
    // U = B·V·Σ⁻¹. About 3.5x faster at m = 50,000, k = 7,276. Squaring loses only eigenvalues below about
    // eps·‖B‖², under the shift that line 8 removes; a zero singular value leaves a zero column of U (λ̂ = 0).
    // Double precision only: squaring B loses the eigenvalues below about eps*||B||^2, which is
    // harmless in double (estimates move by at most 2.8e-14 on 72 cases) but up to 1.3e-4 in
    // single, so single precision keeps the thin SVD whatever the switch says.
    if (use_gram_evd) {
        T* V = ws.VT_B;   // k×k, otherwise gesdd's unused VT output
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, (T)1, ws.Y, m, (T)0, V, k);
        lapack::syevd(lapack::Job::Vec, Uplo::Upper, k, V, k, ws.Sigma);   // ascending
        for (int64_t i = 0; i < k / 2; ++i) {   // descending, as gesdd returns them
            std::swap(ws.Sigma[i], ws.Sigma[k - 1 - i]);
            std::swap_ranges(V + i * k, V + (i + 1) * k, V + (k - 1 - i) * k);
        }
        for (int64_t i = 0; i < k; ++i) {       // σ = sqrt(eigenvalue); scale V's columns by 1/σ before U = B·V
            ws.Sigma[i] = std::sqrt(std::max(ws.Sigma[i], (T)0));
            const T inv = (ws.Sigma[i] > (T)0) ? (T)1 / ws.Sigma[i] : (T)0;
            blas::scal(k, inv, V + i * k, 1);
        }
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, (T)1, ws.Y, m, V, k, (T)0, U_out, m);
    } else {
        lapack::gesdd(lapack::Job::SomeVec, m, k, ws.Y, m,
                      ws.Sigma, U_out, m, ws.VT_B, k);
    }


    // Remove the shift and clamp eigenvalues without changing the recovered columns.
    ws.clamped_eigenvalues = 0;
    for (int64_t i = 0; i < k; ++i) {
        const T raw = ws.Sigma[i] * ws.Sigma[i] - nu;
        if (raw < (T)0) ws.clamped_eigenvalues += 1;
        lambda_out[i] = std::max(raw, (T)0);
    }
    return nu;
}

} // namespace RandLAPACK::detail
