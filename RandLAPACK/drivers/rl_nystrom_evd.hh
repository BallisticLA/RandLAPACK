#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace RandLAPACK {

// --- Phase 1: free-standing NystromEVD ------------------------------------
//
// Reference-aligned sketched Nyström spectral recovery, shifted variant
// (Persson-Kressner / Epperly). Used by `FunNystromPP` in
// `rl_fun_nystrom_pp.hh`.

// Heap-owned workspace for `NystromEVD`. Buffers grow on demand via
// `util::upsize`; existing contents are not preserved across calls. Kept as a
// plain pointer-bag (not `std::vector`-of-`T`) so FunNystromPP can hold it
// alive across many `call()` invocations at amortised allocation cost.
template <typename T>
struct NystromEVD_workspace {
    T* Omega = nullptr; int64_t Omega_sz = 0;   // m × k
    T* Y     = nullptr; int64_t Y_sz     = 0;   // m × k  (holds Y, then Y_ν, then B)
    T* G     = nullptr; int64_t G_sz     = 0;   // k × k  (Gram H = ΩᵀY_ν, then its Cholesky factor)
    T* Sigma = nullptr; int64_t Sigma_sz = 0;   // k      (singular values of B)
    T* VT_B  = nullptr; int64_t VT_B_sz  = 0;   // k × k  (gesdd VT output; unused)
    T* tau   = nullptr; int64_t tau_sz   = 0;   // k      (geqrf reflectors)

    // 11-slot timing vector (microseconds), populated when `times_enabled`.
    // Slots used here: 0 alloc, 1 syrf, 2 matvec, 6 spectral-recovery, 10 total.
    bool times_enabled = false;
    std::vector<long> times = std::vector<long>(11, 0L);

    NystromEVD_workspace() = default;
    NystromEVD_workspace(const NystromEVD_workspace&) = delete;
    NystromEVD_workspace& operator=(const NystromEVD_workspace&) = delete;

    ~NystromEVD_workspace() {
        delete[] Omega;
        delete[] Y;
        delete[] G;
        delete[] Sigma;
        delete[] VT_B;
        delete[] tau;
    }
};


namespace detail {
// Time fn() in microseconds: run it synchronously and return the elapsed
// steady_clock duration.
template <typename Fn>
long measure_us(Fn&& fn) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
}
} // namespace detail


/// Reference-aligned sketched Nyström spectral recovery, shifted variant.
/// Port of davpersson/funNystrom nystrom_epperly.m, numbered to match
/// Algorithm 2 of the funNyström++ writeup. Every code block below is
/// tagged [Alg. 2, line N] with the step it implements:
///   (1)  Ω   ← Ω₁                 take the caller's sketch (NOT orthonormalized)
///        (+ q−1 subspace-iteration passes Ω ← orthonormalize(A·Ω); a
///         generalization of Algorithm 2, which is single-pass)
///   (2)  Y   ← A·Ω
///   (3)  ν   ← sqrt(m)·eps·‖Y‖_F   shift (pseudocode convention)
///   (4)  Y_ν ← Y + ν·Ω             = (A+νI)·Ω
///   (5)  C   ← chol(Ωᵀ·Y_ν)        Ωᵀ(A+νI)Ω is SPD, so Cholesky never fails
///   (6)  B   ← Y_ν·C⁻¹             triangular solve
///   (7)  [U, Σ, ~] ← svd_econ(B)
///   (8)  λ̂   ← max{0, Σ² − ν}      remove the shift
///        ((9)-(10) truncate to rank k: a no-op here, since Ω is drawn at rank k)
///
/// The shift makes Ωᵀ(A+νI)Ω strictly positive definite, so the Cholesky cannot
/// fail on a (near) rank-deficient spectrum and no fall-back path is needed (the
/// earlier dual-path recovery has been removed).
///
/// Outputs `U_out` (m × k, column-major) and `lambda_out` (length k,
/// descending) as raw heap buffers managed via `util::upsize`. The caller owns
/// the buffers (FunNystromPP holds them as members and frees them in its dtor).
///
/// `force_fallback` is retained for call-site compatibility but ignored: with
/// the shift there is only one recovery path.
template <typename T, linops::SymmetricLinearOperator SLO>
void NystromEVD(
    SLO &A_op,
    int64_t k,
    int64_t q,
    const T *Omega1_in,
    T*& U_out,        int64_t& U_out_sz,
    T*& lambda_out,   int64_t& lambda_out_sz,
    NystromEVD_workspace<T> &ws,
    bool /*force_fallback*/ = false,
    double *t_specrec_ms_out = nullptr
) {
    using namespace blas;
    int64_t m = A_op.dim;

    using clk = std::chrono::steady_clock;
    auto t_total_start = clk::now();
    long t_alloc = 0, t_syrf = 0, t_matvec = 0;

    // [setup] Allocate/grow workspace (not an Algorithm-2 line).
    t_alloc = detail::measure_us([&] {
        util::upsize(ws.Omega, ws.Omega_sz, m * k);
        util::upsize(ws.Y,     ws.Y_sz,     m * k);
        util::upsize(ws.G,     ws.G_sz,     k * k);
        util::upsize(ws.Sigma, ws.Sigma_sz, k);
        util::upsize(ws.VT_B,  ws.VT_B_sz,  k * k);
        util::upsize(ws.tau,   ws.tau_sz,   k);
        util::upsize(U_out,      U_out_sz,      m * k);
        util::upsize(lambda_out, lambda_out_sz, k);
    });

    // [Alg. 2, line 1] Ω ← Ω₁  (take the caller's sketch; NOT orthonormalized, per
    //   nystrom_epperly.m and the basis-invariance of the shifted recovery).
    //   Beyond Algorithm 2: q−1 subspace-iteration passes Ω ← orthonormalize(A·Ω)
    //   sharpen the captured subspace. Re-orthonormalizing A·Ω each pass is the
    //   range-finder and is required for q > 1 (else the iterated columns collapse
    //   onto the dominant eigenvector). q = 1 does none of this (matches Epperly).
    t_syrf = detail::measure_us([&] {
        std::copy(Omega1_in, Omega1_in + m * k, ws.Omega);

        for (int64_t iter = 1; iter < q; ++iter) {
            A_op(Layout::ColMajor, k, (T)1, ws.Omega, m, (T)0, ws.Y, m);
            std::copy(ws.Y, ws.Y + m * k, ws.Omega);
            lapack::geqrf(m, k, ws.Omega, m, ws.tau);
            lapack::ungqr(m, k, k, ws.Omega, m, ws.tau);
        }
    });

    // [Alg. 2, line 2] Y ← A·Ω
    t_matvec = detail::measure_us([&] {
        A_op(Layout::ColMajor, k, (T)1, ws.Omega, m, (T)0, ws.Y, m);
    });

    // ---- Shifted Nyström spectral recovery (Algorithm 2, lines 3-8) ----
    auto t_specrec_start = clk::now();

    const T eps_mach = std::numeric_limits<T>::epsilon();
    // [Alg. 2, line 3] ν ← sqrt(m)·eps·‖Y‖_F  (pseudocode convention; NB
    //   nystrom_epperly.m uses eps·‖Y‖_F / sqrt(m), a factor-of-m difference,
    //   pinned to the pseudocode per project decision 2026-07-08).
    const T nu = std::sqrt((T)m) * eps_mach * blas::nrm2(m * k, ws.Y, 1);

    // [Alg. 2, line 4] Y_ν ← Y + ν·Ω = (A+νI)·Ω  (overwrites ws.Y).
    blas::axpy(m * k, nu, ws.Omega, 1, ws.Y, 1);

    // [Alg. 2, line 5a] H ← Ωᵀ·Y_ν  (k×k = Ωᵀ(A+νI)Ω, SPD since A+νI is); symmetrize.
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
               (T)1, ws.Omega, m, ws.Y, m, (T)0, ws.G, k);
    RandLAPACK::util::symmetrize(k, ws.G, k);

    // [Alg. 2, line 5b] C ← chol(H), upper. SPD by construction; guard defensively.
    int chol_status = lapack::potrf(Uplo::Upper, k, ws.G, k);
    if (chol_status != 0)
        throw std::runtime_error(
            "NystromEVD: shifted Cholesky failed (potrf status != 0); "
            "the shift nu was too small for this operator.");
    RandLAPACK::util::get_U(k, k, ws.G, k);   // zero strict lower of the factor

    // [Alg. 2, line 6] B ← Y_ν·C⁻¹  (triangular solve; B overwrites ws.Y).
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
               m, k, (T)1, ws.G, k, ws.Y, m);

    // [Alg. 2, line 7] [U, Σ, ~] ← svd_econ(B). U_out ← left singular vectors (m×k).
    lapack::gesdd(lapack::Job::SomeVec, m, k, ws.Y, m,
                  ws.Sigma, U_out, m, ws.VT_B, k);

    // [Alg. 2, line 8] λ̂ ← max{0, Σ² − ν}  (remove the shift; clamp negatives to 0).
    //   Lines 9-10 (truncate to rank k) are a no-op: Ω is drawn at rank k, so B is
    //   m×k and U_out / lambda_out already have exactly k columns / entries.
    for (int64_t i = 0; i < k; ++i)
        lambda_out[i] = std::max(ws.Sigma[i] * ws.Sigma[i] - nu, (T)0);

    auto t_specrec_end = clk::now();
    if (t_specrec_ms_out) {
        *t_specrec_ms_out = std::chrono::duration<double, std::milli>(
            t_specrec_end - t_specrec_start).count();
    }

    if (ws.times_enabled) {
        long t_total = std::chrono::duration_cast<std::chrono::microseconds>(
                           clk::now() - t_total_start).count();
        long t_specrec = std::chrono::duration_cast<std::chrono::microseconds>(
                             t_specrec_end - t_specrec_start).count();
        ws.times[0]  = t_alloc;
        ws.times[1]  = t_syrf;
        ws.times[2]  = t_matvec;
        ws.times[6]  = t_specrec;
        ws.times[10] = t_total;
    }
}


} // namespace RandLAPACK
