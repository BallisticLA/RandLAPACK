#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace RandLAPACK {

/// FunNyström++ v2 — reference-aligned trace estimator.
///
/// From-scratch C++ port of the Persson-Kressner MATLAB reference
/// (`davpersson/funNystrom/Other/{nystrom,funnystrompp}.m`), intended as
/// a bit-exact baseline against MATLAB at fixed RNG. Deliberately does
/// not share infrastructure with the production funnystrompp PR — no
/// SYPS, no SYRF, no internal_stab plumbing — so that each algorithmic
/// knob can be added back one at a time with verification.
///
/// Phase 1 (NystromEVD_v2, implemented inline below): Gaussian sketch +
/// q − 1 subspace-iteration passes (with QR stabilization between
/// passes) + SVD-pseudoinverse recovery on the k × k Gram. Mirrors
/// `nystrom.m` line-for-line.
///
/// Phase 2 (this class's `call`): Hutchinson on the residual
/// f(A) − f(Â) using a caller-supplied fAfun oracle. Mirrors
/// `funnystrompp.m` line-for-line. Uses Gaussian Ω₂ to match the
/// reference. A Rademacher option will be added in Phase 6 of the
/// project plan.
///
/// The class takes Ω₁ (Phase 1 sketch) and Ω₂ (Phase 2 Hutchinson)
/// externally — this is the cross-validation harness contract. The
/// caller generates or loads them via the same RNG as the MATLAB side.
/// See util::load_dense_bin for the on-disk format.
///
/// See [funnystrompp-v2-baseline-plan.md](Claude_files/randnla/project-plans/funnystrompp-v2-baseline-plan.md)
/// for the multi-phase development plan.
template <typename T>
class FunNystromPP_v2 {
public:
    bool verbose = false;

    // After call(), these hold Phase 1's eigenpairs of Â. Exposed as
    // public so tests can inspect them; in production code you'd treat
    // them as read-only output.
    std::vector<T> U;        // m × k_out, column-major
    std::vector<T> lambda;   // k_out descending eigenvalues
    int64_t k_out = 0;

    FunNystromPP_v2() = default;
    FunNystromPP_v2(const FunNystromPP_v2&) = delete;
    FunNystromPP_v2& operator=(const FunNystromPP_v2&) = delete;

    /// Returns the estimate t = t1 + t2 of tr(f(A)).
    ///
    /// fAfun(int64_t s, const T* B, T* Y) is a callable that computes
    /// Y := f(A) * B for column-major B, Y of shape m × s with lda = m.
    /// In Phase 1 tests we pass a dense exact-f(A) oracle; in Phase 4
    /// we will swap it for block_lanczos_fa.
    ///
    /// @param[in]  A_op     Symmetric linop providing A * X.
    /// @param[in]  fAfun    Callable B ↦ f(A) * B.
    /// @param[in]  fscalar  Scalar f operating on each eigenvalue.
    /// @param[in]  k        Phase 1 Nyström rank.
    /// @param[in]  s        Phase 2 Hutchinson sample count.
    /// @param[in]  q        Phase 1 number of A applications (q = 1 single
    ///                      pass; q = 2 = 1 subspace-iter pass; etc.).
    /// @param[in]  Omega1   Caller-supplied m × k sketch for Phase 1.
    /// @param[in]  Omega2   Caller-supplied m × s sketch for Phase 2
    ///                      (Gaussian in the v2 baseline).
    /// @param[out] t1_out   Phase 1 contribution Σ f(λ̂ᵢ).
    /// @param[out] t2_out   Phase 2 stochastic correction.
    /// @return     t = t1 + t2.
    template <linops::SymmetricLinearOperator SLO, typename FAFun, typename FScalar>
    T call(
        SLO &A_op,
        FAFun &&fAfun,
        FScalar &&fscalar,
        int64_t k,
        int64_t s,
        int64_t q,
        const T *Omega1,
        const T *Omega2,
        T &t1_out,
        T &t2_out
    );
};


// --- Phase 1: free-standing NystromEVD_v2 ----------------------------------

/// Reference-aligned sketched Nyström spectral recovery.
/// Direct port of davpersson/funNystrom/Other/nystrom.m.
///
/// Computes Â = Y (ΩᵀY)† Yᵀ implicitly via:
///   Ω ← qr(Ω, 0)
///   for iter = 1..q-1:  Ω ← qr(A·Ω, 0)
///   Y ← A · Ω
///   [V_G, D, _] ← svd(Ωᵀ · Y, 'econ')
///   D[D < 5e-16 · D[0]] ← 0                       (explicit pinv threshold)
///   B ← Y · V_G · pinv(diag(√D)) · V_Gᵀ
///   [U, Σ, _] ← svd(B, 'econ')
///   λ ← Σ²
/// and stores U (n × k) and λ (length k, descending) in the
/// caller-supplied vectors. Workspace is owned by the caller through
/// the `workspace` struct so it can be reused across calls.
template <typename T>
struct NystromEVD_v2_workspace {
    std::vector<T> Omega;     // m × k
    std::vector<T> Y;         // m × k
    std::vector<T> G;         // k × k
    std::vector<T> V_G;       // k × k (left singular vectors of G)
    std::vector<T> VT_G;      // k × k (right singular vectors of G, transposed)
    std::vector<T> D;         // k (singular values of G)
    std::vector<T> tmp_kk;    // k × k (for V_G · pinv(√D) · V_Gᵀ)
    std::vector<T> Sigma;     // k (singular values of B)
    std::vector<T> VT_B;      // k × k (right singular vectors of B, unused but gesdd writes it)
    std::vector<T> tau;       // k (geqrf reflectors)
};

template <typename T, linops::SymmetricLinearOperator SLO>
void NystromEVD_v2(
    SLO &A_op,
    int64_t k,
    int64_t q,
    const T *Omega1_in,
    std::vector<T> &U_out,            // populated to size m·k, column-major
    std::vector<T> &lambda_out,        // populated to size k, descending
    NystromEVD_v2_workspace<T> &ws
) {
    using namespace blas;
    int64_t m = A_op.dim;

    // Allocate / resize workspace.
    ws.Omega.assign(m * k, (T)0);
    ws.Y.assign(m * k, (T)0);
    ws.G.assign(k * k, (T)0);
    ws.V_G.assign(k * k, (T)0);
    ws.VT_G.assign(k * k, (T)0);
    ws.D.assign(k, (T)0);
    ws.tmp_kk.assign(k * k, (T)0);
    ws.Sigma.assign(k, (T)0);
    ws.VT_B.assign(k * k, (T)0);
    ws.tau.assign(k, (T)0);
    U_out.assign(m * k, (T)0);
    lambda_out.assign(k, (T)0);

    // Step 1: Ω ← qr(Ω₁_in, 0).
    std::copy(Omega1_in, Omega1_in + m * k, ws.Omega.data());
    lapack::geqrf(m, k, ws.Omega.data(), m, ws.tau.data());
    lapack::ungqr(m, k, k, ws.Omega.data(), m, ws.tau.data());

    // Steps 2–3: q − 1 subspace-iteration passes, each: Ω ← qr(A · Ω, 0).
    for (int64_t iter = 1; iter < q; ++iter) {
        A_op(Layout::ColMajor, k, (T)1, ws.Omega.data(), m, (T)0, ws.Y.data(), m);
        std::copy(ws.Y.begin(), ws.Y.end(), ws.Omega.begin());
        lapack::geqrf(m, k, ws.Omega.data(), m, ws.tau.data());
        lapack::ungqr(m, k, k, ws.Omega.data(), m, ws.tau.data());
    }

    // Step 4: Y ← A · Ω.
    A_op(Layout::ColMajor, k, (T)1, ws.Omega.data(), m, (T)0, ws.Y.data(), m);

    // Step 5: [V_G, D, V_Gᵀ] ← svd(Ωᵀ · Y, 'econ').
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
               (T)1, ws.Omega.data(), m, ws.Y.data(), m, (T)0, ws.G.data(), k);
    lapack::gesdd(lapack::Job::SomeVec, k, k, ws.G.data(), k,
                  ws.D.data(), ws.V_G.data(), k, ws.VT_G.data(), k);

    // Step 6: thresholded pinv. D[j] < 5·ε·D[0] → 0.
    const T D_max  = ws.D[0];
    const T thresh = (T)5e-16 * D_max;
    for (int64_t j = 0; j < k; ++j) {
        if (ws.D[j] < thresh) ws.D[j] = (T)0;
    }

    // Step 7: B ← Y · V_G · pinv(diag(√D)) · V_Gᵀ.
    // Compute pinv(diag(√D)): scale columns of V_G by 1/√D[j] where D[j] > 0.
    for (int64_t j = 0; j < k; ++j) {
        T scale = (ws.D[j] > (T)0) ? (T)1 / std::sqrt(ws.D[j]) : (T)0;
        for (int64_t i = 0; i < k; ++i) {
            ws.tmp_kk[i + j * k] = ws.V_G[i + j * k] * scale;
        }
    }
    // tmp_kk now holds V_G · pinv(diag(√D)). Multiply by V_Gᵀ on the right.
    // Reuse ws.V_G as the output buffer: G ← tmp_kk · V_Gᵀ. Read VT_G (which
    // is the V^T factor returned by gesdd, i.e. V_Gᵀ already).
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, k, k,
               (T)1, ws.tmp_kk.data(), k, ws.VT_G.data(), k, (T)0, ws.G.data(), k);
    // ws.G now holds V_G · pinv(diag(√D)) · V_Gᵀ (k × k, symmetric).
    // B ← Y · ws.G. Reuse ws.Y as output (overwrite).
    // But ws.Y is m × k input AND m × k output — same shape — and we cannot
    // alias in gemm. Compute into a scratch buffer (tmp_kk is wrong size).
    // Reuse U_out: it'll be overwritten in step 8 anyway.
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k,
               (T)1, ws.Y.data(), m, ws.G.data(), k, (T)0, U_out.data(), m);
    // Now U_out holds B (m × k); ws.Y is free.

    // Step 8: [U, Σ, _] ← svd(B, 'econ'). B is in U_out; gesdd overwrites it
    // with the left singular vectors when called with SomeVec.
    // gesdd(SomeVec, m, n, A, lda, S, U, ldu, VT, ldvt) for m >= n writes U
    // into A in place (with min(m,n) columns). We need a separate U buffer
    // because gesdd writes the leading singular vectors into the provided U.
    // Copy B from U_out into ws.Y first so gesdd's input/output don't alias.
    std::copy(U_out.begin(), U_out.end(), ws.Y.begin());
    lapack::gesdd(lapack::Job::SomeVec, m, k, ws.Y.data(), m,
                  ws.Sigma.data(), U_out.data(), m, ws.VT_B.data(), k);

    // Step 9: λ ← Σ².
    for (int64_t i = 0; i < k; ++i) {
        lambda_out[i] = ws.Sigma[i] * ws.Sigma[i];
    }
}


// --- Phase 2: FunNystromPP_v2::call ---------------------------------------

template <typename T>
template <linops::SymmetricLinearOperator SLO, typename FAFun, typename FScalar>
T FunNystromPP_v2<T>::call(
    SLO &A_op,
    FAFun &&fAfun,
    FScalar &&fscalar,
    int64_t k,
    int64_t s,
    int64_t q,
    const T *Omega1,
    const T *Omega2,
    T &t1_out,
    T &t2_out
) {
    int64_t m = A_op.dim;

    // Phase 1.
    NystromEVD_v2_workspace<T> ws;
    NystromEVD_v2<T>(A_op, k, q, Omega1, this->U, this->lambda, ws);
    this->k_out = k;

    // t1 = Σ fscalar(λᵢ).
    t1_out = (T)0;
    for (int64_t i = 0; i < k; ++i) t1_out += fscalar(this->lambda[i]);

    // Phase 2: t2 = ( tr(Ω₂ᵀ · fAfun(Ω₂)) − tr(Yᵀ · diag(f(λ)) · Y) ) / s
    //   where Y = Uᵀ · Ω₂ (shape k × s).

    // Step a: Y_2 ← Uᵀ · Ω₂  (k × s)
    std::vector<T> Y_2(k * s, (T)0);
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, s, m, (T)1, this->U.data(), m, Omega2, m, (T)0, Y_2.data(), k);

    // Step b: fAOmega ← f(A) · Ω₂  (m × s, caller-supplied oracle)
    std::vector<T> fAOmega(m * s, (T)0);
    fAfun(m, s, Omega2, fAOmega.data());

    // Step c: tr_AΩ ← tr(Ω₂ᵀ · fAOmega) = Σⱼ ⟨Ω₂[:,j], fAOmega[:,j]⟩.
    T tr_AOmega = (T)0;
    for (int64_t j = 0; j < s; ++j) {
        tr_AOmega += blas::dot(m, Omega2 + j * m, 1, fAOmega.data() + j * m, 1);
    }

    // Step d: tr_AhatΩ ← tr(Y_2ᵀ · diag(f(λ)) · Y_2)
    //                 = Σⱼ Σᵢ f(λᵢ) · Y_2[i,j]².
    T tr_AhatOmega = (T)0;
    for (int64_t j = 0; j < s; ++j) {
        for (int64_t i = 0; i < k; ++i) {
            T v = Y_2[i + j * k];
            tr_AhatOmega += fscalar(this->lambda[i]) * v * v;
        }
    }

    t2_out = (tr_AOmega - tr_AhatOmega) / (T)s;
    return t1_out + t2_out;
}


} // namespace RandLAPACK
