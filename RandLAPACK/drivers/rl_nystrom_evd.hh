#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "rl_nystrom_recovery.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
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
    T* Q     = nullptr; int64_t Q_sz     = 0;   // m × k  (dense orthonormalized iterate, ping-pong partner of Y; q > 1 only, unallocated at q = 1)
    T* Y     = nullptr; int64_t Y_sz     = 0;   // m × k  (holds Y, then Y_ν, then B)
    T* G     = nullptr; int64_t G_sz     = 0;   // k × k  (Gram H = ΩᵀY_ν, then its Cholesky factor)
    T* Sigma = nullptr; int64_t Sigma_sz = 0;   // k      (singular values of B)
    T* VT_B  = nullptr; int64_t VT_B_sz  = 0;   // k × k  (gesdd VT output; unused)
    T* tau   = nullptr; int64_t tau_sz   = 0;   // k      (geqrf reflectors; q > 1 only)
    T* Om    = nullptr; int64_t Om_sz    = 0;   // m × k  (dense Gaussian sketch; vec_nnz = -1 only)
    // Diagnostic, written by the spectral recovery, read by nobody inside the algorithm:
    // how many of the k eigenvalues went negative when the shift was removed and were
    // clamped to zero. Purely observational, so it carries no _sz partner and needs no
    // allocation.
    int64_t clamped_eigenvalues = 0;

    // 11-slot timing vector (microseconds), populated when `times_enabled`.
    // Slots used here: 0 alloc, 1 syrf, 2 matvec, 6 spectral-recovery, 10 total.
    bool times_enabled = false;
    std::vector<long> times = std::vector<long>(11, 0L);

    NystromEVD_workspace() = default;
    NystromEVD_workspace(const NystromEVD_workspace&) = delete;
    NystromEVD_workspace& operator=(const NystromEVD_workspace&) = delete;

    ~NystromEVD_workspace() {
        delete[] Q;
        delete[] Y;
        delete[] G;
        delete[] Sigma;
        delete[] VT_B;
        delete[] tau;
        delete[] Om;
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
/// Numbered to match Algorithm 2 of the funNyström++ writeup. Every code
/// block below is tagged [Alg. 2, line N] with the step it implements:
///   (1)  draw sparse Ω ∈ R^{m×k}   SASO with `vec_nnz` nonzeros
///                                  per ROW (Axis::Short with k < m: the short
///                                  axis is the k columns, so each of the m
///                                  rows scatters vec_nnz ±1 entries across
///                                  them; nnz(Ω) = vec_nnz·m), generated
///                                  internally from `state` (CQRRPT-style;
///                                  `state` is advanced). Matches Alg. 1
///                                  line 1 of the writeup, which specifies a
///                                  SASO sketch (not the partitioned SparseStack distribution).
///                                  vec_nnz = 0 means
///                                  "auto" (resolved below to ~log(k)).
///        (+ q−1 subspace-iteration passes Ω ← orthonormalize(A·Ω); a
///         generalization of Algorithm 2, which is single-pass)
///   (2)  Y   ← A·Ω                 in sparse arithmetic when the operator
///                                  supports a SkOp matvec (ExplicitSymLinOp:
///                                  right_spmm, needs BOTH triangles populated)
///   (3)  ν   ← sqrt(m)·eps·‖Y‖_F   shift (pseudocode convention)
///   (4)  Y_ν ← Y + ν·Ω             = (A+νI)·Ω
///   (5)  C   ← chol(Ωᵀ·Y_ν)        Ωᵀ(A+νI)Ω is SPD iff rank(Ω) = k
///   (6)  B   ← Y_ν·C⁻¹             triangular solve
///   (7)  [U, Σ, ~] ← svd_econ(B)
///   (8)  λ̂   ← max{0, Σ² − ν}      remove the shift
///        ((9)-(10) truncate to rank k: a no-op here, since Ω is drawn at rank k)
///
/// The shift makes Ωᵀ(A+νI)Ω positive definite whenever Ω has full column
/// rank k: the shift adds ν·Ω (not ν·I) to Y, so it regularizes a (near)
/// rank-deficient SPECTRUM of A but cannot repair a rank-deficient SKETCH.
/// With the Axis::Short SASO a column of Ω is empty with probability
/// (1 − vec_nnz/k)^m, so a fixed small vec_nnz makes an exactly-zero column
/// (hence an exactly singular Gram, hence a genuine potrf failure)
/// increasingly likely as k grows toward m; ~log(k) nonzeros per row keeps
/// that probability negligible, which is what vec_nnz = 0 ("auto") resolves
/// to at sketch time.
///
/// Ω stays in sparse form wherever it appears - it is never densified. At
/// q = 1: line 2 applies it through the operator's SkOp matvec, line 4's ν·Ω
/// update is a scatter-add over the m·vec_nnz stored nonzeros, and line 5's
/// Gram applies Ωᵀ as a sketching operator via RandBLAS::sketch_general
/// (O(m·vec_nnz·k), vs O(m·k²) for a dense GEMM). At q > 1 the recovery
/// consumes the dense orthonormalized iterate Q instead - which is where the
/// sketch naturally *becomes* dense, through Q ← orthonormalize(A·…); the
/// subspace-iteration passes ping-pong between ws.Q and ws.Y via pointer
/// swaps; no m×k copies anywhere. In code the sketching operator has exactly
/// one name (S); ws.Q is the dense iterate, never an alias of the sketch.
/// The operator type must provide the SkOp matvec overload (as
/// linops::ExplicitSymLinOp does); one without it fails to compile at the
/// A_op(..., S, ...) call.
///
/// Outputs `U_out` (m × k, column-major) and `lambda_out` (length k,
/// descending) as raw heap buffers managed via `util::upsize`. The caller owns
/// the buffers (FunNystromPP holds them as members and frees them in its dtor).
template <typename T, linops::SymmetricLinearOperator SLO, typename RNG>
void NystromEVD(
    SLO &A_op,
    int64_t k,
    int64_t q,
    int64_t vec_nnz,
    RandBLAS::RNGState<RNG> &state,
    T*& U_out,        int64_t& U_out_sz,
    T*& lambda_out,   int64_t& lambda_out_sz,
    NystromEVD_workspace<T> &ws,
    double *t_specrec_ms_out = nullptr
) {
    using namespace blas;
    int64_t m = A_op.dim;

    if (k < 1 || k > m)
        throw std::invalid_argument(
            "NystromEVD: sketch rank k must be in [1, n]; got k = " +
            std::to_string(k) + " with n = " + std::to_string(m) +
            ". gesdd fills only min(n, k) singular values, so k > n would "
            "leave the recovery reading uninitialized data.");
    if (q < 1)
        throw std::invalid_argument(
            "NystromEVD: q must be >= 1 (total A-applications in Phase 1); got q = " +
            std::to_string(q) + ".");
    // The SASO is SparseDist(m, k, vec_nnz) with Axis::Short, so RandBLAS
    // requires vec_nnz <= min(m, k) = k (each row scatters vec_nnz entries
    // across the k columns). 0 opts in to the auto policy resolved below.
    // vec_nnz = -1 selects a dense Gaussian sketch (RandBLAS::DenseDist, as BQRRP draws its
    // sketch) applied by ordinary matrix products: the comparison series for what sparsity buys.
    const bool dense_sketch = (vec_nnz == -1);
    if (!dense_sketch && (vec_nnz < 0 || vec_nnz > std::min(m, k)))
        throw std::invalid_argument(
            "NystromEVD: vec_nnz must be -1 (dense Gaussian), 0 (auto) or in [1, min(n, k)] = [1, " +
            std::to_string(std::min(m, k)) + "]; got vec_nnz = " +
            std::to_string(vec_nnz) + " with n = " + std::to_string(m) +
            ", k = " + std::to_string(k) + ".");

    // Resolve vec_nnz = 0 ("auto") at sketch time. A column of the Axis::Short
    // SASO is empty with probability (1 − vec_nnz/k)^m; an empty column makes
    // the shifted Gram Ωᵀ(A+νI)Ω exactly singular (the shift adds ν·Ω, not
    // ν·I, so it cannot help) and potrf genuinely fails. A fixed small
    // vec_nnz therefore fails increasingly often as k grows toward m;
    // ~log(k) nonzeros per row keeps the empty-column probability negligible.
    // Explicit caller values pass through unchanged (the guard above already
    // enforces vec_nnz <= k).
    const int64_t vnz = dense_sketch ? (int64_t)1   // unused: the SASO is never filled on the dense path
        : (vec_nnz == 0)
        ? std::min(std::max((int64_t)4, (int64_t)std::ceil(std::log((double)k))), k)
        : vec_nnz;

    using clk = std::chrono::steady_clock;
    auto t_total_start = clk::now();
    long t_alloc = 0, t_syrf = 0, t_matvec = 0;

    // [Alg. 2, line 1] Draw the SASO sketch Ω internally
    //   (CQRRPT-style; `state` is advanced past the draw). NOT orthonormalized,
    //   per Algorithm 2 and the basis-invariance of the shifted recovery.
    RandBLAS::SparseDist DS(m, k, vnz);
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);

    // The operator must support the RandBLAS SketchingOperator matvec overload
    // (like linops::ExplicitSymLinOp); an operator without it fails to compile
    // at the A_op(..., S, ...) calls below. At q = 1 the recovery consumes Ω in
    // sparse form (scatter-add + sketch_general) and no dense image of the
    // sketch ever exists; ws.Q is needed only for the q > 1 QR iterates.

    // [setup] Allocate/grow workspace (not an Algorithm-2 line). ws.Q holds
    // the orthonormalized iterate at q > 1 (ping-pong partner of ws.Y);
    // ws.Q and ws.tau are not allocated at q = 1.
    t_alloc = detail::measure_us([&] {
        util::upsize(ws.Y,     ws.Y_sz,     m * k);
        util::upsize(ws.G,     ws.G_sz,     k * k);
        util::upsize(ws.Sigma, ws.Sigma_sz, k);
        util::upsize(ws.VT_B,  ws.VT_B_sz,  k * k);
        util::upsize(U_out,      U_out_sz,      m * k);
        util::upsize(lambda_out, lambda_out_sz, k);
        if (dense_sketch) util::upsize(ws.Om, ws.Om_sz, m * k);
        if (q > 1) {
            util::upsize(ws.Q,   ws.Q_sz,   m * k);
            util::upsize(ws.tau, ws.tau_sz, k);
        }
    });

    //   Beyond Algorithm 2: q−1 subspace-iteration passes Q ← orthonormalize(A·Q)
    //   (the first starting from A·S) sharpen the captured subspace. This is
    //   where the sketch naturally becomes a dense object: the iterate Q, not Ω.
    //   Re-orthonormalizing each pass is the range-finder and is required for
    //   q > 1 (else the iterated columns collapse onto the dominant
    //   eigenvector). q = 1 does none of this. The iterates ping-pong between
    //   ws.Y and ws.Q via pointer swaps (no m×k copies); the q-th (final)
    //   application happens in the t_matvec block below.
    t_syrf = detail::measure_us([&] {
        if (dense_sketch) {
            RandBLAS::DenseDist D(m, k);
            state = RandBLAS::fill_dense(D, ws.Om, state);
        } else {
            RandBLAS::fill_sparse(S);
            state = S.next_state;
        }
        if (q > 1) {
            if (dense_sketch) A_op(Layout::ColMajor, k, (T)1, ws.Om, m, (T)0, ws.Y, m);
            else              A_op(Layout::ColMajor, k, (T)1, S, (T)0, ws.Y, m);
            for (int64_t iter = 1; iter < q; ++iter) {
                lapack::geqrf(m, k, ws.Y, m, ws.tau);
                lapack::ungqr(m, k, k, ws.Y, m, ws.tau);
                std::swap(ws.Q, ws.Y);
                std::swap(ws.Q_sz, ws.Y_sz);
                if (iter < q - 1)
                    A_op(Layout::ColMajor, k, (T)1, ws.Q, m, (T)0, ws.Y, m);
            }
        }
    });

    // [Alg. 2, line 2] Y ← A·Ω  (the application whose output the recovery
    //   consumes: the single sparse pass at q = 1, else the q-th application
    //   A·Q on the last orthonormalized iterate).
    t_matvec = detail::measure_us([&] {
        if (q == 1 && dense_sketch)
            A_op(Layout::ColMajor, k, (T)1, ws.Om, m, (T)0, ws.Y, m);
        else if (q == 1)
            A_op(Layout::ColMajor, k, (T)1, S, (T)0, ws.Y, m);
        else
            A_op(Layout::ColMajor, k, (T)1, ws.Q, m, (T)0, ws.Y, m);
    });

    // ---- Shifted Nyström spectral recovery (Algorithm 2, lines 3-8) ----
    auto t_specrec_start = clk::now();

    detail::nystrom_recovery<T>(m, k,
        {ws.Y, ws.G, ws.Sigma, ws.VT_B, ws.clamped_eigenvalues},
        U_out, lambda_out,
        [&](T nu) {
            // [Alg. 2, line 4] Y_nu = Y + nu*Omega. Keep the SASO sparse.
            if (q == 1 && dense_sketch) {
                blas::axpy(m * k, nu, ws.Om, 1, ws.Y, 1);
            } else if (q == 1) {
                auto S_coo = RandBLAS::coo_view_of_skop(S);
                for (int64_t t = 0; t < S_coo.nnz; ++t)
                    ws.Y[S_coo.rows[t] + S_coo.cols[t] * m] += nu * S_coo.vals[t];
            } else {
                blas::axpy(m * k, nu, ws.Q, 1, ws.Y, 1);
            }
        },
        [&] {
            // [Alg. 2, line 5] Form both triangles of Omega^T*Y_nu.
            if (q == 1 && dense_sketch) {
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
                           (T)1, ws.Om, m, ws.Y, m, (T)0, ws.G, k);
            } else if (q == 1) {
                RandBLAS::sketch_general(Layout::ColMajor, Op::Trans, Op::NoTrans,
                                         k, k, m, (T)1, S, 0, 0, ws.Y, m, (T)0, ws.G, k);
            } else {
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
                           (T)1, ws.Q, m, ws.Y, m, (T)0, ws.G, k);
            }
        }, vnz);

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
