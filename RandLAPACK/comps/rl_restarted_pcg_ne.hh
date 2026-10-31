#pragma once

// Public API: restarted_pcg_ne, restarted PCG on the right-preconditioned normal equations
// for min_x ||b - A x||_2, with an optional upper-triangular right preconditioner R.
//
// For a tall LinearOperator A (m x n, m >= n) it runs CG on
//     H z = g,    H = R^{-T} A^T A R^{-1},    g = R^{-T} A^T b,
// and returns x = R^{-1} z (R = nullptr: H = A^T A). Each round solves the correction
// equation H dz = r_ne with pcg_inner until its residual drops by `round_drop`, adds dz to z,
// and recomputes the true residual b - A x and the normal-equation residual in the stable
// form R^{-T} A^T (b - A x) (Epperly, Meier and Nakatsukasa, arXiv:2406.03468, Alg. 1)
// rather than g - H z, which cancels catastrophically on hard problems. The convergence test
// is on the true residual; round_drop only paces the rounds.

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_thread_caps.hh"
#include "rl_exceptions.hh"
#include "rl_pcg_inner.hh"
#include "rl_lsqr.hh"
#include "rl_concepts.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>


namespace RandLAPACK {


/// How a restarted_pcg_ne run ended.
enum class RestartedPCGNETermination {
    converged,          ///< ||b - A x|| / ||b|| <= tol
    iteration_budget,   ///< max_iters inner iterations spent
    breakdown,          ///< the inner CG broke down (R unusable or H not SPD in floating point)
    round_budget,       ///< max_restarts rounds after the first spent
    ls_floor,           ///< the true residual stopped improving (outer stagnation, or a zero
                        ///< normal-equation residual with tol unmet)
    oracle_met          ///< the oracle met be_tol
};

inline const char* to_string(RestartedPCGNETermination t) {
    switch (t) {
        case RestartedPCGNETermination::converged:        return "converged";
        case RestartedPCGNETermination::iteration_budget: return "iteration_budget";
        case RestartedPCGNETermination::breakdown:        return "breakdown";
        case RestartedPCGNETermination::round_budget:     return "round_budget";
        case RestartedPCGNETermination::ls_floor:         return "ls_floor";
        case RestartedPCGNETermination::oracle_met:       return "oracle_met";
    }
    return "unknown";
}


/// Caller-supplied convergence measure, such as a sketched backward-error estimate (Epperly,
/// Meier and Nakatsukasa, arXiv:2406.03468, eq. 4.2). It receives the iterate x (length n), the
/// true residual r = b - A x (length m) and A^T r (length n), all formed by the solver, and
/// returns a nonnegative value. It runs under the solver's level-2 thread cap (Blas2ThreadGuard).
template <typename T>
using BackwardErrorOracle = std::function<T(const T* x, const T* r, const T* ATr)>;


/// Controls of one restarted_pcg_ne run.
template <typename T>
struct RestartedPCGNEControls {
    /// Target on ||b - A x|| / ||b||, finite and >= 0 (0 needs an exactly zero residual).
    T   tol = (T)0;
    int max_iters = 0;               ///< inner iterations over all rounds; must be set (>= 1)
    int max_iters_per_round = 200;   ///< inner iteration cap of one round
    T   round_drop = (T)1e-4;        ///< relative drop of the round's residual that ends it, in (0,1)
    int max_restarts = -1;           ///< rounds allowed after the first (0: one round); negative: unlimited
    int inner_stag_window = 20;      ///< inner stagnation window (see PCGInnerControls); <= 0 disables
    /// Relative drop that counts as progress, in [0, 1], both for the inner stagnation test and
    /// for the outer floor test on the true residual.
    T   stag_rel_improve = (T)1e-3;
    /// When > 0, a round also stops once its normal-equation residual is below
    /// inner_abs_tol * ||g||, even short of the round_drop factor: a run-wide floor relative
    /// to ||g||, unlike round_drop, which is relative to each round's starting residual.
    T   inner_abs_tol = (T)0;
    /// Consecutive rounds whose true residual does not drop by stag_rel_improve (against the
    /// last such drop) that end the run at the floor; <= 0 disables.
    int outer_stag_window = 2;
    /// Optional convergence measure, active when set and be_tol >= 0. It is evaluated on a
    /// supplied x0 and after every round; meeting tol outranks it, and so does a breakdown in
    /// the same round. Its time is excluded from times[3] and reported in t_be_us.
    BackwardErrorOracle<T> oracle;
    T   be_tol = (T)-1;
    /// When > 0 and the oracle is active, also evaluate it every be_poll_every iterations
    /// within each round, on the trial iterate R^{-1}(z + dz); each poll costs one triangular
    /// solve and two applies, counted as solve time. A poll that ends the round is also that
    /// round's final evaluation (it saw the same x).
    int be_poll_every = 0;
};


/// Results of one restarted_pcg_ne run: totals, then one entry per round in each vector.
template <typename T>
struct RestartedPCGNEReport {
    int  iters_done   = 0;            ///< inner CG iterations over all rounds
    int  rounds_done  = 0;            ///< rounds run
    T    final_relres = (T)-1;        ///< ||b - A x|| / ||b|| at return
    std::array<long, 4> times = {0, 0, 0, 0};   ///< [fwd_us, adj_us, trsv_us, total_us]; total excludes t_be_us

    std::vector<int> cg_iters;                  ///< inner CG iterations of the round
    std::vector<PCGInnerTermination> cg_status; ///< how the round's inner CG ended
    std::vector<T>   cg_relres;       ///< recursive CG relres (correction equation) of the round's returned iterate
    std::vector<T>   cg_best_relres;  ///< smallest recursive CG relres seen in the round
    std::vector<int> cg_best_iter;    ///< iteration reaching cg_best_relres
    std::vector<T>   ls_relres;       ///< true ||b - A x|| / ||b|| after the round
    std::vector<T>   be;              ///< oracle value after the round (-1 when inactive)
    std::vector<int> polls;           ///< in-round oracle polls of the round
    T    be_x0           = (T)-1;     ///< oracle value of a supplied x0 (-1: none or inactive)
    long t_inner_us      = 0;         ///< inside pcg_inner, excluding the oracle's share of polls
    long t_fwd_inner_us  = 0;         ///< A applies inside pcg_inner (including polls)
    long t_adj_inner_us  = 0;         ///< A^T applies inside pcg_inner (including polls)
    long t_trsv_inner_us = 0;         ///< triangular solves inside pcg_inner (including polls)
    long t_be_us         = 0;         ///< inside the oracle, at round ends and in polls
    long t_poll_us       = 0;         ///< inside in-round polls (applies and oracle)

    void clear() { *this = RestartedPCGNEReport(); }
};


/// @brief Restarted PCG on the right-preconditioned normal equations for min ||b - A x||.
///
/// @param[in]  A     tall LinearOperator (m x n), applied as A*v and A^T*u.
/// @param[in]  m,n   dimensions; must equal A's, with m >= n.
/// @param[in]  R     upper-triangular right preconditioner (n x n, ColMajor), or nullptr for
///                   none. Its diagonal must be nonzero; an ill-conditioned R may end in a
///                   breakdown.
/// @param[in]  ldr   leading dimension of R (>= n when R is given).
/// @param[in]  b     right-hand side (length m).
/// @param[out] x     solution (length n).
/// @param[in]  ctl   controls (see RestartedPCGNEControls).
/// @param[out] report totals and per-round records (see RestartedPCGNEReport); reset on entry.
/// @param[in]  x0    optional initial guess (length n); nullptr starts from x = 0. When given,
///                   z = R x0 and the first normal-equation residual comes from b - A x0.
/// @returns how the run ended. The tolerance outranks every other exit, and a breakdown
///          outranks the oracle. A NaN residual never meets tol, so NaN data ends in a
///          breakdown for any tol < 1.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
RestartedPCGNETermination restarted_pcg_ne(
    GLO& A, int64_t m, int64_t n,
    const T* R, int64_t ldr,
    const T* b, T* x,
    const RestartedPCGNEControls<T>& ctl,
    RestartedPCGNEReport<T>& report,
    const T* x0 = nullptr)
{
    using Term = RestartedPCGNETermination;
    randlapack_require(m == A.n_rows && n == A.n_cols && m >= n)
        << "restarted_pcg_ne: (m, n) = (" << m << ", " << n << ") must equal A's ("
        << A.n_rows << ", " << A.n_cols << ") with m >= n";
    randlapack_require(R == nullptr || ldr >= n)
        << "restarted_pcg_ne: ldr=" << ldr << " must be >= n=" << n;
    randlapack_require(R == nullptr || RandLAPACK::util::diag_is_nonzero(n, R, ldr))
        << "restarted_pcg_ne: R has a zero diagonal entry";
    randlapack_require(std::isfinite(ctl.tol) && ctl.tol >= (T)0)
        << "restarted_pcg_ne: tol must be finite and >= 0 (got " << ctl.tol << ")";
    randlapack_require(ctl.stag_rel_improve >= (T)0 && ctl.stag_rel_improve <= (T)1)
        << "restarted_pcg_ne: stag_rel_improve must lie in [0, 1] (got " << ctl.stag_rel_improve << ")";
    randlapack_require(ctl.max_iters >= 1)
        << "restarted_pcg_ne: max_iters must be >= 1 (got " << ctl.max_iters << ")";
    randlapack_require(ctl.round_drop > (T)0 && ctl.round_drop < (T)1)
        << "restarted_pcg_ne: round_drop must lie in (0,1) (got " << ctl.round_drop << ")";
    randlapack_require(ctl.max_iters_per_round >= 1)
        << "restarted_pcg_ne: max_iters_per_round must be >= 1 (got " << ctl.max_iters_per_round << ")";
    randlapack_require(ctl.be_poll_every >= 0)
        << "restarted_pcg_ne: be_poll_every must be >= 0 (got " << ctl.be_poll_every << ")";

    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    SolveWidthScope solve_scope(n);
    report.clear();
    long t_fwd = 0, t_adj = 0, t_trsv = 0;
    long t_fwd_in = 0, t_adj_in = 0, t_trsv_in = 0, t_kernel = 0;
    long t_be = 0, t_poll = 0, t_be_poll = 0;
    auto total_start = clock::now();

    const bool prec        = (R != nullptr);
    const bool be_active   = static_cast<bool>(ctl.oracle) && ctl.be_tol >= (T)0;
    const bool poll_active = be_active && ctl.be_poll_every > 0;

    // Zero-initialized: z (the cold start is z = 0) and every operator output (an operator may
    // read its output even when beta == 0). The rest is overwritten before use.
    std::unique_ptr<T[]> zeroed(new T[4 * n]()), scratch(new T[5 * n]), wm_buf(new T[m]());
    std::unique_ptr<T[]> rp_buf(poll_active ? new T[m]() : nullptr),
                         ATrp_buf(poll_active ? new T[n]() : nullptr);
    T* z    = zeroed.get();      // preconditioned solution, x = R^{-1} z
    T* g    = z + n;             // R^{-T} A^T b
    T* r_ne = g + n;             // normal-equation residual R^{-T} A^T (b - A x)
    T* q    = r_ne + n;          // H p
    T* dz   = scratch.get();     // inner CG correction
    T* p    = dz + n;            // CG direction
    T* r    = p + n;             // inner CG recursive residual
    T* zb   = r + n;             // inner CG best-iterate snapshot
    T* sc   = zb + n;            // triangular-solve scratch
    T* wm   = wm_buf.get();      // length-m scratch for A applies
    T* rp   = rp_buf.get();      // in-round poll: b - A x_trial
    T* ATrp = ATrp_buf.get();    // in-round poll: A^T (b - A x_trial)

    // Timed pieces (only the triangular solves are thread-capped). `inner` (when non-null) also
    // receives the time, for work done inside pcg_inner.
    auto add = [](long& total, long* inner, long dt) { total += dt; if (inner) *inner += dt; };
    auto solve_R = [&](blas::Op op, T* v, long* inner) {   // v = R^{-1} v or R^{-T} v
        if (!prec) return;
        long dt = 0;
        detail::capped_trsv(op, n, R, ldr, v, &dt);
        add(t_trsv, inner, dt);
    };
    auto apply_fwd = [&](const T* v, T* out, long* inner) {   // out = A v
        auto tf = clock::now();
        A(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, 1, n, (T)1.0, v, n, (T)0.0, out, m);
        add(t_fwd, inner, duration_cast<microseconds>(clock::now() - tf).count());
    };
    auto apply_adj = [&](const T* u, T* out, long* inner) {   // out = A^T u
        auto ta = clock::now();
        A(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, n, 1, m, (T)1.0, u, m, (T)0.0, out, n);
        add(t_adj, inner, duration_cast<microseconds>(clock::now() - ta).count());
    };
    auto residual_into = [&](const T* xq, T* out, long* inner) {   // out = b - A xq
        apply_fwd(xq, out, inner);
        blas::scal(m, (T)-1.0, out, 1);
        blas::axpy(m, (T)1.0, b, 1, out, 1);
    };

    // H v = R^{-T} (A^T (A (R^{-1} v))); runs only inside pcg_inner.
    auto apply_H = [&](const T* vin, T* out) {
        const T* fwd_in = vin;
        if (prec) {
            std::copy(vin, vin + n, sc);
            solve_R(blas::Op::NoTrans, sc, &t_trsv_in);
            fwd_in = sc;
        }
        apply_fwd(fwd_in, wm, &t_fwd_in);
        apply_adj(wm, out, &t_adj_in);
        solve_R(blas::Op::Trans, out, &t_trsv_in);
    };

    // x = R^{-1} z, wm = b - A x; returns ||b - A x|| / ||b||.
    T bnorm = blas::nrm2(m, b, 1);
    T bden = std::max(bnorm, std::numeric_limits<T>::min());
    auto recover_x_and_relres = [&]() -> T {
        std::copy(z, z + n, x);
        solve_R(blas::Op::NoTrans, x, nullptr);
        residual_into(x, wm, nullptr);
        return blas::nrm2(m, wm, 1) / bden;
    };

    // g = R^{-T} A^T b.
    apply_adj(b, g, nullptr);
    solve_R(blas::Op::Trans, g, nullptr);

    auto eval_oracle_at = [&](const T* xq, const T* rq, const T* ATrq) -> T {
        auto tb = clock::now();
        T v;
        { Blas2ThreadGuard tg(n);   // at the solve's width: no team re-formation
            v = ctl.oracle(xq, rq, ATrq);
        }
        t_be += duration_cast<microseconds>(clock::now() - tb).count();
        return v;
    };
    // In-round poll: the trial iterate R^{-1}(z + dz) with the round-end arithmetic, its
    // residual and adjoint product in dedicated buffers. Nothing the kernel reads is touched,
    // so a poll that does not fire leaves the trajectory unchanged.
    int polls_round = 0;
    T poll_be = (T)-1;   // the last poll's oracle value
    auto poll_hook = [&](const T* dz_cur, int) -> bool {
        auto tp = clock::now();
        std::copy(z, z + n, x);
        blas::axpy(n, (T)1.0, dz_cur, 1, x, 1);
        solve_R(blas::Op::NoTrans, x, &t_trsv_in);
        residual_into(x, rp, &t_fwd_in);
        apply_adj(rp, ATrp, &t_adj_in);
        long tb0 = t_be;
        T v = eval_oracle_at(x, rp, ATrp);
        poll_be = v;
        t_be_poll += t_be - tb0;
        ++polls_round;
        t_poll += duration_cast<microseconds>(clock::now() - tp).count();
        return v <= ctl.be_tol;
    };

    Term status = Term::iteration_budget;
    bool be_done = false;
    T relres;
    // r_ne is brought up to date at the top of the next round (ne_current), so a run that ends
    // skips its last adjoint apply; have_ATr: r_ne already holds A^T (b - A x) for the oracle.
    bool ne_current = true, have_ATr = false;
    if (x0 == nullptr) {
        // z = 0: x = 0 and the normal-equation residual is g, with no extra apply.
        std::copy(g, g + n, r_ne);
        std::fill(x, x + n, (T)0);
        relres = (bnorm == (T)0) ? (T)0 : (T)1;   // a NaN b keeps the run going to a breakdown
    } else {
        // z = R x0, so x = R^{-1} z is x0; the residual comes from b - A x0 in stable form.
        std::copy(x0, x0 + n, z);
        if (prec) {
            auto ts = clock::now();
            { Blas2ThreadGuard tg(n);
                blas::trmv(blas::Layout::ColMajor, blas::Uplo::Upper,
                           blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, z, 1);
            }
            t_trsv += duration_cast<microseconds>(clock::now() - ts).count();
        }
        relres = recover_x_and_relres();
        if (be_active) {
            apply_adj(wm, r_ne, nullptr);
            have_ATr = true;
            T be0 = eval_oracle_at(x, wm, r_ne);
            report.be_x0 = be0;
            if (relres > ctl.tol && be0 <= ctl.be_tol) { be_done = true; status = Term::oracle_met; }
        }
        ne_current = false;
    }

    const T g0_norm = blas::nrm2(n, g, 1);
    int iters_done = 0, rounds = 0;

    // Outer stagnation: when tol is below the achievable floor, rounds stop improving the
    // true residual. The reference value advances only on a significant drop, so steady slow
    // descent is not mistaken for a floor.
    T   ls_stag_ref    = relres;
    int ls_flat_rounds = 0;

    // !(relres <= tol), so a NaN residual never meets tol.
    while (!be_done && !(relres <= ctl.tol) && iters_done < ctl.max_iters) {
        if (ctl.max_restarts >= 0 && rounds > ctl.max_restarts) { status = Term::round_budget; break; }
        if (!ne_current) {   // R^{-T} A^T (b - A x) of the current x, wm holding b - A x
            if (!have_ATr) apply_adj(wm, r_ne, nullptr);
            solve_R(blas::Op::Trans, r_ne, nullptr);
            ne_current = true;
        }
        T ne_norm = blas::nrm2(n, r_ne, 1);
        if (ne_norm == (T)0) { status = Term::ls_floor; break; }   // x is the minimizer; tol is unmet

        ++rounds;
        // Inner CG on H dz = r_ne from dz = 0. Stagnated and capped rounds continue to the
        // next true-residual restart; a breakdown ends the run.
        PCGInnerControls<T> ictl;
        ictl.tol = ctl.round_drop;
        if (ctl.inner_abs_tol > (T)0) {   // the kernel tolerance is relative to this round's ||r_ne||
            T floor_rel = ctl.inner_abs_tol * g0_norm / ne_norm;
            if (floor_rel > ictl.tol) ictl.tol = floor_rel;
        }
        ictl.max_iters        = std::min(ctl.max_iters_per_round, ctl.max_iters - iters_done);
        ictl.stag_window      = ctl.inner_stag_window;
        ictl.stag_rel_improve = ctl.stag_rel_improve;
        ictl.poll_every       = poll_active ? ctl.be_poll_every : 0;
        if (poll_active) ictl.poll = poll_hook;
        polls_round = 0;
        PCGInnerReport<T> rep;
        auto tk0 = clock::now();
        PCGInnerTermination kterm = pcg_inner<T>(apply_H, r_ne, n, dz, r, p, q, zb, ictl, rep);
        t_kernel += duration_cast<microseconds>(clock::now() - tk0).count();

        blas::axpy(n, (T)1.0, dz, 1, z, 1);
        iters_done += rep.iters;

        // Recompute the true residual exactly, and A^T r now only for the oracle, which wants
        // it raw. A poll that ended the round already evaluated this x with the same arithmetic.
        T be_round = (T)-1;
        if (kterm == PCGInnerTermination::polled) {
            std::swap(wm, rp);
            std::copy(ATrp, ATrp + n, r_ne);
            relres = blas::nrm2(m, wm, 1) / bden;
            be_round = poll_be;
            have_ATr = true;
        } else {
            relres = recover_x_and_relres();
            have_ATr = be_active;
            if (be_active) {
                apply_adj(wm, r_ne, nullptr);
                be_round = eval_oracle_at(x, wm, r_ne);
            }
        }
        ne_current = false;

        report.cg_iters.push_back(rep.iters);
        report.cg_status.push_back(rep.status);
        report.cg_relres.push_back(rep.relres);
        report.cg_best_relres.push_back(rep.best_relres);
        report.cg_best_iter.push_back(rep.best_iter);
        report.ls_relres.push_back(relres);
        report.be.push_back(be_round);
        report.polls.push_back(polls_round);

        if (relres <= ctl.tol) { status = Term::converged; break; }
        if (kterm == PCGInnerTermination::breakdown) { status = Term::breakdown; break; }
        if (be_active && (be_round <= ctl.be_tol || rep.status == PCGInnerTermination::polled)) {
            status = Term::oracle_met; break;
        }
        if (ctl.outer_stag_window > 0) {
            if (relres >= ls_stag_ref * ((T)1 - ctl.stag_rel_improve)) {
                if (++ls_flat_rounds >= ctl.outer_stag_window) { status = Term::ls_floor; break; }
            } else {
                ls_flat_rounds = 0;
                ls_stag_ref = relres;
            }
        }
    }
    // Covers runs that never enter the loop (b = 0, or a converged x0).
    if (status != Term::oracle_met && relres <= ctl.tol) status = Term::converged;

    report.iters_done      = iters_done;
    report.rounds_done     = rounds;
    report.final_relres    = relres;
    const long t_total     = static_cast<long>(duration_cast<microseconds>(clock::now() - total_start).count()) - t_be;
    report.times           = {t_fwd, t_adj, t_trsv, t_total};
    report.t_inner_us      = t_kernel - t_be_poll;
    report.t_fwd_inner_us  = t_fwd_in;
    report.t_adj_inner_us  = t_adj_in;
    report.t_trsv_inner_us = t_trsv_in;
    report.t_be_us         = t_be;
    report.t_poll_us       = t_poll;
    return status;
}


} // namespace RandLAPACK
