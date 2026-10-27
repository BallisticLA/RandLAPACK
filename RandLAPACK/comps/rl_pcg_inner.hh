#pragma once

// Public API: pcg_inner, plain conjugate gradients on the caller's SPD operator, M z = c, for
// one right-hand side given only a matvec; any preconditioning lives in apply_M. It stops on a
// relative-residual target, a stagnation window, an iteration cap or a caller poll, returns the
// best iterate seen on the stagnation, cap and breakdown exits, and reports what happened.
// restarted_pcg_ne runs it once per round on the right-preconditioned normal equations.

#include "rl_blaspp.hh"
#include "rl_exceptions.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>


namespace RandLAPACK {


/// How one pcg_inner solve ended.
enum class PCGInnerTermination {
    converged,   ///< reached the relative-residual target
    hit_cap,     ///< exhausted max_iters; z is the best iterate
    breakdown,   ///< p^T M p <= 0 or a non-finite step (non-SPD M, lost orthogonality); z is the best iterate
    stagnated,   ///< no significant progress for stag_window iterations; z is the best iterate
    polled       ///< the caller's poll ended the solve; z is the current iterate
};

inline const char* to_string(PCGInnerTermination t) {
    switch (t) {
        case PCGInnerTermination::converged: return "converged";
        case PCGInnerTermination::hit_cap:   return "hit_cap";
        case PCGInnerTermination::breakdown: return "breakdown";
        case PCGInnerTermination::stagnated: return "stagnated";
        case PCGInnerTermination::polled:    return "polled";
    }
    return "unknown";
}

/// What one pcg_inner solve did. Residuals are the recursive CG residual ||r|| / ||c||.
template <typename T>
struct PCGInnerReport {
    int                 iters       = 0;      ///< completed iterations
    PCGInnerTermination status      = PCGInnerTermination::converged;
    T                   relres      = (T)0;   ///< of the returned iterate
    T                   best_relres = (T)0;   ///< smallest seen
    int                 best_iter   = 0;      ///< iteration reaching best_relres
};

/// Controls of one pcg_inner solve.
template <typename T>
struct PCGInnerControls {
    T   tol = (T)0;                     ///< stop when ||r|| <= tol * ||c||
    int max_iters = 0;                  ///< iteration cap; must be set (>= 1)
    int stag_window      = 20;          ///< iterations without progress that end the solve; <= 0 disables
    T   stag_rel_improve = (T)1e-3;     ///< relative drop that counts as progress
    /// Optional poll: every poll_every completed iterations, after the convergence test and
    /// before the stagnation test, poll(z, it) sees the current iterate; true ends the solve
    /// with status polled and z unchanged. 0 or an empty hook disables it.
    int poll_every = 0;
    std::function<bool(const T* z, int it)> poll;
};

/// @brief Plain CG on the SPD system M z = c, starting from z = 0.
///
/// @param apply_M    callable void(const T* v, T* out): out = M v, with any preconditioning
///                   already folded into M; v and out do not alias.
/// @param c          right-hand side (length n).
/// @param n          system dimension.
/// @param z          solution (length n).
/// @param cg_r,cg_p,cg_Mp,cg_zbest  caller-allocated length-n workspaces.
/// @param ctl        tolerances and caps (see PCGInnerControls).
/// @param rep        the solve's report (see PCGInnerReport).
/// @returns how the solve ended (also in rep.status).
template <typename T, typename FApplyM>
PCGInnerTermination pcg_inner(FApplyM&& apply_M, const T* c, int64_t n,
                              T* z, T* cg_r, T* cg_p, T* cg_Mp, T* cg_zbest,
                              const PCGInnerControls<T>& ctl, PCGInnerReport<T>& rep)
{
    randlapack_require(ctl.max_iters >= 1)
        << "pcg_inner: max_iters must be >= 1 (got " << ctl.max_iters << ")";
    std::fill(z, z + n, (T)0);
    rep.iters = 0;
    rep.best_iter = 0;
    const T c_norm = blas::nrm2(n, c, 1);
    if (c_norm == (T)0) {   // M is SPD, so M z = 0 has the unique solution 0
        rep.status = PCGInnerTermination::converged;
        rep.relres = (T)0; rep.best_relres = (T)0;
        return rep.status;
    }

    // CG runs on 2^-e c, whose norm lies in [1, 2), so its squared norms neither underflow nor
    // overflow. Power-of-two scaling is exact: normal-range iterates are unchanged. z is scaled
    // back before every return and every poll.
    const int e = std::isfinite(c_norm) ? std::ilogb(c_norm) : 0;
    auto scale_z = [&](int k) {
        if (k != 0)
            for (int64_t i = 0; i < n; ++i) z[i] = std::ldexp(z[i], k);
    };
    const T cs_norm = std::ldexp(c_norm, -e);
    const T tol_abs = ctl.tol * cs_norm;

    std::fill(cg_zbest, cg_zbest + n, (T)0);
    for (int64_t i = 0; i < n; ++i) cg_r[i] = std::ldexp(c[i], -e);   // r = c - M 0
    std::copy(cg_r, cg_r + n, cg_p);
    rep.best_relres = (T)1;

    auto finish_with_best = [&](int it, PCGInnerTermination status) {
        std::copy(cg_zbest, cg_zbest + n, z);
        scale_z(e);
        rep.iters  = it;
        rep.status = status;
        rep.relres = rep.best_relres;
        return status;
    };

    // Stagnation state: the residual at the last drop of at least stag_rel_improve, and its
    // iteration. A merely noisy decrease does not count as progress.
    T   stag_ref        = std::numeric_limits<T>::max();
    int last_improve_it = 0;

    T rs_old = blas::dot(n, cg_r, 1, cg_r, 1);

    for (int it = 0; it < ctl.max_iters; ++it) {
        apply_M(cg_p, cg_Mp);

        // Breakdown (non-SPD M, lost orthogonality, or a non-finite step). rep.iters counts
        // completed iterations, so this one is not included.
        T pMp = blas::dot(n, cg_p, 1, cg_Mp, 1);
        if (!(pMp > 0) || !std::isfinite(pMp))
            return finish_with_best(it, PCGInnerTermination::breakdown);
        T alpha = rs_old / pMp;
        if (!std::isfinite(alpha))
            return finish_with_best(it, PCGInnerTermination::breakdown);

        blas::axpy(n,  alpha, cg_p, 1, z, 1);      // z += alpha p
        blas::axpy(n, -alpha, cg_Mp, 1, cg_r, 1);  // r -= alpha M p

        T rs_new = blas::dot(n, cg_r, 1, cg_r, 1);
        if (!std::isfinite(rs_new))
            return finish_with_best(it, PCGInnerTermination::breakdown);
        T r_norm = std::sqrt(rs_new);
        T relres = r_norm / cs_norm;
        if (relres < rep.best_relres) {
            rep.best_relres = relres;
            rep.best_iter   = it + 1;
            std::copy(z, z + n, cg_zbest);
        }
        if (relres < stag_ref * ((T)1 - ctl.stag_rel_improve)) {
            stag_ref        = relres;
            last_improve_it = it + 1;
        }

        // Convergence is checked before stagnation, so a solve reaching the target reports
        // converged even if its last steps were flat.
        if (r_norm <= tol_abs) {
            scale_z(e);
            rep.iters  = it + 1;
            rep.status = PCGInnerTermination::converged;
            rep.relres = relres;
            return rep.status;
        }
        if (ctl.poll_every > 0 && ctl.poll && ((it + 1) % ctl.poll_every) == 0) {
            scale_z(e);
            if (ctl.poll(z, it + 1)) {
                rep.iters  = it + 1;
                rep.status = PCGInnerTermination::polled;
                rep.relres = relres;
                return rep.status;
            }
            scale_z(-e);
        }
        if (ctl.stag_window > 0 && (it + 1) - last_improve_it >= ctl.stag_window)
            return finish_with_best(it + 1, PCGInnerTermination::stagnated);

        T beta = rs_new / rs_old;
        for (int64_t i = 0; i < n; ++i) cg_p[i] = cg_r[i] + beta * cg_p[i];
        rs_old = rs_new;
    }
    return finish_with_best(ctl.max_iters, PCGInnerTermination::hit_cap);
}


} // namespace RandLAPACK
