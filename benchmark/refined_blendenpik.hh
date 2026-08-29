#pragma once

// Shared dispatch for the Blendenpik_refine / Blendenpik_cold_refine benchmark
// rows, used by BOTH least-squares benchmarks (bench_toeplitz_ls and
// bench_CQRRT_linops). Extracted after the three previous per-benchmark
// implementations of these rows drifted into three DIFFERENT wrong
// accountings (phase-1 solve time missing from all columns in the Toeplitz
// benchmark; refinement time missing and the warm row silently cold in the
// FEM2 benchmark), which is the standing failure mode of duplicated dispatch
// blocks in this suite.
//
// ROW SEMANTICS. A refined row is "Blendenpik's
// preconditioner and its sketch-and-solve initial guess, refined by the shared
// restarted PCG-NE engine":
//
//   phase 1: Blendenpik in init_only mode: sketch, QR, and the sketch-and-
//            solve x0. NO internal LSQR runs, so every iterative step of the
//            row happens in the SAME engine every Q-less method uses and the
//            row's iteration count has one unit (engine inner CG iterations).
//   phase 2: restarted_pcg_ne from x0 (warm row) or from zero (cold row),
//            with the full per-round history recorded.
//
// Accounting contract (every microsecond of the row lands in exactly one slot):
//   qr_us    = sketch + QR                     (bp.times[0] + bp.times[1])
//   setup_us = x0 build: ormqr + trsv + one operator apply (bp.times[4]);
//              reported for the warm row only. The cold row still executes the
//              x0 build so its Blendenpik phase is bit-identical to the warm
//              row's (same state, same sketch, same R); that executed-but-
//              unused work is deliberately excluded from the cold row's
//              accounting and callers' whole-row timers should exclude it too
//              (use qr_us + setup_us + solve_us, not a wall clock around this
//              call).
//   solve_us = engine total (times[3]), with the fwd/adj/trsm splits and the
//              inner-kernel/overhead split available from the history.

#include "rl_blendenpik.hh"
#include "rl_restarted_pcg_ne.hh"

#include <cstdint>
#include <vector>


namespace RandLAPACK {
namespace bench {


template <typename T>
struct RefinedBlendenpikResult {
    // ---- phase 1: Blendenpik build ----
    int  qr_status = 1;        ///< 0 = usable R; nonzero = rank-deficient sketch
    long qr_us     = 0;        ///< sketch + QR
    long setup_us  = 0;        ///< x0 build (warm row; 0 for the cold row)
    T    x0_relres = (T)-1;    ///< true ||b - A x0||/||b|| of the handed-off x0
                               ///< (warm row; -1 for the cold row)
    // ---- phase 2: shared engine ----
    int  status        = 1;    ///< restarted_pcg_ne status (0/1/2/3/4)
    int  iters         = 0;    ///< engine inner CG iterations ONLY
    int  rounds        = 0;    ///< outer rounds run
    T    solver_relres = (T)-1;///< true LS relres at engine exit
    long solve_us      = 0;    ///< engine wall time (times[3])
    long t_fwd_us  = 0;        ///< all-inclusive engine op splits (kernel + outer)
    long t_adj_us  = 0;
    long t_trsm_us = 0;
    PCGRoundHistory<T> history;///< per-round records + inner-kernel time split
    /// The sketch R factor (n x n, ColMajor), for orth_error. Owned by this
    /// result (transferred from Blendenpik_linops::R_out, not copied); moved
    /// out on copy/return like the rest of this struct's implicit members.
    T* R = nullptr;
    int64_t R_sz = 0;

    RefinedBlendenpikResult() = default;
    RefinedBlendenpikResult(const RefinedBlendenpikResult&) = delete;
    RefinedBlendenpikResult& operator=(const RefinedBlendenpikResult&) = delete;
    RefinedBlendenpikResult(RefinedBlendenpikResult&& other) noexcept
        : qr_status(other.qr_status), qr_us(other.qr_us), setup_us(other.setup_us),
          x0_relres(other.x0_relres), status(other.status), iters(other.iters),
          rounds(other.rounds), solver_relres(other.solver_relres), solve_us(other.solve_us),
          t_fwd_us(other.t_fwd_us), t_adj_us(other.t_adj_us), t_trsm_us(other.t_trsm_us),
          history(std::move(other.history)), R(other.R), R_sz(other.R_sz) {
        other.R = nullptr; other.R_sz = 0;
    }
    RefinedBlendenpikResult& operator=(RefinedBlendenpikResult&& other) noexcept {
        if (this != &other) {
            delete[] R;
            qr_status = other.qr_status; qr_us = other.qr_us; setup_us = other.setup_us;
            x0_relres = other.x0_relres; status = other.status; iters = other.iters;
            rounds = other.rounds; solver_relres = other.solver_relres; solve_us = other.solve_us;
            t_fwd_us = other.t_fwd_us; t_adj_us = other.t_adj_us; t_trsm_us = other.t_trsm_us;
            history = std::move(other.history);
            R = other.R; R_sz = other.R_sz;
            other.R = nullptr; other.R_sz = 0;
        }
        return *this;
    }
    ~RefinedBlendenpikResult() { delete[] R; }
};


/// Run one refined-Blendenpik row. `warm` selects Blendenpik_refine (engine
/// starts from the sketch-and-solve x0) vs Blendenpik_cold_refine (engine
/// starts from zero; same R). `state` is taken by value: callers rebuild their
/// per-(row, run) state anyway, and the sketch must not depend on which rows
/// preceded this one.
template <typename T, typename RNG, RandLAPACK::linops::LinearOperator GLO>
RefinedBlendenpikResult<T> run_refined_blendenpik(
    GLO& A, const T* b, int64_t m, T* x, int64_t n,
    T d_factor, int64_t sketch_nnz, RandBLAS::RNGState<RNG> state,
    bool warm,
    T tol, int max_iters,
    int restart_maxit, T restart_drop, int max_restarts,
    int stag_window = 20, T stag_rel_improve = (T)1e-3,
    T inner_abs_tol = (T)0, int outer_stag_window = 2)
{
    RefinedBlendenpikResult<T> res;

    // Phase 1: sketch + QR + x0, no LSQR.
    Blendenpik_linops<T, RNG> bp(/*timing=*/true, tol);
    bp.nnz       = sketch_nnz;
    bp.max_iters = max_iters;   // unused in init_only mode; set for safety
    bp.init_only = true;
    std::vector<T> x0(n, (T)0);
    res.qr_status = bp.call(A, b, m, x0.data(), n, d_factor, state);
    if (res.qr_status != 0) return res;

    res.qr_us     = bp.times[0] + bp.times[1];
    res.setup_us  = warm ? bp.times[4] : 0;
    res.x0_relres = warm ? bp.final_relres : (T)-1;
    // Transfer ownership of the R buffer from bp (about to go out of scope)
    // rather than copying it; bp's destructor then sees a null R_out.
    res.R    = bp.R_out;
    res.R_sz = bp.R_out_sz;
    bp.R_out = nullptr;
    bp.R_out_sz = 0;

    // Phase 2: the shared engine does ALL iterative work of the row.
    long lt[4] = {0, 0, 0, 0};
    res.status = restarted_pcg_ne<T>(A, m, n, res.R, n, b, x,
        tol, max_iters, res.iters,
        restart_maxit, restart_drop, max_restarts,
        &res.rounds, lt, &res.solver_relres,
        stag_window, stag_rel_improve, inner_abs_tol,
        &res.history,
        warm ? x0.data() : nullptr,
        outer_stag_window);
    res.solve_us  = lt[3];
    res.t_fwd_us  = lt[0];
    res.t_adj_us  = lt[1];
    res.t_trsm_us = lt[2];
    return res;
}


} // namespace bench
} // namespace RandLAPACK
