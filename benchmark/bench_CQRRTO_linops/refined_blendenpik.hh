#pragma once

// The Blendenpik_refine and Blendenpik_cold_refine rows: Blendenpik's preconditioner, refined by
// the solver the benchmark's QR rows use.
//
//   phase 1: Blendenpik_linops in init_only mode (sketch, QR and the sketch-and-solve x0;
//            no LSQR);
//   phase 2: the caller's solver with R as the right preconditioner, from x0
//            (Blendenpik_refine) or from zero (Blendenpik_cold_refine).
//
// qr_us is sketch + QR and setup_us the x0 build (warm row only); the cold row's unused x0
// build, which keeps its phase 1 identical to the warm row's, is in neither slot.

#include "rl_blendenpik.hh"

#include <cstdint>
#include <memory>
#include <vector>


namespace RandLAPACK {
namespace bench {


template <typename T>
struct RefinedBlendenpikResult {
    int  qr_status = 1;          ///< Blendenpik_linops::call's return (0 = usable R)
    long qr_us     = -1;         ///< sketch + QR; -1 when phase 1 failed
    long setup_us  = 0;          ///< x0 build (warm row; 0 for the cold row)
    T    x0_relres = (T)-1;      ///< ||b - A x0|| / ||b|| (warm row; -1 for the cold row)
    std::unique_ptr<T[]> R;      ///< the n x n sketch factor (ColMajor, ld n)
};


/// Runs one refined row. phase 1 sketches with `state` as given, so it does not depend on the
/// rows that ran before; then solve(R, x0) runs the row's solver with the n x n factor R
/// (ColMajor, ld n) from x0, which is nullptr for the cold row. On a failed phase 1
/// (qr_status != 0) solve is not called.
template <typename T, typename RNG, RandLAPACK::linops::LinearOperator GLO, typename Solve>
RefinedBlendenpikResult<T> run_refined_blendenpik(
    GLO& A, const T* b, int64_t m, int64_t n,
    T d_factor, int64_t sketch_nnz, RandBLAS::RNGState<RNG> state, bool warm, Solve&& solve)
{
    RefinedBlendenpikResult<T> res;

    Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, T(0));
    bp.nnz       = sketch_nnz;
    bp.init_only = true;
    std::vector<T> x0(n);
    res.qr_status = bp.call(A, b, m, x0.data(), n, d_factor, state);
    if (res.qr_status != 0) return res;

    res.qr_us     = bp.times[0] + bp.times[1];
    res.setup_us  = warm ? bp.times[4] : 0;
    res.x0_relres = warm ? bp.final_relres : (T)-1;
    res.R.reset(bp.R_out);
    bp.R_out = nullptr;
    bp.R_out_sz = 0;

    solve((const T*)res.R.get(), warm ? (const T*)x0.data() : (const T*)nullptr);
    return res;
}


} // namespace bench
} // namespace RandLAPACK
