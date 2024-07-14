#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_preconditioners.hh"
#include "rl_rpchol.hh"
#include "rl_pdkernels.hh"
#include "rl_determiter.hh"

#include <RandBLAS.hh>
#include <limits>
#include <vector>

namespace RandLAPACK {

/**
 * Fun thing about the name KRILLx:
 * 
 *      we can do KRILLrs for KRILL with lockstep PCG for regularization sweep.
 * 
 *      we can do KRILLb (?) for "random lifting + block" version.
 */

using std::vector;

template <typename T, typename FUNC, typename SEMINORM, typename STATE>
STATE krill_full_rpchol(
    int64_t n, FUNC &G, vector<T> &H, vector<T> &X, T tol,
    STATE state, SEMINORM seminorm, int64_t rpchol_block_size = -1, int64_t max_iters = 20, int64_t k = -1
) {
    auto mus = G.regs;
    int64_t ell = ((int64_t) H.size()) / n;
    randblas_require(ell * n == (int64_t) H.size());
    int64_t mu_size = mus.size();
    randblas_require(mu_size == 1 || mu_size == ell);

    if (rpchol_block_size < 0)
        rpchol_block_size = std::min((int64_t) 64, n/4);
    if (k < 0)
        k = (int64_t) std::sqrt(n);
    
    vector<T> V(n*k, 0.0);
    vector<T> eigvals(k, 0.0);
    G.set_eval_includes_reg(false);
    state = rpchol_pc_data(n, G, k, rpchol_block_size, V.data(), eigvals.data(), state);
    linops::SpectralPrecond<T> invP(n);
    invP.prep(V, eigvals, mus, ell);
    G.set_eval_includes_reg(true);
    lockorblock_pcg(G, H, tol, max_iters, invP, seminorm, X, true);

    return state;
}


// template <typename T, typename FUNC, typename STATE>
// STATE krill_block(
//
// ) {
//
// }


} // end namespace RandLAPACK
