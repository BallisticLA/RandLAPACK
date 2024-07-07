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

// TODO: make an interface for objects that be called as G(i,j)
// or G.block_eval( ... arguments for computing submatrices of kernel matrices ... )
//
// ^ Maybe that'll just be an extension of the LinearOperator interface?
//
//   Uhh ... probably not, since that interface already has an operator() definition.
//   maybe overloading saves us?

template <typename T, typename FUNC, typename SEMINORM, typename STATE>
STATE krill_regularization_sweep(
    int64_t n, const FUNC &G, vector<T> &mus, vector<T> &h, vector<T> &X, T tol,
    STATE state, int64_t rpchol_block_size = -1, int64_t max_iters = 20
) {
    if (rpchol_block_size < 0)
        rpchol_block_size = std::min(64, n/3);
    int64_t ell = mus.size();
    vector<T> H{};
    H.reserve(n * ell);
    for (int64_t i = 0; i < ell, ++i)
        H.insert(H.end(), h.begin(), h.end());
    // H can now be interpreted as a column-major matrix of size n-by-ell,
    // where each column is a copy of h.
    int64_t k = (int64_t) std::sqrt(n);
    vector<T> V(n*k, 0.0);
    vector<T> eigvals(k, 0.0);
    state = rpchol_pc_data(n, G, k, rpchol_block_size, V.data(), eigvals.data(), state);
    // Define the preconditioner as an abstract function handle
    OOPreconditioners::SpectralPrecond<T> invP(n);
    invP.prep(V, eigvals, mus, ell);
    //
    SEMINORM seminorm{};
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
