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

/**
 * 
 * TODO:
 *  (1) finish and test krill_restricted_rpchol
 *  (2) write and test a krill_restricted function that accepts the centers as inputs
 *      in advance.
 *  (3) See also, rl_preconditioners.hh
 * 
 */

namespace RandLAPACK {

/**
 * Fun thing about the name KRILLx:
 * 
 *      we can do KRILLrs for KRILL with lockstep PCG for regularization sweep.
 * 
 *      we can do KRILLb (?) for "random lifting + block" version.
 */


template <typename T, typename FUNC, typename SEMINORM, typename STATE>
STATE krill_full_rpchol(
    int64_t n, FUNC &G, std::vector<T> &H, std::vector<T> &X, T tol,
    STATE state, SEMINORM &seminorm, int64_t rpchol_block_size = -1, int64_t max_iters = 20, int64_t k = -1
) {
    using std::vector;
    int64_t mu_size = G.num_ops;
    vector<T> mus(mu_size);
    std::copy(G.regs, G.regs + mu_size, mus.data());
    int64_t ell = ((int64_t) H.size()) / n;
    randblas_require(ell * n == (int64_t) H.size());
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
    pcg(G, H.data(), ell, seminorm, tol, max_iters, invP, X.data(), true);

    return state;
}

/**
 * We start with a regularized kernel linear operator G and target data H.
 * We use "K" to denote the unregularized version of G, which can be accessed
 * by calling G.set_eval_includes_reg(false);
 * 
 * If G.regs.size() == 1, then the nominal KRR problem reduces to computing
 * 
 *     (K + G.regs[0] * I) X = H.       (*)
 * 
 * If G.regs.size() > 1, then KRR is nominally about solving the independent
 * collection of problems
 * 
 *      (K + mu_i * I) x_i = h_i,       (**)
 * 
 * where K is the unregularized version of G, mu_i = G.regs[i], and x_i, h_i
 * are the i-th columns of X and H respectively. In this situation we need
 * H to have exactly G.regs.size() columns.
 *      
 * This function produces __approximate__ solutions to KRR problems. It does so
 * by finding a set of indices for which
 * 
 *      K_hat = K(:,inds) * inv(K(inds, inds)) * K(inds, :) 
 * 
 * is a good low-rank approximation of K. We spend O(n*k^2) arithmetic operations and
 * O(n*k_ evaluations of K(i,j) to get our hands on "inds" and a factored representation
 * of K_hat.
 * 
 * Given inds, we turn our attention to solving the problem
 * 
 *      min{ || K(:,inds) x - H ||_2^2 + mu || sqrtm(K(inds, inds)) x ||_2^2 : x  }.
 *      
 * We don't store K(:,inds) explicitly. Instead, we have access to a matrix V where
 * 
 *      (i)   K_hat = VV',
 *      (ii)  V(inds,:)V(inds,:)' = K(inds, inds), and
 *      (iii) V*V(inds,:)' = K_hat(:,inds) = K(:, inds).
 * 
 * If we abbreviate M := V(inds, :), then the restricted KRR problem can be framed as 
 * 
 *      min{ || V M' x - H ||_2^2 + mu || M' X ||_2^2  :  x  }.
 * 
 * We approach this by a change of basis, solving problems like
 * 
 *      min{ ||V y - H||_2^2 + mus || y ||_2^2 : y }        (***)
 * 
 *  and then returning x = inv(M') y.
 * 
 * Note that since we spend O(n*k^2) time getting our hands on V and inds, it would be
 * reasonable to spend O(n*k^2) additional time to solve (***) by a direct method.
 * However, it is easy enough to reduce the cost of solving (***) to o(n*k^2)
 * (that is, little-o of n*k^2) by a sketch and precondition approach. 
 *
 */
// template <typename T, typename FUNC, typename SEMINORM, typename STATE>
// STATE krill_restricted_rpchol(
//     int64_t n, FUNC &G, std::vector<T> &H, std::vector<T> &X, T tol,
//     STATE state, SEMINORM seminorm, int64_t rpchol_block_size = -1, int64_t max_iters = 20, int64_t k = -1
// ) {
//     // NOTE: on entry, X is n-by-s for some integer s. That's way bigger than it needs to be, since the
//     // solution we return can be written down with k*s nonzeros plus k indices to indicate which rows of X
//     // are nonzero.
//     vector<T> V(n*k, 0.0);
//     vector<T> eigvals(k, 0.0);
//     G.set_eval_includes_reg(false);

//     vector<int64_t> inds(k, -1);
//     state = rp_cholesky(n, G, k, inds.data(), V.data(), rpchol_block_size, state);
//     inds.resize(k);
//     // ^ VV' defines a rank-k Nystrom approximation of G. The approximation satisfies
//     //
//     //          VV' = G(:,inds) * inv(G(inds, inds)) * G(inds, :) 
//     //   and
//     //          (VV')(inds, inds) = G(inds, inds).
//     //
//     //   That second identity can be written as MM' = G(inds, inds) for M = V(inds, :).
//     //


//     vector<T> M(k * k);
//     _rpchol_impl::pack_selected_rows(blas::Layout::ColMajor, n, k, V.data(), inds, M.data());
//     //
//     //
//     //

//     linops::SpectralPrecond<T> invP(n);
//     // invP.prep(V, eigvals, mus, ell);
//     return state;
// }

// template <typename T, typename FUNC, typename STATE>
// STATE krill_block(
//
// ) {
//
// }


} // end namespace RandLAPACK