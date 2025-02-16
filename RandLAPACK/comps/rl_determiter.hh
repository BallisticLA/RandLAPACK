#pragma once

#include "rl_blaspp.hh"
#include "rl_linops.hh"

#include <iostream>
#include <vector>
#include <cstdint>

namespace RandLAPACK {

/*  Solve the saddle point problem
    (A'A + mu*I)x = A'b - c

    Have access to a matrix M such that
    (A'A + mu*I)MM' is well-conditioned.
*/
template <typename T>
void pcg_saddle(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    const T* b, // length m
    const T* c, // length n
    T delta, // >= 0
    int64_t iter_lim,
    T* resid_vec, // length iter_lim
    T tol, //  > 0
    int64_t k,
    const T* M, // n-by-k
    int64_t ldm,
    const T* x0, // length n
    T* x,  // length n
    T* y // length m
) {

    T* allwork = new T[m + 5*n + k]{};

    T* out_a1  = allwork;
    T* out_at1 = out_a1  + m;
    T* r       = out_at1 + n;
    T* d       = r       + n;
    T* b1      = d       + n;
    T* out_m1  = b1      + n;
    T* out_mt1 = out_m1  + n;

    //  b1 = A'b - c
    blas::copy(n, c, 1, b1, 1);
    blas::gemv(Layout::ColMajor, Op::Trans, m, n, 1.0, A, lda, b, 1, -1.0, b1, 1);

    // r = b1 - (A'(A x0) + delta x0)
    //		out_a1 = A x0
    //		out_at1 = A'out_a1
    //		out_at1 += delta x0
    //		r -= out_at1
    blas::copy(n, b1, 1, r, 1);
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, 1.0, A, lda, x0, 1, delta, out_a1, 1);
    blas::gemv(Layout::ColMajor, Op::Trans, m, n, 1.0, A, lda, out_a1, 1, 0.0, out_at1, 1);
    blas::axpy(n, delta, x0, 1, out_at1, 1);
    blas::axpy(n, -1.0, out_at1, 1, r, 1);

    // d = M (M' r);
    blas::gemv(Layout::ColMajor, Op::Trans, n, k, 1.0, M, ldm, r, 1, 0.0, out_mt1, 1);
    blas::gemv(Layout::ColMajor, Op::NoTrans, n, k, 1.0, M, ldm, out_mt1, 1, 0.0, d, 1);

    bool reg = delta > 0;
    blas::copy(n, x0, 1, x, 1);
    T delta1_old = blas::dot(n, d, 1, r, 1);
    T delta1_new = delta1_old;
    T rel_sq_tol = (delta1_old * tol) * tol;

    int64_t iter = 0;
    T alpha = 0.0;
    T beta = 0.0;
    while (iter < iter_lim && delta1_new > rel_sq_tol) {
        resid_vec[iter] = delta1_new;

        // q = A'(A d) + delta d
        //		q = out_at1
        blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, 1.0,  A, lda, d, 1, 0.0, out_a1, 1);
        blas::gemv(Layout::ColMajor, Op::Trans, m, n, 1.0, A, lda, out_a1, 1, 0.0, out_at1, 1);
        if (reg) blas::axpy(n, delta,  d, 1, out_at1, 1);

        // alpha = delta1_new / (d' q)
        alpha = delta1_new / blas::dot(n, d, 1, out_at1, 1);

        // x += alpha d
        blas::axpy(n, alpha, d, 1, x, 1);

        // update r
        if (iter % 25 == 1) {
            // r = b1 - (A'(A x) + delta x)
            //		out_a1 = A x
            //		out_at1 = A' out_a1
            //		r = b1
            //		r -= out_at1
            //		r -= delta x
            blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, 1.0,  A, lda, x, 1, 0.0, out_a1, 1);
            blas::gemv(Layout::ColMajor, Op::Trans, m, n, 1.0, A, lda, out_a1, 1, 0.0, out_at1, 1);
            blas::copy(n, b1, 1, r, 1);
            blas::axpy(n, -1.0, out_at1, 1, r, 1 );
            if (reg) blas::axpy(n, -delta, x, 1, r, 1);
        } else {
            // r -= alpha q
            blas::axpy(n, -alpha, out_at1, 1, r, 1);
        }

        // s = M (M' r)
        //		out_mt1 = M' r
        //		out_m1 = M out_mt1
        //		s = out_m1
        blas::gemv(Layout::ColMajor, Op::Trans, n, k, 1.0, M, ldm, r, 1, 0.0, out_mt1, 1);
        blas::gemv(Layout::ColMajor, Op::NoTrans, n, k, 1.0, M, ldm, out_mt1, 1, 0.0, out_m1, 1);

        // scalars and update d
        delta1_old = delta1_new;
        delta1_new = blas::dot(n, r, 1, out_m1, 1);
        beta = delta1_new / delta1_old;
        for (int i = 0; i < n; ++i) {
            d[i] = beta*d[i] + out_m1[i];
        }

        ++iter;
    }

    resid_vec[iter] = delta1_new;

    // recover y = b - Ax
    blas::copy(m, b, 1, y, 1);
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, -1.0, A, lda, x, 1, 1.0, y, 1);
    delete [] allwork;
    return;
}


// MARK: [L/B]PCG helpers

template <typename T>
struct StatefulFrobeniusNorm {
    std::vector<T> history;
    // default constructor
    StatefulFrobeniusNorm() : history() {};
    // move constructor
    StatefulFrobeniusNorm(StatefulFrobeniusNorm<T> &&other) noexcept : history(std::move(other.history)) {};
    inline T operator()(int64_t n, int64_t s, const T* NR) { 
        T nrm = blas::nrm2(n * s, NR, 1);
        this->history.push_back(nrm);
        return nrm;
    };
};

template <typename T>
void zero_off_diagonal(T* mat, int64_t s) {
    for (int64_t i = 0; i < s - 1; ++i) {
        T* ptr_to_next_diag = mat + i + i*s;
        //blas::scal(s, 0.0, ptr_to_next_diag + 1, 1);
        for (int64_t j = 0; j < s; ++j)
            ptr_to_next_diag[j+1] = (T) 0.0;
    }
}

/**
 * A is a symmetric column-major matrix represented by its lower triangle.
 * 
 * If A is not PSD then this function returns an error code -(n+2).
 * If A is (near) zero then this function returns an error code -(n+1).
 * In all other cases this function returns k = dim(ker(A)).
 * 
 * If A is PSD then its trailing n - k columns will be overwritten by a 
 * matrix B where pinv(A) = BB'.
 *
 * @param[in] n matrix dimension
 * @param[in,out] A buffer for symmetric n-by-n matrix stored in host memory.
 * @param[in] lda leading dimension of A.
 * @param[out] work buffer of length >= n; overwritten by the eigenvalues of A.
 *
 * @returns k = dim(ker(A))
 */
template <typename T>
int64_t psd_sqrt_pinv(
    int64_t n,
    T* A,
    int64_t lda,
    T* work
) {
    lapack::syevd(lapack::Job::Vec, blas::Uplo::Lower, n, A, lda, work);
    T rel_tol = 10 * std::numeric_limits<T>::epsilon();
    T abs_tol = rel_tol * std::max(1.0, work[n - 1]);
    if (work[0] < -abs_tol) {
        std::cout << "The input matrix was not positive semidefinite." << std::endl;
        return -(n + 1);
    } else if (work[n - 1] < abs_tol) {
        std::cout << "The input matrix is zero, up to numerical precision." << std::endl;
        return -(n + 2);
    }
    int ker = n;
    while(ker > 0) {
        if (work[ker - 1] > abs_tol) {
            blas::scal(n, 1/std::sqrt(work[ker - 1]), &A[(ker - 1) * n], 1);
            ker = ker - 1;
        } else {
            break;
        }
    }
    return ker;
}


/** 
 * Check if LHS is PSD. If it is, then update RHS <- pinv(LHS)*RHS.
 * 
 * First we try to Cholesky decompose LHS. If that fails, we compute
 * its eigendecomposition. If the eigendecomposition shows that LHS
 * is (close to) the zero matrix or has negative eigenvalues then we
 * return an error code. Otherwise, we use the eigendecomposition to
 * perform the update for RHS.
 * 
 * @param[in] n
 *      Matrix dimension
 * @param[in,out] LHS
 *      buffer for an n-by-n matrix.
 *      Contents of this buffer are destroyed.
 * @param[in,out] RHS
 *      buffer for n-by-n matrix.
 * @param[out] work
 *     buffer of size >= n*n.
 * 
 * @returns k = rank(LHS), or an error code.
 */
template <typename T>
int64_t posm_square(
    int64_t n,
    T* LHS,
    T* RHS,
    T* work
) {
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Lower;
    using blas::Op;
    using blas::Side;
    using blas::Diag;

    // Try Cholesky (store a backup of LHS into "work")
    std::copy(LHS, LHS + n*n, work);
    int chol_err = lapack::potrf(uplo, n, LHS, n);
    if (!chol_err) {
        blas::trsm(
            layout, Side::Left, uplo, Op::NoTrans,
            Diag::NonUnit, n, n, 1.0, LHS, n, RHS, n
        ); // L y = b
        blas::trsm(
            layout, Side::Left, uplo, Op::Trans,
            Diag::NonUnit, n, n, 1.0, LHS, n, RHS, n
        ); // L^T x = y
        return n;
    } 
    // Cholesky failed.
    //      apply pinv(LHS) * RHS by computing an eigendecomposition of LHS.
    T* LHS_eigvecs = work;
    T* LHS_eigvals = LHS;
    int ker = psd_sqrt_pinv(n, LHS_eigvecs, n, LHS_eigvals);
    if (ker < 0) {
        return ker;
    } else if (ker == n) {
        for (int i = 0; i < n*n; ++i) {
            RHS[i] = 0.0;
        }
        return 0;
    }
    int rank = n - ker;
    T* pinv_sqrt = &LHS_eigvecs[ker * n];
    
    // pinv_sqrt is n-by-rank, and pinv(LHS) = pinv_sqrt * (pinv_sqrt').
    blas::gemm(
        layout, Op::Trans, Op::NoTrans, rank, n, n, 1.0, pinv_sqrt, n, RHS, n,  0.0, work, rank
    ); // work <- pinv_sqrt' * RHS
    blas::gemm(
        layout, Op::NoTrans, Op::NoTrans, n, n, rank, 1.0, pinv_sqrt, n, work, rank, 0.0, RHS, n
    ); // RHS <- pinv_sqrt * work
    return rank;
}


// MARK: [L/B]PCG

/**
 * Uses (a variant of) preconditioned conjugate gradient to solve systems of linear equations
 * 
 *  G_i * x_i = h_i for i in {1, ..., s}        (Eq. 1)
 *
 * with preconditioners N_1, ..., N_s. The linear systems all have the same number 
 * of variables, dim, so X = [x_1,..., x_s] and H = [h_1, ..., h_s] are passed as dim-by-s matrices. 
 * 
 * Arguments "G" and "N" act as operator-overloaded containers for the G_i and N_i.
 * We template these arguments abstractly. See below for details.
 *
 * PCG iterations continue until a maximum number of steps is reached OR an error metric
 * computed from the "seminorm" callable argument falls below the specified tolerance.
 * 
 * @param[in] G
 *
 *      A callable object storing one or more positive semidefinite linear operators.
 *      We have three requirements on the type of this object:
 * 
 *              G.num_ops = the number of distinct linear operators stored in G.
 *                  If G.num_ops == 1, then G_i = G_1 for all i in {1, ..., s}.
 *                  If G.num_ops >  1, then we assume each G_i is distinct, and 
 *                  we require that G.num_ops == s.
 *
 *              G.dim = the number of rows and columns in each G_i.
 *
 *              G(blas::Layout::ColMajor, s, alpha, B, ldb, beta, C, ldc) updates
 *                  c_i := alpha G_i * b_i + beta c_i
 *              where C = [c_1, ..., c_s] and B = [b_1, ..., b_s] are dim-by-s matrices stored
 *              in column-major order with strides ldc and ldb, respectively.
 * 
 *      If G.num_ops == 1 then we solve (Eq. 1) using block PCG with block size s. If G.num_ops > 1
 *      then we run an algorithm which is equivalent to s embarassingly parallel single-vector
 *      PCG, but which takes care to keep iterations of the embarassingly parallel solves in 
 *      lockstep synchronization. The lockstep mode is useful when the G_i share a common 
 *      structure, such as having the form G_i = A + mu_i*I for a psd matrix A and regularization
 *      parameters (mu_1,...,mu_s). In this case one can compute the s matrix-vector products with
 *      (G_1,...,G_s) with one matrix-matrix multiply (against A) followed by cheap postprocessing. 
 * 
 * @param[in] H
 *      Pointer to a column-major matrix of shape (G.dim, s) with stride parameter G.dim.
 *      Defines the right-hand sides in (Eq. 1).
 * 
 * @param[in] s
 *      The number of linear systems to be solved.
 *
 * @param[in,out] seminorm
 *      Callable as val = seminorm(dim, s, R), where R is a pointer to an array of type T
 *      read as a column-major matrix of shape (dim, s) with stride equal to dim. One valid
 *      implementation is 
 *
 *          auto seminorm = [](int64_t rows, int64_t cols, T* R) {
 *              return blas::nrm2(rows * cols, R, 1);
 *          };
 *
 *      in which case seminorm is actually a proper norm. (The "semi" in "seminorm" allows
 *      for the possibility that val == 0 if R belongs to fixed linear subspace, such as
 *      the space of matrices whose columns sum to the zero vector.)
 *
 *      We call seminorm twice at each iteration of PCG. Even calls (i.e., call 0, call 2, etc..)
 *      input the raw residual matrix, H - G(X). Odd calls (i.e., call 1, call 3, etc..)
 *      input the preconditioned residual matrix, equal to N(H - G(X)).
 * 
 * @param[in] tol
 *      All PCG iterations stop if normNR < tol*(1.0 + normNR0), normNR is the value returned
 *      by seminorm applied to the current preconditioned residual, and normNR0 is the value
 *      returned by seminorm applied to the preconditioned residual before iterations began.
 *
 * @param[in] max_iters
 *      An upper bound on the number of calls to G's operator() function.
 * 
 * @param[in] N
 *     This is any object that implements operator() with the same semantics of G's implementation.
 *     Each N_i should be an approximate inverse for G_i.
 *
  * @param[in,out] X
 *      Pointer to a column-major matrix of shape (G.dim, s) with stride parameter G.dim.
 *      The values of its columns on entry are used to initialize PCG solves of (Eq. 1).
 *      The values of its columns on exit contain the results of the PCG solves.
 *
 * @param[in] verbose
 *      A boolean. If true, then logging information is printed to stdout.
 * 
 */
template <typename T, typename FG, typename FSeminorm, typename FN>
void pcg(
    FG &G,
    const T* H,
    int64_t s,
    FSeminorm &seminorm,
    T tol,
    int64_t max_iters,
    FN &N,
    T* X,
    bool verbose
) {
    int64_t n = G.dim;
    int64_t ns = n*s;
    int64_t ss = s*s;
    bool treat_as_separable = G.num_ops > 1;
    if (treat_as_separable) randblas_require(s == G.num_ops);

    // All workspace gets zero-initialized; this is only
    // overridden for "R".
    T* allwork = new T[4*ns + 5*ss]{};
    
    // block vector work; n-by-s matrices.
    T* R  = allwork; std::copy(H, H + ns, R);
    T* P  = R  + ns;
    T* GP = P  + ns;
    T* NR = GP + ns;
    // block "scalar" work; s-by-s matrices.
    T* RNR     = NR      + ns;
    T* alpha   = RNR     + ss;
    T* beta    = alpha   + ss;
    T* scratch = beta    + ss;
    T* ableft  = scratch + ss; 

    T normNR = INFINITY, prevnormNR = INFINITY;

    auto layout = blas::Layout::ColMajor;
    using blas::Op;

    G(layout, s, 1.0, X, n, 0.0, GP, n
    ); // GP <- G X
    blas::axpy(ns, -1.0, GP, 1, R, 1
    ); // R <- R - GP
    N(layout, s, 1.0, R, n, 0.0, P, n
    ); // P <- N R
    blas::gemm(layout, Op::Trans, Op::NoTrans, s, s, n, 1.0, R, n, P, n, 0.0, RNR, s
    ); // RNR <- R^T P = R^T N R
    if (treat_as_separable) zero_off_diagonal(RNR, s);

    std::copy(RNR, RNR + ss, alpha
    ); // alpha <- RNR

    int64_t k = 0;
    T normR0  = seminorm(n, s, R);
    T normNR0 = seminorm(n, s, P);
    T stop_abstol = tol*(1.0 + normNR0);
    int64_t subspace_dim = 0;
    if (verbose)
        std::cout << "normNR : " << normNR0 << "\tnormR : " << normR0 << "\tk: 0\tdim : 0\n";
    while (subspace_dim < n && k < max_iters) {
        // 
        // Update X and R
        //
        k++;

        G(layout, s, (T) 1.0, P, n, (T) 0.0, GP, n
        ); // ^ GP <- G P
        blas::gemm(layout, Op::Trans, Op::NoTrans, s, s, n, 1.0, P, n, GP, n, 0.0, ableft, s
        ); // ableft <- P^T G P
        if (treat_as_separable) zero_off_diagonal(ableft, s);
        int64_t subspace_incr = posm_square(s, ableft, alpha, scratch
        ); // alpha <- (ableft)^(-1) alpha
        if (treat_as_separable && subspace_incr > 0)
            subspace_incr = 1;

        if (subspace_incr < - ((int64_t) s) )
            break;
        subspace_dim = subspace_dim + subspace_incr;

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, n, s, s, 1.0, P, n, alpha, s, 1.0, X, n
        ); // X <- X + P alpha
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, n, s, s, -1.0, GP, n, alpha, s, 1.0, R, n
        ); // R <- R - GP alpha

        //
        //  Check termination criteria
        //
        //      TODO: change how we check termination criteria in the event that we're working
        //            with treat_as_separable = true.
        T normR = seminorm(n, s, R);

        N(layout, s, 1.0, R, n, 0.0, NR, n
        ); // NR <- N R
        prevnormNR = normNR;
        normNR = seminorm(n, s, NR);
        if (verbose)
            std::cout << "normNR : " << normNR << "\tnormR : " << normR << "\tk: " << k << "\tdim : " << subspace_dim << '\n';
        if (normNR < stop_abstol)
            break;
        // 
        //  Update P, beta, and alpha
        //
        std::copy(RNR, RNR + ss, ableft
        ); // ableft <- RNR
        blas::gemm(layout, Op::Trans, Op::NoTrans, s, s, n, 1.0, R, n, NR, n, 0.0, RNR, s
        ); // RNR <- R^T NR
        if (treat_as_separable) zero_off_diagonal(RNR, s);
        std::copy(RNR, RNR + ss, alpha
        ); // alpha <- RNR
        std::copy(alpha, alpha + ss, beta
        ); // beta <- alpha
        int err = posm_square( s, ableft, beta, scratch
        ); // beta <- (ableft)^-1 beta
        if (err < - ((int64_t) s))
            break;
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, n, s, s, 1.0, P, n, beta, s, 1.0, NR, n
        ); // NR <- P beta
        std::copy(NR, NR + ns, P
        ); // P <- NR
    }

    delete [] allwork;
    return;
}


/**
 * This wraps a function of the same name that has a lower-level API. Refer to that lower-level
 * function for detailed documentation. The key __differences__ between that lower-level function
 * and this wrapper are as follows.
 * 
 *    1. This wrapper accepts std::vector instead of raw pointers.
 * 
 *    2. This wrapper doesn't require the user to specifify the number of right-hand-sides
 *       (since that can be inferred from the sizes of the input std::vector objects).
 * 
 *    3. This wrapper has a default method for the seminorm used in PCG stopping criteria.
 *       That default is to use the StatefulFrobeniusNorm type (defined above), which logs
 *       residuals both before and after preconditioning.
 *       
 * The main purpose of this wrapper is backward-compatibility with existing code.
 */
template <typename T, typename FG, typename FN, typename FSeminorm = StatefulFrobeniusNorm<T>>
FSeminorm pcg(
    FG &G,
    const std::vector<T> &H,
    T tol,
    int64_t max_iters,
    FN &N,
    std::vector<T> &X,
    bool verbose
) {
    FSeminorm seminorm{};
    int64_t s = ((int64_t) H.size()) / G.dim;
    pcg(G, H.data(), s, seminorm, tol, max_iters, N, X.data(), verbose);
    return seminorm;
}


} // end namespace RandLAPACK
