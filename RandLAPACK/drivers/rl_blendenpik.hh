#pragma once

// Public API: Blendenpik_linops — sketch-and-precondition least-squares solver.
//
// Classical Blendenpik (Avron, Maymounkov, Toledo 2010) for min ||b - A x||_2 on
// a tall LinearOperator A (m x n, m >= n):
//   1. sketch  Ask = S A   (S a d x m sparse SASO map, d = d_factor * n)
//   2. unpivoted Householder QR of the sketch: [~, R] = qr(Ask)
//   3. R is a right preconditioner: A R^{-1} is nearly orthonormal (kappa ~ 1)
//   4. solve min ||b - (A R^{-1}) y|| by matrix-free LSQR; return x = R^{-1} y.
//
// This is the sparse-projection variant Oleg asked for (SASO instead of Blendenpik's
// SRFT). It is an INDEPENDENT solver: no mu-regularization, no iterative refinement.
// Reference: knowledge-base/least-squares.md 2.2 "Algorithm SPO1" [BOOK p.101-105].

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_lsqr.hh"
#include "../linops/rl_concepts.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace RandLAPACK {


/// @brief Blendenpik (sparse-sketch + QR preconditioner + LSQR) for tall LS.
template <typename T, typename RNG = RandBLAS::DefaultRNG>
class Blendenpik_linops {
    public:
        bool timing;
        T tol;               ///< LSQR stopping tolerance (atol = btol = tol).
        int max_iters;       ///< LSQR iteration cap.
        int64_t nnz;         ///< SASO nonzeros per column (sparse projection).
        int lsqr_iters;      ///< LSQR iterations used on the last call (output).

        // [0]=sketch, [1]=qr, [2]=lsqr, [3]=total  (microseconds)
        std::vector<long> times;

        Blendenpik_linops(bool time_subroutines, T ep) {
            timing    = time_subroutines;
            tol       = (ep > (T)0) ? ep : std::numeric_limits<T>::epsilon();
            max_iters = 0;   // 0 => default cap (4n) chosen in call()
            nnz       = 4;   // sparse projection, 4 nnz/col (as CQRRT)
            lsqr_iters = 0;
        }

        /// Solve min ||b - A x||_2. A is m x n; b length m; x length n (output).
        /// d_factor sets the sketch size d = d_factor * n (>= 1, typ. 4).
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(GLO& A, const T* b, int64_t m, T* x, int64_t n,
                 T d_factor, RandBLAS::RNGState<RNG>& state)
        {
            using clock = std::chrono::steady_clock;
            using std::chrono::duration_cast; using std::chrono::microseconds;
            long t_sketch = 0, t_qr = 0, t_lsqr = 0;
            auto total_start = clock::now();

            int64_t d = (int64_t)(d_factor * (T)n);
            if (d < n) d = n;

            T* Ask = new T[d * n]();   // sketch (d x n, ColMajor)
            T* tau = new T[n]();
            T* R   = new T[n * n]();   // preconditioner (upper triangular)
            auto cleanup = [&]() { delete[] Ask; delete[] tau; delete[] R; };

            // ---- Step 1: Ask = S A   (sparse SASO, applied from the right by the operator) ----
            auto t0 = clock::now();
            RandBLAS::SparseDist DS(d, m, this->nnz);
            RandBLAS::SparseSkOp<T, RNG> S(DS, state);
            state = S.next_state;
            RandBLAS::fill_sparse(S);
            A(blas::Side::Right, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
              d, n, m, (T)1.0, S, (T)0.0, Ask, d);
            if (timing) t_sketch = duration_cast<microseconds>(clock::now() - t0).count();

            // ---- Step 2: unpivoted Householder QR of the sketch; R = upper(Ask) ----
            t0 = clock::now();
            lapack::geqrf(d, n, Ask, d, tau);
            lapack::lacpy(MatrixType::Upper, n, n, Ask, d, R, n);
            if (n > 1) lapack::laset(MatrixType::Lower, n - 1, n - 1, (T)0, (T)0, R + 1, n);
            if (timing) t_qr = duration_cast<microseconds>(clock::now() - t0).count();

            if (!RandLAPACK::util::diag_is_nonzero(n, R, n)) {
                std::fprintf(stderr, "[Blendenpik] FAIL: sketch R has a ~0 diagonal (rank-deficient sketch)\n");
                cleanup();
                return 1;
            }

            // ---- Steps 3-4: LSQR on A with right preconditioner R; x = R^{-1} y ----
            int cap = (max_iters > 0) ? max_iters : (int)std::min<int64_t>(4 * n, 1000);
            long lsqr_times[4] = {0, 0, 0, 0};
            t0 = clock::now();
            int st = RandLAPACK::lsqr<T>(A, m, n, R, n, b, x, tol, tol, cap, lsqr_iters, lsqr_times);
            if (timing) t_lsqr = duration_cast<microseconds>(clock::now() - t0).count();
            // LSQR always returns a valid iterate x (best-so-far), so hitting the cap is
            // NOT a hard failure -- x is reported and its residual shows the accuracy.
            converged = (st == 0);

            if (timing) {
                long total = duration_cast<microseconds>(clock::now() - total_start).count();
                this->times = {t_sketch, t_qr, t_lsqr, total};
            }
            // Expose the sketch R factor so the caller can report Q = A R^{-1} orthogonality.
            R_out.assign(R, R + n * n);

            cleanup();
            return 0;   // x is a valid iterate (converged or capped); only the rank-deficient
                        // sketch guard above returns 1 (hard failure, no usable R).
        }

        /// Sketch R factor from the last call (n x n, ColMajor upper-triangular),
        /// for Q = A R^{-1} orthogonality reporting.
        std::vector<T> R_out;
        /// Whether LSQR met its tolerance (false = hit the iteration cap).
        bool converged = false;
};


} // namespace RandLAPACK
