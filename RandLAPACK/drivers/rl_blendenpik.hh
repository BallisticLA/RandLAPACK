#pragma once

// Public API: Blendenpik_linops: sketch-and-precondition least-squares solver.
//
// Classical Blendenpik (Avron, Maymounkov, Toledo 2010) for min ||b - A x||_2 on
// a tall LinearOperator A (m x n, m >= n):
//   1. sketch  Ask = S A   (S a d x m sparse SASO map, d = d_factor * n)
//   2. unpivoted Householder QR of the sketch: [~, R] = qr(Ask)
//   3. R is a right preconditioner: A R^{-1} is nearly orthonormal (kappa ~ 1)
//   4. solve min ||b - (A R^{-1}) y|| by matrix-free LSQR; return x = R^{-1} y.
//
// This is the sparse-projection variant (SASO instead of Blendenpik's SRFT). It is
// an INDEPENDENT solver: no mu-regularization, no iterative refinement.
// Reference: Algorithm SPO1 in Avron, Maymounkov, Toledo (2010).
//
// Sketch-and-solve initialization (`warm_start`, default ON):
// LSQR is started from x0 = R^{-1}(Q^T(S b)), the solution of the sketched problem,
// instead of from zero. Epperly, Meier and Nakatsukasa (arXiv:2406.03468v3, sec. 3.1)
// note this initialization is "necessary for the method to be forward stable" and that
// it is an optional setting in the original Blendenpik code; starting from zero is a
// documented cause of stagnating short of the attainable accuracy. Because rl_lsqr
// always starts from zero internally, the warm start is applied here as an equivalent
// shift: solve for the correction dx against the residual r0 = b - A x0, then return
// x = x0 + dx. That leaves rl_lsqr untouched, so the five Q-less QR methods that share
// it are provably unaffected.
//
// NOTE this remains one-shot sketch-and-precondition (no iterative refinement), which
// the same reference proves is not backward stable. Residual stagnation ABOVE the
// backward-stable level is therefore expected behaviour, not a defect.

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_blas2_threads.hh"
#include "rl_exceptions.hh"
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
        /// Start LSQR from the sketch-and-solve solution rather than from zero.
        /// Required for forward stability (see the file header); on by default.
        bool warm_start;
        /// Stop after the sketch-and-solve initial guess and return it as x, skipping
        /// LSQR entirely (implies warm_start). Lets a caller reuse this class as a
        /// standalone sketch-and-solve solver, e.g. to warm-start IterRefineLSQ with
        /// the exact same x0 Blendenpik uses, isolating the initialization effect.
        bool init_only;

        // [0]=sketch, [1]=qr, [2]=lsqr, [3]=total, [4]=x0 setup  (microseconds).
        // Slot 4 is the sketch-and-solve x0 build (ormqr + trsv + one operator
        // apply for r0); it is not included in slot [3].
        std::vector<long> times;

        Blendenpik_linops(bool time_subroutines, T ep) {
            timing    = time_subroutines;
            tol       = (ep > (T)0) ? ep : std::numeric_limits<T>::epsilon();
            max_iters = 0;   // callers MUST set this before call(); see the require below
            nnz       = 4;   // sparse projection, 4 nnz/col (benchmark CLI overrides;
                             // NOT the CQRRT default, which is 2)
            lsqr_iters = 0;
            warm_start = true;
            init_only  = false;
        }

        ~Blendenpik_linops() {
            delete[] R_out;
        }

        // R_out is an owned raw pointer with no callers copying this object
        // (only constructed, called, and read); disable copy so a copy never
        // silently double-frees or aliases the buffer.
        Blendenpik_linops(const Blendenpik_linops&) = delete;
        Blendenpik_linops& operator=(const Blendenpik_linops&) = delete;

        /// Solve min ||b - A x||_2. A is m x n; b length m; x length n (output).
        /// d_factor sets the sketch size d = d_factor * n (>= 1, typ. 4).
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(GLO& A, const T* b, int64_t m, T* x, int64_t n,
                 T d_factor, RandBLAS::RNGState<RNG>& state)
        {
            // Reset every output member up front so an early-return path
            // (including the rank-deficient sketch guard below) never leaves
            // a previous call's values on a reused object.
            converged      = false;
            final_relres   = (T)-1;
            lsqr_iters     = 0;
            lsqr_stop_test = 0;
            lsqr_op_times.clear();
            times.clear();
            delete[] R_out;
            R_out    = nullptr;
            R_out_sz = 0;

            // Hoisted ahead of every allocation below (was checked only at
            // Step 4, after six new[] and the sketch+QR had already run,
            // so the throw leaked all of it).
            randlapack_require(init_only || max_iters > 0)
                << "Blendenpik_linops: set max_iters before call() unless init_only is set (no default cap)";

            using clock = std::chrono::steady_clock;
            using std::chrono::duration_cast; using std::chrono::microseconds;
            long t_sketch = 0, t_qr = 0, t_lsqr = 0, t_x0 = 0;
            auto total_start = clock::now();
            // Local view only: init_only implies a warm start (x0 IS the output),
            // but must not mutate the member, or a reused object would be
            // silently warm-started on later calls.
            const bool ws = warm_start || init_only;

            int64_t d = (int64_t)(d_factor * (T)n);
            if (d < n) d = n;

            // Allocation (and its first-touch zeroing) is timed INTO the sketch
            // slot: the Q-less drivers all time their allocations into the
            // build total, so leaving Blendenpik's outside every slot would
            // make its build bar incomparable.
            auto t0 = clock::now();
            T* Ask = new T[d * n]();   // sketch (d x n, ColMajor)
            T* tau = new T[n]();
            T* R   = new T[n * n]();   // preconditioner (upper triangular)
            T* Sb  = ws ? new T[d]() : nullptr;   // sketched RHS, for x0
            T* x0  = ws ? new T[n]() : nullptr;
            T* r0  = ws ? new T[m]() : nullptr;
            auto cleanup = [&]() {
                delete[] Ask; delete[] tau; delete[] R;
                delete[] Sb; delete[] x0; delete[] r0;
            };

            // ---- Step 1: Ask = S A   (sparse SASO, applied from the right by the operator) ----
            RandBLAS::SparseDist DS(d, m, this->nnz);
            RandBLAS::SparseSkOp<T, RNG> S(DS, state);
            state = S.next_state;
            RandBLAS::fill_sparse(S);
            A(blas::Side::Right, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
              d, n, m, (T)1.0, S, (T)0.0, Ask, d);
            // Sketch the RHS with the SAME S, so x0 solves the sketched LS problem
            // min ||Ask x - Sb||. Treat b as an m x 1 matrix.
            if (ws) {
                RandBLAS::sketch_general(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                         d, 1, m, (T)1.0, S, 0, 0, b, m, (T)0.0, Sb, d);
            }
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

            // ---- Step 3: sketch-and-solve initial guess  x0 = R^{-1} (Q^T (S b)) ----
            // Q is the implicit factor from geqrf(Ask); apply Q^T with ormqr rather than
            // forming it, then one triangular solve. The dominant cost is the one full
            // operator apply for r0. Timed into its own slot (times[4]) so
            // callers reporting sketch+qr and lsqr do not drop it.
            if (ws) {
                t0 = clock::now();
                lapack::ormqr(blas::Side::Left, blas::Op::Trans, d, 1, n, Ask, d, tau, Sb, d);
                std::copy(Sb, Sb + n, x0);
                { Blas2ThreadGuard tg(n);   // cap threads: see rl_blas2_threads.hh
                    blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::NoTrans,
                               blas::Diag::NonUnit, n, R, n, x0, 1);
                }
                // r0 = b - A x0: LSQR then solves for the correction against this residual.
                A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                  m, 1, n, (T)1.0, x0, n, (T)0.0, r0, m);
                for (int64_t i = 0; i < m; ++i) r0[i] = b[i] - r0[i];
                if (timing) t_x0 = duration_cast<microseconds>(clock::now() - t0).count();
            }

            // init_only: the sketch-and-solve x0 IS the answer; skip LSQR. Report the
            // true relative residual of x0 so callers can log the warm start's quality.
            if (init_only) {
                std::copy(x0, x0 + n, x);
                lsqr_iters = 0;
                converged  = false;   // no tolerance was pursued
                T nb = blas::nrm2(m, b, 1);
                final_relres = (nb > (T)0) ? blas::nrm2(m, r0, 1) / nb : (T)-1;
                if (timing) {
                    long total = duration_cast<microseconds>(clock::now() - total_start).count();
                    this->times = {t_sketch, t_qr, 0, total, t_x0};
                }
                // Hand R to the caller instead of copying it: R_out now owns the
                // buffer, so cleanup() below must not free it too.
                R_out = R; R_out_sz = n * n; R = nullptr;
                cleanup();
                return 0;
            }

            // ---- Step 4: LSQR on A with right preconditioner R; x = R^{-1} y ----
            long lt[4] = {0, 0, 0, 0};
            t0 = clock::now();
            int st = 0;
            if (ws) {
                T nb  = blas::nrm2(m, b, 1);
                T nr0 = blas::nrm2(m, r0, 1);
                if (nr0 <= tol * nb) {
                    // x0 already meets the caller's tolerance on the TRUE
                    // residual ||b - A x0|| / ||b||. Running LSQR from here
                    // would otherwise pursue S1 (||dx-residual|| <= btol *
                    // ||r0||) which is orders stricter than tol whenever
                    // ||r0|| << ||b|| (an unintentionally harder target
                    // than every other method's stop criterion pursues).
                    std::copy(x0, x0 + n, x);
                    lsqr_iters     = 0;
                    lsqr_stop_test = 1;   // S1 (residual) already satisfied
                    lsqr_op_times.assign(4, 0L);
                    // -1 sentinel for ||b|| == 0, matching init_only above:
                    // the ratio is undefined, not zero.
                    final_relres = (nb > (T)0) ? nr0 / nb : (T)-1;
                    converged = true;
                } else {
                    // LSQR solves for the correction dx against r0 = b - A x0.
                    // btol_eff makes its S1 test on ||r0 - Ã dx|| / ||r0||
                    // equivalent to the caller's ||b - A x|| / ||b|| <= tol,
                    // since b - A(x0 + dx) = r0 - A dx.
                    T btol_eff = tol * nb / nr0;
                    st = RandLAPACK::lsqr<T>(A, m, n, R, n, r0, x,
                                             tol, btol_eff, max_iters, lsqr_iters, lt,
                                             &final_relres, &lsqr_stop_test);
                    lsqr_op_times.assign(lt, lt + 4);
                    // Undo the shift: LSQR solved for the correction, so add the
                    // initial guess back.
                    blas::axpy(n, (T)1.0, x0, 1, x, 1);
                    // LSQR normalized its residual by ||r0||, not ||b||. Rescale
                    // so the reported number means the same thing as for every
                    // other method (||b - A x|| / ||b||); the numerator is
                    // already the true residual (same identity as above).
                    if (final_relres >= (T)0 && nb > (T)0) final_relres *= nr0 / nb;
                    // LSQR always returns a valid iterate x (best-so-far), so
                    // hitting the cap is NOT a hard failure for the SOLUTION,
                    // but it does mean the requested tolerance was not met, and
                    // callers must be able to see that.
                    converged = (st == 0);
                }
            } else {
                st = RandLAPACK::lsqr<T>(A, m, n, R, n, b, x,
                                         tol, tol, max_iters, lsqr_iters, lt,
                                         &final_relres, &lsqr_stop_test);
                lsqr_op_times.assign(lt, lt + 4);
                converged = (st == 0);
            }
            if (timing) t_lsqr = duration_cast<microseconds>(clock::now() - t0).count();

            if (timing) {
                long total = duration_cast<microseconds>(clock::now() - total_start).count();
                this->times = {t_sketch, t_qr, t_lsqr, total, t_x0};
            }
            // Expose the sketch R factor so the caller can report Q = A R^{-1}
            // orthogonality. Hand off the buffer (cleanup() must not free it).
            R_out = R; R_out_sz = n * n; R = nullptr;

            cleanup();
            return 0;   // x is a valid iterate (converged or capped); only the rank-deficient
                        // sketch guard above returns 1 (hard failure, no usable R).
        }

        /// Sketch R factor from the last call (n x n, ColMajor upper-triangular),
        /// for Q = A R^{-1} orthogonality reporting. Owned by this object (the
        /// destructor frees it); nullptr / 0 when no successful call has run yet.
        T* R_out = nullptr;
        int64_t R_out_sz = 0;
        /// Whether LSQR met its tolerance (false = hit the iteration cap).
        bool converged = false;
        /// LSQR's own ||b - A x|| / ||b|| at termination. Previously Blendenpik never
        /// requested this from lsqr, so its CSV column was structurally -1 while every
        /// other method carried a real number.
        T final_relres = (T)-1;
        /// Which LSQR stopping test ended the run: 1 = S1 (residual), 2 = S2
        /// (normal-equation test; fires at the LS floor), 0 = iteration cap.
        int lsqr_stop_test = 0;
        /// LSQR's internal operator split [fwd_us, adj_us, trsm_us, total_us] from
        /// the last call. If discarded instead, the benchmarks' op-split
        /// columns would be forced to -1 sentinels.
        std::vector<long> lsqr_op_times;
};


} // namespace RandLAPACK
