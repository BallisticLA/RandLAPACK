#pragma once

// Public API: Blendenpik_linops, sketch-and-precondition least squares.
//
// Blendenpik (Avron, Maymounkov, Toledo 2010) for min ||b - A x||_2 on a tall
// LinearOperator A (m x n, m >= n) that also provides the Side::Right sketching overload
// (wrap an operator without one in VStackLinOp), with a sparse sign sketch in place of the SRFT:
//   1. Ask = S A, with S a d x m sparse sign sketch and d = d_factor * n;
//   2. R = the triangular factor of an unpivoted Householder QR of Ask;
//   3. LSQR on A R^{-1}, which is nearly orthonormal; x = R^{-1} y.
//
// By default LSQR solves for the correction to the sketch-and-solve x0 = R^{-1} Q^T (S b), as
// forward stability needs (Epperly, Meier and Nakatsukasa, arXiv:2406.03468, sec. 3.1). One-shot
// (no refinement), so not backward stable.

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_thread_caps.hh"
#include "rl_exceptions.hh"
#include "rl_lapackpp.hh"
#include "rl_lsqr.hh"
#include "rl_concepts.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace RandLAPACK {


/// @brief Blendenpik (sparse sketch + QR preconditioner + LSQR) for tall least squares.
template <typename T, typename RNG = RandBLAS::DefaultRNG>
class Blendenpik_linops {
    public:
        bool timing;
        T tol;               ///< LSQR tolerance: the relative residual target (S1) and S2's atol;
                             ///< a constructor value <= 0 selects machine epsilon.
        int max_iters;       ///< LSQR iteration cap; must be set before call() unless init_only.
        /// Nonzeros per short-axis vector of the d x m sparse sign sketch (per column when
        /// d <= m), capped at min(d, m). Default 4, RandBLAS's SparseDist default.
        int64_t nnz;
        int lsqr_iters;      ///< LSQR iterations of the last call (output).
        /// Start LSQR from the sketch-and-solve solution rather than from zero (default on).
        bool warm_start;
        /// Return the sketch-and-solve solution without running LSQR (implies a warm start),
        /// e.g. to give another solver the same initial guess.
        bool init_only;

        /// [0] sketch (including workspace allocation), [1] QR, [2] LSQR, [3] total
        /// (including slot 4), [4] x0 setup, in microseconds. Filled only when timing is on.
        std::vector<long> times;

        /// Triangular factor of the last call (n x n, ColMajor, upper), for reporting
        /// A R^{-1} orthogonality. Owned by this object; nullptr unless the last call succeeded.
        T* R_out = nullptr;
        int64_t R_out_sz = 0;
        /// Whether the returned x met tol (by LSQR's test, or x0 already meeting it).
        bool converged = false;
        /// ||b - A x|| / ||b|| at termination (0 when b == 0): computed on the init_only and
        /// skip paths, the LSQR recurrence's estimate otherwise; -1 when not computed (no call
        /// yet, or the last call returned 1).
        T final_relres = (T)-1;
        /// Which LSQR test ended the last call: 1 = S1 (also when x0 already met tol), 2 = S2,
        /// 0 = cap hit, init_only, or a call that returned 1.
        int lsqr_stop_test = 0;
        /// LSQR's [fwd_us, adj_us, trsv_us, total_us] from the last call.
        std::vector<long> lsqr_op_times;

        Blendenpik_linops(bool time_subroutines, T ep) {
            timing    = time_subroutines;
            tol       = (ep > (T)0) ? ep : std::numeric_limits<T>::epsilon();
            max_iters = 0;
            nnz       = 4;
            lsqr_iters = 0;
            warm_start = true;
            init_only  = false;
        }

        ~Blendenpik_linops() {
            delete[] R_out;
        }

        Blendenpik_linops(const Blendenpik_linops&) = delete;
        Blendenpik_linops& operator=(const Blendenpik_linops&) = delete;

        /// Solve min ||b - A x||_2 for A m x n, b of length m, x of length n (output).
        /// The sketch has d = max(n, d_factor * n) rows.
        /// @returns 0 with x a valid iterate (check `converged`); 1 if the sketch's triangular
        ///          factor has a zero diagonal entry (rank-deficient sketch), x untouched.
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(GLO& A, const T* b, int64_t m, T* x, int64_t n,
                 T d_factor, RandBLAS::RNGState<RNG>& state)
        {
            converged      = false;
            final_relres   = (T)-1;
            lsqr_iters     = 0;
            lsqr_stop_test = 0;
            lsqr_op_times.clear();
            times.clear();
            delete[] R_out;
            R_out    = nullptr;
            R_out_sz = 0;

            randlapack_require(init_only || max_iters > 0)
                << "Blendenpik_linops: max_iters=" << max_iters
                << " must be set > 0 before call() unless init_only is set";
            randlapack_require(m == A.n_rows && n == A.n_cols && m >= n)
                << "Blendenpik_linops: (m, n) = (" << m << ", " << n << ") must equal A's ("
                << A.n_rows << ", " << A.n_cols << ") with m >= n";

            using clock = std::chrono::steady_clock;
            using std::chrono::duration_cast; using std::chrono::microseconds;
            long t_sketch = 0, t_qr = 0, t_lsqr = 0, t_x0 = 0;
            auto total_start = clock::now();
            const bool ws = warm_start || init_only;

            randlapack_require(d_factor > 0) << "Blendenpik_linops: d_factor=" << d_factor << " must be positive";
            const int64_t d = cqrrt_sketch_rows(d_factor, n);
            const int64_t dn = RandBLAS::safe_int_product<int64_t>(d, n);
            const int64_t nn = RandBLAS::safe_int_product<int64_t>(n, n);

            auto t0 = clock::now();
            std::unique_ptr<T[]> Ask_buf(new T[dn]()), tau_buf(new T[n]()), R_buf(new T[nn]());
            std::unique_ptr<T[]> Sb_buf(ws ? new T[d]() : nullptr), x0_buf(ws ? new T[n]() : nullptr),
                                 r0_buf(ws ? new T[m]() : nullptr);
            T* Ask = Ask_buf.get(); T* tau = tau_buf.get(); T* R = R_buf.get();
            T* Sb = Sb_buf.get(); T* x0 = x0_buf.get(); T* r0 = r0_buf.get();

            // Step 1: Ask = S A, and Sb = S b with the same S.
            RandBLAS::SparseDist DS(d, m, std::min({this->nnz, d, m}));
            RandBLAS::SparseSkOp<T, RNG> S(DS, state);
            state = S.next_state;
            RandBLAS::fill_sparse(S);
            A(blas::Side::Right, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
              d, n, m, (T)1.0, S, (T)0.0, Ask, d);
            if (ws) {
                RandBLAS::sketch_general(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                         d, 1, m, (T)1.0, S, 0, 0, b, m, (T)0.0, Sb, d);
            }
            if (timing) t_sketch = duration_cast<microseconds>(clock::now() - t0).count();

            // Step 2: R = upper triangle of the Householder QR of Ask (R's strict lower stays 0).
            t0 = clock::now();
            lapack::geqrf(d, n, Ask, d, tau);
            lapack::lacpy(MatrixType::Upper, n, n, Ask, d, R, n);
            if (timing) t_qr = duration_cast<microseconds>(clock::now() - t0).count();

            if (!RandLAPACK::util::diag_is_nonzero(n, R, n))
                return 1;

            // Step 3: x0 = R^{-1} (Q^T (S b)), with Q applied implicitly by ormqr, and
            // r0 = b - A x0.
            if (ws) {
                t0 = clock::now();
                lapack::ormqr(blas::Side::Left, blas::Op::Trans, d, 1, n, Ask, d, tau, Sb, d);
                std::copy(Sb, Sb + n, x0);
                detail::capped_trsv(blas::Op::NoTrans, n, R, n, x0);
                A(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                  m, 1, n, (T)1.0, x0, n, (T)0.0, r0, m);
                for (int64_t i = 0; i < m; ++i) r0[i] = b[i] - r0[i];
                if (timing) t_x0 = duration_cast<microseconds>(clock::now() - t0).count();
            }

            // Step 4, unless init_only or x0 already meets tol: LSQR on A with right preconditioner
            // R. With a warm start it solves for the correction dx against r0, and
            // btol = tol ||b|| / ||r0|| turns its S1 test into ||b - A x|| / ||b|| <= tol.
            t0 = clock::now();
            const T nb  = ws ? blas::nrm2(m, b, 1) : (T)0;
            const T nr0 = ws ? blas::nrm2(m, r0, 1) : (T)0;
            if (init_only || (ws && nr0 <= tol * nb)) {
                std::copy(x0, x0 + n, x);
                final_relres = (nb > (T)0) ? nr0 / nb : (T)0;
                if (!init_only) {   // LSQR would stop on S1 at once
                    lsqr_stop_test = 1;
                    lsqr_op_times.assign(4, 0L);
                    converged = true;
                }
            } else {
                long lt[4] = {0, 0, 0, 0};
                const T* rhs = ws ? r0 : b;
                const T btol = ws ? tol * nb / nr0 : tol;
                int st = RandLAPACK::lsqr<T>(A, m, n, R, n, rhs, x,
                                             tol, btol, max_iters, lsqr_iters, lt,
                                             &final_relres, &lsqr_stop_test);
                lsqr_op_times.assign(lt, lt + 4);
                if (ws) {
                    blas::axpy(n, (T)1.0, x0, 1, x, 1);
                    final_relres *= nr0 / nb;   // lsqr normalized by ||r0||
                }
                converged = (st == 0);
            }
            if (timing) {
                t_lsqr = init_only ? 0 : duration_cast<microseconds>(clock::now() - t0).count();
                long total = duration_cast<microseconds>(clock::now() - total_start).count();
                this->times = {t_sketch, t_qr, t_lsqr, total, t_x0};
            }
            R_out = R_buf.release(); R_out_sz = nn;
            return 0;
        }
};


} // namespace RandLAPACK
