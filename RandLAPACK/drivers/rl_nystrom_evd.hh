#pragma once

#include "rl_syps.hh"
#include "rl_syrf.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cstdint>
#include <vector>

namespace RandLAPACK {


// -----------------------------------------------------------------------------
/// Power scheme for error estimation, based on Algorithm E.1 from https://arxiv.org/pdf/2110.02820.pdf.
/// p - number of algorithm iterations
/// vector_buf - buffer for vector operations (must hold at least 4*m elements)
/// Mat_buf    - buffer for matrix operations (must hold at least m*k elements)
/// All other parameters come from NystromEVD
template <typename T, linops::SymmetricLinearOperator SLO>
T power_error_est(
    SLO &A,
    int64_t k,
    int p,
    T* vector_buf,
    T* V,
    T* Mat_buf,
    T* eigvals
) {
    int64_t m = A.dim;
    T err = 0;
    for(int iter = 0; iter < p; ++iter) {
        T g_norm = blas::nrm2(m, vector_buf, 1);
        blas::scal(m, (T)1 / g_norm, vector_buf, 1);

        // V' * g/||g|| — result in column 1 of vector_buf
        gemv(Layout::ColMajor, Op::Trans, m, k, (T)1, V, m, vector_buf, 1, (T)0, &vector_buf[m], 1);

        // V * diag(eigvals) into Mat_buf — column-major: column col occupies rows [0, m)
        for (int64_t col = 0; col < k; ++col)
            for (int64_t row = 0; row < m; ++row)
                Mat_buf[col * m + row] = V[col * m + row] * eigvals[col];

        // V * diag(eigvals) * V' * g/||g|| — result in column 2 of vector_buf
        gemv(Layout::ColMajor, Op::NoTrans, m, k, (T)1, Mat_buf, m, &vector_buf[m], 1, (T)0, &vector_buf[2 * m], 1);
        // A * g/||g|| — result in column 3 of vector_buf
        A(Layout::ColMajor, 1, (T)1, vector_buf, m, (T)0, &vector_buf[3*m], m);

        // w = A*g/||g|| - V*diag(eigvals)*V'*g/||g||
        blas::axpy(m, (T)-1, &vector_buf[2 * m], 1, &vector_buf[3 * m], 1);
        // error estimate: (g/||g||)' * w
        err = blas::dot(m, vector_buf, 1, &vector_buf[3 * m], 1);
        std::copy(&vector_buf[3 * m], &vector_buf[4 * m], vector_buf);
    }
    return err;
}


template <typename SYRF_t>
class NystromEVD {
    public:
        using T   = typename SYRF_t::T;
        using RNG = typename SYRF_t::RNG;
        SYRF_t &syrf;
        int error_est_p;
        bool verbose;

        // Internal working buffers — owned by this object, grown with new[]/delete[] via util::resize
        T* Y          = nullptr; int64_t Y_sz          = 0;
        T* Omega      = nullptr; int64_t Omega_sz      = 0;
        T* R          = nullptr; int64_t R_sz          = 0;
        T* S          = nullptr; int64_t S_sz          = 0;
        T* symrf_work = nullptr; int64_t symrf_work_sz = 0;

        bool timing = false;
        std::vector<long> times;  // populated after call() when timing==true
        // Slots: alloc, syrf, matvec, gram, potrf, trsm, svd, post_svd, error_est, rest, total

        NystromEVD(
            SYRF_t &syrf_obj,
            int error_est_power_iters,
            bool verb = false
        ) : syrf(syrf_obj) {
            error_est_p = error_est_power_iters;
            verbose = verb;
        }

        NystromEVD(const NystromEVD&)            = delete;
        NystromEVD& operator=(const NystromEVD&) = delete;

        ~NystromEVD() {
            delete[] Y;
            delete[] Omega;
            delete[] R;
            delete[] S;
            delete[] symrf_work;
        }

        /// Computes a rank-k approximation to an EVD of a symmetric positive semidefinite matrix:
        ///     A_hat = V diag(eigvals) V^*,
        /// where V is a matrix of approximate eigenvectors and eigvals holds the eigenvalues.
        ///
        /// Adaptive: if tolerance is not met, doubles k until convergence or k == m.
        /// Set error_est_power_iters=0 and tol=0 for fixed-rank single-pass behavior.
        ///
        /// This code is identical to Algorithm E2 from https://arxiv.org/pdf/2110.02820.pdf.
        ///
        /// @param[in]     uplo     Triangle of A that holds the data (Upper or Lower).
        /// @param[in]     m        Number of rows/cols of A.
        /// @param[in]     A        m×m matrix, column-major.
        /// @param[in,out] k        Sketch rank on entry; final rank on exit (may grow adaptively).
        /// @param[in]     tol      Convergence tolerance. Use 0.0 for fixed-rank.
        /// @param[in,out] V        Caller-owned buffer; on exit holds m×k eigenvectors.
        /// @param[in,out] V_sz     Capacity of V in elements; grown by util::resize as needed.
        /// @param[in,out] eigvals  Caller-owned buffer; on exit holds k eigenvalues.
        /// @param[in,out] eigvals_sz  Capacity of eigvals; grown by util::resize as needed.
        ///
        int call(
            Uplo uplo,
            int64_t m,
            const T* A,
            int64_t &k,
            T tol,
            T*& V, int64_t& V_sz,
            T*& eigvals, int64_t& eigvals_sz,
            RandBLAS::RNGState<RNG> &state
        ) {
            linops::ExplicitSymLinOp<T> A_linop(m, uplo, A, m, Layout::ColMajor);
            return this->call(A_linop, k, tol, V, V_sz, eigvals, eigvals_sz, state);
        }

        template <linops::SymmetricLinearOperator SLO>
        int call(
            SLO &A,
            int64_t &k,
            T tol,
            T*& V, int64_t& V_sz,
            T*& eigvals, int64_t& eigvals_sz,
            RandBLAS::RNGState<RNG> &state
        ) {
            using namespace std::chrono;
            int64_t m = A.dim;
            T err = 0;
            RandBLAS::RNGState<RNG> error_est_state(state.counter, state.key);
            error_est_state.key.incr(1);

            steady_clock::time_point t0, t1, total_t_start, total_t_stop;
            long alloc_t_dur     = 0;
            long syrf_t_dur      = 0;
            long matvec_t_dur    = 0;
            long gram_t_dur      = 0;
            long potrf_t_dur     = 0;
            long trsm_t_dur      = 0;
            long svd_t_dur       = 0;
            long post_svd_t_dur  = 0;
            long error_est_t_dur = 0;
            long total_t_dur     = 0;

            if (this->timing) total_t_start = steady_clock::now();

            while(true) {
                if (this->timing) t0 = steady_clock::now();
                util::resize(eigvals, eigvals_sz, k);
                util::resize(V, V_sz, m * k);
                T* V_dat = V;
                util::resize(Y,          Y_sz,          m * k);
                util::resize(Omega,      Omega_sz,      std::max(m * k, m * (int64_t)4));
                util::resize(R,          R_sz,          k * k);
                util::resize(S,          S_sz,          k * k);
                util::resize(symrf_work, symrf_work_sz, m * k);
                if (this->timing) { t1 = steady_clock::now(); alloc_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                if (this->timing) t0 = steady_clock::now();
                this->syrf.call(A, k, this->Omega, state, symrf_work);
                if (this->timing) { t1 = steady_clock::now(); syrf_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                if (this->timing) t0 = steady_clock::now();
                A(Layout::ColMajor, k, (T)1, Omega, m, (T)0, Y, m);
                if (this->timing) { t1 = steady_clock::now(); matvec_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                // ----- Phase 1 spectral recovery -----
                //
                // Two paths sharing the same Gram-matrix construction:
                //
                // FAST PATH (well-conditioned Gram — the common case):
                //   G    = Ωᵀ · Y                              (k×k sym PSD)
                //   G    = Lᵀ · L                              (Cholesky)
                //   Y    := Y · L⁻¹                            (TRSM)
                //   G_y  = Yᵀ · Y                              (small Gram)
                //   G_y  = V · Σ² · Vᵀ                         (syevd)
                //   U    = Y · V · diag(1/σ)                   (gemm + scal)
                //   eigvals = Σ²
                //
                //   The eig-of-Yᵀ·Y trick (Halko-Martinsson-Tropp §5.1) avoids
                //   the more expensive gesdd of n×k while keeping eigenvector
                //   accuracy fine because Y is well-conditioned post-TRSM.
                //
                // FALL-BACK (Cholesky failure on near-singular Gram):
                //   syevd(G) → V_G, D                          (pseudoinverse)
                //   B    = Y · V_G · diag(1/√D) · V_Gᵀ          (3 GEMMs)
                //   gesdd(B) → U, Σ
                //   eigvals = Σ²
                //
                //   This is the safe path. It handles rank-deficient Gram via
                //   thresholded pseudoinverse. Mirrors the Persson-Kressner
                //   reference nystrom.m. Slightly more expensive than the
                //   fast path but always succeeds.
                //
                // No regularization shift in either path — both paths preserve
                // the ε_mach accuracy of the SVD-pseudoinverse rewrite.

                T eps_mach = std::numeric_limits<T>::epsilon();

                // Form Gram G := Ωᵀ · Y into R (k×k). G = Ωᵀ·A·Ω in exact
                // arithmetic and is symmetric, but the two non-trivial GEMMs
                // (sketch + power-iter Y, then Ωᵀ Y) introduce small
                // asymmetry; symmetrize so potrf and the later syevd see a
                // numerically symmetric matrix.
                if (this->timing) t0 = steady_clock::now();
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
                           (T)1, Omega, m, Y, m, (T)0, R, k);
                RandLAPACK::util::symmetrize(k, R, k);
                // Backup G into symrf_work so we can restore it if Cholesky
                // fails (potrf is destructive). symrf_work is m×k = enough for
                // the k×k Gram.
                lapack::lacpy(lapack::MatrixType::General, k, k, R, k, symrf_work, k);
                if (this->timing) { t1 = steady_clock::now(); gram_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                int64_t r = 0;

                // Try Cholesky.
                if (this->timing) t0 = steady_clock::now();
                int chol_status = lapack::potrf(Uplo::Upper, k, R, k);
                if (this->timing) { t1 = steady_clock::now(); potrf_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                if (chol_status == 0) {
                    // -------- FAST PATH --------
                    RandLAPACK::util::get_U(k, k, R, k);   // zero strict lower

                    // Y := Y · R⁻¹  (R is upper-tri Cholesky factor)
                    if (this->timing) t0 = steady_clock::now();
                    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper,
                               Op::NoTrans, Diag::NonUnit, m, k,
                               (T)1, R, k, Y, m);
                    if (this->timing) { t1 = steady_clock::now(); trsm_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                    // G_y = Yᵀ · Y (k×k symmetric); reuse R buffer. syrk
                    // writes only the upper triangle; the strict lower stays
                    // unspecified, which is fine — the following syevd call
                    // is invoked with Uplo::Upper and reads only the upper
                    // triangle (then overwrites all of R with eigenvectors).
                    if (this->timing) t0 = steady_clock::now();
                    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                               k, m, (T)1, Y, m, (T)0, R, k);
                    // syevd(G_y) → R (eigvecs V), S (Σ², ASCENDING order).
                    lapack::syevd(lapack::Job::Vec, Uplo::Upper, k, R, k, S);
                    // Reverse to descending order to match the rest of the
                    // pipeline (post-SVD threshold logic expects S[0]=largest).
                    for (int64_t ii = 0; ii < k / 2; ++ii) std::swap(S[ii], S[k - 1 - ii]);
                    for (int64_t jj = 0; jj < k / 2; ++jj) {
                        blas::swap(k, R + jj * k, 1, R + (k - 1 - jj) * k, 1);
                    }
                    // U = Y · V  (left singular vectors, unscaled).
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k,
                               (T)1, Y, m, R, k, (T)0, V_dat, m);
                    // Scale columns by 1/σ (= 1/sqrt(σ²)). Threshold tiny σ² as 0.
                    T sig_sq_max    = (k > 0) ? std::max(S[0], (T)0) : (T)0;
                    T sig_sq_thresh = (T)4 * eps_mach * sig_sq_max;
                    for (int64_t jj = 0; jj < k; ++jj) {
                        if (S[jj] > sig_sq_thresh) {
                            T inv_sig = (T)1 / std::sqrt(S[jj]);
                            blas::scal(m, inv_sig, V_dat + jj * m, 1);
                            ++r;
                        } else {
                            std::fill(V_dat + jj * m, V_dat + (jj + 1) * m, (T)0);
                        }
                    }
                    // eigvals = σ² (already descending).
                    for (int64_t i = 0; i < k; ++i)
                        eigvals[i] = std::max(S[i], (T)0);
                    if (this->timing) { t1 = steady_clock::now(); svd_t_dur += duration_cast<microseconds>(t1 - t0).count(); }
                } else {
                    // -------- FALL-BACK: syevd-pseudoinverse --------
                    // Restore Gram into R from backup.
                    lapack::lacpy(lapack::MatrixType::General, k, k, symrf_work, k, R, k);

                    if (this->timing) t0 = steady_clock::now();
                    lapack::syevd(lapack::Job::Vec, Uplo::Lower, k, R, k, S);
                    if (this->timing) { t1 = steady_clock::now(); potrf_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                    if (this->timing) t0 = steady_clock::now();
                    T D_max    = (k > 0) ? std::max(S[k - 1], (T)0) : (T)0;
                    T D_thresh = (T)2 * eps_mach * D_max;
                    // symrf_work was the Gram backup; safe to overwrite now.
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k,
                               (T)1, Y, m, R, k, (T)0, symrf_work, m);
                    for (int64_t jj = 0; jj < k; ++jj) {
                        T scale = (S[jj] > D_thresh) ? (T)1 / std::sqrt(S[jj]) : (T)0;
                        blas::scal(m, scale, symrf_work + jj * m, 1);
                    }
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, k, k,
                               (T)1, symrf_work, m, R, k, (T)0, Y, m);
                    if (this->timing) { t1 = steady_clock::now(); trsm_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                    if (this->timing) t0 = steady_clock::now();
                    lapack::gesdd(Job::SomeVec, m, k, Y, m, S, V_dat, m, R, k);
                    if (this->timing) { t1 = steady_clock::now(); svd_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                    T S_max    = (k > 0) ? S[0] : (T)0;
                    T S_thresh = (T)2 * eps_mach * S_max;
                    for (int64_t i = 0; i < k; ++i) {
                        eigvals[i] = std::pow(S[i], 2);
                        if (S[i] > S_thresh) ++r;
                    }
                    std::fill(&V_dat[m * r], &V_dat[m * k], (T)0);
                }

                if (this->timing) t0 = steady_clock::now();
                if (this->timing) { t1 = steady_clock::now(); post_svd_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                // Fixed-rank mode: skip error estimation entirely.
                if (this->error_est_p == 0 && tol == (T)0) break;

                if (this->timing) t0 = steady_clock::now();
                RandBLAS::DenseDist g(m, 1);
                error_est_state = RandBLAS::fill_dense(g, Omega, error_est_state);
                err = power_error_est(A, k, this->error_est_p, Omega, V_dat, Y, eigvals);
                if (this->timing) { t1 = steady_clock::now(); error_est_t_dur += duration_cast<microseconds>(t1 - t0).count(); }

                // Convergence: stop when the residual estimate hits the
                // numerical noise floor (~ ε_mach · λ_max ≈ ε_mach · eigvals[0]
                // since gesdd returned descending Σ → eigvals[0] is largest)
                // or when the rank has saturated.
                T noise_floor = eps_mach * (k > 0 ? eigvals[0] : (T)0);
                if (err <= 5 * std::max(tol, noise_floor) || k == m) {
                    break;
                } else if (2 * k > m) {
                    k = m;
                } else {
                    k = 2 * k;
                }
            }

            if (this->timing) {
                total_t_stop = steady_clock::now();
                total_t_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                long t_rest = total_t_dur - (alloc_t_dur + syrf_t_dur + matvec_t_dur + gram_t_dur +
                                             potrf_t_dur + trsm_t_dur + svd_t_dur + post_svd_t_dur + error_est_t_dur);
                this->times = {alloc_t_dur, syrf_t_dur, matvec_t_dur, gram_t_dur,
                               potrf_t_dur, trsm_t_dur, svd_t_dur, post_svd_t_dur,
                               error_est_t_dur, t_rest, total_t_dur};
            }
            return 0;
        }

};


} // end namespace RandLAPACK
