// Toeplitz least-squares benchmark (Oleg's autoregression / system-identification
// experiment) ported to RandLAPACK.
//
// Solves the regularized LS  min_x ||T x - b||^2 + lambda ||x||^2  =  min_x ||A x - rhs||,
// with A = [T; sqrt(lambda) I], rhs = [b; 0], T an m x n prolate-kernel Toeplitz matrix.
// T is matrix-free (FFT circulant embedding, ext_toeplitz_linop.hh). Q-less QR right
// preconditioners (unpreconditioned / CholQR / CholQR2 / sCholQR3 / sCholQR3_basic /
// CQRRT / Blendenpik) build R; the solve is matrix-free LSQR on A with right precond R.
//
// v1: double precision only, LSQR solver. Records accuracy (Oleg's metrics) + speed
// (build/solve time) + storage (peak RSS, analytical) + Cholesky shift-retry count.
//
// CLI: <prec> <outdir> <m> <n> <omega> <lambda_rel> <method_mask> <tol> <maxit>
//      <d_factor> <sketch_nnz> [seed]
//   method_mask bits: 1 CQRRT, 2 CholQR, 4 sCholQR3, 8 sCholQR3_basic, 16 CholQR2,
//                     32 Blendenpik, 64 unpreconditioned.

#include "RandLAPACK.hh"
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "rl_blendenpik.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "../../extras/linops/ext_toeplitz_linop.hh"

#include <RandBLAS.hh>
#include <blas.hh>
#include <lapack.hh>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <chrono>

using RNG = RandBLAS::DefaultRNG;
namespace rl  = RandLAPACK;
namespace rle = RandLAPACK_extras::linops;
using blas::Layout; using blas::Op; using blas::Side;
using std::chrono::steady_clock; using std::chrono::duration_cast; using std::chrono::microseconds;

static double prolate(int64_t k, double omega) {
    if (k == 0) return 2.0 * omega;
    double kk = (double)k;
    return std::sin(2.0 * M_PI * omega * kk) / (M_PI * kk);
}

// Power iteration for lambda_max(T'T): x <- T'(T x), normalize; Rayleigh quotient.
template <typename TOp>
static double power_lambda_max(TOp& T, int64_t m, int64_t n, int iters) {
    std::vector<double> x(n), Tx(m), Gx(n);
    std::mt19937 rng(12345); std::normal_distribution<double> nd(0,1);
    for (auto& v : x) v = nd(rng);
    double nrm = blas::nrm2(n, x.data(), 1); blas::scal(n, 1.0/nrm, x.data(), 1);
    double lam = 0;
    for (int it = 0; it < iters; ++it) {
        T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x.data(), n, 0.0, Tx.data(), m);
        T(Side::Left, Layout::ColMajor, Op::Trans,   Op::NoTrans, n, 1, m, 1.0, Tx.data(), m, 0.0, Gx.data(), n);
        lam = blas::dot(n, x.data(), 1, Gx.data(), 1);   // x'(T'T)x with ||x||=1
        double gn = blas::nrm2(n, Gx.data(), 1);
        if (gn == 0) break;
        blas::scal(n, 1.0/gn, Gx.data(), 1);
        std::copy(Gx.begin(), Gx.end(), x.begin());
    }
    return lam;
}

// Preconditioner quality of R via Q = A R^{-1}. Materializes Q one column-block at a
// time (mtot x n peak), one Side::Right TRSM -- the blocked pattern the FEM benchmark
// uses at ISAAC scale -- then forms the Gram G = Q^T Q (n x n) once and reads two of
// Oleg's metrics from it:
//   orth_out = ||G - I||_F / sqrt(n)                 (orthogonality loss)
//   cond_out = sqrt(lambda_max(G) / lambda_min(G))   = cond(A R^{-1})
// The cond estimate needs an n x n symmetric eigendecomposition (O(n^3)); it is skipped
// (returned as NaN) when n > cond_cap, since that eig is infeasible at ISAAC-large n.
template <typename AOp>
static void compute_orth_and_cond(AOp& A, const double* R, int64_t mtot, int64_t n,
                                  int64_t block_size, int64_t cond_cap,
                                  double& orth_out, double& cond_out) {
    int64_t b = (block_size > 0 && block_size < n) ? block_size : n;
    std::vector<double> Q(mtot * n), E(n * b);
    for (int64_t j0 = 0; j0 < n; j0 += b) {
        int64_t bk = std::min(b, n - j0);
        std::fill(E.begin(), E.begin() + n * b, 0.0);
        for (int64_t j = 0; j < bk; ++j) E[(j0 + j) + j * n] = 1.0;
        A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mtot, bk, n, 1.0, E.data(), n, 0.0, Q.data() + j0 * mtot, mtot);
    }
    blas::trsm(Layout::ColMajor, Side::Right, blas::Uplo::Upper, Op::NoTrans, blas::Diag::NonUnit,
               mtot, n, 1.0, R, n, Q.data(), mtot);   // Q = A R^{-1}

    // G = Q^T Q (upper triangle), n x n.
    std::vector<double> G(n * n, 0.0);
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, Op::Trans, n, mtot, 1.0, Q.data(), mtot, 0.0, G.data(), n);

    // orth = ||G - I||_F / sqrt(n) (from a copy; leaves G intact for the eig).
    {
        std::vector<double> GmI(G);
        for (int64_t j = 0; j < n; ++j) GmI[j + j * n] -= 1.0;
        orth_out = lapack::lansy(lapack::Norm::Fro, blas::Uplo::Upper, n, GmI.data(), n) / std::sqrt((double)n);
    }

    // cond(A R^{-1}) = sqrt(lambda_max/lambda_min) of G, only if within the size cap.
    cond_out = std::numeric_limits<double>::quiet_NaN();
    if (cond_cap <= 0 || n <= cond_cap) {
        std::vector<double> evals(n);
        int info = lapack::syevd(lapack::Job::NoVec, blas::Uplo::Upper, n, G.data(), n, evals.data());
        if (info == 0 && evals[0] > 0) cond_out = std::sqrt(evals[n - 1] / evals[0]);  // ascending
    }
}

int main(int argc, char** argv) {
    if (argc < 12) {
        std::fprintf(stderr, "usage: %s <prec> <outdir> <m> <n> <omega> <lambda_rel> <method_mask> "
                             "<tol> <maxit> <d_factor> <sketch_nnz> [seed] [qless_warm_start] [bp_warm_start]\n", argv[0]);
        return 1;
    }
    std::string prec = argv[1];          // "double" (v1)
    std::string outdir = argv[2];
    int64_t m = std::stoll(argv[3]);
    int64_t n = std::stoll(argv[4]);
    double omega = std::stod(argv[5]);
    double lambda_rel = std::stod(argv[6]);
    int64_t method_mask = std::stoll(argv[7]);
    double tol = std::stod(argv[8]);
    int maxit = std::stoi(argv[9]);
    double d_factor = std::stod(argv[10]);
    int64_t sketch_nnz = std::stoll(argv[11]);
    int64_t seed = (argc > 12) ? std::stoll(argv[12]) : 1;
    // Warm-start ablation (2026-07-31, mirrors CQRRT_linop_applications):
    //   qless_warm_start: 1 => every Q-less QR method's LSQR starts from the
    //     sketch-and-solve x0 (built with an AUXILIARY sketch via Blendenpik
    //     init_only; the method's own QR stays sketch-free). Cost recorded in
    //     the setup_us CSV column. Default 0 (historical behavior).
    //   bp_warm_start: 0 => Blendenpik's own warm start disabled. Default 1
    //     (its class default since 2026-07-27 -- which silently made the
    //     07-29 campaign MIXED: Blendenpik warm, everyone else cold).
    bool qless_warm = (argc > 13) && std::stoi(argv[13]) != 0;
    bool bp_warm    = (argc > 14) ? (std::stoi(argv[14]) != 0) : true;
    int64_t block_size = 256;
    if (m < n) { std::fprintf(stderr, "require m >= n\n"); return 1; }

    std::printf("=== Toeplitz LS benchmark: m=%lld n=%lld omega=%.4f lambda_rel=%.1e mask=%lld ===\n",
                (long long)m, (long long)n, omega, lambda_rel, (long long)method_mask);

    // 1. Prolate kernel generators c (m), r (n).
    std::vector<double> c(m), r(n);
    for (int64_t i = 0; i < m; ++i) c[i] = prolate(i, omega);
    for (int64_t j = 0; j < n; ++j) r[j] = prolate(j, omega);

    // 2. Toeplitz operator + augmented A = [T; sqrt(lambda) I].
    rle::ToeplitzLinOp<double> T(c.data(), m, r.data(), n, 8);

    double lam_max = power_lambda_max(T, m, n, 50);
    double lambda  = lambda_rel * lam_max;
    std::printf("lambda_max(T'T)~=%.6e  lambda=%.6e\n", lam_max, lambda);

    rl::linops::ScaledIdentityOp<double> regI(n, std::sqrt(lambda));
    rl::linops::VStackOp<rle::ToeplitzLinOp<double>, rl::linops::ScaledIdentityOp<double>> A_hat(T, regI);
    A_hat.block_size = block_size;
    int64_t mtot = m + n;

    // 3. Synthetic data: x_true random-smooth, b = T x_true + noise, rhs = [b; 0].
    std::mt19937 rng((unsigned)seed); std::normal_distribution<double> nd(0,1);
    std::vector<double> x_true(n);
    { std::vector<double> z(n); for (auto& v : z) v = nd(rng);
      // light smoothing (Gaussian filter, half-width 18)
      int hw = std::min<int64_t>(n-1, 80);
      std::vector<double> filt(hw+1); double fs = 0;
      for (int i = 0; i <= hw; ++i) { filt[i] = std::exp(-std::pow(i/18.0,2)); fs += (i==0?filt[i]:2*filt[i]); }
      for (int i = 0; i < n; ++i) { double acc = 0;
          for (int j = -hw; j <= hw; ++j) { int idx = i+j; if (idx>=0 && idx<n) acc += z[idx]*filt[std::abs(j)]; }
          x_true[i] = acc/fs; }
      double xn = blas::nrm2(n, x_true.data(), 1); blas::scal(n, 1.0/std::max(xn,1e-300), x_true.data(), 1);
    }
    std::vector<double> b(m);
    T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x_true.data(), n, 0.0, b.data(), m);
    double b_clean_norm = blas::nrm2(m, b.data(), 1);
    { double relnoise = 1e-11; std::vector<double> e(m); for (auto& v : e) v = nd(rng);
      double en = blas::nrm2(m, e.data(), 1);
      for (int64_t i = 0; i < m; ++i) b[i] += e[i]/std::max(en,1e-300)*relnoise*b_clean_norm; }
    double b_norm = blas::nrm2(m, b.data(), 1);
    std::vector<double> rhs(mtot, 0.0); std::copy(b.begin(), b.end(), rhs.begin());
    double rhs_norm = blas::nrm2(mtot, rhs.data(), 1);
    double x_true_norm = blas::nrm2(n, x_true.data(), 1);
    // normalRhs = A^T rhs (for normalRelres = ||A^T(Ax-rhs)|| / ||A^T rhs||).
    std::vector<double> normalRhs(n);
    A_hat(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, 1, mtot, 1.0, rhs.data(), mtot, 0.0, normalRhs.data(), n);
    double normalRhs_norm = blas::nrm2(n, normalRhs.data(), 1);
    // cond(A R^{-1}) needs an n x n eig; skip it above this size (infeasible at ISAAC-large n).
    const int64_t cond_est_max_n = 16384;   // includes small (n=8256), excludes large (n=33024)

    // 4. Method list from the mask.
    std::vector<std::string> algs;
    if (method_mask & 1)  algs.push_back("CQRRT_linop");
    if (method_mask & 2)  algs.push_back("CholQR");
    if (method_mask & 4)  algs.push_back("sCholQR3");
    if (method_mask & 8)  algs.push_back("sCholQR3_basic");
    if (method_mask & 16) algs.push_back("CholQR2");
    if (method_mask & 32) algs.push_back("Blendenpik");
    if (method_mask & 64) algs.push_back("unpreconditioned");

    // 5. CSV.
    std::string tstamp; { std::time_t t = std::time(nullptr); char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t)); tstamp = buf; }
    std::string csv = outdir + "/" + tstamp + "_toeplitz_ls_results.csv";
    std::ofstream out(csv);
    out << "# Toeplitz LS benchmark m=" << m << " n=" << n << " omega=" << omega
        << " lambda=" << lambda << " tol=" << tol << " d_factor=" << d_factor << "\n";
    out << "# qless_warm_start=" << (qless_warm ? 1 : 0) << " bp_warm_start=" << (bp_warm ? 1 : 0) << "\n";
    out << "algorithm,m,n,qr_status,qr_time_us,solve_time_us,peak_rss_kb,analytical_kb,"
           "orth_error,iterations,solver_flag,solver_relres,aug_relres,normal_relres,"
           "data_relres,recovery_error,cond_estimate,chol_retries,"
           "solve_fwd_us,solve_adj_us,solve_trsm_us,setup_us\n";   // fwd/adj/trsm 2026-07-29; setup_us 2026-07-31

    std::vector<double> R(n*n), x(n), Tx(m), Ax(mtot);

    for (const auto& alg : algs) {
        std::printf("\n=== %s ===\n", alg.c_str());
        std::fill(R.begin(), R.end(), 0.0);
        std::fill(x.begin(), x.end(), 0.0);
        auto state = RandBLAS::RNGState<RNG>((uint32_t)seed);
        int qr_status = 0, iters = 0, flag = 0, chol_retries = 0;
        long qr_us = 0, solve_us = 0, peak_kb = 0, analytical_kb = 0;
        // Solve-time DECOMPOSITION (added 2026-07-29 to localize a timing anomaly).
        // rl_lsqr already computes these into times[0..2] but the benchmark previously
        // recorded only times[3] (the total), which made the solve column impossible to
        // interpret: the 07-29 campaign showed per-iteration solve cost FALLING as the
        // circulant FFT length L grew 32768 -> 131072 -> 524288, which no cost model allows.
        //
        // Two things must be separated before any solve-time claim is trustworthy:
        //   * operator cost (t_fwd + t_adj) vs preconditioner cost (t_trsm) vs LSQR's own
        //     vector work and one-time setup, which is total - (fwd + adj + trsm);
        //   * fixed setup vs per-iteration cost. Dividing the TOTAL by the iteration count
        //     is invalid at 4 iterations (setup dominates) yet fine at 471, which by itself
        //     manufactures a large apparent gap between preconditioned and unpreconditioned
        //     runs. Reporting the parts removes that trap.
        // -1 means "not measured for this method" (Blendenpik exposes only a lumped solve).
        long solve_fwd_us = -1, solve_adj_us = -1, solve_trsm_us = -1;
        long setup_us = 0;   // warm-start x0 build; 0 = cold (or embedded, for warm Blendenpik)
        double solver_relres = -1;   // LSQR's own ||b - Ã y|| / ||b|| at termination
        bool have_R = false, is_bp = (alg == "Blendenpik"), is_unprec = (alg == "unpreconditioned");

        rl::PeakRSSTracker mem; mem.start();
        auto t0 = steady_clock::now();

        if (is_bp) {
            rl::Blendenpik_linops<double, RNG> bp(true, tol);
            bp.nnz = sketch_nnz;
            bp.warm_start = bp_warm;
            // Give Blendenpik the SAME iteration budget as every other method. It
            // previously fell back to its internal min(4n,1000) default while the others
            // received the CLI maxit (3000 in the campaigns) -- a silent 3x handicap.
            bp.max_iters = maxit;
            qr_status = bp.call(A_hat, rhs.data(), mtot, x.data(), n, d_factor, state);
            peak_kb = mem.stop();
            if (qr_status == 0) { qr_us = bp.times[0] + bp.times[1]; solve_us = bp.times[2];
                iters = bp.lsqr_iters; std::copy(bp.R_out.begin(), bp.R_out.end(), R.begin()); have_R = true;
                // Report the same convergence signals as the other methods: flag 0/1 for
                // met-tolerance / hit-cap, and a real solver residual instead of -1.
                flag = bp.converged ? 0 : 1;
                solver_relres = bp.final_relres; }
        } else if (is_unprec) {
            long lt[4] = {0};
            flag = rl::lsqr<double>(A_hat, mtot, n, nullptr, 0, rhs.data(), x.data(), tol, tol, maxit, iters, lt, &solver_relres);
            solve_us = lt[3]; peak_kb = mem.stop();
            solve_fwd_us = lt[0]; solve_adj_us = lt[1]; solve_trsm_us = lt[2];  // t_trsm is 0 here (no R)
        } else {
            // Build R via the selected Q-less QR method, then shared LSQR with right precond R.
            if (alg == "CholQR") {
                rl::CholQR_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::cholqr_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "CholQR2") {
                rl::CholQR2_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::cholqr2_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "sCholQR3") {
                rl::sCholQR3_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::scholqr3_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "sCholQR3_basic") {
                rl::sCholQR3_linops_basic<double> qr(true, tol);
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::scholqr3_linops_basic_analytical_kb<double>(mtot, n); }
            } else { // CQRRT_linop
                rl::CQRRT_linops<double, RNG> qr(true, tol);
                qr.nnz = sketch_nnz; qr.block_size = block_size;
                qr.precond_method = rl::CQRRTLinopPrecond::TRSM_IDENTITY;
                qr_status = qr.call(A_hat, R.data(), n, d_factor, state);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::cqrrt_linops_analytical_kb<double>(mtot, n, d_factor, block_size); }
            }
            if (qr_status == 0) {
                have_R = true;
                long lt[4] = {0};
                if (qless_warm) {
                    // Sketch-and-solve x0 via an auxiliary sketch (Blendenpik init_only,
                    // fresh deterministic state), then the shift trick around rl_lsqr:
                    // solve for the correction dx against r0 = rhs - A_hat x0 and return
                    // x = x0 + dx, exactly as rl_blendenpik.hh does internally.
                    auto ws_t0 = steady_clock::now();
                    std::vector<double> x0(n, 0.0), r0(mtot, 0.0);
                    auto ws_state = RandBLAS::RNGState<RNG>((uint32_t)(seed + 7919));
                    rl::Blendenpik_linops<double, RNG> ss(false, tol);
                    ss.nnz = sketch_nnz; ss.init_only = true;
                    int ss_status = ss.call(A_hat, rhs.data(), mtot, x0.data(), n, d_factor, ws_state);
                    if (ss_status != 0) {
                        std::fprintf(stderr, "warning: sketch-and-solve x0 failed (%d); falling back to x0=0\n", ss_status);
                        std::fill(x0.begin(), x0.end(), 0.0);
                    }
                    A_hat(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mtot, 1, n, 1.0, x0.data(), n, 0.0, r0.data(), mtot);
                    for (int64_t i = 0; i < mtot; ++i) r0[i] = rhs[i] - r0[i];
                    setup_us = duration_cast<microseconds>(steady_clock::now() - ws_t0).count();
                    flag = rl::lsqr<double>(A_hat, mtot, n, R.data(), n, r0.data(), x.data(), tol, tol, maxit, iters, lt, &solver_relres);
                    blas::axpy(n, 1.0, x0.data(), 1, x.data(), 1);
                    // LSQR normalized its residual by ||r0||; rescale to mean the same
                    // thing as every other row (||rhs - A x|| / ||rhs||).
                    if (solver_relres >= 0) {
                        double nr0 = blas::nrm2(mtot, r0.data(), 1);
                        if (rhs_norm > 0) solver_relres *= nr0 / rhs_norm;
                    }
                } else {
                    flag = rl::lsqr<double>(A_hat, mtot, n, R.data(), n, rhs.data(), x.data(), tol, tol, maxit, iters, lt, &solver_relres);
                }
                solve_us = lt[3];
                solve_fwd_us = lt[0]; solve_adj_us = lt[1]; solve_trsm_us = lt[2];
            }
            peak_kb = mem.stop();
        }
        (void)duration_cast<microseconds>(steady_clock::now() - t0).count();

        // Metrics (Oleg's set: solver/data/aug/normal relres, recovery, orth, cond).
        double orth_err = -1, aug_relres = -1, data_relres = -1, normal_relres = -1, recov = -1;
        double cond_est = std::numeric_limits<double>::quiet_NaN();
        if (qr_status == 0) {
            if (have_R) compute_orth_and_cond(A_hat, R.data(), mtot, n, block_size, cond_est_max_n, orth_err, cond_est);
            // augResidual = A_hat x - rhs  (length mtot).
            A_hat(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mtot, 1, n, 1.0, x.data(), n, 0.0, Ax.data(), mtot);
            double ar = 0; for (int64_t i = 0; i < mtot; ++i) { Ax[i] -= rhs[i]; ar += Ax[i]*Ax[i]; }
            aug_relres = std::sqrt(ar) / std::max(rhs_norm, 1e-300);
            // normalResidual = A_hat^T (A_hat x - rhs); normalRelres = ||.|| / ||A^T rhs||.
            std::vector<double> nres(n);
            A_hat(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, 1, mtot, 1.0, Ax.data(), mtot, 0.0, nres.data(), n);
            normal_relres = blas::nrm2(n, nres.data(), 1) / std::max(normalRhs_norm, 1e-300);
            T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x.data(), n, 0.0, Tx.data(), m);
            double dr = 0; for (int64_t i = 0; i < m; ++i) { double d = Tx[i] - b[i]; dr += d*d; }
            data_relres = std::sqrt(dr) / std::max(b_norm, 1e-300);
            double re = 0; for (int64_t i = 0; i < n; ++i) { double d = x[i] - x_true[i]; re += d*d; }
            recov = std::sqrt(re) / std::max(x_true_norm, 1e-300);
        }

        std::printf("  qr_status=%d iters=%d flag=%d qr_us=%ld solve_us=%ld peak_kb=%ld\n",
                    qr_status, iters, flag, qr_us, solve_us, peak_kb);
        // Solve breakdown, also echoed to the job log so the anomaly is visible without
        // pulling the CSV. `other` = LSQR's own vector work + one-time setup.
        if (solve_fwd_us >= 0) {
            long other = solve_us - (solve_fwd_us + solve_adj_us + solve_trsm_us);
            std::printf("  solve breakdown: fwd=%ld adj=%ld trsm=%ld other=%ld us"
                        "  (per-iter over %d iters: fwd=%.3f adj=%.3f trsm=%.3f ms)\n",
                        solve_fwd_us, solve_adj_us, solve_trsm_us, other, iters,
                        iters > 0 ? solve_fwd_us  / 1000.0 / iters : 0.0,
                        iters > 0 ? solve_adj_us  / 1000.0 / iters : 0.0,
                        iters > 0 ? solve_trsm_us / 1000.0 / iters : 0.0);
        }
        std::printf("  orth=%.3e solver_relres=%.3e aug=%.3e normal=%.3e data=%.3e recovery=%.3e cond=%.3e retries=%d\n",
                    orth_err, solver_relres, aug_relres, normal_relres, data_relres, recov, cond_est, chol_retries);

        out << alg << "," << m << "," << n << "," << qr_status << "," << qr_us << "," << solve_us << ","
            << peak_kb << "," << analytical_kb << ","
            << std::scientific << orth_err << "," << iters << "," << flag << ","
            << solver_relres << "," << aug_relres << "," << normal_relres << ","
            << data_relres << "," << recov << "," << cond_est << "," << chol_retries << ","
            << solve_fwd_us << "," << solve_adj_us << "," << solve_trsm_us << "," << setup_us << "\n";
    }
    out.close();
    std::printf("\nresults -> %s\n", csv.c_str());
    return 0;
}
