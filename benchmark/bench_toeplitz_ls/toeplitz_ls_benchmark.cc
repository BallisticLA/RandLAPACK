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

// Q-factor orthogonality loss ||Q^T Q - I||_F / sqrt(n), Q = A R^{-1}. Materializes Q
// one column-block at a time (mtot x n peak, not mtot x n x extra), then a single
// Side::Right TRSM -- the same blocked pattern the FEM benchmark uses at ISAAC scale.
template <typename AOp>
static double compute_orth_error(AOp& A, const double* R, int64_t mtot, int64_t n, int64_t block_size) {
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
    return RandLAPACK::testing::orthogonality_error<double>(Q.data(), mtot, n);
}

int main(int argc, char** argv) {
    if (argc < 12) {
        std::fprintf(stderr, "usage: %s <prec> <outdir> <m> <n> <omega> <lambda_rel> <method_mask> "
                             "<tol> <maxit> <d_factor> <sketch_nnz> [seed]\n", argv[0]);
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
    out << "algorithm,m,n,qr_status,qr_time_us,solve_time_us,peak_rss_kb,analytical_kb,"
           "orth_error,iterations,solver_flag,aug_relres,data_relres,recovery_error,chol_retries\n";

    std::vector<double> R(n*n), x(n), Tx(m), Ax(mtot);

    for (const auto& alg : algs) {
        std::printf("\n=== %s ===\n", alg.c_str());
        std::fill(R.begin(), R.end(), 0.0);
        std::fill(x.begin(), x.end(), 0.0);
        auto state = RandBLAS::RNGState<RNG>((uint32_t)seed);
        int qr_status = 0, iters = 0, flag = 0, chol_retries = 0;
        long qr_us = 0, solve_us = 0, peak_kb = 0, analytical_kb = 0;
        bool have_R = false, is_bp = (alg == "Blendenpik"), is_unprec = (alg == "unpreconditioned");

        rl::PeakRSSTracker mem; mem.start();
        auto t0 = steady_clock::now();

        if (is_bp) {
            rl::Blendenpik_linops<double, RNG> bp(true, tol);
            bp.nnz = sketch_nnz;
            qr_status = bp.call(A_hat, rhs.data(), mtot, x.data(), n, d_factor, state);
            peak_kb = mem.stop();
            if (qr_status == 0) { qr_us = bp.times[0] + bp.times[1]; solve_us = bp.times[2];
                iters = bp.lsqr_iters; std::copy(bp.R_out.begin(), bp.R_out.end(), R.begin()); have_R = true; }
        } else if (is_unprec) {
            long lt[4] = {0};
            flag = rl::lsqr<double>(A_hat, mtot, n, nullptr, 0, rhs.data(), x.data(), tol, tol, maxit, iters, lt);
            solve_us = lt[3]; peak_kb = mem.stop();
        } else {
            // Build R via the selected Q-less QR method, then shared LSQR with right precond R.
            if (alg == "CholQR") {
                rl::CholQR_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.times[5]; chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::cholqr_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "CholQR2") {
                rl::CholQR2_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.times[10]; chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::cholqr2_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "sCholQR3") {
                rl::sCholQR3_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.times[17]; chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::scholqr3_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "sCholQR3_basic") {
                rl::sCholQR3_linops_basic<double> qr(true, tol);
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.times[14]; chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::scholqr3_linops_basic_analytical_kb<double>(mtot, n); }
            } else { // CQRRT_linop
                rl::CQRRT_linops<double, RNG> qr(true, tol);
                qr.nnz = sketch_nnz; qr.block_size = block_size;
                qr.precond_method = rl::CQRRTLinopPrecond::TRSM_IDENTITY;
                qr_status = qr.call(A_hat, R.data(), n, d_factor, state);
                if (qr_status == 0) { qr_us = qr.times[10]; chol_retries = qr.n_chol_retries;
                    analytical_kb = rl::cqrrt_linops_analytical_kb<double>(mtot, n, d_factor, block_size); }
            }
            if (qr_status == 0) {
                have_R = true;
                long lt[4] = {0};
                flag = rl::lsqr<double>(A_hat, mtot, n, R.data(), n, rhs.data(), x.data(), tol, tol, maxit, iters, lt);
                solve_us = lt[3];
            }
            peak_kb = mem.stop();
        }
        (void)duration_cast<microseconds>(steady_clock::now() - t0).count();

        // Metrics.
        double orth_err = -1, aug_relres = -1, data_relres = -1, recov = -1;
        if (qr_status == 0) {
            if (have_R) orth_err = compute_orth_error(A_hat, R.data(), mtot, n, block_size);
            A_hat(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mtot, 1, n, 1.0, x.data(), n, 0.0, Ax.data(), mtot);
            double ar = 0; for (int64_t i = 0; i < mtot; ++i) { double d = Ax[i] - rhs[i]; ar += d*d; }
            aug_relres = std::sqrt(ar) / std::max(rhs_norm, 1e-300);
            T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x.data(), n, 0.0, Tx.data(), m);
            double dr = 0; for (int64_t i = 0; i < m; ++i) { double d = Tx[i] - b[i]; dr += d*d; }
            data_relres = std::sqrt(dr) / std::max(b_norm, 1e-300);
            double re = 0; for (int64_t i = 0; i < n; ++i) { double d = x[i] - x_true[i]; re += d*d; }
            recov = std::sqrt(re) / std::max(x_true_norm, 1e-300);
        }

        std::printf("  qr_status=%d iters=%d flag=%d qr_us=%ld solve_us=%ld peak_kb=%ld\n",
                    qr_status, iters, flag, qr_us, solve_us, peak_kb);
        std::printf("  orth=%.3e aug_relres=%.3e data_relres=%.3e recovery=%.3e retries=%d\n",
                    orth_err, aug_relres, data_relres, recov, chol_retries);

        out << alg << "," << m << "," << n << "," << qr_status << "," << qr_us << "," << solve_us << ","
            << peak_kb << "," << analytical_kb << ","
            << std::scientific << orth_err << "," << iters << "," << flag << ","
            << aug_relres << "," << data_relres << "," << recov << "," << chol_retries << "\n";
    }
    out.close();
    std::printf("\nresults -> %s\n", csv.c_str());
    return 0;
}
