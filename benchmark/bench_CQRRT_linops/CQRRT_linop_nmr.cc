// 2D NMR Relaxometry Benchmark — Q-less QR + iterative-refinement LSQ.
//
// Test problem: PRnmr from the IR Tools toolbox (Gazzola, Hansen, Nagy 2018).
// Forward operator A = A2 ⊗ A1 with separable Laplace-type kernel:
//
//     (A1)_{l,k} = 1 - 2 * exp(-tau1[l] / T1[k])     (m1 × n1)
//     (A2)_{l,k} =     exp(-tau2[l] / T2[k])         (m2 × n2)
//
// Time grids are logspace(log_lo, log_hi). Default range here is (-2, 1) =
// 3 decades; the IR Tools paper uses (-4, 1) = 5 decades, but unregularized
// Q-less QR cannot handle that conditioning even with stabilized
// preconditioners. Operator dimensions:
//     M = m1*m2 (rows), N = n1*n2 (cols), default m_i = 2*n_i.
//
// Pipeline:
//     1. Generate A1, A2, phantom x_true ∈ ℝ^{n1*n2}, b = A x_true + noise.
//     2. Wrap A as KroneckerOperator (no materialization). When λ > 0, also
//        wrap that in RegularizedLinOp so the QR step runs on A_aug = [J; λI].
//     3. For each Q-less QR variant selected by `method_mask`, run it on the
//        operator → R (n × n upper triangular).
//     4. For each (algorithm, run): run IterRefineLSQ(A_eff, R, b_eff) → x.
//     5. Record per-(algorithm, run) ||x - x_true||/||x_true||, ||b - Ax||/||b||,
//        IR iter counts, timing, and analytical-memory predictions.
//
// Usage:
//     ./CQRRT_linop_nmr <prec> <output_dir> <num_runs> <n>
//                       [phantom] [noise_level] [d_factor] [sketch_nnz] [block_size]
//                       [log_lo] [log_hi] [method_mask] [lambda]
// where:
//     prec        = "double" | "float"
//     n           = scalar; sets n1 = n2 = n, m1 = m2 = 2*n
//     phantom     = "two_blob" (default) | "one_blob"
//     noise_level = ||noise||/||b|| (default 0.05)
//     method_mask = bitmask selecting Q-less QR variants to run, ALL piped
//                   through the IR-LSQ pipeline. Default 0b1101111 (all
//                   linop variants; CQRRT_expl excluded since materializing
//                   the Kronecker would defeat the purpose).
//                     bit 0 ( 1): CQRRT_linop (TRSM_IDENTITY)
//                     bit 1 ( 2): CholQR
//                     bit 2 ( 4): sCholQR3
//                     bit 3 ( 8): sCholQR3_basic
//                     bit 5 ( 32): CQRRT_linop_stb         (GEQP3)
//                     bit 6 ( 64): CQRRT_linop_stb_bqrrp   (BQRRP)
//     lambda      = Tikhonov regularization. min ||Jx-b||² + λ²||x||².
//                   When λ>0 the QR is run on the augmented operator
//                   A_aug = [J; λI] so it succeeds even when J alone is
//                   too ill-conditioned for Q-less QR.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;


// =============================================================================
// Grid generation
// =============================================================================
// logspace(a, b, n) -> n points evenly spaced in log10 from 10^a to 10^b.
template <typename T>
static std::vector<T> logspace(T a, T b, int64_t n) {
    std::vector<T> v(n);
    if (n == 1) { v[0] = std::pow((T)10, a); return v; }
    T step = (b - a) / (T)(n - 1);
    for (int64_t i = 0; i < n; ++i) v[i] = std::pow((T)10, a + step * (T)i);
    return v;
}


// =============================================================================
// Build A1 and A2 per the PRnmr formulas.
// =============================================================================
template <typename T>
static void build_A_factors(int64_t m1, int64_t n1, int64_t m2, int64_t n2,
                            T log_lo, T log_hi,
                            std::vector<T>& A1, std::vector<T>& A2)
{
    auto T1   = logspace<T>(log_lo, log_hi, n1);
    auto T2   = logspace<T>(log_lo, log_hi, n2);
    auto tau1 = logspace<T>(log_lo, log_hi, m1);
    auto tau2 = logspace<T>(log_lo, log_hi, m2);

    A1.assign(m1 * n1, (T)0);
    A2.assign(m2 * n2, (T)0);
    // ColMajor: A1[i + j*m1] = (A1)_{i,j} = 1 - 2*exp(-tau1[i] / T1[j])
    for (int64_t j = 0; j < n1; ++j)
        for (int64_t i = 0; i < m1; ++i)
            A1[i + j * m1] = (T)1.0 - (T)2.0 * std::exp(-tau1[i] / T1[j]);
    for (int64_t j = 0; j < n2; ++j)
        for (int64_t i = 0; i < m2; ++i)
            A2[i + j * m2] = std::exp(-tau2[i] / T2[j]);
}


// =============================================================================
// Phantom generation: synthetic 2D T1-T2 distribution.
// =============================================================================
// "two_blob": sum of two 2D Gaussians in (log T1, log T2) space — coarse
//             approximation of multi-population NMR phantoms (carbonate / methane).
// "one_blob": single Gaussian (simpler reference).
template <typename T>
static std::vector<T> make_phantom(int64_t n1, int64_t n2,
                                    T log_lo, T log_hi,
                                    const std::string& kind)
{
    std::vector<T> x(n1 * n2, (T)0);
    auto T1 = logspace<T>(log_lo, log_hi, n1);
    auto T2 = logspace<T>(log_lo, log_hi, n2);

    auto add_gaussian = [&](T mu1, T mu2, T sigma1, T sigma2, T amp) {
        for (int64_t j2 = 0; j2 < n2; ++j2) {
            T lt2 = std::log10(T2[j2]);
            for (int64_t j1 = 0; j1 < n1; ++j1) {
                T lt1 = std::log10(T1[j1]);
                T d1 = (lt1 - mu1) / sigma1;
                T d2 = (lt2 - mu2) / sigma2;
                x[j1 + j2 * n1] += amp * std::exp(-(T)0.5 * (d1 * d1 + d2 * d2));
            }
        }
    };

    if (kind == "one_blob") {
        add_gaussian((T)-1.0, (T)-1.0, (T)0.5, (T)0.5, (T)1.0);
    } else {
        // "two_blob" (default)
        add_gaussian((T)-2.5, (T)-2.5, (T)0.4, (T)0.4, (T)1.0);  // short relaxation
        add_gaussian((T) 0.0, (T)-0.5, (T)0.5, (T)0.5, (T)0.7);  // long relaxation
    }
    return x;
}


// =============================================================================
// Result struct
// =============================================================================
template <typename T>
struct nmr_result {
    int64_t m, n, m1, n1, m2, n2;
    int64_t run_idx;
    std::string alg_name;     // "CQRRT_linop", "CholQR", "sCholQR3", "sCholQR3_basic", "CQRRT_linop_stb", "CQRRT_linop_stb_bqrrp"
    std::string phantom;
    T noise_level;
    int qr_status;            // 0 on success; nonzero indicates QR breakdown (no IR-LSQ run)

    // Q-less QR (CQRRT_linop)
    long qr_time_us;
    T orth_error;       // ||Q^T Q - I||_F / sqrt(n) (only if cheap to check)
    long peak_rss_kb;
    long analytical_kb;

    // IR LSQ
    long ir_total_us;
    int  ir_outer_iters;
    int  ir_inner_iters_total;
    T    ls_residual_norm;     // ||b - A x|| / ||b||
    T    ls_solution_error;    // ||x - x_true|| / ||x_true||

    // QR breakdown (CQRRT_linop times: 11 entries)
    std::vector<long> qr_breakdown;
    // IR breakdown (6 entries: outer, inner, trsm, fwd, adj, other)
    std::vector<long> ir_breakdown;
};


template <typename T>
static void print_summary(const nmr_result<T>& r) {
    std::printf("\n  [%s] Run %ld (phantom=%s, noise=%.3f):\n",
                r.alg_name.c_str(), (long)r.run_idx, r.phantom.c_str(), (double)r.noise_level);
    if (r.qr_status != 0) {
        std::printf("    QR returned status %d — IR-LSQ skipped.\n", r.qr_status);
        return;
    }
    std::printf("    QR: %ld us, peak_RSS=%ld KB, predicted=%ld KB\n",
                r.qr_time_us, r.peak_rss_kb, r.analytical_kb);
    if (r.orth_error >= 0) std::printf("    orth_err = %.3e\n", (double)r.orth_error);
    std::printf("    IR-LSQ: total=%ld us, outer=%d, inner_total=%d\n",
                r.ir_total_us, r.ir_outer_iters, r.ir_inner_iters_total);
    std::printf("    ||r||/||b||  = %.3e\n", (double)r.ls_residual_norm);
    std::printf("    ||x-x_true||/||x_true|| = %.3e\n", (double)r.ls_solution_error);
}


// =============================================================================
// Core templated runner
// =============================================================================
template <typename T, typename RNG = r123::Philox4x32>
int run_benchmark(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_runs> <n>"
                  << " [phantom] [noise_level] [d_factor] [sketch_nnz] [block_size]"
                  << " [log_lo] [log_hi] [method_mask] [lambda]\n"
                  << "  method_mask: bitmask of Q-less QR variants (default = 0b1101111)\n"
                  << "    bit 0 ( 1): CQRRT_linop (TRSM_IDENTITY)\n"
                  << "    bit 1 ( 2): CholQR\n"
                  << "    bit 2 ( 4): sCholQR3\n"
                  << "    bit 3 ( 8): sCholQR3_basic\n"
                  << "    bit 5 ( 32): CQRRT_linop_stb         (GEQP3)\n"
                  << "    bit 6 ( 64): CQRRT_linop_stb_bqrrp   (BQRRP)\n"
                  << "  lambda:    Tikhonov regularization. min ||Jx-b||² + λ²||x||²; default 0.0\n"
                  << "  log_lo, log_hi: τ and T grids on logspace(log_lo, log_hi).\n"
                  << "    Default (-2, 1) = 3 decades. The IR Tools paper uses (-4, 1) = 5 decades,\n"
                  << "    but unregularized Q-less QR (any preconditioner) cannot handle that conditioning.\n";
        return 1;
    }

    std::string output_dir = argv[2];
    int64_t num_runs       = std::stol(argv[3]);
    int64_t n              = std::stol(argv[4]);
    std::string phantom    = (argc >= 6) ? std::string(argv[5]) : std::string("two_blob");
    T noise_level          = (argc >= 7) ? (T)std::stod(argv[6]) : (T)0.05;
    T d_factor             = (argc >= 8) ? (T)std::stod(argv[7]) : (T)2.0;
    int64_t sketch_nnz     = (argc >= 9) ? std::stol(argv[8])    : 4;
    int64_t block_size     = (argc >= 10) ? std::stol(argv[9])   : 0;
    T log_lo               = (argc >= 11) ? (T)std::stod(argv[10]) : (T)-2.0;
    T log_hi               = (argc >= 12) ? (T)std::stod(argv[11]) : (T)1.0;
    int64_t method_mask    = (argc >= 13) ? std::stol(argv[12]) : 0b1101111;
    T lambda_reg           = (argc >= 14) ? (T)std::stod(argv[13]) : (T)0.0;

    int64_t n1 = n, n2 = n;
    int64_t m1 = 2 * n1, m2 = 2 * n2;
    int64_t M = m1 * m2;
    int64_t N = n1 * n2;

    std::cout << "=== NMR Relaxometry Benchmark (Kronecker + CQRRT + IR-LSQ) ===\n"
              << "  m1, n1, m2, n2 = " << m1 << ", " << n1 << ", " << m2 << ", " << n2 << "\n"
              << "  M (=m1*m2) = " << M << ",  N (=n1*n2) = " << N
              << "  (aspect M:N = " << (double)M / (double)N << ":1)\n"
              << "  phantom    = " << phantom << "\n"
              << "  noise_lvl  = " << noise_level << "\n"
              << "  method_mask= " << method_mask
              << " (linop=" << (method_mask&1)
              << " CholQR=" << ((method_mask>>1)&1)
              << " sCholQR3=" << ((method_mask>>2)&1)
              << " sCholQR3_basic=" << ((method_mask>>3)&1)
              << " linop_stb=" << ((method_mask>>5)&1)
              << " linop_stb_bqrrp=" << ((method_mask>>6)&1) << ")\n"
              << "  lambda     = " << lambda_reg << "\n"
              << "  d_factor   = " << d_factor << "\n"
              << "  sketch_nnz = " << sketch_nnz << "\n"
              << "  block_size = " << block_size << "\n"
              << "  num_runs   = " << num_runs << "\n"
#ifdef _OPENMP
              << "  OpenMP threads: " << omp_get_max_threads() << "\n";
#else
              << "  OpenMP threads: 1\n";
#endif

    if (M < N) {
        std::cerr << "Error: NMR setup requires M >= N (got M=" << M << ", N=" << N << ")\n";
        return 1;
    }

    std::cout << "  log range = [10^" << log_lo << ", 10^" << log_hi << "] ("
              << (log_hi - log_lo) << " decades)\n";

    // -------- Build A1, A2, KroneckerOperator --------
    std::vector<T> A1, A2;
    std::cout << "Building A1, A2 (logspace formulas)... " << std::flush;
    auto build_t0 = steady_clock::now();
    build_A_factors(m1, n1, m2, n2, log_lo, log_hi, A1, A2);
    auto build_t1 = steady_clock::now();
    std::cout << "done (" << duration_cast<microseconds>(build_t1 - build_t0).count() << " us)\n";

    RandLAPACK::linops::KroneckerOperator<T> J(m1, n1, m2, n2, A1.data(), A2.data());
    std::cout << "Kronecker operator J = A2 ⊗ A1: " << M << " × " << N << "\n";

    // When lambda > 0, the QR step runs on the augmented operator
    //   A_aug = [J; λI]  (M+N rows × N cols),
    // which is positive-definite-Gram by construction (σ_min ≥ λ). This
    // bypasses the Cholesky-breakdown problem that hits unregularized Q-less
    // QR on ill-conditioned operators (e.g., NMR Fredholm kernels).
    // Solving min ||A_aug x − [b; 0]||² is mathematically the Tikhonov
    // regularized problem  min ||Jx − b||² + λ²||x||².
    const bool use_augmented = (lambda_reg > (T)0);
    RandLAPACK::linops::RegularizedLinOp<RandLAPACK::linops::KroneckerOperator<T>>
        J_aug(J, lambda_reg);
    if (use_augmented) {
        std::cout << "Augmented operator A_aug = [J; λI]: " << (M + N) << " × " << N
                  << "  (λ = " << lambda_reg << ")\n";
    }

    // -------- Phantom + RHS --------
    std::vector<T> x_true = make_phantom<T>(n1, n2, log_lo, log_hi, phantom);
    T x_true_norm = blas::nrm2(N, x_true.data(), 1);
    if (x_true_norm == 0) { std::cerr << "Phantom is zero — aborting\n"; return 1; }

    // b = A * x_true (clean)
    std::vector<T> b_clean(M, (T)0), b(M, (T)0), noise_vec(M, (T)0);
    J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      M, 1, N, (T)1.0, x_true.data(), N, (T)0.0, b_clean.data(), M);
    T b_clean_norm = blas::nrm2(M, b_clean.data(), 1);

    // Add Gaussian noise scaled to give ||noise||/||b_clean|| = noise_level.
    {
        std::mt19937 noise_rng(13);
        std::normal_distribution<T> N01(0, 1);
        for (auto& v : noise_vec) v = N01(noise_rng);
        T raw_noise_norm = blas::nrm2(M, noise_vec.data(), 1);
        T scale = noise_level * b_clean_norm / raw_noise_norm;
        for (int64_t i = 0; i < M; ++i) b[i] = b_clean[i] + scale * noise_vec[i];
    }
    T b_norm = blas::nrm2(M, b.data(), 1);
    std::cout << "Synthetic LS problem: ||x_true|| = " << x_true_norm
              << ",  ||b|| = " << b_norm << "\n\n";

    // Augmented RHS  b_aug = [b; 0]  (length M+N), used when lambda > 0.
    std::vector<T> b_aug;
    if (use_augmented) {
        b_aug.assign(M + N, (T)0);
        std::copy(b.begin(), b.end(), b_aug.begin());
    }

    // -------- RNG states for runs --------
    RandBLAS::RNGState<RNG> main_state(123);
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        run_states[r] = main_state;
        if (r > 0) run_states[r].key.incr(r);
    }
    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);

    // -------- Warmup --------
    std::cout << "Running warmup... " << std::flush;
    {
        auto warm_state = run_states[0];
        std::vector<T> R_warm(N * N, (T)0);
        RandLAPACK::CQRRT_linops<T, RNG> warm_algo(false, tol, false);
        warm_algo.nnz = sketch_nnz;
        warm_algo.block_size = block_size;
        warm_algo.call(J, R_warm.data(), N, d_factor, warm_state);
    }
    std::cout << "done\n\n";

    // -------- Per-(algorithm, run) lambda --------
    // Same body for J (unregularized) or J_aug (regularized) via the JLO template.
    // m_eff = number of rows of the operator (M for J, M+N for J_aug).
    // b_eff = synthetic RHS of length m_eff.
    // ir_lambda = regularization strength to pass to IterRefineLSQ. When the QR
    //   ran on J_aug we set this to 0 (the regularization is built into A_aug);
    //   when the QR ran on J alone we pass lambda_reg through.
    auto run_one = [&]<typename JLO>(const std::string& alg_name,
                                      JLO& J_eff, const T* b_eff, int64_t m_eff,
                                      T ir_lambda, int64_t r) -> nmr_result<T>
    {
        nmr_result<T> res{};
        res.m = M; res.n = N;
        res.m1 = m1; res.n1 = n1; res.m2 = m2; res.n2 = n2;
        res.run_idx = r;
        res.alg_name = alg_name;
        res.phantom = phantom;
        res.noise_level = noise_level;
        res.qr_status = 0;

        // ---- Q-less QR on J_eff: dispatch on alg_name ----
        std::cout << "[Run " << r << ", " << alg_name << "] QR ... " << std::flush;
        std::vector<T> R(N * N, (T)0);
        auto state = run_states[r];

        RandLAPACK::PeakRSSTracker mem; mem.start();
        if (alg_name == "sCholQR3") {
            // Fully-blocked shifted Cholesky-QR-3. Times has 18 entries; total at [17].
            RandLAPACK::sCholQR3_linops<T> qr_algo(/*time_subroutines=*/true, tol);
            qr_algo.block_size = block_size;
            res.qr_status = qr_algo.call(J_eff, R.data(), N);
            res.peak_rss_kb = mem.stop();
            if (res.qr_status == 0) {
                res.qr_time_us  = qr_algo.times[17];
                // First 11 of 18 entries are kept; full breakdown table is elsewhere.
                res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m_eff, N, block_size);
            }
        } else if (alg_name == "sCholQR3_basic") {
            // Non-blocked variant (matches the textbook sCholQR3 pseudocode).
            // Times has 15 entries; total at [14].
            RandLAPACK::sCholQR3_linops_basic<T> qr_algo(/*time_subroutines=*/true, tol);
            res.qr_status = qr_algo.call(J_eff, R.data(), N);
            res.peak_rss_kb = mem.stop();
            if (res.qr_status == 0) {
                res.qr_time_us  = qr_algo.times[14];
                res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<T>(m_eff, N);
            }
        } else if (alg_name == "CholQR") {
            RandLAPACK::CholQR_linops<T> qr_algo(/*time_subroutines=*/true, tol);
            qr_algo.block_size = block_size;
            res.qr_status = qr_algo.call(J_eff, R.data(), N);
            res.peak_rss_kb = mem.stop();
            if (res.qr_status == 0) {
                res.qr_time_us = qr_algo.times[5];
                res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 6);
                res.qr_breakdown.resize(11, 0);  // pad to 11 for breakdown CSV uniformity
                res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m_eff, N, block_size);
            }
        } else {
            // CQRRT_linops with selectable preconditioner.
            RandLAPACK::CQRRT_linops<T, RNG> qr_algo(/*time_subroutines=*/true, tol);
            qr_algo.nnz = sketch_nnz;
            qr_algo.block_size = block_size;
            if      (alg_name == "CQRRT_linop")           qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::TRSM_IDENTITY;
            else if (alg_name == "CQRRT_linop_stb")       qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::GEQP3;
            else /* CQRRT_linop_stb_bqrrp */              qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::BQRRP;
            res.qr_status = qr_algo.call(J_eff, R.data(), N, d_factor, state);
            res.peak_rss_kb = mem.stop();
            if (res.qr_status == 0) {
                res.qr_time_us = qr_algo.times[10];
                res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                res.analytical_kb = (alg_name == "CQRRT_linop_stb_bqrrp")
                    ? RandLAPACK::cqrrt_linops_bqrrp_analytical_kb<T>(N, d_factor)
                    : RandLAPACK::cqrrt_linops_analytical_kb<T>(m_eff, N, d_factor, block_size);
            }
        }

        if (res.qr_status != 0) {
            std::cerr << "\n  [" << alg_name << "] Run " << r
                      << ": QR returned status " << res.qr_status
                      << " (likely Cholesky breakdown).\n"
                      << "  Try a different algorithm, larger d_factor, or increase lambda.\n";
            res.qr_time_us = -1;
            res.qr_breakdown.assign(11, 0);
            res.analytical_kb = 0;
            res.orth_error = (T)-1.0;
            res.ir_total_us = 0;
            res.ir_outer_iters = 0;
            res.ir_inner_iters_total = 0;
            res.ls_residual_norm = (T)-1.0;
            res.ls_solution_error = (T)-1.0;
            print_summary(res);
            return res;
        }
        res.orth_error = (T)-1.0;
        std::cout << "done (" << res.qr_time_us << " us). IR-LSQ ... " << std::flush;

        // ---- IterRefineLSQ ----
        std::vector<T> x_ls(N, (T)0);
        RandLAPACK::IterRefineLSQ<T> ir(/*tol=*/tol,
                                        /*max_inner=*/200,
                                        /*n_steps=*/2,
                                        /*timing=*/true,
                                        /*verbose=*/false,
                                        /*lambda=*/ir_lambda);
        auto ls_t0 = steady_clock::now();
        int ir_status = ir.call(J_eff, R.data(), N, b_eff, m_eff, x_ls.data(), N);
        auto ls_t1 = steady_clock::now();
        if (ir_status != 0) {
            std::cerr << "Warning: IterRefineLSQ status " << ir_status << " (CG breakdown)\n";
        }

        res.ir_total_us = duration_cast<microseconds>(ls_t1 - ls_t0).count();
        res.ir_outer_iters = ir.outer_iters_done;
        res.ir_inner_iters_total = 0;
        for (int v : ir.inner_iters_per_step) res.ir_inner_iters_total += v;
        res.ls_residual_norm = ir.final_residual_norm;
        if (!ir.times.empty()) res.ir_breakdown = ir.times;

        // Solution error: ||x_ls - x_true|| / ||x_true||
        T err_sq = 0;
        for (int64_t i = 0; i < N; ++i) {
            T d = x_ls[i] - x_true[i];
            err_sq += d * d;
        }
        res.ls_solution_error = std::sqrt(err_sq) / x_true_norm;
        std::cout << "done (" << res.ir_total_us << " us)\n";

        print_summary(res);
        return res;
    };

    // Build the ordered list of selected algorithm names from the bitmask.
    // Bit assignments match CQRRT_linop_applications.cc (CQRRT_expl, bit 4, is
    // intentionally excluded — materializing the Kronecker would defeat the
    // purpose). sCholQR3_basic shares the implementation with sCholQR3 here.
    std::vector<std::string> selected_algs;
    if (method_mask & 1)   selected_algs.push_back("CQRRT_linop");
    if (method_mask & 2)   selected_algs.push_back("CholQR");
    if (method_mask & 4)   selected_algs.push_back("sCholQR3");
    if (method_mask & 8)   selected_algs.push_back("sCholQR3_basic");
    if (method_mask & 32)  selected_algs.push_back("CQRRT_linop_stb");
    if (method_mask & 64)  selected_algs.push_back("CQRRT_linop_stb_bqrrp");

    if (selected_algs.empty()) {
        std::cerr << "Error: method_mask selects no algorithms (got " << method_mask << ").\n";
        return 1;
    }

    std::vector<nmr_result<T>> all_results;
    for (const auto& alg_name : selected_algs) {
        std::cout << "\n=== Algorithm: " << alg_name << " ===\n";
        for (int64_t r = 0; r < num_runs; ++r) {
            nmr_result<T> res;
            if (use_augmented) {
                res = run_one(alg_name, J_aug, b_aug.data(), M + N, (T)0, r);
            } else {
                res = run_one(alg_name, J, b.data(), M, lambda_reg, r);
            }
            all_results.push_back(std::move(res));
        }
    }

    // -------- CSV output --------
    char time_buf[64];
    time_t now = time(nullptr);
    strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&now));

    std::string results_file   = output_dir + "/" + time_buf + "_nmr_results.csv";
    std::string breakdown_file = output_dir + "/" + time_buf + "_nmr_breakdown.csv";

    {
        std::ofstream out(results_file);
        out << "# NMR Relaxometry Benchmark results\n"
            << "# Date: " << ctime(&now)
            << "# m1=" << m1 << " n1=" << n1 << " m2=" << m2 << " n2=" << n2 << "\n"
            << "# M=" << M << " N=" << N << "\n"
            << "# phantom=" << phantom << " noise_level=" << noise_level << "\n"
            << "# d_factor=" << d_factor << " sketch_nnz=" << sketch_nnz
            << " block_size=" << block_size << "\n"
            << "# method_mask=" << method_mask << " lambda=" << lambda_reg << "\n"
            << "# log_lo=" << log_lo << " log_hi=" << log_hi << "\n"
#ifdef _OPENMP
            << "# OpenMP threads: " << omp_get_max_threads() << "\n"
#else
            << "# OpenMP threads: 1\n"
#endif
            ;
        out << "algorithm,run,m,n,qr_status,qr_time_us,peak_rss_kb,analytical_kb,"
               "ir_total_us,ir_outer_iters,ir_inner_iters_total,"
               "ls_residual_norm,ls_solution_error\n";
        for (const auto& r : all_results) {
            out << r.alg_name << "," << r.run_idx << "," << r.m << "," << r.n << ","
                << r.qr_status << "," << r.qr_time_us << "," << r.peak_rss_kb << "," << r.analytical_kb << ","
                << r.ir_total_us << "," << r.ir_outer_iters << "," << r.ir_inner_iters_total << ","
                << std::scientific << std::setprecision(6) << r.ls_residual_norm << ","
                << std::scientific << std::setprecision(6) << r.ls_solution_error
                << "\n";
        }
        std::cout << "\nResults written to " << results_file << "\n";
    }

    {
        std::ofstream out(breakdown_file);
        out << "# NMR Benchmark runtime breakdown (microseconds)\n"
            << "# QR breakdown layout depends on algorithm:\n"
            << "#   CQRRT_linop[_stb*] (11): alloc, sketch, qr, tri_inv, fwd, adj, trmm, chol, finalize, rest, total\n"
            << "#   sCholQR3 / sCholQR3_basic — full 18 entries truncated to first 11 (alloc, fwd1, adj1, chol1, upd1, fwd2, adj2, gemm2, chol2, upd2, fwd3)\n"
            << "#   CholQR (6, padded to 11): alloc, fwd, adj, chol, rest, total\n"
            << "# IR-LSQ breakdown (6): outer_total, inner_cg_total, trsm_total, fwd_total, adj_total, other\n"
            << "algorithm,run,phase,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10\n";
        for (const auto& r : all_results) {
            out << r.alg_name << "," << r.run_idx << ",QR";
            for (size_t i = 0; i < r.qr_breakdown.size(); ++i) out << "," << r.qr_breakdown[i];
            for (size_t i = r.qr_breakdown.size(); i < 11; ++i) out << ",0";
            out << "\n";
            out << r.alg_name << "," << r.run_idx << ",IR";
            for (size_t i = 0; i < r.ir_breakdown.size(); ++i) out << "," << r.ir_breakdown[i];
            for (size_t i = r.ir_breakdown.size(); i < 11; ++i) out << ",0";
            out << "\n";
        }
        std::cout << "Breakdown written to " << breakdown_file << "\n";
    }

    return 0;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_runs> <n>"
                  << " [phantom] [noise_level] [d_factor] [sketch_nnz] [block_size]\n";
        return 1;
    }
    std::string prec = argv[1];
    if (prec == "double") return run_benchmark<double>(argc, argv);
    if (prec == "float")  return run_benchmark<float>(argc, argv);
    std::cerr << "Unknown precision: " << prec << " (use 'double' or 'float')\n";
    return 1;
}
