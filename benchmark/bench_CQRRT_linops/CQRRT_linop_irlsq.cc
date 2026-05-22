// Sparse-input IR-LSQ Benchmark — Q-less QR + sketch-and-solve x₀ + 2-step IR.
//
// Loads a tall sparse matrix A from a Matrix Market (.mtx) file (M ≥ N),
// wraps it as a SparseLinOp, generates a synthetic LS problem
//     x_true ~ U(-1, 1)^N,  b = A x_true + Gaussian noise  (||noise||/||A x_true|| = noise_level),
// and for each Q-less QR variant selected by `method_mask`:
//     1. Run the QR variant on J → R  (n × n upper triangular).
//     2. Draw a fresh sparse sketch S₂ (independent of CQRRT's S₁), form
//        SA = S₂·J and Sb = S₂·b, then  x₀ = R⁻¹ R⁻ᵀ (SA)ᵀ Sb  (paper Alg. 1, line 3).
//     3. Run IterRefineLSQ(J, R, b) starting from x₀.
//     4. Record per-(algorithm, run) ||x − x_true||/||x_true||, ||b − Ax||/||b||,
//        IR iter counts, timing, peak vs predicted memory.
//
// Usage:
//     ./CQRRT_linop_irlsq <prec> <output_dir> <num_runs> <mtx_path>
//                         [noise_level] [d_factor] [sketch_nnz] [block_size] [method_mask]
// where:
//     prec        = "double" | "float"
//     mtx_path    = path to a tall (M ≥ N) sparse matrix in Matrix Market format
//     noise_level = ||noise|| / ||A x_true||  (default 0.05)
//     d_factor    = sketch oversampling for both CQRRT and x₀ (default 2.0)
//     sketch_nnz  = nonzeros per column of the SASO sketch (default 4)
//     block_size  = blocking parameter for CQRRT / sCholQR3 (default 0 = library default)
//     method_mask = bitmask of Q-less QR variants (default 0b1001111 = 79)
//                     bit 0 ( 1): CQRRT_linop (TRSM_IDENTITY)
//                     bit 1 ( 2): CholQR
//                     bit 2 ( 4): sCholQR3
//                     bit 3 ( 8): sCholQR3_basic
//                     bit 6 ( 64): CQRRT_linop_bqrrp   (BQRRP)

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"
#include "cqrrt_bench_common.hh"

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
// Result struct
// =============================================================================
template <typename T>
struct irlsq_result {
    int64_t m, n;
    int64_t run_idx;
    std::string alg_name;   // "CQRRT_linop", "CholQR", "sCholQR3", "sCholQR3_basic", "CQRRT_linop_bqrrp"
    T noise_level;
    int qr_status;          // 0 on success; nonzero indicates QR breakdown (no IR-LSQ run)

    // Q-less QR
    long qr_time_us;
    T orth_error;
    long peak_rss_kb;
    long analytical_kb;

    // IR LSQ
    long ir_total_us;       // includes the sketch-and-solve x₀ computation
    int  ir_outer_iters;
    int  ir_inner_iters_total;
    T    ls_residual_norm;
    T    ls_solution_error;

    // Breakdowns
    std::vector<long> qr_breakdown;
    std::vector<long> ir_breakdown;
};


template <typename T>
static void print_summary(const irlsq_result<T>& r) {
    std::printf("\n  [%s] Run %ld (noise=%.3f):\n",
                r.alg_name.c_str(), (long)r.run_idx, (double)r.noise_level);
    if (r.qr_status != 0) {
        std::printf("    QR returned status %d — IR-LSQ skipped.\n", r.qr_status);
        return;
    }
    std::printf("    QR: %ld us, peak_RSS=%ld KB, predicted=%ld KB\n",
                r.qr_time_us, r.peak_rss_kb, r.analytical_kb);
    if (r.orth_error >= 0) std::printf("    orth_err = %.3e\n", (double)r.orth_error);
    std::printf("    IR-LSQ (with x_0): total=%ld us, outer=%d, inner_total=%d\n",
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
                  << " <precision> <output_dir> <num_runs> <mtx_path>"
                  << " [noise_level] [d_factor] [sketch_nnz] [block_size] [method_mask]\n"
                  << "  mtx_path:    tall (M >= N) sparse matrix in Matrix Market format\n"
                  << "  noise_level: ||noise||/||A x_true|| (default 0.05)\n"
                  << "  method_mask: bitmask of Q-less QR variants (default = 0b1001111 = 79)\n"
                  << "    bit 0 ( 1): CQRRT_linop (TRSM_IDENTITY)\n"
                  << "    bit 1 ( 2): CholQR\n"
                  << "    bit 2 ( 4): sCholQR3\n"
                  << "    bit 3 ( 8): sCholQR3_basic\n"
                  << "    bit 6 ( 64): CQRRT_linop_bqrrp   (BQRRP)\n";
        return 1;
    }

    std::string output_dir = argv[2];
    int64_t num_runs       = std::stol(argv[3]);
    std::string mtx_path   = argv[4];
    T noise_level          = (argc >= 6) ? (T)std::stod(argv[5]) : (T)0.05;
    T d_factor             = (argc >= 7) ? (T)std::stod(argv[6]) : (T)2.0;
    int64_t sketch_nnz     = (argc >= 8) ? std::stol(argv[7])    : 4;
    int64_t block_size     = (argc >= 9) ? std::stol(argv[8])    : 0;
    int64_t method_mask    = (argc >= 10) ? std::stol(argv[9])   : 0b1001111;

    std::cout << "=== Sparse IR-LSQ Benchmark (SparseLinOp + Q-less QR + IR-LSQ) ===\n"
              << "  mtx_path   = " << mtx_path << "\n"
              << "  noise_lvl  = " << noise_level << "\n"
              << "  method_mask= " << method_mask
              << " (linop=" << (method_mask&1)
              << " CholQR=" << ((method_mask>>1)&1)
              << " sCholQR3=" << ((method_mask>>2)&1)
              << " sCholQR3_basic=" << ((method_mask>>3)&1)
              << " linop_bqrrp=" << ((method_mask>>6)&1) << ")\n"
              << "  d_factor   = " << d_factor << "\n"
              << "  sketch_nnz = " << sketch_nnz << "\n"
              << "  block_size = " << block_size << "\n"
              << "  num_runs   = " << num_runs << "\n"
#ifdef _OPENMP
              << "  OpenMP threads: " << omp_get_max_threads() << "\n";
#else
              << "  OpenMP threads: 1\n";
#endif

    // -------- Load matrix --------
    int64_t M = 0, N = 0, nnz = 0;
    std::cout << "Loading " << mtx_path << " ... " << std::flush;
    auto load_t0 = steady_clock::now();
    auto A_csr = load_csr<T>(mtx_path, M, N, nnz);
    auto load_t1 = steady_clock::now();
    std::cout << "done (" << duration_cast<microseconds>(load_t1 - load_t0).count() << " us)\n"
              << "  M=" << M << "  N=" << N << "  nnz=" << nnz
              << "  density=" << ((double)nnz / ((double)M * (double)N)) << "\n";

    if (M < N) {
        std::cerr << "Error: need tall matrix (M >= N), got M=" << M << " N=" << N << "\n";
        return 1;
    }

    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> J(M, N, A_csr);

    // -------- Synthetic LS problem: x_true ~ U(-1,1), b = J x_true + noise --------
    std::vector<T> x_true(N), b_clean(M, (T)0), b(M, (T)0), noise_vec(M, (T)0);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<T> dist((T)-1.0, (T)1.0);
        for (auto& v : x_true) v = dist(rng);
    }
    T x_true_norm = blas::nrm2(N, x_true.data(), 1);
    if (x_true_norm == 0) { std::cerr << "x_true is zero — aborting\n"; return 1; }

    J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      M, 1, N, (T)1.0, x_true.data(), N, (T)0.0, b_clean.data(), M);
    T b_clean_norm = blas::nrm2(M, b_clean.data(), 1);

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
    auto run_one = [&](const std::string& alg_name, int64_t r) -> irlsq_result<T>
    {
        irlsq_result<T> res{};
        res.m = M; res.n = N;
        res.run_idx = r;
        res.alg_name = alg_name;
        res.noise_level = noise_level;
        res.qr_status = 0;

        // ---- Q-less QR on J: dispatch on alg_name ----
        std::cout << "[Run " << r << ", " << alg_name << "] QR ... " << std::flush;
        std::vector<T> R(N * N, (T)0);
        auto state = run_states[r];

        RandLAPACK::PeakRSSTracker mem; mem.start();
        if (alg_name == "sCholQR3") {
            RandLAPACK::sCholQR3_linops<T> qr_algo(/*time_subroutines=*/true, tol);
            qr_algo.block_size = block_size;
            res.qr_status = qr_algo.call(J, R.data(), N);
            res.peak_rss_kb = mem.stop();
            if (res.qr_status == 0) {
                res.qr_time_us  = qr_algo.times[17];
                res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(M, N, block_size);
            }
        } else if (alg_name == "sCholQR3_basic") {
            RandLAPACK::sCholQR3_linops_basic<T> qr_algo(/*time_subroutines=*/true, tol);
            res.qr_status = qr_algo.call(J, R.data(), N);
            res.peak_rss_kb = mem.stop();
            if (res.qr_status == 0) {
                res.qr_time_us  = qr_algo.times[14];
                res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<T>(M, N);
            }
        } else if (alg_name == "CholQR") {
            RandLAPACK::CholQR_linops<T> qr_algo(/*time_subroutines=*/true, tol);
            qr_algo.block_size = block_size;
            res.qr_status = qr_algo.call(J, R.data(), N);
            res.peak_rss_kb = mem.stop();
            if (res.qr_status == 0) {
                res.qr_time_us = qr_algo.times[5];
                res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 6);
                res.qr_breakdown.resize(11, 0);
                res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(M, N, block_size);
            }
        } else {
            RandLAPACK::CQRRT_linops<T, RNG> qr_algo(/*time_subroutines=*/true, tol);
            qr_algo.nnz = sketch_nnz;
            qr_algo.block_size = block_size;
            if (alg_name == "CQRRT_linop") qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::TRSM_IDENTITY;
            else /* CQRRT_linop_bqrrp */ qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::BQRRP;
            res.qr_status = qr_algo.call(J, R.data(), N, d_factor, state);
            res.peak_rss_kb = mem.stop();
            if (res.qr_status == 0) {
                res.qr_time_us = qr_algo.times[10];
                res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                res.analytical_kb = (alg_name == "CQRRT_linop_bqrrp")
                    ? RandLAPACK::cqrrt_linops_bqrrp_analytical_kb<T>(M, N, d_factor, block_size)
                    : RandLAPACK::cqrrt_linops_analytical_kb<T>(M, N, d_factor, block_size);
            }
        }

        if (res.qr_status != 0) {
            std::cerr << "\n  [" << alg_name << "] Run " << r
                      << ": QR returned status " << res.qr_status
                      << " (likely Cholesky breakdown).\n"
                      << "  The input may be too ill-conditioned for the chosen variant;\n"
                      << "  try a more stabilized variant (CQRRT_linop_bqrrp / sCholQR3) or a larger d_factor.\n";
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

        // ---- Sketch-and-solve initial guess (Algorithm 1, line 3 of
        //      Epperly–Meier–Nakatsukasa 2025), with a fresh sparse sketch
        //      S₂ independent of CQRRT's S₁:
        //          SA = S₂·J     (d_init × N)
        //          Sb = S₂·b     (d_init × 1)
        //          x₀ = R⁻¹ R⁻ᵀ (SA)ᵀ Sb     (Q-less form; R reused as preconditioner)
        //      Timing folded into ir_total_us alongside the IR steps.
        auto ls_t0 = steady_clock::now();
        const int64_t d_init = (int64_t)(d_factor * (T)N);
        RandBLAS::SparseDist DS_init(d_init, M, sketch_nnz);
        auto x0_state = state;
        x0_state.key.incr(0xA1B2C3D4u);
        RandBLAS::SparseSkOp<T, RNG> S2(DS_init, x0_state);
        RandBLAS::fill_sparse(S2);

        std::vector<T> SA(d_init * N, (T)0);
        J(blas::Side::Right, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          d_init, N, M, (T)1.0, S2, (T)0.0, SA.data(), d_init);

        std::vector<T> Sb(d_init, (T)0);
        RandBLAS::sketch_general(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                 d_init, (int64_t)1, M, (T)1.0,
                                 S2, b.data(), M, (T)0.0, Sb.data(), d_init);

        std::vector<T> x_ls(N, (T)0);
        blas::gemv(blas::Layout::ColMajor, blas::Op::Trans, d_init, N,
                   (T)1.0, SA.data(), d_init, Sb.data(), 1,
                   (T)0.0, x_ls.data(), 1);
        blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                   blas::Op::Trans, blas::Diag::NonUnit, N, 1,
                   (T)1.0, R.data(), N, x_ls.data(), N);
        blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                   blas::Op::NoTrans, blas::Diag::NonUnit, N, 1,
                   (T)1.0, R.data(), N, x_ls.data(), N);

        // ---- IterRefineLSQ ----
        RandLAPACK::IterRefineLSQ<T> ir(/*tol=*/tol,
                                        /*max_inner=*/200,
                                        /*n_steps=*/2,
                                        /*timing=*/true,
                                        /*verbose=*/false);
        int ir_status = ir.call(J, R.data(), N, b.data(), M, x_ls.data(), N);
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
    std::vector<std::string> selected_algs;
    if (method_mask & 1)   selected_algs.push_back("CQRRT_linop");
    if (method_mask & 2)   selected_algs.push_back("CholQR");
    if (method_mask & 4)   selected_algs.push_back("sCholQR3");
    if (method_mask & 8)   selected_algs.push_back("sCholQR3_basic");
    if (method_mask & 64)  selected_algs.push_back("CQRRT_linop_bqrrp");

    if (selected_algs.empty()) {
        std::cerr << "Error: method_mask selects no algorithms (got " << method_mask << ").\n";
        return 1;
    }

    std::vector<irlsq_result<T>> all_results;
    for (const auto& alg_name : selected_algs) {
        std::cout << "\n=== Algorithm: " << alg_name << " ===\n";
        for (int64_t r = 0; r < num_runs; ++r) {
            all_results.push_back(run_one(alg_name, r));
        }
    }

    // -------- CSV output --------
    char time_buf[64];
    time_t now = time(nullptr);
    strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&now));

    std::string results_file   = output_dir + "/" + time_buf + "_irlsq_results.csv";
    std::string breakdown_file = output_dir + "/" + time_buf + "_irlsq_breakdown.csv";

    {
        std::ofstream out(results_file);
        out << "# Sparse IR-LSQ Benchmark results\n"
            << "# Date: " << ctime(&now)
            << "# mtx_path=" << mtx_path << "\n"
            << "# M=" << M << " N=" << N << " nnz=" << nnz << "\n"
            << "# noise_level=" << noise_level << "\n"
            << "# d_factor=" << d_factor << " sketch_nnz=" << sketch_nnz
            << " block_size=" << block_size << "\n"
            << "# method_mask=" << method_mask << "\n"
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
        out << "# Sparse IR-LSQ Benchmark runtime breakdown (microseconds)\n"
            << "# QR breakdown layout depends on algorithm (see CQRRT_linop_applications.cc).\n"
            << "# IR-LSQ breakdown (6): outer_total, inner_cg_total, trsm_total, fwd_total, adj_total, other\n"
            << "#   (sketch-and-solve x_0 time is folded into the difference between ir_total_us\n"
            << "#    in the results CSV and outer_total here)\n"
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
                  << " <precision> <output_dir> <num_runs> <mtx_path>"
                  << " [noise_level] [d_factor] [sketch_nnz] [block_size] [method_mask]\n";
        return 1;
    }
    std::string prec = argv[1];
    if (prec == "double") return run_benchmark<double>(argc, argv);
    if (prec == "float")  return run_benchmark<float>(argc, argv);
    std::cerr << "Unknown precision: " << prec << " (use 'double' or 'float')\n";
    return 1;
}
