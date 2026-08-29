// CQRRT_linops scaling study - measures performance and accuracy as matrix size varies
// Tests both CQRRT (preconditioned) and CholQR (unpreconditioned) on tall sparse matrices
// Supports two modes:
//   1. Fixed aspect ratio: n = m / aspect_ratio (both dimensions scale together)
//   2. Fixed columns: n is constant while m varies

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Extras utilities for Matrix Market I/O
#include "../../extras/misc/ext_util.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "cqrrt_bench_common.hh"

// Linops algorithms (now in main RandLAPACK)
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// Shared helpers (cqrrt_bench_common.hh): env-knob provenance, the
// per-pass Cholesky-shift fold, and argv/host CSV provenance.
using RandLAPACK::bench::write_env_line;
using RandLAPACK::bench::fold_chol_shift;
using RandLAPACK::bench::quote_join_argv;
using RandLAPACK::bench::get_hostname;

// Common quality + timing fields shared by all algorithms. Every field
// defaults to a -1/failure sentinel (never a silently-perfect 0) so a
// skipped or failed algorithm reads unambiguously in the CSV.
template <typename T>
struct alg_quality {
    int qr_status = -1;         // 0 = success; driver's own code on failure;
                                 // -1 = not yet run (skip_dense, or a bug if
                                 // still -1 after run_algorithms returns)
    T orth_error = (T)-1;       // ||Q^T Q - I|| / sqrt(n)
    bool is_orthonormal = false; // Is full Q block orthonormal?
    int64_t max_orth_cols = -1; // Maximum orthonormal prefix
    long time = -1;              // Total computation time (microseconds)
    long peak_rss_kb = -1;       // Peak RSS increase during algorithm call (KB)
    long analytical_kb = -1;     // Analytical peak working memory (KB)
    std::vector<long> breakdown; // Per-subroutine timings (excludes total);
                                  // always sized to the algorithm's fixed
                                  // slot count (10/5/17/10) regardless of
                                  // success, failure, or skip_dense, so the
                                  // breakdown CSV row is always full-width.
    int chol_retries = -1;       // -1 = the row failed, or was skipped
                                  // (dense_cqrrt under skip_dense=1); every
                                  // algorithm in this file, including dense
                                  // CQRRT_expl, has an adaptive Cholesky-shift
                                  // retry mechanism, so -1 no longer means
                                  // "no mechanism"
    T chol_shift_abs = (T)-1;
    T chol_shift_rel = (T)-1;
};

template <typename T>
struct scaling_result {
    int64_t m;                // Number of rows
    int64_t n;                // Number of columns
    T cond_num;               // Target condition number
    T density;                // Sparse matrix density (computed from actual nnz)
    T aspect_ratio;           // m / n
    int64_t run_idx;          // Run index (0-based)

    alg_quality<T> cqrrt;
    alg_quality<T> cholqr;
    alg_quality<T> scholqr3;
    alg_quality<T> dense_cqrrt;
};

// Compute Q = A * R^{-1} uniformly for all algorithms.
// Materializes A into Q_out, then solves Q * R = A via trsm (avoids forming R^{-1}).
// This is more numerically stable than forming the explicit inverse via trtri or trsm.
// R is NOT destroyed.
template <typename T, typename GLO>
static void compute_Q_from_R(
    GLO& A_op, T* R, int64_t ldr,
    T* Q_out, int64_t m, int64_t n) {
    T* Eye = new T[n * n]();
    RandLAPACK::util::eye(n, n, Eye);
    A_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         m, n, n, (T)1.0, Eye, n, (T)0.0, Q_out, m);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0, R, ldr, Q_out, m);
    delete[] Eye;
}

// Core algorithm runner: operates on a pre-constructed SparseLinOp.
// Called by both the generate-mode and file-input-mode entry points.
template <typename T, typename RNG>
static std::vector<scaling_result<T>> run_algorithms(
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>>& A_linop,
    int64_t m,
    int64_t n,
    T cond_num,
    T density,
    T d_factor,
    int64_t block_size,
    int64_t sketch_nnz,
    int64_t num_runs,
    std::vector<RandBLAS::RNGState<RNG>>& run_states,
    bool skip_dense) {

    std::vector<scaling_result<T>> results(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        results[r].m = m;
        results[r].n = n;
        results[r].cond_num = cond_num;
        results[r].density = density;
        results[r].aspect_ratio = static_cast<T>(m) / static_cast<T>(n);
        results[r].run_idx = r;
    }

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);

    // Single reusable Q buffer for uniform Q = A * R^{-1} computation across all algorithms
    T* Q_uniform = new T[m * n];

    // RSS window (unified): every algorithm's peak_rss_kb is measured around
    // its OWN single timed call() below, all with test_mode=false. CQRRT_linop
    // and CholQR used to run a SEPARATE untimed pre-call for the RSS
    // measurement, whose stated rationale ("excludes the test_mode=true
    // Q-factor allocation") never applied, since the timed call was already
    // test_mode=false, so that extra pass measured nothing the timed call's
    // own window wouldn't. Dropped in favor of the single-window scheme
    // sCholQR3 already used.

    // ============================================================
    // Run CQRRT (preconditioned Cholesky QR) - multiple runs
    // ============================================================
    {
        T* R_cqrrt = new T[n * n];
        for (int64_t run = 0; run < num_runs; ++run) {
            std::fill(R_cqrrt, R_cqrrt + n * n, (T)0);
            auto state_copy = run_states[run];  // Per-run RNG state

            RandLAPACK::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, false);  // timing=true, test_mode=false
            CQRRT_QR.nnz = sketch_nnz;
            CQRRT_QR.block_size = block_size;

            RandLAPACK::PeakRSSTracker cqrrt_mem;
            cqrrt_mem.start();
            int status = CQRRT_QR.call(A_linop, R_cqrrt, n, d_factor, state_copy);
            results[run].cqrrt.peak_rss_kb = cqrrt_mem.stop();
            results[run].cqrrt.qr_status = status;

            if (status == 0) {
                results[run].cqrrt.time = CQRRT_QR.total_us();
                results[run].cqrrt.breakdown.assign(CQRRT_QR.times.begin(), CQRRT_QR.times.end() - 1);
                results[run].cqrrt.chol_retries = CQRRT_QR.n_chol_retries;
                fold_chol_shift(results[run].cqrrt.chol_shift_abs, results[run].cqrrt.chol_shift_rel,
                                 CQRRT_QR.chol_applied_shifts, CQRRT_QR.chol_gram_traces, 1);

                // Uniform Q computation for every run: Q = A * R^{-1} via operator
                compute_Q_from_R(A_linop, R_cqrrt, n, Q_uniform, m, n);
                results[run].cqrrt.orth_error    = RandLAPACK::testing::orthogonality_error<T>(Q_uniform, m, n);
                results[run].cqrrt.is_orthonormal = (results[run].cqrrt.orth_error <= std::pow(std::numeric_limits<T>::epsilon(), (T)0.75));
                results[run].cqrrt.max_orth_cols  = RandLAPACK::testing::max_orthonormal_cols<T>(Q_uniform, m, n);
            } else {
                // Failure: no valid R, so no Q and no orthogonality metrics.
                // Full-width -1 breakdown (never a truncated/empty vector).
                results[run].cqrrt.breakdown.assign(10, -1L);
                std::cerr << "Warning: CQRRT_linop call() failed (run=" << run
                          << ", status=" << status << "); quality/timing fields set to -1.\n";
            }
        }
        delete[] R_cqrrt;
    }

    // ============================================================
    // Run CholQR (unpreconditioned Cholesky QR) - multiple runs
    // ============================================================
    {
        T* R_cholqr = new T[n * n];
        for (int64_t run = 0; run < num_runs; ++run) {
            std::fill(R_cholqr, R_cholqr + n * n, (T)0);

            RandLAPACK::CholQR_linops<T> CholQR_alg(true, tol, false);  // timing=true, test_mode=false
            CholQR_alg.block_size = block_size;

            RandLAPACK::PeakRSSTracker cholqr_mem;
            cholqr_mem.start();
            int status = CholQR_alg.call(A_linop, R_cholqr, n);
            results[run].cholqr.peak_rss_kb = cholqr_mem.stop();
            results[run].cholqr.qr_status = status;

            if (status == 0) {
                results[run].cholqr.time = CholQR_alg.total_us();
                results[run].cholqr.breakdown.assign(CholQR_alg.times.begin(), CholQR_alg.times.end() - 1);
                results[run].cholqr.chol_retries = CholQR_alg.n_chol_retries;
                fold_chol_shift(results[run].cholqr.chol_shift_abs, results[run].cholqr.chol_shift_rel,
                                 CholQR_alg.chol_applied_shifts, CholQR_alg.chol_gram_traces, 1);

                // Uniform Q computation for every run
                compute_Q_from_R(A_linop, R_cholqr, n, Q_uniform, m, n);
                results[run].cholqr.orth_error    = RandLAPACK::testing::orthogonality_error<T>(Q_uniform, m, n);
                results[run].cholqr.is_orthonormal = (results[run].cholqr.orth_error <= std::pow(std::numeric_limits<T>::epsilon(), (T)0.75));
                results[run].cholqr.max_orth_cols  = RandLAPACK::testing::max_orthonormal_cols<T>(Q_uniform, m, n);
            } else {
                results[run].cholqr.breakdown.assign(5, -1L);
                std::cerr << "Warning: CholQR call() failed (run=" << run
                          << ", status=" << status << "); quality/timing fields set to -1.\n";
            }
        }
        delete[] R_cholqr;
    }

    // ============================================================
    // Run sCholQR3 (shifted Cholesky QR with 3 iterations) - multiple runs
    // ============================================================
    {
        T* R_scholqr3 = new T[n * n];
        for (int64_t run = 0; run < num_runs; ++run) {
            std::fill(R_scholqr3, R_scholqr3 + n * n, (T)0);

            RandLAPACK::sCholQR3_linops<T> sCholQR3_alg(true, tol, false);  // timing=true, test_mode=false
            sCholQR3_alg.block_size = block_size;

            RandLAPACK::PeakRSSTracker scholqr3_mem;
            scholqr3_mem.start();
            int status = sCholQR3_alg.call(A_linop, R_scholqr3, n);
            results[run].scholqr3.peak_rss_kb = scholqr3_mem.stop();
            results[run].scholqr3.qr_status = status;

            if (status == 0) {
                results[run].scholqr3.time = sCholQR3_alg.total_us();
                // breakdown (17): alloc, fwd1, adj1, chol1, upd1, fwd2, adj2, gemm2, chol2, upd2, fwd3, adj3, gemm3, chol3, upd3, q_mat, rest
                results[run].scholqr3.breakdown.assign(sCholQR3_alg.times.begin(), sCholQR3_alg.times.end() - 1);
                results[run].scholqr3.chol_retries = sCholQR3_alg.n_chol_retries;
                fold_chol_shift(results[run].scholqr3.chol_shift_abs, results[run].scholqr3.chol_shift_rel,
                                 sCholQR3_alg.chol_applied_shifts, sCholQR3_alg.chol_gram_traces, 3);

                // Uniform Q computation (same as all other algorithms)
                compute_Q_from_R(A_linop, R_scholqr3, n, Q_uniform, m, n);
                results[run].scholqr3.orth_error    = RandLAPACK::testing::orthogonality_error<T>(Q_uniform, m, n);
                results[run].scholqr3.is_orthonormal = (results[run].scholqr3.orth_error <= std::pow(std::numeric_limits<T>::epsilon(), (T)0.75));
                results[run].scholqr3.max_orth_cols  = RandLAPACK::testing::max_orthonormal_cols<T>(Q_uniform, m, n);
            } else {
                results[run].scholqr3.breakdown.assign(17, -1L);
                std::cerr << "Warning: sCholQR3 call() failed (run=" << run
                          << ", status=" << status << "); quality/timing fields set to -1.\n";
            }
        }
        delete[] R_scholqr3;
    }

    // ============================================================
    // Run CQRRT_expl (materialize operator, then call rl_cqrrt) - multiple runs
    // ============================================================
    // Peak RSS with compute_Q=true is correct: Q overwrites A_materialized in-place (no extra allocation).
    // Skipped when skip_dense=true (e.g., for large file-input matrices where m*n dense doesn't fit in memory).
    if (!skip_dense) {
        T* I_mat   = new T[n * n]();
        RandLAPACK::util::eye(n, n, I_mat);
        T* R_dense = new T[n * n];
        for (int64_t run = 0; run < num_runs; ++run) {
            RandLAPACK::PeakRSSTracker dense_mem;
            dense_mem.start();

            // Step 1: Materialize the operator by multiplying with identity
            T* A_materialized = new T[m * n]();

            auto materialize_start = steady_clock::now();
            A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    m, n, n, (T)1.0, I_mat, n, (T)0.0, A_materialized, m);
            auto materialize_stop = steady_clock::now();
            long materialize_time = duration_cast<microseconds>(materialize_stop - materialize_start).count();

            // Step 2: Call rl_cqrrt with timing, Q-factor disabled (computed uniformly below)
            // Uses same per-run RNG state as CQRRT_linop for fair comparison
            std::fill(R_dense, R_dense + n * n, (T)0);
            auto state_copy = run_states[run];  // Same RNG state as CQRRT_linop's run
            RandLAPACK::CQRRT<T, RNG> dense_alg(true, tol);  // timing=true
            dense_alg.compute_Q = false;
            dense_alg.orthogonalization = false;
            dense_alg.nnz = sketch_nnz;
            int status = dense_alg.call(m, n, A_materialized, m, R_dense, n, d_factor, state_copy);

            results[run].dense_cqrrt.peak_rss_kb = dense_mem.stop();
            results[run].dense_cqrrt.qr_status = status;

            delete[] A_materialized;  // No longer needed (Q computed via operator)

            if (status == 0) {
                // Total = materialization + algorithm total (Q excluded from algo total)
                results[run].dense_cqrrt.time = materialize_time + dense_alg.total_us();
                // Breakdown matches linop CQRRT layout: materialize, saso, qr, trtri(=0), precond, gram, trmm_gram(=0), potrf, finalize, rest
                results[run].dense_cqrrt.breakdown = {
                    materialize_time,
                    dense_alg.times[0],  // saso
                    dense_alg.times[1],  // qr
                    0L,                  // trtri (always 0 for dense)
                    dense_alg.times[3],  // precond
                    dense_alg.times[4],  // gram
                    0L,                  // trmm_gram (always 0 for dense)
                    dense_alg.times[6],  // potrf
                    dense_alg.times[7],  // finalize
                    dense_alg.times[8],  // rest
                };
                // Dense CQRRT (rl_cqrrt.hh CQRRT, not CQRRT_linops) has
                // the same adaptive Cholesky-shift retry as the other three
                // algorithms; fold its record the same
                // way the other rows do (single shift-record entry, npasses=1,
                // same as CQRRT_linops's own single-pass record).
                results[run].dense_cqrrt.chol_retries = dense_alg.n_chol_retries;
                fold_chol_shift(results[run].dense_cqrrt.chol_shift_abs, results[run].dense_cqrrt.chol_shift_rel,
                                 dense_alg.chol_applied_shifts, dense_alg.chol_gram_traces, 1);

                // Uniform Q computation for every run
                compute_Q_from_R(A_linop, R_dense, n, Q_uniform, m, n);
                results[run].dense_cqrrt.orth_error    = RandLAPACK::testing::orthogonality_error<T>(Q_uniform, m, n);
                results[run].dense_cqrrt.is_orthonormal = (results[run].dense_cqrrt.orth_error <= std::pow(std::numeric_limits<T>::epsilon(), (T)0.75));
                results[run].dense_cqrrt.max_orth_cols  = RandLAPACK::testing::max_orthonormal_cols<T>(Q_uniform, m, n);
            } else {
                results[run].dense_cqrrt.breakdown.assign(10, -1L);
                std::cerr << "Warning: CQRRT_expl (dense) call() failed (run=" << run
                          << ", status=" << status << "); quality/timing fields set to -1.\n";
            }
        }
        delete[] I_mat;
        delete[] R_dense;
    } else {
        // skip_dense=true: CQRRT_expl never runs. Every field carries the -1
        // sentinel (never a value-initialized 0, which would misreport as
        // "measured, perfectly orthogonal") so a reader can tell "skipped"
        // apart from "ran and was flawless". qr_status=-1 marks
        // "not run"; it is deliberately not one of CQRRT's own nonzero
        // failure codes, which mean the algorithm genuinely tried and failed.
        for (int64_t r = 0; r < num_runs; ++r) {
            results[r].dense_cqrrt.breakdown.assign(10, -1L);
            results[r].dense_cqrrt.qr_status = -1;
        }
    }

    // Compute analytical peak working memory for each algorithm (same for all runs).
    // dense_cqrrt_akb is gated the same way as the rest of the dense row:
    // skip_dense=true means CQRRT_expl never ran, so its analytical model is
    // -1 ("not applicable"), not a number nobody measured against.
    long cqrrt_akb = RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
    long cholqr_akb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
    long scholqr3_akb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
    long dense_cqrrt_akb = skip_dense ? -1L : RandLAPACK::dense_cqrrt_analytical_kb<T>(m, n, d_factor);
    for (int64_t r = 0; r < num_runs; ++r) {
        results[r].cqrrt.analytical_kb      = cqrrt_akb;
        results[r].cholqr.analytical_kb     = cholqr_akb;
        results[r].scholqr3.analytical_kb   = scholqr3_akb;
        results[r].dense_cqrrt.analytical_kb = dense_cqrrt_akb;
    }

    delete[] Q_uniform;
    return results;
}

// Generate-mode entry point: generates a synthetic sparse matrix, then runs algorithms.
template <typename T, typename RNG>
static std::vector<scaling_result<T>> run_single_test(
    int64_t m,
    int64_t n,
    T cond_num,
    T density,
    T d_factor,
    int64_t block_size,
    int64_t sketch_nnz,
    int64_t num_runs,
    RandBLAS::RNGState<RNG>& state) {

    // Pre-generate per-run RNG states
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        run_states[r] = state;
        if (r > 0) run_states[r].key.incr(r);
    }

    // Generate sparse matrix A: m × n with controlled condition number.
    auto A_coo = RandLAPACK::gen::gen_sparse_cond_coo<T>(m, n, cond_num, state, density);
    T actual_density = static_cast<T>(A_coo.nnz) / (static_cast<T>(m) * n);
    RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

    return run_algorithms(A_linop, m, n, cond_num, actual_density, d_factor,
                          block_size, sketch_nnz, num_runs, run_states, false);
}

// Compute the 2-norm condition number of a sparse linear operator by
// materializing it and computing singular values via LAPACK gesdd.
template <typename T, typename SpLinOp>
static T compute_condition_number(SpLinOp& A_linop, int64_t m, int64_t n) {
    T* A_dense = new T[m * n]();
    RandLAPACK::testing::materialize_linop<T>(A_linop, A_dense);
    auto sigma = RandLAPACK::testing::compute_singular_values<T>(A_dense, m, n);
    delete[] A_dense;
    T cond = sigma[0] / sigma[n - 1];
    std::cout << "  Condition number: " << std::scientific << std::setprecision(6) << (double)cond << " (sigma_max=" << std::setprecision(6) << (double)sigma[0] << ", sigma_min=" << std::setprecision(6) << (double)sigma[n - 1] << ")\n";
    return cond;
}

// File-input entry point: loads a Matrix Market file, then runs algorithms.
template <typename T, typename RNG>
static std::vector<scaling_result<T>> run_single_test_from_file(
    const std::string& filename,
    T d_factor,
    int64_t block_size,
    int64_t sketch_nnz,
    int64_t num_runs,
    bool compute_cond,
    bool skip_dense,
    RandBLAS::RNGState<RNG>& state) {

    // Pre-generate per-run RNG states
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        run_states[r] = state;
        if (r > 0) run_states[r].key.incr(r);
    }

    // Load matrix from Matrix Market file
    int64_t m, n, nnz;
    auto A_csr = load_csr<T>(filename, m, n, nnz);
    T actual_density = static_cast<T>(nnz) / (static_cast<T>(m) * n);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

    T cond_num = std::numeric_limits<T>::quiet_NaN();
    if (compute_cond) {
        std::cout << "Computing condition number via SVD (" << m << " x " << n << ")...\n";
        cond_num = compute_condition_number<T>(A_linop, m, n);
    }

    return run_algorithms(A_linop, m, n, cond_num, actual_density, d_factor,
                          block_size, sketch_nnz, num_runs, run_states, skip_dense);
}

// Forward declarations for shared helpers (defined below run_benchmark)
template <typename T>
static void write_results_to_csv(
    const std::vector<scaling_result<T>>& all_runs, int64_t num_runs,
    std::ofstream& out, std::ofstream& breakdown);
template <typename T>
static void print_console_summary(
    const std::vector<scaling_result<T>>& all_runs, int64_t num_runs, int64_t n);
static void write_csv_headers(
    std::ofstream& out, std::ofstream& breakdown,
    const std::string& precision, double d_factor,
    int64_t sketch_nnz, int64_t block_size, int64_t num_runs, int num_threads,
    const std::string& extra_comment, const std::string& argv_line);
static void prepend_runtime(const std::string& filepath, double seconds);

template <typename T>
static int run_benchmark(int argc, char *argv[]) {

    if (argc < 11 || argc > 13) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_sizes> <num_runs> <m_start> <m_end> <aspect_ratio> <cond_num> <density> <d_factor> [sketch_nnz] [block_size]"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  precision        : 'double' or 'float'" << std::endl;
        std::cerr << "  output_dir       : Directory to write output files" << std::endl;
        std::cerr << "  num_sizes        : Number of matrix sizes to test" << std::endl;
        std::cerr << "  num_runs         : Number of runs per matrix size (for timing)" << std::endl;
        std::cerr << "  m_start          : Starting number of rows (smallest matrix)" << std::endl;
        std::cerr << "  m_end            : Ending number of rows (largest matrix)" << std::endl;
        std::cerr << "  aspect_ratio     : Ratio m/n (e.g., 20 means n = m/20)" << std::endl;
        std::cerr << "  cond_num         : Target condition number for the sparse matrix (e.g., 1e4)" << std::endl;
        std::cerr << "  density          : Target density (e.g., 0.1); bandwidth derived as round(density*n - 1)" << std::endl;
        std::cerr << "  d_factor         : Sketching dimension factor for CQRRT_linop (e.g., 2.0)" << std::endl;
        std::cerr << "  sketch_nnz       : (Optional) Nonzeros per column in SASO sketch (default: 4)" << std::endl;
        std::cerr << "  block_size       : (Optional) Column-block size for CQRRT_linop/CholQR/sCholQR3 Gram (0 = full, default: 256 = paper b)" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " double ./output 30 3 1000 30000 100 1e9 0.05 2.0 4 100" << std::endl;
        std::cerr << "  (Tests 30 matrices from 1000x10 to 30000x300, aspect ratio 100:1, κ=1e9, density≈0.05, 3 runs each)" << std::endl;
        return 1;
    }

    // Parse arguments
    std::string precision = argv[1];
    std::string output_dir = argv[2];
    int64_t num_sizes = std::stoll(argv[3]);
    int64_t num_runs = std::stoll(argv[4]);
    int64_t m_start = std::stoll(argv[5]);
    int64_t m_end = std::stoll(argv[6]);
    double aspect_ratio = std::stod(argv[7]);
    double cond_num = std::stod(argv[8]);
    double density = std::stod(argv[9]);
    double d_factor = std::stod(argv[10]);
    // Default sketch_nnz=4: the Givens-based matrix generator produces
    // high-coherence matrices (non-uniform leverage scores), so nnz >= 4
    // is needed for reliable SASO sketching (nnz=2 causes sporadic spikes).
    int64_t sketch_nnz = (argc >= 12) ? std::stoll(argv[11]) : 4;
    // Default block_size=256 matches the paper's b=256 (blocked Gram, tall
    // intermediate never fully materialized); pass 0 explicitly for unblocked.
    int64_t block_size = (argc >= 13) ? std::stoll(argv[12]) : 256;

    // Loud validation: num_sizes=0 leaves `sizes` empty and
    // sizes.front()/.back() below is UB; num_runs<=0 leaves `results` empty
    // and every per-run write is UB (num_runs=0 was silently accepted before).
    if (num_sizes < 1) {
        std::cerr << "Error: num_sizes must be >= 1 (got " << num_sizes << ")\n";
        return 1;
    }
    if (num_runs < 1) {
        std::cerr << "Error: num_runs must be >= 1 (got " << num_runs << ")\n";
        return 1;
    }

    std::string argv_line = quote_join_argv(argc, argv);
    auto benchmark_start = steady_clock::now();

    // Generate date/time prefix
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    // Build list of (m, n) pairs to test
    std::vector<std::pair<int64_t, int64_t>> sizes;
    for (int64_t i = 0; i < num_sizes; ++i) {
        // Linear interpolation from m_start to m_end
        int64_t m = (num_sizes == 1) ? m_start : m_start + (m_end - m_start) * i / (num_sizes - 1);
        int64_t n = static_cast<int64_t>(m / aspect_ratio);
        if (n < 1) n = 1;  // Ensure at least 1 column
        sizes.push_back({m, n});
    }

    // Get OpenMP thread count
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    std::cout << "\n=== CQRRT_linop vs CholQR vs sCholQR3 vs CQRRT_expl Scaling Study ===\n";
    std::cout << "Precision: " << precision.c_str() << "\n";
    std::cout << "Fixed aspect ratio: " << std::fixed << std::setprecision(1) << aspect_ratio << ":1 (m/n)\n";
    std::cout << "Matrix sizes: " << sizes.front().first << " x " << sizes.front().second << " to " << sizes.back().first << " x " << sizes.back().second << "\n";
    std::cout << "Number of test sizes: " << sizes.size() << "\n";
    std::cout << "Condition number: " << std::scientific << std::setprecision(2) << cond_num << "\n";
    std::cout << "Target density: " << std::fixed << std::setprecision(3) << density << "\n";
    std::cout << "d_factor (CQRRT_linop): " << std::fixed << std::setprecision(2) << d_factor << "\n";
    std::cout << "Sketch nnz (CQRRT_linop): " << sketch_nnz << "\n";
    std::cout << "Block size (CQRRT_linop, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    std::cout << "Runs per size: " << num_runs << "\n";
    std::cout << "OpenMP threads: " << num_threads << "\n";
    std::cout << "=====================================\n\n";

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Prepare output files with date/time prefix
    std::string output_file = output_dir + "/" + date_prefix + "scaling_results.csv";
    std::string breakdown_file = output_dir + "/" + date_prefix + "scaling_breakdown.csv";
    std::ofstream out(output_file);
    std::ofstream breakdown(breakdown_file);

    std::ostringstream extra;
    extra << "# Fixed aspect ratio: " << aspect_ratio << ":1\n"
          << "# Condition number: " << cond_num << "\n"
          << "# Target density: " << density << "\n";
    write_csv_headers(out, breakdown, precision, d_factor,
                      sketch_nnz, block_size, num_runs, num_threads, extra.str(), argv_line);

    // Warmup run to trigger library initialization (MKL thread pools, memory allocators, etc.)
    // This ensures first reported iteration has accurate memory measurements.
    {
        std::cout << "Performing warmup run (not reported)...\n";
        int64_t warmup_m = sizes[0].first;
        int64_t warmup_n = sizes[0].second;
        auto warmup_state = state;  // Use copy to not affect main RNG sequence
        run_single_test<T>(warmup_m, warmup_n, (T)cond_num, (T)density, (T)d_factor, block_size, sketch_nnz, 1, warmup_state);
        std::cout << "Warmup complete, starting measurements.\n\n";
    }

    // Run scaling study
    for (size_t i = 0; i < sizes.size(); ++i) {
        int64_t m = sizes[i].first;
        int64_t n = sizes[i].second;
        std::cout << "Testing " << m << " x " << n << " (aspect ratio " << std::fixed << std::setprecision(1) << static_cast<double>(m) / n << ") [" << i + 1 << "/" << sizes.size() << "]...\n";

        auto all_runs = run_single_test<T>(m, n, (T)cond_num, (T)density, (T)d_factor, block_size, sketch_nnz, num_runs, state);

        print_console_summary(all_runs, num_runs, n);
        write_results_to_csv(all_runs, num_runs, out, breakdown);
    }

    out.close();
    breakdown.close();

    auto benchmark_end = steady_clock::now();
    double total_runtime_s = duration_cast<microseconds>(benchmark_end - benchmark_start).count() / 1e6;

    prepend_runtime(output_file, total_runtime_s);
    prepend_runtime(breakdown_file, total_runtime_s);

    std::cout << "========================================\n";
    std::cout << "Scaling study complete! (" << std::fixed << std::setprecision(1) << total_runtime_s << " seconds)\n";
    std::cout << "Results saved to: " << output_file.c_str() << "\n";
    std::cout << "Runtime breakdown saved to: " << breakdown_file.c_str() << "\n";
    std::cout << "========================================\n";

    return 0;
}

// Write a vector of scaling_results to the two CSV files (results + breakdown).
// Shared by both generate and file-input modes.
template <typename T>
static void write_results_to_csv(
    const std::vector<scaling_result<T>>& all_runs,
    int64_t num_runs,
    std::ofstream& out,
    std::ofstream& breakdown) {

    for (int64_t run = 0; run < num_runs; ++run) {
        const auto& result = all_runs[run];

        const alg_quality<T>* algos[] = {&result.cqrrt, &result.cholqr, &result.scholqr3, &result.dense_cqrrt};

        // Full-width guard: every algorithm's breakdown must be exactly its
        // fixed slot count (10/5/17/10) on every path (success, driver
        // failure, or skip_dense), or the breakdown row drifts out of
        // alignment with the header. run_algorithms guarantees this by
        // construction; a short vector here is not a corrupted campaign, so
        // it is padded (with the same -1 sentinel the failure paths use) and
        // warned about rather than discarding a finished run. A vector LONGER
        // than the header width cannot be safely truncated without silently
        // dropping real data, so that case still throws: it means something
        // upstream is producing more fields than the schema declares.
        auto fit_breakdown = [](std::vector<long> v, size_t expected, const char* name) {
            if (v.size() > expected) {
                randlapack_require(false)
                    << name << " breakdown width mismatch: expected " << expected
                    << " fields, got " << v.size() << " (exceeds the header width, "
                    << "which indicates real corruption, not a short/skipped row)";
            } else if (v.size() < expected) {
                std::cerr << "Warning: " << name << " breakdown has " << v.size()
                          << " fields, expected " << expected
                          << "; padding with -1 to keep the CSV row full width.\n";
                v.resize(expected, -1L);
            }
            return v;
        };
        std::vector<long> cqrrt_bd    = fit_breakdown(result.cqrrt.breakdown, 10, "CQRRT_linop");
        std::vector<long> cholqr_bd   = fit_breakdown(result.cholqr.breakdown, 5, "CholQR");
        std::vector<long> scholqr3_bd = fit_breakdown(result.scholqr3.breakdown, 17, "sCholQR3");
        std::vector<long> dense_bd    = fit_breakdown(result.dense_cqrrt.breakdown, 10, "CQRRT_expl");

        out << std::fixed << std::setprecision(1)
            << result.m << "," << result.n << "," << run << "," << result.aspect_ratio << ","
            << std::scientific << std::setprecision(6) << result.cond_num << ","
            << std::fixed << std::setprecision(6) << result.density << ","
            << std::scientific << std::setprecision(6);
        for (const auto* q : algos)
            out << q->orth_error << "," << q->max_orth_cols << "," << (q->is_orthonormal ? 1 : 0) << "," << q->time << ",";
        for (int i = 0; i < 4; ++i)
            out << algos[i]->peak_rss_kb << "," << algos[i]->analytical_kb << ",";
        for (int i = 0; i < 4; ++i)
            out << algos[i]->qr_status << ",";
        for (int i = 0; i < 4; ++i)
            out << algos[i]->chol_retries << ","
                << algos[i]->chol_shift_abs << "," << algos[i]->chol_shift_rel << (i < 3 ? "," : "\n");

        breakdown << result.m << "," << result.n << "," << run << ",";
        for (const auto& t : cqrrt_bd)    breakdown << t << ",";
        breakdown << result.cqrrt.time << ",";
        for (const auto& t : cholqr_bd)   breakdown << t << ",";
        breakdown << result.cholqr.time << ",";
        for (const auto& t : scholqr3_bd) breakdown << t << ",";
        breakdown << result.scholqr3.time << ",";
        for (const auto& t : dense_bd)    breakdown << t << ",";
        breakdown << result.dense_cqrrt.time << ","
                  << result.cqrrt.peak_rss_kb << "," << result.cqrrt.analytical_kb << ","
                  << result.cholqr.peak_rss_kb << "," << result.cholqr.analytical_kb << ","
                  << result.scholqr3.peak_rss_kb << "," << result.scholqr3.analytical_kb << ","
                  << result.dense_cqrrt.peak_rss_kb << "," << result.dense_cqrrt.analytical_kb << ","
                  << result.cqrrt.qr_status << "," << result.cholqr.qr_status << ","
                  << result.scholqr3.qr_status << "," << result.dense_cqrrt.qr_status << "\n";
    }
    out.flush();
    breakdown.flush();
}

// Print console summary for a single size's results. Only compares runs
// with qr_status == 0: a failed run's `.time` carries the -1 sentinel, which
// must never be picked as "fastest". best_* is -1 when every run
// for that algorithm failed (or, for CQRRT_expl, when skip_dense=1).
template <typename T>
static void print_console_summary(
    const std::vector<scaling_result<T>>& all_runs,
    int64_t num_runs, int64_t n) {

    int64_t best_cqrrt = -1, best_cholqr = -1, best_scholqr3 = -1, best_dense = -1;
    for (int64_t r = 0; r < num_runs; ++r) {
        if (all_runs[r].cqrrt.qr_status == 0 &&
            (best_cqrrt < 0 || all_runs[r].cqrrt.time < all_runs[best_cqrrt].cqrrt.time)) best_cqrrt = r;
        if (all_runs[r].cholqr.qr_status == 0 &&
            (best_cholqr < 0 || all_runs[r].cholqr.time < all_runs[best_cholqr].cholqr.time)) best_cholqr = r;
        if (all_runs[r].scholqr3.qr_status == 0 &&
            (best_scholqr3 < 0 || all_runs[r].scholqr3.time < all_runs[best_scholqr3].scholqr3.time)) best_scholqr3 = r;
        if (all_runs[r].dense_cqrrt.qr_status == 0 &&
            (best_dense < 0 || all_runs[r].dense_cqrrt.time < all_runs[best_dense].dense_cqrrt.time)) best_dense = r;
    }

    auto print_alg = [&](const char* label, int64_t best, T orth, int64_t max_orth, long time) {
        if (best < 0) { std::cout << "  " << label << ": FAILED on every run\n"; return; }
        std::cout << "  " << label << ": orth_err=" << std::scientific << std::setprecision(2) << orth
                   << ", max_orth=" << max_orth << "/" << n << ", time=" << time << " us (run " << best << ")\n";
    };
    if (best_cqrrt < 0) print_alg("CQRRT_linop", -1, T(0), 0, 0L);
    else print_alg("CQRRT_linop", best_cqrrt, all_runs[best_cqrrt].cqrrt.orth_error, all_runs[best_cqrrt].cqrrt.max_orth_cols, all_runs[best_cqrrt].cqrrt.time);
    if (best_cholqr < 0) print_alg("CholQR     ", -1, T(0), 0, 0L);
    else print_alg("CholQR     ", best_cholqr, all_runs[best_cholqr].cholqr.orth_error, all_runs[best_cholqr].cholqr.max_orth_cols, all_runs[best_cholqr].cholqr.time);
    if (best_scholqr3 < 0) print_alg("sCholQR3   ", -1, T(0), 0, 0L);
    else print_alg("sCholQR3   ", best_scholqr3, all_runs[best_scholqr3].scholqr3.orth_error, all_runs[best_scholqr3].scholqr3.max_orth_cols, all_runs[best_scholqr3].scholqr3.time);
    if (best_dense < 0) std::cout << "  CQRRT_expl : FAILED or skipped (skip_dense) on every run\n";
    else print_alg("CQRRT_expl ", best_dense, all_runs[best_dense].dense_cqrrt.orth_error, all_runs[best_dense].dense_cqrrt.max_orth_cols, all_runs[best_dense].dense_cqrrt.time);

    // mem_str takes `best` only to decide N/A vs formatted; callers pass 0 for
    // rss/akb on the N/A path since those values are never read there (the
    // ternary just needs to typecheck without indexing all_runs[-1]).
    auto mem_str = [](int64_t best, long rss, long akb) {
        return best < 0 ? std::string("N/A") : (std::to_string(rss) + " / " + std::to_string(akb));
    };
    long cqrrt_rss    = best_cqrrt    < 0 ? 0L : all_runs[best_cqrrt].cqrrt.peak_rss_kb;
    long cqrrt_akb    = best_cqrrt    < 0 ? 0L : all_runs[best_cqrrt].cqrrt.analytical_kb;
    long cholqr_rss   = best_cholqr   < 0 ? 0L : all_runs[best_cholqr].cholqr.peak_rss_kb;
    long cholqr_akb   = best_cholqr   < 0 ? 0L : all_runs[best_cholqr].cholqr.analytical_kb;
    long scholqr3_rss = best_scholqr3 < 0 ? 0L : all_runs[best_scholqr3].scholqr3.peak_rss_kb;
    long scholqr3_akb = best_scholqr3 < 0 ? 0L : all_runs[best_scholqr3].scholqr3.analytical_kb;
    long dense_rss    = best_dense    < 0 ? 0L : all_runs[best_dense].dense_cqrrt.peak_rss_kb;
    long dense_akb    = best_dense    < 0 ? 0L : all_runs[best_dense].dense_cqrrt.analytical_kb;
    std::cout << "  Memory (peak RSS / analytical KB):\n";
    std::cout << "    CQRRT_linop: " << mem_str(best_cqrrt, cqrrt_rss, cqrrt_akb)
              << ",  CholQR: " << mem_str(best_cholqr, cholqr_rss, cholqr_akb)
              << ",  sCholQR3: " << mem_str(best_scholqr3, scholqr3_rss, scholqr3_akb)
              << ",  CQRRT_expl: " << mem_str(best_dense, dense_rss, dense_akb)
              << "\n\n";
}

// Write CSV headers shared by both modes. Provenance lines: argv,
// host, timestamp, and the env knobs that change sCholQR3/CQRRT numerics
// without changing any row label (RANDLAPACK_GRAM_LEFT, RANDLAPACK_SCHOLQR3_
// SHIFT, RANDLAPACK_GIT_COMMIT among them, via the shared write_env_line).
static void write_csv_headers(
    std::ofstream& out, std::ofstream& breakdown,
    const std::string& precision, double d_factor,
    int64_t sketch_nnz, int64_t block_size, int64_t num_runs, int num_threads,
    const std::string& extra_comment, const std::string& argv_line) {

    const std::string run_ts = make_run_timestamp();
    const std::string host = get_hostname();

    out << "# CQRRT_linop vs CholQR vs sCholQR3 vs CQRRT_expl Results\n";
    out << "# Date: " << run_ts << "\n";
    out << "# host: " << host << "\n";
    out << "# argv: " << argv_line << "\n";
    out << "# Precision: " << precision << "\n";
    if (!extra_comment.empty()) out << extra_comment;
    out << "# d_factor (CQRRT_linop only): " << d_factor << "\n";
    out << "# sketch_nnz (CQRRT_linop only): " << sketch_nnz << "\n";
    out << "# block_size (CQRRT_linop, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    out << "# num_runs: " << num_runs << "\n";
    out << "# OpenMP threads: " << num_threads << "\n";
    write_env_line(out);
    out << "# qr_status: 0 = success; nonzero = the driver's own failure code.\n";
    out << "#   On failure the algorithm's orth/max_orth/time/chol_* fields for that row\n";
    out << "#   carry the -1 sentinel (never a value-initialized 0, which would misreport\n";
    out << "#   as a perfect result). dense_cqrrt_qr_status = -1 also means skip_dense=1\n";
    out << "#   (CQRRT_expl was never run, distinct from a real driver failure code).\n";
    out << "# chol_retries/chol_shift_abs/chol_shift_rel: every algorithm here (including\n";
    out << "#   dense CQRRT_expl) has an adaptive Cholesky-shift retry mechanism. -1 = the\n";
    out << "#   row failed before a shift record existed, or was skipped (dense_cqrrt under\n";
    out << "#   skip_dense=1); 0 = the mechanism ran and used an unshifted pass.\n";
    out << "# Format: per-run per-algorithm quality metrics (orth_error, max_orth_cols, orth_flag, time), memory (KB)\n";
    out << "m,n,run,aspect_ratio,cond_num,density,"
        << "cqrrt_orth_error,cqrrt_max_orth_cols,cqrrt_is_orth,cqrrt_time_us,"
        << "cholqr_orth_error,cholqr_max_orth_cols,cholqr_is_orth,cholqr_time_us,"
        << "scholqr3_orth_error,scholqr3_max_orth_cols,scholqr3_is_orth,scholqr3_time_us,"
        << "dense_cqrrt_orth_error,dense_cqrrt_max_orth_cols,dense_cqrrt_is_orth,dense_cqrrt_time_us,"
        << "cqrrt_peak_rss_kb,cqrrt_analytical_kb,"
        << "cholqr_peak_rss_kb,cholqr_analytical_kb,"
        << "scholqr3_peak_rss_kb,scholqr3_analytical_kb,"
        << "dense_cqrrt_peak_rss_kb,dense_cqrrt_analytical_kb,"
        << "cqrrt_qr_status,cholqr_qr_status,scholqr3_qr_status,dense_cqrrt_qr_status,"
        << "cqrrt_chol_retries,cqrrt_chol_shift_abs,cqrrt_chol_shift_rel,"
        << "cholqr_chol_retries,cholqr_chol_shift_abs,cholqr_chol_shift_rel,"
        << "scholqr3_chol_retries,scholqr3_chol_shift_abs,scholqr3_chol_shift_rel,"
        << "dense_cqrrt_chol_retries,dense_cqrrt_chol_shift_abs,dense_cqrrt_chol_shift_rel\n";

    breakdown << "# Runtime Breakdown for All Algorithms\n";
    breakdown << "# Date: " << run_ts << "\n";
    breakdown << "# host: " << host << "\n";
    breakdown << "# argv: " << argv_line << "\n";
    breakdown << "# Precision: " << precision << "\n";
    if (!extra_comment.empty()) breakdown << extra_comment;
    breakdown << "# d_factor (CQRRT_linop only): " << d_factor << "\n";
    breakdown << "# sketch_nnz (CQRRT_linop only): " << sketch_nnz << "\n";
    breakdown << "# block_size (CQRRT_linop, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    breakdown << "# num_runs: " << num_runs << "\n";
    breakdown << "# OpenMP threads: " << num_threads << "\n";
    write_env_line(breakdown);
    breakdown << "# Times are in microseconds. On a failed or skipped row every field in\n";
    breakdown << "#   that algorithm's slot group is -1 (full width preserved);\n";
    breakdown << "#   see the per-algorithm *_qr_status column appended at the end.\n";
    // Slot semantics verified against the driver headers (RandLAPACK/drivers/
    // rl_cqrrt.hh, rl_cholqr_linops.hh). The column NAMES below are
    // frozen for CSV-name-based-reader compatibility even where a name now
    // reads stale: cqrrt_trtri really holds the configured precond_method's
    // inversion time (this benchmark always runs TRSM_IDENTITY, never TRTRI);
    // cqrrt_linop_precond/cqrrt_linop_gram really hold the Gram-build's fwd/adj
    // operator applies; cqrrt_trmm_gram really holds the Gram-forming GEMM
    // combine (cholqr_primitive's gemm_dur), not a TRMM; cholqr_materialize/
    // cholqr_gram really hold CholQR's own fwd/adj operator applies.
    breakdown << "# CQRRT_linop: alloc, saso, qr, precond_inv, fwd(gram), adj(gram), gemm(gram combine), chol(potrf), finalize, rest, total\n";
    breakdown << "# CholQR: alloc, fwd(gram), adj(gram), chol(potrf), rest, total\n";
    breakdown << "# sCholQR3: alloc, fwd1, adj1, chol1, upd1, fwd2, adj2, gemm2, chol2, upd2, fwd3, adj3, gemm3, chol3, upd3, q_mat, rest, total\n";
    breakdown << "# CQRRT_expl: materialize, saso, qr, trtri(=0), precond, gram, trmm_gram(=0), potrf, finalize, rest, total\n";
    breakdown << "m,n,run,"
              << "cqrrt_alloc,cqrrt_saso,cqrrt_qr,cqrrt_trtri,cqrrt_linop_precond,cqrrt_linop_gram,cqrrt_trmm_gram,cqrrt_potrf,cqrrt_finalize,cqrrt_rest,cqrrt_total,"
              << "cholqr_alloc,cholqr_materialize,cholqr_gram,cholqr_potrf,cholqr_rest,cholqr_total,"
              << "scholqr3_alloc,scholqr3_fwd1,scholqr3_adj1,scholqr3_chol1,scholqr3_upd1,scholqr3_fwd2,scholqr3_adj2,scholqr3_gemm2,scholqr3_chol2,scholqr3_upd2,scholqr3_fwd3,scholqr3_adj3,scholqr3_gemm3,scholqr3_chol3,scholqr3_upd3,scholqr3_q_mat,scholqr3_rest,scholqr3_total,"
              << "dense_materialize,dense_saso,dense_qr,dense_trtri,dense_precond,dense_gram,dense_trmm_gram,dense_potrf,dense_finalize,dense_rest,dense_total,"
              << "cqrrt_peak_rss_kb,cqrrt_analytical_kb,"
              << "cholqr_peak_rss_kb,cholqr_analytical_kb,"
              << "scholqr3_peak_rss_kb,scholqr3_analytical_kb,"
              << "dense_cqrrt_peak_rss_kb,dense_cqrrt_analytical_kb,"
              << "cqrrt_qr_status,cholqr_qr_status,scholqr3_qr_status,dense_cqrrt_qr_status\n";
}

// Prepend total runtime to a CSV file
static void prepend_runtime(const std::string& filepath, double seconds) {
    std::ifstream fin(filepath);
    std::string content;
    std::string line;
    while (std::getline(fin, line)) {
        content += line + "\n";
    }
    fin.close();
    std::ofstream fout(filepath);
    fout << std::fixed << std::setprecision(1);
    fout << "# Total benchmark runtime: " << seconds << " seconds\n";
    fout << content;
    fout.close();
}

// File-input mode: benchmark a single external Matrix Market matrix
template <typename T>
static int run_benchmark_from_file(int argc, char *argv[]) {
    // Args: <precision> <output_dir> <num_runs> <input_file> <d_factor> [sketch_nnz] [block_size] [compute_cond] [skip_dense]
    std::string precision = argv[1];
    std::string output_dir = argv[2];
    int64_t num_runs = std::stoll(argv[3]);
    std::string input_file = argv[4];
    double d_factor = std::stod(argv[5]);
    int64_t sketch_nnz = (argc >= 7) ? std::stoll(argv[6]) : 4;
    int64_t block_size = (argc >= 8) ? std::stoll(argv[7]) : 0;
    bool compute_cond = (argc >= 9) ? (std::stoi(argv[8]) != 0) : false;
    bool skip_dense = (argc >= 10) ? (std::stoi(argv[9]) != 0) : false;

    // Loud validation: num_runs<=0 leaves `results` empty and
    // every per-run write in run_algorithms/write_results_to_csv is UB
    // (num_runs=0 was silently accepted before).
    if (num_runs < 1) {
        std::cerr << "Error: num_runs must be >= 1 (got " << num_runs << ")\n";
        return 1;
    }

    std::string argv_line = quote_join_argv(argc, argv);
    auto benchmark_start = steady_clock::now();

    // Generate date/time prefix
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    std::cout << "\n=== CQRRT_linop vs CholQR vs sCholQR3 vs CQRRT_expl (File Input) ===\n";
    std::cout << "Precision: " << precision.c_str() << "\n";
    std::cout << "Input file: " << input_file.c_str() << "\n";
    std::cout << "d_factor (CQRRT_linop): " << std::fixed << std::setprecision(2) << d_factor << "\n";
    std::cout << "Sketch nnz (CQRRT_linop): " << sketch_nnz << "\n";
    std::cout << "Block size (CQRRT_linop, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    std::cout << "Compute condition number: " << (compute_cond ? "yes" : "no") << "\n";
    std::cout << "Skip dense CQRRT: " << (skip_dense ? "yes" : "no") << "\n";
    std::cout << "Runs: " << num_runs << "\n";
    std::cout << "OpenMP threads: " << num_threads << "\n";
    std::cout << "=====================================\n\n";

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Extract base name from input file path for output naming
    std::string base_name = input_file;
    auto last_slash = base_name.find_last_of('/');
    if (last_slash != std::string::npos) base_name = base_name.substr(last_slash + 1);
    auto last_dot = base_name.find_last_of('.');
    if (last_dot != std::string::npos) base_name = base_name.substr(0, last_dot);

    std::string output_file = output_dir + "/" + date_prefix + base_name + "_results.csv";
    std::string breakdown_file = output_dir + "/" + date_prefix + base_name + "_breakdown.csv";
    std::ofstream out(output_file);
    std::ofstream breakdown(breakdown_file);

    std::string extra_comment = "# Input file: " + input_file + "\n";
    write_csv_headers(out, breakdown, precision, d_factor,
                      sketch_nnz, block_size, num_runs, num_threads, extra_comment, argv_line);

    // Warmup run with a small synthetic matrix
    {
        std::cout << "Performing warmup run (not reported)...\n";
        auto warmup_state = state;
        run_single_test<T>(1000, 50, (T)1e4, (T)0.1, (T)d_factor, block_size, sketch_nnz, 1, warmup_state);
        std::cout << "Warmup complete, starting measurements.\n\n";
    }

    // Run benchmark on the file-loaded matrix
    std::cout << "Loading matrix from " << input_file.c_str() << "...\n";
    auto all_runs = run_single_test_from_file<T>(input_file, (T)d_factor, block_size, sketch_nnz, num_runs, compute_cond, skip_dense, state);

    int64_t m = all_runs[0].m;
    int64_t n = all_runs[0].n;
    std::cout << "Matrix loaded: " << m << " x " << n << ", density=" << std::fixed << std::setprecision(6) << (double)all_runs[0].density << "\n";

    print_console_summary(all_runs, num_runs, n);
    write_results_to_csv(all_runs, num_runs, out, breakdown);

    out.close();
    breakdown.close();

    auto benchmark_end = steady_clock::now();
    double total_runtime_s = duration_cast<microseconds>(benchmark_end - benchmark_start).count() / 1e6;

    prepend_runtime(output_file, total_runtime_s);
    prepend_runtime(breakdown_file, total_runtime_s);

    std::cout << "========================================\n";
    std::cout << "Benchmark complete! (" << std::fixed << std::setprecision(1) << total_runtime_s << " seconds)\n";
    std::cout << "Results saved to: " << output_file.c_str() << "\n";
    std::cout << "Runtime breakdown saved to: " << breakdown_file.c_str() << "\n";
    std::cout << "========================================\n";

    return 0;
}

int main(int argc, char *argv[]) {
    // Detect mode from argument count:
    //   File-input mode: 6-10 args (precision, output_dir, num_runs, input_file, d_factor, [sketch_nnz], [block_size], [compute_cond], [skip_dense])
    //   Generate mode:  11-13 args (precision, output_dir, num_sizes, num_runs, m_start, m_end, aspect_ratio, cond_num, density, d_factor, [sketch_nnz], [block_size])
    bool is_file_mode = (argc >= 6 && argc <= 10);
    bool is_generate_mode = (argc >= 11 && argc <= 13);

    if (!is_file_mode && !is_generate_mode) {
        std::cerr << "Usage (generate mode):" << std::endl;
        std::cerr << "  " << argv[0]
                  << " <precision> <output_dir> <num_sizes> <num_runs> <m_start> <m_end> <aspect_ratio> <cond_num> <density> <d_factor> [sketch_nnz] [block_size]"
                  << std::endl;
        std::cerr << "\nUsage (file-input mode):" << std::endl;
        std::cerr << "  " << argv[0]
                  << " <precision> <output_dir> <num_runs> <input_file.mtx> <d_factor> [sketch_nnz] [block_size] [compute_cond] [skip_dense]"
                  << std::endl;
        std::cerr << "\n  precision    : 'double' or 'float'" << std::endl;
        std::cerr << "  compute_cond : (Optional, file mode only) 1 = compute condition number via SVD (default: 0)" << std::endl;
        std::cerr << "  skip_dense   : (Optional, file mode only) 1 = skip dense CQRRT_expl (default: 0)" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  " << argv[0] << " double ./output 30 3 1000 30000 100 1e9 0.05 2.0 4 100" << std::endl;
        std::cerr << "  " << argv[0] << " double ./output 3 ./matrix.mtx 2.0" << std::endl;
        std::cerr << "  " << argv[0] << " double ./output 3 ./matrix.mtx 2.0 4 0 1 1  # with cond number, skip dense" << std::endl;
        return 1;
    }

    std::string precision = argv[1];
    if (precision != "double" && precision != "float") {
        std::cerr << "Error: precision must be 'double' or 'float', got '" << precision << "'" << std::endl;
        return 1;
    }

    if (is_file_mode) {
        if (precision == "double") return run_benchmark_from_file<double>(argc, argv);
        else return run_benchmark_from_file<float>(argc, argv);
    } else {
        if (precision == "double") return run_benchmark<double>(argc, argv);
        else return run_benchmark<float>(argc, argv);
    }
}
