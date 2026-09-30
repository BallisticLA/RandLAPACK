/*
ABRIK speed comparison benchmark: residual against matvec cost for ABRIK, Spectra's
partial SVD and RSVD on one input, dense or sparse.

Every method is evaluated at the same matvec checkpoints: powers of two times the
smallest block size, then the full budget. ABRIK produces its whole curve in one pass
through BK's call and resume (ABRIK::call_with_checkpoints); Spectra and RSVD make one
independent call per checkpoint. GESDD, dense input only, runs once as the reference.

Output CSV (long format, one data point per row):
  run, method, b_sz, total_matvecs, actual_matvecs, err, elapsed_us, k_res, status

  run            = run index in [0, num_runs). ABRIK and RSVD draw a fresh sketch per run;
                   Spectra always starts from the same vector, so its rows repeat.
  method         = ABRIK | Spectra | RSVD | GESDD
  b_sz           = block size (0 for Spectra and GESDD)
  total_matvecs  = the checkpoint budget the method was handed; for ABRIK rounded down
                   to whole blocks
  actual_matvecs = operator applications the method made, counted in columns:
                   ABRIK   block size times completed Krylov iterations, the initial block
                           A*Omega not counted and every block at full width; below the
                           budget only when BK stopped before the checkpoint
                   Spectra twice its A'A (or AA') applications in the Lanczos iteration;
                           recovering the second set of singular vectors is not counted
                   RSVD    metered: two power passes, the range finder and B = Q'A over a
                           sketch of min(budget/2, min(m, n)) columns, about twice the
                           budget while that is below twice the smaller dimension
                   GESDD   0, a direct factorization
  err            = sqrt(||A V S^{-1} - U||_F^2 + ||A' U S^{-1} - V||_F^2) over the leading
                   k_res triplets
  elapsed_us     = ABRIK: cumulative BK time plus the SVD extraction at this and every
                   earlier checkpoint, residual evaluation excluded;
                   Spectra: its Lanczos iteration, forming the singular vectors excluded;
                   RSVD, GESDD: wall clock of that one call
  k_res          = triplets the residual covers, min(target_rank, triplets available)
  status         = ABRIK: why BK stopped at the checkpoint (budget, norm_converged,
                   rank_deficient, saturated); RSVD and GESDD: done or failed; Spectra: done

Usage:
  ABRIK_speed_comparisons <precision> <output_dir> <input_file> <target_rank> <run_gesdd>
                          <budget> <num_runs> <num_block_sizes> <block_sizes...>
                          [sub_ratio] [use_cqrrt]

  precision    = double | float
  input_file   = .mtx (array or coordinate), .bin, or whitespace-delimited text
  target_rank  = triplets the residual is taken over
  run_gesdd    = 1 to run GESDD once on dense input (ignored for sparse input)
  budget       = total matvec budget, at least the smallest block size
  block_sizes  = ABRIK block sizes; RSVD uses the largest
  sub_ratio    = keep the top-left fraction of rows and columns (default 1.0)
  use_cqrrt    = 1 to use CQRRT instead of Householder QR inside ABRIK (default 0)
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_svd_residual.hh"
#include "ext_matrix_io.hh"
#include "budgeted_svd_solver.hh"
#include "abrik_bench_common.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <Eigen/Dense>
#include <algorithm>
#include <climits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

template <typename T> using EMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> using EVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

static const char* kUsage =
    "<precision> <output_dir> <input_file> <target_rank> <run_gesdd> <budget> <num_runs>"
    " <num_block_sizes> <block_sizes...> [sub_ratio] [use_cqrrt]";

// The reference chain RSVD is built on, plus ABRIK.
template <typename T, typename RNG>
struct AlgorithmObjects {
    RandLAPACK::PLUL<T> Stab;
    RandLAPACK::RS<T, RNG> RS;
    RandLAPACK::CholQRQ<T> Orth_RF;
    RandLAPACK::RF<T, RNG> RF;
    RandLAPACK::CholQRQ<T> Orth_QB;
    RandLAPACK::QB<T, RNG> QB;
    RandLAPACK::RSVD<T, RNG> RSVD;
    RandLAPACK::ABRIK<T, RNG> ABRIK;

    AlgorithmObjects(int64_t rsvd_block_sz, T tol)
        : Stab(false, false),
          RS(Stab, 2, 1, false, false),
          Orth_RF(false, false),
          RF(RS, Orth_RF, false, false),
          Orth_QB(false, false),
          QB(RF, Orth_QB, false, false),
          RSVD(QB, rsvd_block_sz),
          ABRIK(false, false, tol)
    {}
};

// Meters the operator applications a solver makes, in columns of the block it is applied
// to. RSVD takes its rank by reference and QB may lower it when it stops early, so an
// analytic count of its passes can overstate what was spent; this counts them.
template <RandLAPACK::linops::LinearOperator LinOp>
struct CountingLinOp {
    using scalar_t = typename LinOp::scalar_t;
    const int64_t n_rows;
    const int64_t n_cols;
    LinOp& inner;
    int64_t columns_applied = 0;

    explicit CountingLinOp(LinOp& op) : n_rows(op.n_rows), n_cols(op.n_cols), inner(op) {}

    void operator()(Layout layout, Op trans_A, Op trans_B, int64_t m, int64_t n, int64_t k,
                    scalar_t alpha, const scalar_t* B, int64_t ldb, scalar_t beta,
                    scalar_t* C, int64_t ldc) {
        columns_applied += n;
        inner(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }
};

static const char* bk_reason_name(RandLAPACK::BKTermination r) {
    switch (r) {
        case RandLAPACK::BKTermination::max_iters_reached: return "budget";
        case RandLAPACK::BKTermination::norm_converged:    return "norm_converged";
        case RandLAPACK::BKTermination::rank_deficient:    return "rank_deficient";
        case RandLAPACK::BKTermination::saturated:         return "saturated";
    }
    return "unknown";
}

// Checkpoints in matvecs: powers of two times the smallest block size, then the budget.
static std::vector<int64_t> make_checkpoint_matvecs(int64_t step, int64_t budget) {
    std::vector<int64_t> cps;
    for (int64_t mv = step; mv < budget; mv *= 2)
        cps.push_back(mv);
    cps.push_back(budget);
    return cps;
}

// Spectra's partial SVD under a matvec budget. Returns the residual over the leading
// target_rank triplets; dur_us and actual_mv report what the call cost and spent. The
// matrix is taken by Eigen reference, so a dense Map is not copied.
template <typename T, typename EigenMatType, RandLAPACK::linops::LinearOperator LinOp>
static T run_svds(const Eigen::Ref<const EigenMatType>& A_eigen, LinOp& A_op,
                  int64_t budget_mv, int64_t target_rank, int64_t& dur_us, int64_t& actual_mv) {
    int64_t nev = target_rank;
    int64_t ncv_default = std::min(2 * nev + 1, std::min(A_op.n_rows, A_op.n_cols) - 1);
    int64_t ncv = BenchmarkUtil::effective_ncv(budget_mv, nev, ncv_default);
    int64_t max_restarts = BenchmarkUtil::budget_to_restarts(budget_mv, nev, ncv);

    auto t0 = steady_clock::now();
    BenchmarkUtil::BudgetedPartialSVDSolver<EigenMatType> svds(A_eigen, nev, ncv);
    svds.compute(max_restarts);
    dur_us = duration_cast<microseconds>(steady_clock::now() - t0).count();
    // Each A'A (or AA') application is two applications of A.
    actual_mv = 2 * (int64_t) svds.num_operations();

    EMatrix<T> U = svds.matrix_U(nev);
    EMatrix<T> V = svds.matrix_V(nev);
    EVector<T> S = svds.singular_values();
    return RandLAPACK::linops::svd_residual<T>(A_op, U.data(), V.data(), S.data(), nev);
}

// Runs every method at every checkpoint, num_runs times, one CSV row per point.
// gesdd_input is the dense matrix for the GESDD reference, or nullptr to skip it.
template <typename T, typename RNG, RandLAPACK::linops::LinearOperator LinOp, typename SvdsFn>
static void run_with_budget(
    LinOp& A_op, SvdsFn svds_fn, T norm_A, T tol, int64_t target_rank, bool use_cqrrt,
    T* gesdd_input, const std::vector<int64_t>& block_sizes, int64_t budget, int num_runs,
    AlgorithmObjects<T, RNG>& algs, std::ofstream& outfile)
{
    using Checkpoint = typename RandLAPACK::ABRIK<T, RNG>::Checkpoint;
    int64_t m = A_op.n_rows;
    int64_t n = A_op.n_cols;
    int64_t min_b = *std::min_element(block_sizes.begin(), block_sizes.end());
    int64_t max_b = *std::max_element(block_sizes.begin(), block_sizes.end());
    std::vector<int64_t> checkpoint_matvecs = make_checkpoint_matvecs(min_b, budget);

    if (use_cqrrt)
        algs.ABRIK.qr_exp = RandLAPACK::ABRIKSubroutines::QR_explicit::cqrrt;

    for (int run = 0; run < num_runs; ++run) {
        printf("\n########## Run %d/%d ##########\n", run + 1, num_runs);
        auto state_run = RandBLAS::RNGState<RNG>(static_cast<uint32_t>(run));

        // ABRIK: one traced run per block size. Checkpoints below one block are skipped,
        // and a budget that is not a multiple of the block size can repeat the previous
        // iteration count, which the trace must not see twice.
        for (auto b_sz : block_sizes) {
            printf("\n=== ABRIK b=%ld (run %d) ===\n", (long) b_sz, run);
            std::vector<int64_t> cp_iters;
            for (auto mv : checkpoint_matvecs) {
                int64_t iters = mv / b_sz;
                if (iters >= 1 && (cp_iters.empty() || iters > cp_iters.back()))
                    cp_iters.push_back(iters);
            }
            if (cp_iters.empty()) {
                fprintf(stderr, "ABRIK b=%ld: no checkpoint reaches one block, skipped\n", (long) b_sz);
                continue;
            }
            auto state_alg = state_run;
            int status = algs.ABRIK.call_with_checkpoints(A_op, b_sz, target_rank, cp_iters,
                [&](const Checkpoint& cp) {
                    outfile << run << ", ABRIK, " << b_sz << ", " << b_sz * cp.iters_requested
                            << ", " << b_sz * cp.iters_done << ", " << cp.residual << ", "
                            << cp.elapsed_us << ", " << cp.k_residual << ", "
                            << bk_reason_name(cp.reason) << "\n";
                    outfile.flush();
                    printf("  mv=%ld  err=%e  t=%lld us  [%s]\n", (long) (b_sz * cp.iters_done),
                           (double) cp.residual, (long long) cp.elapsed_us, bk_reason_name(cp.reason));
                }, state_alg);
            if (status != 0)
                fprintf(stderr, "ABRIK b=%ld run %d: BK failed with status %d, trace ends\n",
                        (long) b_sz, run, status);
        }

        // Spectra: one independent call per checkpoint budget.
        printf("\n=== Spectra (run %d) ===\n", run);
        for (auto budget_mv : checkpoint_matvecs) {
            int64_t dur_svds = 0;
            int64_t actual_mv = 0;
            T err_svds = svds_fn(budget_mv, dur_svds, actual_mv);
            outfile << run << ", Spectra, 0, " << budget_mv << ", " << actual_mv << ", "
                    << err_svds << ", " << dur_svds << ", " << target_rank << ", done\n";
            outfile.flush();
            printf("  mv_req=%ld  mv_actual=%ld  err=%e  t=%lld us\n",
                   (long) budget_mv, (long) actual_mv, (double) err_svds, (long long) dur_svds);
        }

        // RSVD: one independent call per checkpoint budget, largest block size, rank
        // budget/2 capped at the smaller dimension, its operator applications metered.
        // RSVD may deliver fewer triplets than asked; the residual covers the leading
        // min(target_rank, delivered).
        printf("\n=== RSVD b=%ld (run %d) ===\n", (long) max_b, run);
        for (auto budget_mv : checkpoint_matvecs) {
            int64_t k_r = std::min(std::max((int64_t) 1, budget_mv / 2), std::min(m, n));
            T *U_r = nullptr, *V_r = nullptr, *S_r = nullptr;
            auto state_rsvd = state_run;
            CountingLinOp<LinOp> counted(A_op);
            auto t0 = steady_clock::now();
            int status = algs.RSVD.call(counted, norm_A, k_r, tol, U_r, S_r, V_r, state_rsvd);
            int64_t dur_rsvd = duration_cast<microseconds>(steady_clock::now() - t0).count();
            int64_t k_res = (status == 0) ? std::min(target_rank, k_r) : 0;
            T err_rsvd = (status == 0)
                ? RandLAPACK::linops::svd_residual<T>(A_op, U_r, V_r, S_r, k_res)
                : std::numeric_limits<T>::infinity();
            free(U_r); free(V_r); free(S_r);
            outfile << run << ", RSVD, " << max_b << ", " << budget_mv << ", "
                    << counted.columns_applied << ", " << err_rsvd << ", " << dur_rsvd << ", "
                    << k_res << ", " << (status == 0 ? "done" : "failed") << "\n";
            outfile.flush();
            printf("  mv_req=%ld  mv_actual=%ld  k_r=%ld  err=%e  t=%lld us\n", (long) budget_mv,
                   (long) counted.columns_applied, (long) k_r, (double) err_rsvd, (long long) dur_rsvd);
        }

        // GESDD: dense input only, once; deterministic, so reported under run 0.
        if (run == 0 && gesdd_input) {
            printf("\n=== GESDD ===\n");
            // Economy shapes: U is m x r, S has r entries, V^T is r x n, r = min(m, n).
            int64_t r = std::min(m, n);
            T* A_svd = new T[m * n];
            lapack::lacpy(MatrixType::General, m, n, gesdd_input, m, A_svd, m);
            T* U_g  = new T[m * r];
            T* S_g  = new T[r];
            T* VT_g = new T[r * n];
            T* V_g  = new T[n * r];

            auto t0 = steady_clock::now();
            int64_t info = lapack::gesdd(Job::SomeVec, m, n, A_svd, m, S_g, U_g, m, VT_g, r);
            int64_t dur_svd = duration_cast<microseconds>(steady_clock::now() - t0).count();

            T err_SVD = std::numeric_limits<T>::infinity();
            if (info == 0) {
                RandLAPACK::util::transposition(r, n, VT_g, r, V_g, n, 0);
                err_SVD = RandLAPACK::linops::svd_residual<T>(A_op, U_g, V_g, S_g, target_rank);
            } else {
                fprintf(stderr, "GESDD failed with info %ld; err recorded as inf\n", (long) info);
            }
            printf("  err=%e  t=%lld us\n", (double) err_SVD, (long long) dur_svd);
            outfile << "0, GESDD, 0, 0, 0, " << err_SVD << ", " << dur_svd << ", "
                    << target_rank << ", " << (info == 0 ? "done" : "failed") << "\n";
            outfile.flush();
            delete[] A_svd; delete[] U_g; delete[] S_g; delete[] VT_g; delete[] V_g;
        }
    }
}

template <typename T>
static int run_benchmark(int argc, char* argv[]) {
    if (argc < 10) return abrik_usage(argv[0], kUsage);

    std::string output_dir = argv[2];
    std::string input_path = argv[3];
    int64_t target_rank    = std::stoll(argv[4]);
    bool run_gesdd         = (std::stoi(argv[5]) != 0);
    int64_t budget         = std::stoll(argv[6]);
    int num_runs           = std::stoi(argv[7]);
    int num_b_sz           = std::stoi(argv[8]);
    if (num_runs < 1 || num_b_sz < 1 || argc < 9 + num_b_sz) {
        std::cerr << "Error: num_runs and num_block_sizes must be >= 1, with every block size given\n";
        return abrik_usage(argv[0], kUsage);
    }
    std::vector<int64_t> block_sizes;
    for (int i = 0; i < num_b_sz; ++i)
        block_sizes.push_back(std::stoll(argv[9 + i]));
    int args_consumed = 9 + num_b_sz;
    double sub_ratio = (argc > args_consumed)     ? std::stod(argv[args_consumed])     : 1.0;
    bool use_cqrrt   = (argc > args_consumed + 1) ? (std::stoi(argv[args_consumed + 1]) != 0) : false;

    int64_t min_b = *std::min_element(block_sizes.begin(), block_sizes.end());
    int64_t max_b = *std::max_element(block_sizes.begin(), block_sizes.end());
    if (min_b < 1 || target_rank < 1 || budget < min_b || budget > INT_MAX) {
        std::cerr << "Error: need block sizes >= 1, target_rank >= 1, and smallest block size"
                  << " <= budget <= " << INT_MAX << "\n";
        return 2;
    }

    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T) 0.85);
    auto mat = BenchIO::load_matrix<T>(input_path, sub_ratio);
    int64_t m = mat.m;
    int64_t n = mat.n;
    // Spectra needs target_rank < ncv <= min(m, n); this driver caps ncv at min(m, n) - 1,
    // so target_rank + 2 <= min(m, n).
    if (target_rank + 2 > std::min(m, n)) {
        std::cerr << "Error: target_rank " << target_rank << " is too large for a "
                  << m << " x " << n << " input (needs target_rank + 2 <= min(m, n))\n";
        return 2;
    }
    AlgorithmObjects<T, r123::Philox4x32> algs(max_b, tol);

    std::ofstream outfile;
    std::string out_path = abrik_open_csv(output_dir, "ABRIK_speed_comparisons", outfile);
    if (!outfile) return 1;
    outfile << std::setprecision(10);

    std::ostringstream oss_b;
    for (auto v : block_sizes) oss_b << v << ", ";   // comma list, as the readers split it

    outfile << "# ABRIK speed comparisons: residual against matvec cost\n"
            << "# RANDLAPACK_GIT_COMMIT=" << abrik_build_commit() << "\n"
            << "# Precision: " << argv[1] << "\n"
            << "# Input matrix: " << input_path << "\n"
            << "# Input size: " << m << " x " << n << "\n"
            << "# Format: " << (mat.is_sparse ? "sparse" : "dense") << "\n"
            << "# Target rank: " << target_rank << "\n"
            << "# Budget (total matvecs): " << budget << "\n"
            << "# Num runs: " << num_runs << " (ABRIK and RSVD seeds 0..num_runs-1; Spectra is deterministic)\n"
            << "# Block sizes: " << oss_b.str() << "\n"
            << "# ABRIK QR: " << (use_cqrrt ? "CQRRT" : "Householder") << "\n"
            << "# RSVD: largest block size, rank min(budget/2, min(m, n)), operator applications metered\n"
            << "# Tolerance: " << tol << "\n"
            << "# total_matvecs = checkpoint budget (ABRIK: rounded down to whole blocks); actual_matvecs = ABRIK b_sz * iterations done (initial block not counted), Spectra 2 * A'A applications, RSVD metered\n"
            << "# err = sqrt(||A V S^-1 - U||_F^2 + ||A' U S^-1 - V||_F^2) over the leading k_res triplets\n"
            << "# elapsed_us: ABRIK cumulative BK + SVD extraction (residual excluded); Spectra its Lanczos iteration; RSVD and GESDD wall clock of the call\n"
            << "# GESDD runs once on dense input, reported under run 0\n"
            << "# k_res = triplets the residual covers; status: ABRIK = why BK stopped (budget = checkpoint reached), RSVD and GESDD = done or failed, Spectra = done; err = inf when the call failed or no triplet exists\n"
            << "run, method, b_sz, total_matvecs, actual_matvecs, err, elapsed_us, k_res, status\n";
    outfile.flush();

    auto t_total = steady_clock::now();

    if (mat.is_sparse) {
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>> A_op(m, n, *mat.csc);
        T norm_A = A_op.fro_nrm();
        auto svds_fn = [&](int64_t budget_mv, int64_t& dur, int64_t& actual_mv) -> T {
            return run_svds<T, Eigen::SparseMatrix<T>>(*mat.eigen_sparse, A_op, budget_mv,
                                                       target_rank, dur, actual_mv);
        };
        run_with_budget<T>(A_op, svds_fn, norm_A, tol, target_rank, use_cqrrt, nullptr,
                           block_sizes, budget, num_runs, algs, outfile);
    } else {
        T* A_dense = mat.data();
        RandLAPACK::linops::DenseLinOp<T> A_op(m, n, A_dense, m, Layout::ColMajor);
        T norm_A = A_op.fro_nrm();
        Eigen::Map<const EMatrix<T>> A_eigen(A_dense, m, n);
        auto svds_fn = [&](int64_t budget_mv, int64_t& dur, int64_t& actual_mv) -> T {
            return run_svds<T, EMatrix<T>>(A_eigen, A_op, budget_mv, target_rank, dur, actual_mv);
        };
        run_with_budget<T>(A_op, svds_fn, norm_A, tol, target_rank, use_cqrrt,
                           run_gesdd ? A_dense : nullptr,
                           block_sizes, budget, num_runs, algs, outfile);
    }

    int64_t total_us = duration_cast<microseconds>(steady_clock::now() - t_total).count();
    printf("\nTOTAL BENCHMARK TIME: %.2f seconds\n", total_us / 1e6);
    outfile.close();
    if (outfile.fail()) {
        std::cerr << "Error: writing " << out_path << " failed\n";
        return 1;
    }
    printf("Results: %s\n", out_path.c_str());
    return 0;
}

int main(int argc, char* argv[]) {
    return abrik_bench_main(argc, argv, kUsage, run_benchmark<double>, run_benchmark<float>);
}
