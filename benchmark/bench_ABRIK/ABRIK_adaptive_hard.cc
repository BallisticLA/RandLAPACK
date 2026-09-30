/*
ABRIK adaptive-termination benchmark on hard instances.

The other ABRIK benchmarks run to a fixed budget and never exercise the adaptive driver.
This one does, on inputs where a fixed budget is known to be insufficient (the hard
SuiteSparse cases of Tomas, Quintana-Orti and Anzt, doi:10.1177/10943420231179699,
Figure 1, whose reported residual plateaus between 1e-4 and 1e-2 at b = 16, r = 256,
p = 2).

Two modes run per invocation, and they answer different questions.

  sweep     Drive the restarts externally: independent non-adaptive calls at increasing
            Krylov-iteration budgets, the residual evaluated after each, up to the first
            budget that certifies tol. This is the residual-versus-work curve and needs no
            cooperation from the driver.

  adaptive  One call with adaptive = true, the driver growing its own budget by
            adaptive_growth until its certificate over the leading target_rank triplets
            meets tol. This tests the stopping decision: whether it fires where the sweep
            says it should, and what it does when it cannot converge.

The adaptive mode can produce an honest refusal: exhausting adaptive_max_retries with the
residual still above tol is a correct outcome, recorded as status max_retries, not an
error. The same holds for every terminal state BK reports.

Both modes are scored on the same quantity: RandLAPACK::linops::svd_residual over the
leading min(target_rank, triplets available) triplets, which is also what the driver
certifies, because assessed_rank is set to target_rank.

Output CSV (long format, one data point per row):
  run, mode, b_sz, krylov_iters, matvecs, triplets, residual, elapsed_us, status

  run          = run index in [0, num_runs)
  mode         = sweep | adaptive
  b_sz         = Krylov block size
  krylov_iters = BK iterations completed at this data point
  matvecs      = krylov_iters * b_sz, the initial block A*Omega not counted and every block
                 at full width
  triplets     = singular triplets available; the residual covers the leading
                 min(target_rank, triplets)
  residual     = svd_residual over those triplets (inf when there are none)
  elapsed_us   = sweep: wall clock of the non-adaptive call; adaptive: wall clock of the
                 adaptive call, including the driver's own certificate checks. The
                 benchmark's residual evaluation is excluded in both.
  status       = sweep: converged (residual <= tol over target_rank triplets), running
                 (budget exhausted), under_delivered (fewer than target_rank triplets), or
                 BK's terminal state (norm_converged, rank_deficient, saturated);
                 adaptive: converged, max_retries, norm_converged, rank_deficient,
                 under_delivered, saturated, not_adaptive;
                 failed in either mode when the call returned an error

Usage:
  ABRIK_adaptive_hard <precision> <output_dir> <input_file> <target_rank>
                      <tol_exponent> <iters_start> <iters_step> <iters_max>
                      <num_runs> <num_block_sizes> <block_sizes...> [sub_ratio]
                      [adaptive_growth]

  precision       = double | float
  target_rank     = triplets certified and scored
  tol_exponent    = positive: tolerance eps^tol_exponent (0.85 matches the other ABRIK
                    benchmarks); zero or negative: tolerance 10^tol_exponent
  iters_start     = first sweep budget, and the initial budget of the adaptive driver;
                    ceil(iters_start / 2) * b_sz must reach target_rank
  iters_step      = increment between sweep budgets
  iters_max       = upper bound of the sweep budgets. The adaptive run is bounded by a
                    retry count instead, the number of growth steps from iters_start to
                    iters_max, so its last budget can exceed iters_max.
  sub_ratio       = keep the top-left fraction of rows and columns (default 1.0)
  adaptive_growth = budget multiplier per adaptive retry (default 2.0)
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_linops.hh"
#include "rl_svd_residual.hh"
#include "ext_matrix_io.hh"
#include "abrik_bench_common.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

static const char* kUsage =
    "<precision> <output_dir> <input_file> <target_rank> <tol_exponent> <iters_start>"
    " <iters_step> <iters_max> <num_runs> <num_block_sizes> <block_sizes...>"
    " [sub_ratio] [adaptive_growth]";

static const char* bk_reason_name(RandLAPACK::BKTermination r) {
    switch (r) {
        case RandLAPACK::BKTermination::max_iters_reached: return "running";
        case RandLAPACK::BKTermination::norm_converged:    return "norm_converged";
        case RandLAPACK::BKTermination::rank_deficient:    return "rank_deficient";
        case RandLAPACK::BKTermination::saturated:         return "saturated";
    }
    return "unknown";
}

static const char* abrik_reason_name(RandLAPACK::ABRIKTermination r) {
    switch (r) {
        case RandLAPACK::ABRIKTermination::not_adaptive:    return "not_adaptive";
        case RandLAPACK::ABRIKTermination::converged:       return "converged";
        case RandLAPACK::ABRIKTermination::max_retries:     return "max_retries";
        case RandLAPACK::ABRIKTermination::norm_converged:  return "norm_converged";
        case RandLAPACK::ABRIKTermination::rank_deficient:  return "rank_deficient";
        case RandLAPACK::ABRIKTermination::under_delivered: return "under_delivered";
        case RandLAPACK::ABRIKTermination::saturated:       return "saturated";
    }
    return "unknown";
}

// ABRIK allocates U, V and Sigma with new[]; deleting nullptrs is fine.
template <typename T>
static void free_factors(T*& U, T*& V, T*& Sigma) {
    delete[] U;     U     = nullptr;
    delete[] V;     V     = nullptr;
    delete[] Sigma; Sigma = nullptr;
}

template <typename T, typename RNG, RandLAPACK::linops::LinearOperator LinOp>
static void run_instance(
    LinOp& A_op, int64_t target_rank, T tol, int64_t b_sz,
    int iters_start, int iters_step, int iters_max, double adaptive_growth,
    int run, RandBLAS::RNGState<RNG> state_run, std::ofstream& outfile)
{
    // Sweep: one independent call per budget. Repeating work keeps the points
    // independent, which is the question here: what a given budget achieves.
    for (int iters = iters_start; iters <= iters_max; iters += iters_step) {
        RandLAPACK::ABRIK<T, RNG> abrik(false, false, tol);
        abrik.max_krylov_iters = iters;

        T *U = nullptr, *V = nullptr, *Sigma = nullptr;
        auto state_alg = state_run;
        auto t0 = steady_clock::now();
        int status = abrik.call(A_op, b_sz, U, V, Sigma, state_alg);
        int64_t dur = duration_cast<microseconds>(steady_clock::now() - t0).count();

        int64_t triplets = (status == 0) ? abrik.singular_triplets_found : 0;
        T residual = RandLAPACK::linops::svd_residual<T>(A_op, U, V, Sigma,
                                                         std::min(target_rank, triplets));
        bool certified = (status == 0) && triplets >= target_rank && residual <= tol;
        const char* st = (status != 0)            ? "failed"
                       : certified                ? "converged"
                       : (triplets < target_rank) ? "under_delivered"
                                                  : bk_reason_name(abrik.bk_termination_reason);
        int iters_done = abrik.num_krylov_iters;

        outfile << run << ", sweep, " << b_sz << ", " << iters_done << ", "
                << (iters_done * b_sz) << ", " << triplets << ", "
                << residual << ", " << dur << ", " << st << "\n";
        outfile.flush();
        printf("  sweep  b=%ld iters=%4d  triplets=%5ld  res=%.3e  t=%lld us  [%s]\n",
               (long) b_sz, iters_done, (long) triplets, (double) residual, (long long) dur, st);
        free_factors(U, V, Sigma);

        // Nothing further to learn once the certificate is met; the adaptive run reports
        // where the driver itself stops.
        if (certified)
            break;
    }

    // Adaptive: the driver decides. It certifies exactly the target rank, and its retry
    // cap is the number of growth steps from iters_start to iters_max.
    {
        RandLAPACK::ABRIK<T, RNG> abrik(false, false, tol);
        abrik.max_krylov_iters     = iters_start;
        abrik.adaptive             = true;
        abrik.assessed_rank        = target_rank;
        abrik.adaptive_growth      = adaptive_growth;
        abrik.adaptive_max_retries = (iters_max > iters_start)
            ? (int) std::ceil(std::log((double) iters_max / (double) iters_start)
                              / std::log(adaptive_growth)) : 0;

        T *U = nullptr, *V = nullptr, *Sigma = nullptr;
        auto state_alg = state_run;
        auto t0 = steady_clock::now();
        int status = abrik.call(A_op, b_sz, U, V, Sigma, state_alg);
        int64_t dur = duration_cast<microseconds>(steady_clock::now() - t0).count();

        int64_t triplets = (status == 0) ? abrik.singular_triplets_found : 0;
        T residual = RandLAPACK::linops::svd_residual<T>(A_op, U, V, Sigma,
                                                         std::min(target_rank, triplets));
        const char* st = (status != 0) ? "failed" : abrik_reason_name(abrik.termination_reason);
        int iters_done = abrik.num_krylov_iters;

        outfile << run << ", adaptive, " << b_sz << ", " << iters_done << ", "
                << (iters_done * b_sz) << ", " << triplets << ", "
                << residual << ", " << dur << ", " << st << "\n";
        outfile.flush();
        printf("  ADAPT  b=%ld iters=%4d  triplets=%5ld  res=%.3e  t=%lld us  [%s]\n",
               (long) b_sz, iters_done, (long) triplets, (double) residual, (long long) dur, st);
        free_factors(U, V, Sigma);
    }
}

template <typename T>
static int run_benchmark(int argc, char* argv[]) {
    if (argc < 12) return abrik_usage(argv[0], kUsage);

    std::string output_dir = argv[2];
    std::string input_path = argv[3];
    int64_t target_rank    = std::stoll(argv[4]);
    double tol_exponent    = std::stod(argv[5]);
    int iters_start        = std::stoi(argv[6]);
    int iters_step         = std::stoi(argv[7]);
    int iters_max          = std::stoi(argv[8]);
    int num_runs           = std::stoi(argv[9]);
    int num_b_sz           = std::stoi(argv[10]);
    if (num_runs < 1 || num_b_sz < 1 || argc < 11 + num_b_sz) {
        std::cerr << "Error: num_runs and num_block_sizes must be >= 1, with every block size given\n";
        return abrik_usage(argv[0], kUsage);
    }
    if (iters_start < 1 || iters_step < 1 || iters_max < iters_start) {
        std::cerr << "Error: need 1 <= iters_start <= iters_max and iters_step >= 1\n";
        return 2;
    }
    std::vector<int64_t> block_sizes;
    for (int i = 0; i < num_b_sz; ++i)
        block_sizes.push_back(std::stoll(argv[11 + i]));
    if (target_rank < 1 || *std::min_element(block_sizes.begin(), block_sizes.end()) < 1) {
        std::cerr << "Error: target_rank and every block size must be >= 1\n";
        return 2;
    }
    int args_consumed = 11 + num_b_sz;
    double sub_ratio       = (argc > args_consumed)     ? std::stod(argv[args_consumed])     : 1.0;
    double adaptive_growth = (argc > args_consumed + 1) ? std::stod(argv[args_consumed + 1]) : 2.0;
    if (adaptive_growth <= 1.0) {
        std::cerr << "Error: adaptive_growth must exceed 1 (got " << adaptive_growth << ")\n";
        return 2;
    }
    for (auto b_sz : block_sizes) {
        int64_t first_pass = ((iters_start + 1) / 2) * b_sz;
        if (first_pass < target_rank) {
            std::cerr << "Error: iters_start " << iters_start << " at block size " << b_sz
                      << " yields " << first_pass << " triplets, fewer than target_rank "
                      << target_rank << "\n";
            return 2;
        }
    }

    T tol = (tol_exponent > 0)
        ? std::pow(std::numeric_limits<T>::epsilon(), (T) tol_exponent)
        : std::pow((T) 10, (T) tol_exponent);

    auto mat = BenchIO::load_matrix<T>(input_path, sub_ratio);
    int64_t m = mat.m;
    int64_t n = mat.n;

    std::ofstream outfile;
    std::string out_path = abrik_open_csv(output_dir, "ABRIK_adaptive_hard", outfile);
    if (!outfile) return 1;

    std::ostringstream oss_b;
    for (auto v : block_sizes) oss_b << v << ", ";   // comma list, as the readers split it

    outfile << "# ABRIK adaptive-termination benchmark\n"
            << "# RANDLAPACK_GIT_COMMIT=" << abrik_build_commit() << "\n"
            << "# Precision: " << argv[1] << "\n"
            << "# Input matrix: " << input_path << "\n"
            << "# Input size: " << m << " x " << n << "\n"
            << "# Format: " << (mat.is_sparse ? "sparse" : "dense") << "\n"
            << "# target_rank: " << target_rank << "\n"
            << "# tol: " << tol << " (" << (tol_exponent > 0 ? "eps^" : "10^") << tol_exponent << ")\n"
            << "# iters_start: " << iters_start << "  iters_step: " << iters_step
            << "  iters_max: " << iters_max << "\n"
            << "# block_sizes: " << oss_b.str() << "\n"
            << "# num_runs: " << num_runs << " (distinct RNG seeds 0..num_runs-1)\n"
            << "# sub_ratio: " << sub_ratio << "\n"
            << "# adaptive_growth: " << adaptive_growth << "\n"
            << "# sweep = independent non-adaptive calls at increasing budgets; adaptive = one call with the driver growing its own budget, assessed_rank = target_rank\n"
            << "# matvecs = krylov_iters * b_sz, initial block not counted; residual over the leading min(target_rank, triplets)\n"
            << "# elapsed_us excludes the benchmark's residual evaluation; adaptive rows include the driver's own checks\n"
            << "# status max_retries = the driver declined to certify tol, a valid outcome; failed = the call returned an error\n"
            << "# residual = inf when the call failed or no triplet exists\n"
            << "run, mode, b_sz, krylov_iters, matvecs, triplets, residual, elapsed_us, status\n";
    outfile.flush();
    outfile << std::scientific << std::setprecision(8);   // data rows only

    printf("ABRIK adaptive benchmark: %ld x %ld, target_rank %ld, tol %.3e\n",
           (long) m, (long) n, (long) target_rank, (double) tol);

    for (int run = 0; run < num_runs; ++run) {
        RandBLAS::RNGState<r123::Philox4x32> state_run(static_cast<uint32_t>(run));
        for (auto b_sz : block_sizes) {
            printf("\n=== run %d, block size %ld ===\n", run, (long) b_sz);
            if (mat.is_sparse) {
                RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>> A_op(m, n, *mat.csc);
                run_instance<T, r123::Philox4x32>(A_op, target_rank, tol, b_sz, iters_start, iters_step,
                                                  iters_max, adaptive_growth, run, state_run, outfile);
            } else {
                RandLAPACK::linops::DenseLinOp<T> A_op(m, n, mat.data(), m, Layout::ColMajor);
                run_instance<T, r123::Philox4x32>(A_op, target_rank, tol, b_sz, iters_start, iters_step,
                                                  iters_max, adaptive_growth, run, state_run, outfile);
            }
        }
    }

    outfile.close();
    if (outfile.fail()) {
        std::cerr << "Error: writing " << out_path << " failed\n";
        return 1;
    }
    printf("\nWrote %s\n", out_path.c_str());
    return 0;
}

int main(int argc, char* argv[]) {
    return abrik_bench_main(argc, argv, kUsage, run_benchmark<double>, run_benchmark<float>);
}
