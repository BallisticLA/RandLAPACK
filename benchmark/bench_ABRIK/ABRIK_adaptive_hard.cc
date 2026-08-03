/*
ABRIK adaptive-termination benchmark on hard instances.

The other ABRIK benchmarks run to a fixed matvec budget, which measures accuracy
per unit of work but never exercises the adaptive driver: they construct
ABRIK(verbose, timing, tol) and leave the `adaptive` member at its default of
false. This benchmark exists to exercise it, on inputs where a fixed budget is
known to be insufficient (the hard SuiteSparse cases of Tomas, Quintana-Orti and
Anzt, doi:10.1177/10943420231179699, Figure 1, whose reported residual plateaus
between 1e-4 and 1e-2 at a fixed b = 16, r = 256, p = 2).

Two modes run per invocation, and they answer different questions.

  sweep     Drive the restarts externally: call ABRIK non-adaptively at a
            sequence of increasing Krylov-iteration budgets and evaluate the
            residual after each. This yields the full residual-versus-work
            curve, which is what the paper figure needs, and it needs no
            cooperation from the driver.

  adaptive  One call with `adaptive = true`, letting the driver decide its own
            budget from the certificate. This tests the stopping decision
            itself: whether it fires where the sweep says it should, and what it
            does when it cannot converge.

The second mode is the one that can produce an honest refusal. If the driver
exhausts adaptive_max_retries with the residual still above tol, that is a
correct outcome and is recorded as such (status = max_retries), not an error.
The same applies to BK terminating on norm convergence or rank deficiency: the
run is reported with the reason it stopped.

Residuals use RandLAPACK::linops::svd_residual, the per-triplet normalized
two-sided residual that the driver itself uses for its adaptive check, so the
sweep and the adaptive run are scored on exactly the same quantity.

Output CSV (long format, one data point per row):
  run, mode, b_sz, krylov_iters, matvecs, triplets, residual, elapsed_us, status

  run           = run index in [0, num_runs)
  mode          = sweep | adaptive
  b_sz          = Krylov block size
  krylov_iters  = BK iterations consumed at this data point
  matvecs       = krylov_iters * b_sz (products with the operator)
  triplets      = singular triplets available (end_cols); the residual is taken
                  over the leading min(target_rank, triplets) of them
  residual      = svd_residual over those leading triplets
  elapsed_us    = cumulative wall clock for factorization and extraction,
                  excluding the residual evaluation itself
  status        = running | converged | max_retries | norm_converged
                  | rank_deficient | not_converged

Usage:
  ABRIK_adaptive_hard <precision> <output_dir> <input_file> <target_rank>
                      <tol_exponent> <iters_start> <iters_step> <iters_max>
                      <num_runs> <num_block_sizes> <block_sizes...> [sub_ratio]

  tol_exponent = tolerance as eps^exponent (e.g. 0.85 matches the other ABRIK
                 benchmarks; use a smaller exponent for a looser target).
                 NEGATIVE values mean an ABSOLUTE tolerance of 10^value
                 (e.g. -5 -> 1e-5, -10 -> 1e-10; added 2026-08-03).
  iters_start  = first Krylov-iteration budget in the sweep, and the initial
                 budget handed to the adaptive driver
  iters_step   = increment between sweep points (the sweep is additive so the
                 curve is evenly sampled; the adaptive driver doubles instead)
  iters_max    = last sweep point; also bounds the adaptive run, whose retry cap
                 is set to ceil(log2(iters_max / iters_start))
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_svd_residual.hh"
#include "ext_matrix_io.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

using namespace std::chrono;

// Free the factor buffers ABRIK returns. ABRIK allocates U, V and Sigma with
// new[], so they are released the same way; calling this on nullptrs is safe.
template <typename T>
static void free_factors(T*& U, T*& V, T*& Sigma) {
    delete[] U;     U     = nullptr;
    delete[] V;     V     = nullptr;
    delete[] Sigma; Sigma = nullptr;
}

template <typename T, typename RNG, RandLAPACK::linops::LinearOperator LinOp>
static void run_instance(
    LinOp& A_op,
    int64_t target_rank,
    T tol,
    int64_t b_sz,
    int iters_start,
    int iters_step,
    int iters_max,
    int run,
    RandBLAS::RNGState<RNG> state_run,
    std::ofstream& outfile
) {
    // ---- Mode 1: external sweep, one independent call per budget ----
    //
    // Each point is an independent call rather than a resumed one. That costs
    // repeated work, but it keeps the points independent of one another, which
    // matters here because the question is what a given budget achieves, not
    // how cheaply a curve can be produced.
    for (int iters = iters_start; iters <= iters_max; iters += iters_step) {
        RandLAPACK::ABRIK<T, RNG> abrik(false, false, tol);
        abrik.max_krylov_iters = iters;
        abrik.adaptive         = false;

        T *U = nullptr, *V = nullptr, *Sigma = nullptr;
        auto state_alg = state_run;

        auto t0 = steady_clock::now();
        int status = abrik.call(A_op, b_sz, U, V, Sigma, state_alg);
        long dur = duration_cast<microseconds>(steady_clock::now() - t0).count();

        int64_t triplets = abrik.singular_triplets_found;
        int64_t k_eval   = std::min(target_rank, triplets);

        T residual = (k_eval >= 1)
            ? RandLAPACK::linops::svd_residual<T>(A_op, U, V, Sigma, k_eval)
            : std::numeric_limits<T>::infinity();

        const char* st = (status != 0) ? "not_converged"
                       : (residual <= tol) ? "converged" : "running";

        outfile << run << ", sweep, " << b_sz << ", " << iters << ", "
                << (iters * b_sz) << ", " << triplets << ", "
                << residual << ", " << dur << ", " << st << "\n";
        outfile.flush();
        printf("  sweep  b=%ld iters=%4d  triplets=%5ld  res=%.3e  t=%ld us\n",
               b_sz, iters, triplets, residual, dur);

        free_factors(U, V, Sigma);

        // Nothing further to learn from a larger budget once the certificate is
        // satisfied; the adaptive run below is what reports where it stopped.
        if (residual <= tol)
            break;
    }

    // ---- Mode 2: the driver decides for itself ----
    {
        RandLAPACK::ABRIK<T, RNG> abrik(false, false, tol);
        abrik.max_krylov_iters     = iters_start;
        abrik.adaptive             = true;
        // The driver doubles its own budget; cap the retries so it cannot run past
        // iters_max. Doubling from iters_start reaches iters_max in log2 steps.
        abrik.adaptive_max_retries = (iters_max > iters_start)
            ? (int)std::ceil(std::log2((double)iters_max / (double)iters_start)) : 0;

        T *U = nullptr, *V = nullptr, *Sigma = nullptr;
        auto state_alg = state_run;

        auto t0 = steady_clock::now();
        int status = abrik.call(A_op, b_sz, U, V, Sigma, state_alg);
        long dur = duration_cast<microseconds>(steady_clock::now() - t0).count();

        int64_t triplets = abrik.singular_triplets_found;
        int64_t k_eval   = std::min(target_rank, triplets);

        T residual = (k_eval >= 1)
            ? RandLAPACK::linops::svd_residual<T>(A_op, U, V, Sigma, k_eval)
            : std::numeric_limits<T>::infinity();

        // The driver now reports why it stopped, so read it rather than guessing.
        const char* st = "not_converged";
        if (status == 0) {
            switch (abrik.termination_reason) {
                case RandLAPACK::ABRIKTermination::converged:      st = "converged";      break;
                case RandLAPACK::ABRIKTermination::max_retries:    st = "max_retries";    break;
                case RandLAPACK::ABRIKTermination::norm_converged: st = "norm_converged"; break;
                case RandLAPACK::ABRIKTermination::rank_deficient: st = "rank_deficient"; break;
                default:                                           st = "not_adaptive";   break;
            }
        }

        int iters_used = abrik.num_krylov_iters;

        outfile << run << ", adaptive, " << b_sz << ", " << iters_used << ", "
                << (iters_used * b_sz) << ", " << triplets << ", "
                << residual << ", " << dur << ", " << st << "\n";
        outfile.flush();
        printf("  ADAPT  b=%ld iters=%4d  triplets=%5ld  assessed=%ld  res=%.3e  t=%ld us  [%s]\n",
               b_sz, iters_used, triplets, (long)abrik.assessed_rank, residual, dur, st);

        free_factors(U, V, Sigma);
    }
}

template <typename T>
static void run_benchmark(int argc, char* argv[]) {
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input_file> <target_rank>"
                  << " <tol_exponent> <iters_start> <iters_step> <iters_max>"
                  << " <num_runs> <num_block_sizes> <block_sizes...> [sub_ratio]\n";
        return;
    }

    std::string output_dir = argv[2];
    std::string input_path = argv[3];
    int64_t target_rank    = std::stol(argv[4]);
    double tol_exponent    = std::stod(argv[5]);
    int iters_start        = std::stoi(argv[6]);
    int iters_step         = std::stoi(argv[7]);
    int iters_max          = std::stoi(argv[8]);
    int num_runs           = std::stoi(argv[9]);
    int num_b_sz           = std::stoi(argv[10]);

    if (num_runs < 1) {
        std::cerr << "Error: num_runs must be >= 1 (got " << num_runs << ")\n";
        return;
    }
    if (iters_step < 1) {
        std::cerr << "Error: iters_step must be >= 1 (got " << iters_step << ")\n";
        return;
    }
    if (iters_max < iters_start) {
        std::cerr << "Error: iters_max (" << iters_max << ") < iters_start ("
                  << iters_start << ")\n";
        return;
    }

    std::vector<int64_t> block_sizes;
    for (int i = 0; i < num_b_sz; ++i)
        block_sizes.push_back(std::stol(argv[11 + i]));

    int args_consumed = 11 + num_b_sz;
    double sub_ratio  = (argc > args_consumed) ? std::stod(argv[args_consumed]) : 1.0;

    // tol_exponent > 0: tolerance = eps^tol_exponent (historical form).
    // tol_exponent < 0: tolerance = 10^tol_exponent, an ABSOLUTE tolerance
    //   (2026-08-03: lets Rob's adaptive configs "10 triplets at 1e-5 / 1e-10"
    //   be expressed exactly -- pass -5 or -10 -- instead of via eps-exponent
    //   rounding).
    T tol = (tol_exponent > 0)
        ? std::pow(std::numeric_limits<T>::epsilon(), (T)tol_exponent)
        : std::pow((T)10.0, (T)tol_exponent);

    auto mat = BenchIO::load_matrix<T>(input_path, sub_ratio);
    int64_t m = mat.m;
    int64_t n = mat.n;

    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    std::string out_filename = std::string(date_prefix) + "ABRIK_adaptive_hard.csv";
    std::string out_path = (output_dir != ".") ? output_dir + "/" + out_filename : out_filename;
    std::ofstream outfile(out_path);

    std::ostringstream oss_b;
    for (size_t i = 0; i < block_sizes.size(); ++i)
        oss_b << (i ? " " : "") << block_sizes[i];

    outfile << std::scientific << std::setprecision(8);
    outfile << "# ABRIK adaptive-termination benchmark\n"
            << "# input: " << input_path << "\n"
            << "# m: " << m << "  n: " << n << "\n"
            << "# target_rank: " << target_rank << "\n"
            << "# tol: " << tol << " (eps^" << tol_exponent << ")\n"
            << "# iters_start: " << iters_start
            << "  iters_step: " << iters_step
            << "  iters_max: " << iters_max << "\n"
            << "# block_sizes: " << oss_b.str() << "\n"
            << "# num_runs: " << num_runs << "\n"
            << "# sub_ratio: " << sub_ratio << "\n"
            << "# sweep    = independent non-adaptive calls at increasing budgets\n"
            << "# adaptive = one call with adaptive = true; the driver doubles its own budget\n"
            << "# status max_retries = the driver declined to certify tol, which is a valid outcome\n"
            << "# elapsed_us excludes the residual evaluation\n"
            << "run, mode, b_sz, krylov_iters, matvecs, triplets, residual, elapsed_us, status\n";

    printf("ABRIK adaptive benchmark: %ld x %ld, target_rank %ld, tol %.3e\n",
           m, n, target_rank, tol);

    for (int run = 0; run < num_runs; ++run) {
        // A distinct seed per run, so repeated runs differ in the sketch rather
        // than merely repeating one draw.
        RandBLAS::RNGState<r123::Philox4x32> state_run(run);

        for (auto b_sz : block_sizes) {
            printf("\n=== run %d, block size %ld ===\n", run, b_sz);
            if (mat.is_sparse) {
                RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>>
                    A_op(m, n, *mat.csc);
                run_instance<T, r123::Philox4x32>(
                    A_op, target_rank, tol, b_sz,
                    iters_start, iters_step, iters_max, run, state_run, outfile);
            } else {
                T* A_dense = mat.data();
                RandLAPACK::linops::DenseLinOp<T> A_op(m, n, A_dense, m, Layout::ColMajor);
                run_instance<T, r123::Philox4x32>(
                    A_op, target_rank, tol, b_sz,
                    iters_start, iters_step, iters_max, run, state_run, outfile);
            }
        }
    }

    outfile.close();
    printf("\nWrote %s\n", out_path.c_str());
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <precision> ...\n";
        return 1;
    }
    std::string precision = argv[1];
    if (precision == "double") {
        run_benchmark<double>(argc, argv);
    } else if (precision == "single") {
        run_benchmark<float>(argc, argv);
    } else {
        std::cerr << "Error: precision must be 'double' or 'single' (got "
                  << precision << ")\n";
        return 1;
    }
    return 0;
}
