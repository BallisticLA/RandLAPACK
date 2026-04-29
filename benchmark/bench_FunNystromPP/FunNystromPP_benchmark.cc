#if defined(__APPLE__)
int main() {return 0;}
#else
/*
FunNystromPP benchmark — compares funNyström++ against plain Hutchinson+LanczosFA
for estimating tr(sqrt(A)) on a random dense PSD matrix A = B'B + n*I.

Two algorithms are timed for each matrix size n:
  1. FunNystromPP(k, s, d): Nyström rank-k approximation plus Hutchinson correction
     with s samples each using d Lanczos steps.
  2. Hutchinson+LanczosFA(s, d): plain Hutchinson with s Rademacher samples,
     each using LanczosFA with d steps to apply sqrt(A). No Nyström component.
     Uses the same s and d as FunNystromPP (correction-phase budget parity).

Both algorithms use a fixed seed so comparisons are reproducible across sizes.
For n <= 500 a syevd reference is computed to measure relative error.

Usage:
  FunNystromPP_benchmark <dir_path> <num_runs> <k_ratio> <s_ratio> <d_steps> <n_1> [n_2 ...]
    dir_path  : output directory (use "." for current directory)
    num_runs  : number of timed repetitions per matrix size
    k_ratio   : k = n / k_ratio  (Nyström rank; e.g., 10 -> k = n/10)
    s_ratio   : s = k * s_ratio  (Hutchinson samples; e.g., 5 -> s = 5*k)
    d_steps   : Lanczos depth per Hutchinson sample
    n_1 ...   : matrix dimensions to sweep
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <sstream>

namespace linops = RandLAPACK::linops;
using RNG = r123::Philox4x32;

// ============================================================================
// Benchmark data struct
// ============================================================================

template <typename T>
struct FunNystromPP_benchmark_data {
    int64_t n;   // current matrix dimension (may be < A's allocation)
    int64_t k;   // Nyström rank
    int64_t s;   // Hutchinson samples
    int64_t d;   // Lanczos steps per sample
    std::vector<T> A;  // n_max × n_max full symmetric matrix

    FunNystromPP_benchmark_data(int64_t n_max, int64_t k_, int64_t s_, int64_t d_) :
        n(n_max), k(k_), s(s_), d(d_),
        A(n_max * n_max, 0.0)
    {}
};

// ============================================================================
// data_regen: build A = B'B + n*I, fully symmetrized, for the current n.
// ============================================================================

template <typename T>
static void data_regen(
    FunNystromPP_benchmark_data<T>& data,
    RandBLAS::RNGState<RNG>& state
) {
    int64_t n = data.n;
    std::vector<T> B(n * n);
    RandBLAS::DenseDist DB(n, n);
    state = RandBLAS::fill_dense(DB, B.data(), state);

    std::fill(data.A.begin(), data.A.begin() + n * n, (T)0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n,
               (T)1.0, B.data(), n, (T)0.0, data.A.data(), n);
    for (int64_t i = 0; i < n; ++i)
        data.A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            data.A[i + j * n] = data.A[j + i * n];
}

// ============================================================================
// LFAOp: wraps LanczosFA as a SymmetricLinearOperator for plain Hutchinson.
// Computes C = alpha * f(A)*B + beta*C using lfa.call internally.
// ============================================================================

template <typename SLO_t, typename LFA_t, typename F_t>
struct LFAOp {
    using scalar_t = double;
    const int64_t dim;
    SLO_t& A;
    LFA_t& lfa;
    F_t f;
    int64_t d;
    std::vector<double> Z_buf;

    LFAOp(int64_t n, SLO_t& A_, LFA_t& lfa_, F_t f_, int64_t d_)
        : dim(n), A(A_), lfa(lfa_), f(f_), d(d_), Z_buf(n) {}

    void operator()([[maybe_unused]] Layout layout, int64_t n_vecs, double alpha,
                    double* const B, [[maybe_unused]] int64_t ldb,
                    double beta, double* C, [[maybe_unused]] int64_t ldc)
    {
        Z_buf.assign(dim * n_vecs, 0.0);
        lfa.call(A, B, dim, n_vecs, f, d, Z_buf.data());
        if (beta == 0.0) {
            for (int64_t i = 0; i < dim * n_vecs; ++i)
                C[i] = alpha * Z_buf[i];
        } else {
            blas::scal(dim * n_vecs, beta, C, 1);
            blas::axpy(dim * n_vecs, alpha, Z_buf.data(), 1, C, 1);
        }
    }
};

// ============================================================================
// call_all_algs: timed comparison for one matrix size.
// Creates algorithm objects fresh each call (buffers start empty, grow as needed).
// ============================================================================

template <typename T>
static void call_all_algs(
    int64_t numruns,
    FunNystromPP_benchmark_data<T>& data,
    RandBLAS::RNGState<RNG>& state,
    const std::string& output_filename
) {
    int64_t n = data.n, k = data.k, s = data.s, d = data.d;

    // Build full algorithm stack for FunNystromPP.
    RandLAPACK::SYPS<T, RNG>                                             syps(3, 1, false, false);
    RandLAPACK::HQRQ<T>                                                  orth(false, false);
    RandLAPACK::SYRF<RandLAPACK::SYPS<T, RNG>, RandLAPACK::HQRQ<T>>    syrf(syps, orth);
    RandLAPACK::REVD2<decltype(syrf)>                                    revd2(syrf, 0);
    RandLAPACK::LanczosFA<T, RNG>                                        lfa_pp;
    RandLAPACK::Hutchinson<T, RNG>                                       hutch_pp;
    RandLAPACK::FunNystromPP<decltype(revd2), decltype(lfa_pp), decltype(hutch_pp)>
                                                                         driver(revd2, lfa_pp, hutch_pp);

    // Separate LanczosFA + Hutchinson for plain baseline.
    RandLAPACK::LanczosFA<T, RNG>   lfa_base;
    RandLAPACK::Hutchinson<T, RNG>  hutch_base;

    auto f_sqrt = [](double x){ return std::sqrt(std::max(x, 0.0)); };

    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d, N=%ld, K=%ld, S=%ld, D=%ld\n", i, n, k, s, d);

        // ---- FunNystromPP --------------------------------------------------
        state_gen = state;
        data_regen(data, state_gen);
        state_alg = state;
        linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, data.A.data(), n, Layout::ColMajor);

        auto start_pp = steady_clock::now();
        double est_pp = driver.call(A_op, f_sqrt, 0.0, k, s, d, state_alg);
        auto stop_pp  = steady_clock::now();
        long dur_pp   = duration_cast<microseconds>(stop_pp - start_pp).count();
        printf("  FunNystromPP:      est=%.6e  time=%ld us\n", est_pp, dur_pp);

        // ---- Hutchinson + LanczosFA (no Nyström) ---------------------------
        state_gen = state;
        data_regen(data, state_gen);
        state_alg = state;
        linops::ExplicitSymLinOp<T> A_op2(n, blas::Uplo::Upper, data.A.data(), n, Layout::ColMajor);
        LFAOp<linops::ExplicitSymLinOp<T>, RandLAPACK::LanczosFA<T, RNG>, decltype(f_sqrt)>
            lfa_op(n, A_op2, lfa_base, f_sqrt, d);

        auto start_h = steady_clock::now();
        double est_h = hutch_base.call(lfa_op, s, state_alg);
        auto stop_h  = steady_clock::now();
        long dur_h   = duration_cast<microseconds>(stop_h - start_h).count();
        printf("  Hutchinson+LFA:    est=%.6e  time=%ld us\n", est_h, dur_h);

        // ---- Syevd reference (only for small n) ----------------------------
        state_gen = state;
        data_regen(data, state_gen);

        double true_tr = -1.0;
        if (n <= 500) {
            std::vector<T> A_copy(data.A.begin(), data.A.begin() + n * n);
            std::vector<T> eigvals(n);
            lapack::syevd(lapack::Job::NoVec, lapack::Uplo::Upper, n,
                          A_copy.data(), n, eigvals.data());
            true_tr = 0.0;
            for (int64_t j = 0; j < n; ++j)
                true_tr += std::sqrt(std::max(eigvals[j], 0.0));
        }

        double err_pp = (true_tr > 0) ? std::abs(est_pp - true_tr) / true_tr : -1.0;
        double err_h  = (true_tr > 0) ? std::abs(est_h  - true_tr) / true_tr : -1.0;
        if (true_tr > 0)
            printf("  Reference:         true=%.6e  err_pp=%e  err_h=%e\n",
                   true_tr, err_pp, err_h);

        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        file << n    << ",  " << k    << ",  " << s    << ",  " << d    << ",  "
             << dur_pp << ",  " << dur_h  << ",  "
             << est_pp << ",  " << est_h  << ",  "
             << true_tr << ",  " << err_pp << ",  " << err_h  << ",\n";
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <dir_path> <num_runs> <k_ratio> <s_ratio> <d_steps>"
                     " <n_1> [n_2 ...]\n"
                  << "  dir_path : output directory (use '.' for current dir)\n"
                  << "  num_runs : timed repetitions per matrix size\n"
                  << "  k_ratio  : k = n / k_ratio  (Nystrom rank)\n"
                  << "  s_ratio  : s = k * s_ratio  (Hutchinson samples)\n"
                  << "  d_steps  : Lanczos depth per sample\n";
        return 1;
    }

    int64_t numruns = std::stol(argv[2]);
    double  k_ratio = std::stod(argv[3]);
    double  s_ratio = std::stod(argv[4]);
    int64_t d_steps = std::stol(argv[5]);

    std::vector<int64_t> n_sizes;
    for (int i = 6; i < argc; ++i)
        n_sizes.push_back(std::stol(argv[i]));

    std::ostringstream oss;
    for (auto v : n_sizes) oss << v << ", ";
    std::string sizes_str = oss.str();

    auto state          = RandBLAS::RNGState<RNG>();
    auto state_constant = state;

    int64_t n_max = *std::max_element(n_sizes.begin(), n_sizes.end());
    int64_t k_max = std::max((int64_t)1, (int64_t)(n_max / k_ratio));
    int64_t s_max = std::max((int64_t)1, (int64_t)(k_max * s_ratio));

    FunNystromPP_benchmark_data<double> all_data(n_max, k_max, s_max, d_steps);

    // Output file
    std::string output_filename = "_FunNystromPP_benchmark_num_info_lines_7.txt";
    std::string path;
    if (std::string(argv[1]) != ".")
        path = std::string(argv[1]) + output_filename;
    else
        path = output_filename;

    std::ofstream file(path, std::ios::out | std::ios::app);
    file << "Description: FunNystromPP vs Hutchinson+LanczosFA benchmark for tr(sqrt(A)), A = B'B + n*I."
            "\nFile format: 11 columns: n, k, s, d, time_fun_pp (us), time_hutch (us),"
            " est_fun_pp, est_hutch, true_tr (syevd; -1 for n>500), err_fun_pp, err_hutch;"
            " rows = numruns repetitions per matrix size."
            "\nNum OMP threads: " + std::to_string(RandLAPACK::util::get_omp_threads()) +
            "\nInput sizes: " + sizes_str +
            "\nParameters: k_ratio=" + std::string(argv[3]) +
            " s_ratio=" + std::string(argv[4]) +
            " d=" + std::to_string(d_steps) +
            "\nNum runs per size: " + std::to_string(numruns) +
            "\n";
    file.flush();

    auto start_all = steady_clock::now();
    for (int64_t n : n_sizes) {
        int64_t k = std::max((int64_t)1, (int64_t)(n / k_ratio));
        int64_t s = std::max((int64_t)1, (int64_t)(k * s_ratio));
        all_data.n = n;
        all_data.k = k;
        all_data.s = s;
        all_data.d = d_steps;

        call_all_algs(numruns, all_data, state_constant, path);
    }
    auto stop_all = steady_clock::now();
    long dur_all = duration_cast<microseconds>(stop_all - start_all).count();
    file << "Total benchmark execution time: " + std::to_string(dur_all) + "\n";
    file.flush();

    return 0;
}
#endif
