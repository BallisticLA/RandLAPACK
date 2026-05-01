#if defined(__APPLE__)
int main() {return 0;}
#else
/*
FunNystromPP benchmark — compares funNyström++ against plain Hutchinson+LanczosFA
for estimating tr(f(A)).

Two algorithms are timed and their matvec counts recorded for each matrix size:
  1. FunNystromPP(k, s, d): Nyström rank-k approximation plus Hutchinson correction
     with s samples, each using d Lanczos steps.
  2. Hutchinson+LanczosFA(s, d): plain Hutchinson with s Rademacher samples,
     each using LanczosFA with d steps to apply f(A). No Nyström component.
     Uses the same s and d as FunNystromPP (equal correction-phase budget).

Matrix input (mat_or_file):
  1         = psd_alg:     A = Q diag(λ) Q', λᵢ = 100 * i^{-2}   (algebraic decay, κ ≈ n²/100)
  2         = psd_exp:     A = Q diag(λ) Q', λᵢ = exp(-i/100)    (exponential decay)
  3         = rbf_kernel:  K[i,j] = exp(-||x_i-x_j||²/(2σ²)), x_i~N(0,I_d), σ=sqrt(d)
  4         = poly_kernel: K[i,j] = (x_i·x_j + 1)², x_i~N(0,I_d)
  /path.txt = load dense square symmetric matrix from space-separated text file
              (row-major, full matrix stored; upper triangle used by algorithms)
              TODO: add .mtx, .m, and other common formats

  For types 1/2: k_mat eigenvalues are nonzero; exact tr(f(A)) is always available.
  For types 3/4: k_mat_ratio reinterpreted as data dimension d = n/k_mat_ratio;
                 reference requires compute_ref=1 (syevd, only feasible for small n).

Function types (func_type):
  0 = sqrt(x)             [f_zero = 0; safe for SPSD matrices]
  1 = log(x)              [requires all eigenvalues > 0; for generated types, noise=1 is applied
                           automatically (A = lowrank + I), so all eigenvalues ≥ 1]
  2 = x*(x + poly_lambda) [degree-2 polynomial; exact in d=2 Lanczos steps; use for tr(K(K+λI))]

Sketch types (sketch_type):
  0 = Gaussian (DenseDist, default)
  1 = SJLT (SparseDist, vec_nnz=4 per column)

compute_ref:
  0 = skip reference (report -1 in output)
  1 = run syevd to compute exact tr(f(A))
  For types 1/2 this flag is ignored; the reference is always available from eigenvalues.
  For types 3/4 and file input, syevd is run only when compute_ref=1.

Both algorithms use a fixed seed so comparisons are reproducible.
Matvec counts reflect actual A applications inside each algorithm (not estimates).
The matrix is constructed/loaded once per n; that cost is not timed.

Output format: 16 comma-separated columns per row (trailing comma):
  n, k, k_mat, s, d,
  matvec_pp, matvec_hutch,
  time_pp_total (us), time_pp_phase1 (us), time_pp_phase2 (us),
  time_hutch (us),
  est_fun_pp, est_hutch,
  true_tr (-1 if unavailable), err_fun_pp, err_hutch

Usage:
  FunNystromPP_benchmark <dir_path> <num_runs> <k_ratio> <k_mat_ratio> <s_ratio>
                         <d_steps> <mat_or_file> <func_type> <compute_ref>
                         <sketch_type> <poly_lambda> [n_1 n_2 ...]
    dir_path    : output directory (use "." for current directory)
    num_runs    : number of timed repetitions per matrix size
    k_ratio     : k = n / k_ratio       (Nyström rank; e.g., 10 -> k = n/10)
    k_mat_ratio : k_mat = n / k_mat_ratio (matrix rank for generated types; ignored for file)
    s_ratio     : s = k * s_ratio       (Hutchinson samples; e.g., 5 -> s = 5*k)
    d_steps     : Lanczos depth per Hutchinson sample
    mat_or_file : 1=psd_alg, 2=psd_exp, or path to .txt file
    func_type   : 0=sqrt, 1=log, 2=poly (x*(x+poly_lambda))
    compute_ref : 0=skip eigendecomp reference, 1=run syevd (file input only)
    sketch_type : 0=Gaussian, 1=SJLT (vec_nnz=4 per column)
    poly_lambda : lambda parameter for func_type=2 (ignored for func_type=0,1)
    n_1 ...     : matrix dimensions (generated types only; ignored for file input)
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace linops = RandLAPACK::linops;
using RNG = r123::Philox4x32;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// ============================================================================
// Benchmark data struct
// ============================================================================

template <typename T>
struct FunNystromPP_benchmark_data {
    int64_t n;      // current matrix dimension
    int64_t k;      // Nyström rank for algorithms
    int64_t k_mat;  // rank used to construct A (0 for file input)
    int64_t s;      // Hutchinson samples
    int64_t d;      // Lanczos steps per sample
    bool from_file; // true when A was loaded from a file
    T noise;        // diagonal shift (noise=1 for func_type=1 on lowrank types; else 0)
    int64_t d_dim;  // data dimension for kernel types (mat_type=3/4); 0 otherwise
    std::vector<T> A;       // n_max × n_max storage (upper triangle; rest zero)
    std::vector<T> eigvals; // k_mat eigenvalues of the low-rank component (pre-shift; empty for kernel/file types)

    FunNystromPP_benchmark_data(int64_t n_max, int64_t k_, int64_t k_mat_,
                                int64_t s_, int64_t d_, bool from_file_) :
        n(n_max), k(k_), k_mat(k_mat_), s(s_), d(d_), from_file(from_file_),
        noise((T)0.0), d_dim(0), A(n_max * n_max, 0.0)
    {}
};

// ============================================================================
// CountingLinOp: wraps any SLO and counts individual matrix-vector products.
// Satisfies the SymmetricLinearOperator concept.
// ============================================================================

template <typename SLO_t>
struct CountingLinOp {
    using scalar_t = typename SLO_t::scalar_t;
    const int64_t dim;
    SLO_t& base;
    int64_t count = 0;

    CountingLinOp(int64_t n, SLO_t& slo) : dim(n), base(slo) {}
    void reset() { count = 0; }

    void operator()(Layout layout, int64_t n_vecs, scalar_t alpha,
                    scalar_t* const B, int64_t ldb,
                    scalar_t beta, scalar_t* C, int64_t ldc) {
        count += n_vecs;
        base(layout, n_vecs, alpha, B, ldb, beta, C, ldc);
    }
};

// ============================================================================
// data_regen: build or load A for the current n.
//   Generated (mat_type 1/2): fills data.eigvals and the upper triangle of data.A.
//   File input: reads the full matrix into data.A via process_input_mat.
// ============================================================================

template <typename T>
static void data_regen(
    FunNystromPP_benchmark_data<T>& data,
    RandBLAS::RNGState<RNG>& state,
    int mat_type,
    int func_type,
    const std::string& file_path
) {
    int64_t n = data.n;
    std::fill(data.A.begin(), data.A.begin() + n * n, (T)0.0);

    if (data.from_file) {
        // Two-pass: dimensions already determined in main; just load data.
        int64_t m = n, ncols = n;
        int wq = 0;
        RandLAPACK::gen::process_input_mat<T>(
            m, ncols, data.A.data(),
            const_cast<char*>(file_path.c_str()), wq);
        data.eigvals.clear();
        data.noise = (T)0.0;
        data.d_dim = 0;
    } else if (mat_type >= 3) {
        // Kernel matrix: k_mat reinterpreted as data dimension d_dim = n / k_mat_ratio.
        data.d_dim = data.k_mat;
        data.noise = (T)0.0;
        data.eigvals.clear();
        T bandwidth = std::sqrt((T)data.d_dim);
        if (mat_type == 3)
            RandLAPACK::gen::gen_kernel_matrix<T, RNG>(
                n, data.d_dim, data.A.data(), n, 0, bandwidth, (T)0.0, 0, state);
        else  // mat_type == 4
            RandLAPACK::gen::gen_kernel_matrix<T, RNG>(
                n, data.d_dim, data.A.data(), n, 1, (T)0.0, (T)1.0, 2, state);
    } else {
        data.d_dim = 0;
        data.eigvals.resize(data.k_mat);
        if (mat_type == 1)
            RandLAPACK::gen::gen_alg_decay_singvals(data.k_mat, (T)100.0, (T)2.0,
                                                     data.eigvals.data());
        else
            RandLAPACK::gen::gen_exp_decay_singvals(data.k_mat, (T)1.0, (T)100.0,
                                                     data.eigvals.data());
        if (func_type == 1) {
            // log requires strict PD; shift by 1 so all eigenvalues >= 1.
            // Models A = I + K (log det(I+K) use case). f(noise) = log(1) = 0,
            // so f_zero=0 remains correct for the Nyström tail approximation.
            data.noise = (T)1.0;
            RandLAPACK::gen::gen_shifted_lowrank_psd(n, data.k_mat, data.A.data(), n,
                                                      data.eigvals.data(), data.noise, state);
        } else {
            data.noise = (T)0.0;
            RandLAPACK::gen::gen_sym_psd_lowrank(n, data.k_mat, data.A.data(), n,
                                                  data.eigvals.data(), state);
        }
    }
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
        : dim(n), A(A_), lfa(lfa_), f(f_), d(d_), Z_buf() {}

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
// call_all_algs: timed + matvec-counted comparison for one matrix size.
// data.A and (for generated types) data.eigvals must already be filled.
// ============================================================================

template <typename T>
static void call_all_algs(
    int64_t numruns,
    FunNystromPP_benchmark_data<T>& data,
    RandBLAS::RNGState<RNG>& state,
    int mat_type,
    int func_type,
    double poly_lambda,
    bool compute_ref,
    int sketch_type,
    const std::string& output_filename
) {
    int64_t n = data.n, k = data.k, s = data.s, d = data.d;

    std::function<double(double)> f;
    const char* fname;
    double f_zero = 0.0;
    if (func_type == 0) {
        f     = [](double x){ return std::sqrt(std::max(x, 0.0)); };
        fname = "sqrt";
        f_zero = 0.0;
    } else if (func_type == 1) {
        f     = [](double x){ return std::log(x); };
        fname = "log";
        f_zero = 0.0;  // placeholder; caller must ensure λ_min > 0
    } else {
        double lam = poly_lambda;
        f     = [lam](double x){ return x * (x + lam); };
        fname = "poly";
        f_zero = 0.0;
    }

    // Reference: exact formula for low-rank types; syevd for kernel/file types.
    bool ref_available = false;
    double true_tr = 0.0;
    if (!data.from_file && mat_type < 3) {
        // Low-rank types: exact reference from eigenvalues.
        // Actual eigenvalues: eigvals[j] + noise (dominant k_mat) and noise (tail).
        // f(noise) = 0 for all func_types: sqrt(0)=0, log(1)=0, 0*(0+lam)=0.
        for (int64_t j = 0; j < data.k_mat; ++j)
            true_tr += f(data.eigvals[j] + data.noise);
        true_tr += (T)(n - data.k_mat) * f(data.noise);
        ref_available = true;
    } else if (compute_ref) {
        // Kernel types and file input: compute reference via full eigendecomposition.
        std::vector<T> A_copy(data.A.begin(), data.A.begin() + n * n);
        std::vector<T> ev(n);
        lapack::syevd(lapack::Job::NoVec, lapack::Uplo::Upper, n,
                      A_copy.data(), n, ev.data());
        for (int64_t j = 0; j < n; ++j)
            true_tr += f(ev[j]);
        ref_available = true;
    }

    // Build full algorithm stack for FunNystromPP.
    RandLAPACK::SYPS<T, RNG>                                             syps(3, 1, false, false);
    syps.sketch_type = sketch_type;
    RandLAPACK::HQRQ<T>                                                  orth(false, false);
    RandLAPACK::SYRF<RandLAPACK::SYPS<T, RNG>, RandLAPACK::HQRQ<T>>    syrf(syps, orth);
    RandLAPACK::NystromEVD<decltype(syrf)>                                    nystrom_evd(syrf, 0);
    RandLAPACK::LanczosFA<T, RNG>                                        lfa_pp;
    RandLAPACK::Hutchinson<T, RNG>                                       hutch_pp;
    RandLAPACK::FunNystromPP<decltype(nystrom_evd), decltype(lfa_pp), decltype(hutch_pp)>
                                                                         driver(nystrom_evd, lfa_pp, hutch_pp);

    // Separate LanczosFA + Hutchinson for plain baseline.
    RandLAPACK::LanczosFA<T, RNG>   lfa_base;
    RandLAPACK::Hutchinson<T, RNG>  hutch_base;

    using ESLO = linops::ExplicitSymLinOp<T>;
    using CSLO = CountingLinOp<ESLO>;
    using FuncT = std::function<double(double)>;

    const char* mat_str = data.from_file ? "file" :
                          mat_type == 1  ? "psd_alg"     :
                          mat_type == 2  ? "psd_exp"     :
                          mat_type == 3  ? "rbf_kernel"  : "poly_kernel";

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d, N=%lld, K=%lld, K_MAT=%lld, S=%lld, D=%lld  [%s, %s]\n",
               i, (long long)n, (long long)k, (long long)data.k_mat,
               (long long)s, (long long)d, fname, mat_str);

        auto state_alg = state;

        // ---- FunNystromPP --------------------------------------------------
        ESLO A_op(n, blas::Uplo::Upper, data.A.data(), n, Layout::ColMajor);
        CSLO A_pp(n, A_op);

        RandLAPACK::FunNystromPP_timing pp_timing;
        auto start_pp = steady_clock::now();
        double est_pp = driver.call(A_pp, f, (T)f_zero, k, s, d, state_alg, &pp_timing);
        auto stop_pp  = steady_clock::now();
        long dur_pp   = duration_cast<microseconds>(stop_pp - start_pp).count();
        int64_t mv_pp = A_pp.count;
        printf("  FunNystromPP:   est=%.6e  time=%lld us  (ph1=%lld ph2=%lld)  matvecs=%lld\n",
               est_pp, (long long)dur_pp,
               (long long)pp_timing.phase1_us, (long long)pp_timing.phase2_us,
               (long long)mv_pp);

        // ---- Hutchinson + LanczosFA (no Nyström) ---------------------------
        state_alg = state;
        ESLO A_op2(n, blas::Uplo::Upper, data.A.data(), n, Layout::ColMajor);
        CSLO A_hutch(n, A_op2);
        LFAOp<CSLO, RandLAPACK::LanczosFA<T, RNG>, FuncT> lfa_op(n, A_hutch, lfa_base, f, d);

        auto start_h = steady_clock::now();
        double est_h = hutch_base.call(lfa_op, s, state_alg);
        auto stop_h  = steady_clock::now();
        long dur_h    = duration_cast<microseconds>(stop_h - start_h).count();
        int64_t mv_h  = A_hutch.count;
        printf("  Hutchinson+LFA: est=%.6e  time=%lld us  matvecs=%lld\n",
               est_h, (long long)dur_h, (long long)mv_h);

        double err_pp = ref_available ? std::abs(est_pp - true_tr) / std::abs(true_tr) : -1.0;
        double err_h  = ref_available ? std::abs(est_h  - true_tr) / std::abs(true_tr) : -1.0;
        if (ref_available)
            printf("  Reference:      true=%.6e  err_pp=%e  err_h=%e\n",
                   true_tr, err_pp, err_h);
        else
            printf("  Reference:      N/A\n");

        double out_tr = ref_available ? true_tr : -1.0;
        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        file << n                     << ",  " << k             << ",  " << data.k_mat  << ",  "
             << s                     << ",  " << d             << ",  "
             << mv_pp                 << ",  " << mv_h          << ",  "
             << dur_pp                << ",  " << pp_timing.phase1_us << ",  " << pp_timing.phase2_us << ",  "
             << dur_h                 << ",  "
             << est_pp                << ",  " << est_h         << ",  "
             << out_tr                << ",  " << err_pp        << ",  " << err_h  << ",\n";
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <dir_path> <num_runs> <k_ratio> <k_mat_ratio> <s_ratio>"
                     " <d_steps> <mat_or_file> <func_type> <compute_ref>"
                     " <sketch_type> <poly_lambda> [n_1 n_2 ...]\n"
                  << "  dir_path    : output directory (use '.' for current dir)\n"
                  << "  num_runs    : timed repetitions per matrix size\n"
                  << "  k_ratio     : k = n / k_ratio       (Nystrom rank)\n"
                  << "  k_mat_ratio : k_mat = n / k_mat_ratio (matrix rank; generated only)\n"
                  << "  s_ratio     : s = k * s_ratio       (Hutchinson samples)\n"
                  << "  d_steps     : Lanczos depth per sample\n"
                  << "  mat_or_file : 1=psd_alg, 2=psd_exp, or path to .txt file\n"
                  << "  func_type   : 0=sqrt, 1=log, 2=poly (x*(x+poly_lambda))\n"
                  << "  compute_ref : 0=skip eigendecomp, 1=run syevd (file input only)\n"
                  << "  sketch_type : 0=Gaussian, 1=SJLT (vec_nnz=4)\n"
                  << "  poly_lambda : lambda for func_type=2 (ignored otherwise)\n"
                  << "  n_1 ...     : matrix sizes (generated types only)\n";
        return 1;
    }

    int64_t numruns     = std::stol(argv[2]);
    double  k_ratio     = std::stod(argv[3]);
    double  k_mat_ratio = std::stod(argv[4]);
    double  s_ratio     = std::stod(argv[5]);
    int64_t d_steps     = std::stol(argv[6]);
    int     func_type   = std::stoi(argv[8]);
    bool    compute_ref = std::stoi(argv[9]) != 0;
    int     sketch_type = std::stoi(argv[10]);
    double  poly_lambda = std::stod(argv[11]);

    // argv[7]: integer mat_type or file path
    bool from_file = false;
    std::string mat_file_path;
    int mat_type = 0;
    try {
        mat_type = std::stoi(argv[7]);
    } catch (...) {
        from_file = true;
        mat_file_path = std::string(argv[7]);
    }

    const char* func_str = func_type == 0 ? "sqrt" : (func_type == 1 ? "log" : "poly");
    const char* mat_str  = from_file    ? mat_file_path.c_str() :
                           mat_type == 1 ? "psd_alg"    :
                           mat_type == 2 ? "psd_exp"    :
                           mat_type == 3 ? "rbf_kernel" : "poly_kernel";
    const char* sketch_str = sketch_type == 1 ? "SJLT" : "Gaussian";

    // Collect n_sizes; for file input, determine n from the file.
    std::vector<int64_t> n_sizes;
    if (from_file) {
        int64_t fm = 0, fn = 0;
        int wq = 1;
        RandLAPACK::gen::process_input_mat<double>(
            fm, fn, nullptr,
            const_cast<char*>(mat_file_path.c_str()), wq);
        if (fm != fn)
            throw std::runtime_error("File matrix must be square (got "
                + std::to_string(fm) + "x" + std::to_string(fn) + ")");
        n_sizes = {fm};
    } else {
        if (argc < 13) {
            std::cerr << "Error: at least one matrix size (n_1) required for generated types.\n";
            return 1;
        }
        for (int i = 12; i < argc; ++i)
            n_sizes.push_back(std::stol(argv[i]));
    }

    std::ostringstream oss;
    for (auto v : n_sizes) oss << v << ", ";
    std::string sizes_str = oss.str();

    auto state          = RandBLAS::RNGState<RNG>();
    auto state_constant = state;

    int64_t n_max     = *std::max_element(n_sizes.begin(), n_sizes.end());
    int64_t k_max     = std::max((int64_t)1, (int64_t)(n_max / k_ratio));
    int64_t k_mat_max = from_file ? 0
                                  : std::max((int64_t)1, (int64_t)(n_max / k_mat_ratio));
    int64_t s_max     = std::max((int64_t)1, (int64_t)(k_max * s_ratio));

    FunNystromPP_benchmark_data<double> all_data(
        n_max, k_max, k_mat_max, s_max, d_steps, from_file);

    std::string output_filename =
        "_FunNystromPP_benchmark_num_info_lines_" + std::to_string(8) + ".txt";
    std::string path;
    if (std::string(argv[1]) != ".")
        path = std::string(argv[1]) + output_filename;
    else
        path = output_filename;

    std::ofstream file(path, std::ios::out | std::ios::app);
    file << "Description: FunNystromPP vs Hutchinson+LanczosFA benchmark for tr(f(A))."
            "\nFile format: 16 columns: n, k, k_mat, s, d, matvec_pp, matvec_hutch,"
            " time_pp_total (us), time_pp_phase1 (us), time_pp_phase2 (us),"
            " time_hutch (us), est_fun_pp, est_hutch,"
            " true_tr (-1 if unavailable), err_fun_pp, err_hutch;"
            " rows = numruns repetitions per matrix size."
            "\nNum OMP threads: " + std::to_string(RandLAPACK::util::get_omp_threads()) +
            "\nInput sizes: " + sizes_str +
            "\nParameters: k_ratio=" + std::string(argv[3]) +
            " k_mat_ratio=" + std::string(argv[4]) +
            " s_ratio=" + std::string(argv[5]) +
            " d=" + std::to_string(d_steps) +
            " mat=" + std::string(mat_str) +
            " func=" + func_str +
            (func_type == 2 ? (" poly_lambda=" + std::to_string(poly_lambda)) : std::string("")) +
            " sketch=" + sketch_str +
            "\nNum runs per size: " + std::to_string(numruns) +
            "\nMatrix construction: " + (from_file ? "loaded from file" :
              mat_type >= 3 ? "gen_kernel_matrix (random data points x_i~N(0,I_d); cost not timed)" :
              "gen_sym_psd_lowrank (Haar-random eigenvectors; cost not timed)") +
            "\n";
    file.flush();

    auto start_all = steady_clock::now();
    for (int64_t n : n_sizes) {
        int64_t k     = std::max((int64_t)1, (int64_t)(n / k_ratio));
        int64_t k_mat = from_file ? 0
                                  : std::max((int64_t)1, (int64_t)(n / k_mat_ratio));
        int64_t s     = std::max((int64_t)1, (int64_t)(k * s_ratio));
        all_data.n     = n;
        all_data.k     = k;
        all_data.k_mat = k_mat;
        all_data.s     = s;
        all_data.d     = d_steps;

        auto state_gen = state_constant;
        data_regen(all_data, state_gen, mat_type, func_type, mat_file_path);

        call_all_algs(numruns, all_data, state_constant, mat_type,
                      func_type, poly_lambda, compute_ref, sketch_type, path);
    }
    auto stop_all = steady_clock::now();
    long dur_all = duration_cast<microseconds>(stop_all - start_all).count();
    file << "Total benchmark execution time: " + std::to_string(dur_all) + "\n";
    file.flush();

    return 0;
}
#endif
