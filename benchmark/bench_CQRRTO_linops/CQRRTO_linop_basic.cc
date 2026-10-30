// Q-less QR scaling study on sparse linear operators: CQRRTO_linop, CholQR and sCholQR3 on the
// operator, and dense CQRRT on its materialization (CQRRT_expl).
//
// Generate mode sweeps synthetic tall sparse matrices of a target condition number at a fixed
// aspect ratio m/n; file mode (--input) runs on one Matrix Market matrix. Every method's R is
// checked through Q = A R^{-1}, formed the same way for all of them. Writes
// <timestamp>_scaling_results.csv and _scaling_breakdown.csv (generate mode), or
// <timestamp>_<matrix>_results.csv and _breakdown.csv (file mode). Run with --help for the
// arguments.

#include "RandLAPACK.hh"
#include "RandLAPACK/testing/rl_gen.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "cqrrto_bench_common.hh"
#include "cqrrto_bench_io.hh"

#include <RandBLAS.hh>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace bench = RandLAPACK::bench;
template <typename T>
using CSRLinOp = RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>>;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// Methods in CSV order: column prefix, prefix of the breakdown slot columns, run_qr_method name
// (empty for the dense row), and slot names.
struct Method {
    const char* prefix;
    const char* slot_prefix;
    const char* method;
    std::vector<const char*> slots;
};
static const std::array<Method, 4> kMethods = {{
    {"cqrrto", "cqrrto", "CQRRTO_linop",
     {"alloc", "sketch", "qr", "tri_inv", "fwd", "adj", "left", "chol", "finalize", "rest", "total"}},
    {"cholqr", "cholqr", "CholQR", {"alloc", "fwd", "adj", "chol", "rest", "total"}},
    {"scholqr3", "scholqr3", "sCholQR3",
     {"alloc", "fwd1", "adj1", "chol1", "upd1", "fwd2", "adj2", "left2", "chol2", "upd2",
      "fwd3", "adj3", "left3", "chol3", "upd3", "q_mat", "rest", "total"}},
    {"dense_cqrrt", "dense", "",
     {"materialize", "sketch", "qr", "tri_inv", "precond", "gram", "left", "chol", "finalize", "rest", "total"}},
}};

// One method's outcome in one run. -1 marks a value that does not apply or was not computed.
template <typename T>
struct AlgResult {
    int     qr_status = -1;          // 0 = success, else the driver's code; -1 = not run
    T       orth_error = -1;         // ||Q^T Q - I||_F / sqrt(n)
    int64_t max_orth_cols = -1;      // longest orthonormal prefix of Q
    bool    is_orthonormal = false;  // orth_error <= eps^0.75
    long    time = -1;               // total (us)
    long    peak_rss_kb = -1;
    long    analytical_kb = -1;
    std::vector<long> breakdown;     // the method's slots, total last; empty unless it succeeded
    int     chol_retries = -1;       // -1 also for the dense row, which has no shift retry
    double  chol_shift_abs = -1, chol_shift_rel = -1;
};

template <typename T>
struct RunResult {
    int64_t m, n, run;
    T cond_num, density;
    std::array<AlgResult<T>, 4> alg;
};

// Orthogonality metrics of Q = A R^{-1} for one successful method.
template <typename T, typename GLO>
static void check_Q(GLO& A, const T* R, int64_t m, int64_t n, std::vector<T>& Q, AlgResult<T>& a) {
    RandLAPACK::materialize(A, m, n, Q.data(), m);
    blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans,
               blas::Diag::NonUnit, m, n, T(1), R, n, Q.data(), m);
    a.orth_error     = RandLAPACK::testing::orthogonality_error<T>(Q.data(), m, n);
    a.is_orthonormal = a.orth_error <= std::pow(std::numeric_limits<T>::epsilon(), (T)0.75);
    a.max_orth_cols  = RandLAPACK::testing::max_orthonormal_cols<T>(Q.data(), m, n);
}

// Runs every method num_runs times on A; run r draws its sketch from state with key + r. Each
// method's peak RSS is measured around its own timed call.
template <typename T, typename RNG>
static std::vector<RunResult<T>> run_algorithms(
    CSRLinOp<T>& A, int64_t m, int64_t n, T cond_num, T density, T d_factor, int64_t block_size,
    int64_t sketch_nnz, int64_t num_runs, const RandBLAS::RNGState<RNG>& state, bool skip_dense) {

    std::vector<RunResult<T>> results(num_runs);
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs, state);
    for (int64_t r = 0; r < num_runs; ++r) {
        results[r] = {m, n, r, cond_num, density, {}};
        if (r > 0) run_states[r].key.incr(r);
    }
    std::vector<T> Q(m * n), R(n * n);

    for (size_t k = 0; k < 3; ++k) {
        for (int64_t r = 0; r < num_runs; ++r) {
            AlgResult<T>& a = results[r].alg[k];
            std::fill(R.begin(), R.end(), T(0));
            RandLAPACK::PeakRSSTracker mem;
            mem.start();
            bench::QRRun q = bench::run_qr_method<T, RNG>(kMethods[k].method, A, R.data(), n, block_size,
                                                          sketch_nnz, d_factor, run_states[r]);
            a.peak_rss_kb    = mem.stop();
            a.qr_status      = q.status;
            a.chol_retries   = q.chol_retries;
            a.chol_shift_abs = q.chol_shift_abs;
            a.chol_shift_rel = q.chol_shift_rel;
            if (q.status != 0) {
                std::cerr << "Warning: " << kMethods[k].method << " failed (run " << r
                          << ", status " << q.status << ", Cholesky status " << q.chol_status << ")\n";
                continue;
            }
            a.time          = q.qr_time_us;
            a.breakdown     = q.breakdown;
            a.analytical_kb = q.analytical_kb;
            check_Q(A, R.data(), m, n, Q, a);
        }
    }

    // Dense CQRRT on the materialized operator, with the same per-run sketches as CQRRTO.
    // skip_dense leaves its rows at qr_status = -1 (the m x n copy may not fit in memory).
    if (!skip_dense) {
        const T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
        for (int64_t r = 0; r < num_runs; ++r) {
            AlgResult<T>& a = results[r].alg[3];
            std::fill(R.begin(), R.end(), T(0));
            RandLAPACK::PeakRSSTracker mem;
            mem.start();
            std::unique_ptr<T[]> A_dense(new T[m * n]);
            auto t0 = steady_clock::now();
            RandLAPACK::materialize(A, m, n, A_dense.get(), m);
            const long materialize_us = duration_cast<microseconds>(steady_clock::now() - t0).count();
            RandLAPACK::CQRRT<T, RNG> dense(true, tol);
            dense.compute_Q = false;
            dense.orthogonalization = false;
            dense.nnz = sketch_nnz;
            auto st = run_states[r];
            a.qr_status = dense.call(m, n, A_dense.get(), m, R.data(), n, d_factor, st);
            a.peak_rss_kb = mem.stop();
            A_dense.reset();
            if (a.qr_status != 0) {
                std::cerr << "Warning: CQRRT_expl failed (run " << r << ", status " << a.qr_status << ")\n";
                continue;
            }
            a.time = materialize_us + dense.total_us();
            a.breakdown.assign(1, materialize_us);
            a.breakdown.insert(a.breakdown.end(), dense.times.begin(), dense.times.end() - 1);
            a.breakdown.push_back(a.time);
            a.analytical_kb = RandLAPACK::dense_cqrrt_analytical_kb<T>(m, n, d_factor);
            check_Q(A, R.data(), m, n, Q, a);
        }
    }
    return results;
}

// ============================================================================
// Settings
// ============================================================================

struct Settings {
    std::string precision, out_dir, input_file;   // input_file empty: generate mode
    int64_t num_runs = 1, sketch_nnz = 4, block_size = RandLAPACK::kDefaultGramBlockSize;
    double  d_factor = 2.0;
    // Generate mode.
    int64_t num_sizes = 1, m_start = 0, m_end = 0;
    double  aspect_ratio = 0, cond_num = 0, density = 0;
    // File mode.
    bool    compute_cond = false, skip_dense = false;
};

static void print_usage(const char* exe) {
    std::cerr
        << "Usage (generate mode): " << exe << " --precision=P --out=DIR --m-start=M --m-end=M\n"
        << "                       --aspect-ratio=F --cond=F --density=F [options]\n"
        << "Usage (file mode):     " << exe << " --precision=P --out=DIR --input=A.mtx [options]\n\n"
        << "  --precision=P      double or single (float)\n"
        << "  --out=DIR          output directory\n"
        << "Generate mode:\n"
        << "  --m-start= --m-end=  row counts of the smallest and largest matrix\n"
        << "  --sizes=N          [1]    sizes from m-start to m-end, linearly spaced\n"
        << "  --aspect-ratio=F   m / n, >= 1\n"
        << "  --cond=F           target condition number, >= 1\n"
        << "  --density=F        target density in [0, 1]; bandwidth round(density * n - 1);\n"
        << "                     0 skips the generator's Givens rotations (the top n rows only)\n"
        << "File mode:\n"
        << "  --input=A.mtx      Matrix Market matrix (m >= n)\n"
        << "  --compute-cond     [off]  report cond(A) from a dense SVD\n"
        << "  --skip-dense       [off]  skip CQRRT_expl (the m x n copy may not fit in memory)\n"
        << "Options (default in brackets):\n"
        << "  --runs=N           [1]    runs per matrix, each with its own sketch\n"
        << "  --d-factor=F       [2.0]  sketch rows per column (CQRRTO_linop, CQRRT_expl), >= 1\n"
        << "  --sketch-nnz=N     [4]    nonzeros per sketch column: the generator's matrices have\n"
        << "                            nonuniform leverage scores, which 2 does not sketch reliably\n"
        << "  --block-size=N     [256]  Gram column-block width; 0 = unblocked\n\n"
        << "QR driver options come from the environment (cqrrto_bench_common.hh).\n"
        << "Example: " << exe << " --precision=double --out=./output --m-start=1000 --m-end=30000"
           " --sizes=30 --runs=3 --aspect-ratio=100 --cond=1e9 --density=0.05\n";
}

// Parses and validates the arguments; throws std::runtime_error on an invalid one.
static Settings parse_settings(const bench::BenchArgs& a) {
    a.reject_unknown({"precision", "out", "input", "runs", "d-factor", "sketch-nnz", "block-size",
                      "sizes", "m-start", "m-end", "aspect-ratio", "cond", "density",
                      "compute-cond", "skip-dense", "help"});
    Settings s;
    s.precision  = a.require_str("precision");
    s.out_dir    = a.require_str("out");
    s.input_file = a.str("input", "");
    s.num_runs   = a.i64("runs", s.num_runs);
    s.d_factor   = a.dbl("d-factor", s.d_factor);
    s.sketch_nnz = a.i64("sketch-nnz", s.sketch_nnz);
    s.block_size = a.i64("block-size", s.block_size);
    auto require = [](bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); };
    require(s.precision == "double" || s.precision == "single" || s.precision == "float",
            "--precision must be double or single");
    require(s.num_runs >= 1, "--runs must be >= 1");
    require(s.d_factor >= 1, "--d-factor must be >= 1");
    require(s.sketch_nnz >= 1, "--sketch-nnz must be >= 1");
    require(s.block_size >= 0, "--block-size must be >= 0");
    if (!s.input_file.empty()) {
        for (const char* k : {"sizes", "m-start", "m-end", "aspect-ratio", "cond", "density"})
            require(!a.has(k), std::string("--") + k + " is a generate-mode argument; drop it or --input");
        s.compute_cond = a.boolean("compute-cond", s.compute_cond);
        s.skip_dense   = a.boolean("skip-dense", s.skip_dense);
        return s;
    }
    for (const char* k : {"compute-cond", "skip-dense"})
        require(!a.has(k), std::string("--") + k + " is a file-mode argument (with --input)");
    s.num_sizes    = a.i64("sizes", s.num_sizes);
    s.m_start      = a.i64("m-start", 0);
    s.m_end        = a.i64("m-end", 0);
    s.aspect_ratio = a.dbl("aspect-ratio", 0);
    s.cond_num     = a.dbl("cond", 0);
    s.density      = a.dbl("density", 0);
    for (const char* k : {"m-start", "m-end", "aspect-ratio", "cond", "density"})
        require(a.has(k), std::string("generate mode needs --") + k + " (or give --input)");
    require(s.num_sizes >= 1, "--sizes must be >= 1");
    require(s.m_start >= 1 && s.m_end >= s.m_start, "need 1 <= --m-start <= --m-end");
    require(s.aspect_ratio >= 1, "--aspect-ratio must be >= 1");
    require(s.cond_num >= 1, "--cond must be >= 1");
    require(s.density >= 0 && s.density <= 1, "--density must lie in [0, 1]");
    return s;
}

// The checks that depend on the precision the values are converted to.
template <typename T>
static void check_in_precision(const Settings& s) {
    auto require = [](bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); };
    require(std::isfinite((T)s.d_factor), "--d-factor is not finite in the run's precision");
    require(std::isfinite((T)s.cond_num), "--cond is not finite in the run's precision");
    require(s.density == 0 || (T)s.density > 0, "--density is 0 in the run's precision; pass 0 for no rotations");
}

// ============================================================================
// Output
// ============================================================================

static void write_csv_headers(std::ofstream& out, std::ofstream& breakdown, const Settings& s,
                              const std::string& stamp, const std::string& extra,
                              const std::string& argv_line) {
    std::ostringstream h;
    h << "# Date: " << stamp << "\n"
      << "# argv: " << argv_line << "\n"
      << "# Precision: " << s.precision << "\n"
      << extra
      << "# d_factor and sketch_nnz (CQRRTO_linop, CQRRT_expl): " << s.d_factor << ", " << s.sketch_nnz << "\n"
      << "# block_size (CQRRTO_linop, CholQR, sCholQR3): " << s.block_size << " (0 = unblocked)\n"
      << "# num_runs: " << s.num_runs << "\n"
      << "# OpenMP threads: " << RandLAPACK::util::get_omp_threads() << "\n";
    bench::write_env_lines(h);
    h << "# qr_status: 0 = success, else the driver's failure code; -1 = not run (CQRRT_expl under\n"
      << "#   --skip-dense). A failed method keeps its qr_status, peak_rss_kb and chol_* (the passes\n"
      << "#   that ran) and writes is_orth 0 and -1 in its other fields; a skipped one writes\n"
      << "#   is_orth 0 and -1 elsewhere. chol_* are -1 for CQRRT_expl, which has no shift retry.\n";

    out << "# CQRRTO_linop vs CholQR vs sCholQR3 vs CQRRT_expl results\n" << h.str() << "m,n,run,aspect_ratio,cond_num,density";
    for (const auto& mt : kMethods)
        out << "," << mt.prefix << "_orth_error," << mt.prefix << "_max_orth_cols,"
            << mt.prefix << "_is_orth," << mt.prefix << "_time_us";
    for (const auto& mt : kMethods) out << "," << mt.prefix << "_peak_rss_kb," << mt.prefix << "_analytical_kb";
    for (const auto& mt : kMethods) out << "," << mt.prefix << "_qr_status";
    for (const auto& mt : kMethods)
        out << "," << mt.prefix << "_chol_retries," << mt.prefix << "_chol_shift_abs," << mt.prefix << "_chol_shift_rel";
    out << "\n";

    breakdown << "# Runtime breakdown (microseconds) of each method's driver slots\n" << h.str() << "m,n,run";
    for (const auto& mt : kMethods)
        for (const char* slot : mt.slots) breakdown << "," << mt.slot_prefix << "_" << slot;
    for (const auto& mt : kMethods) breakdown << "," << mt.prefix << "_peak_rss_kb," << mt.prefix << "_analytical_kb";
    for (const auto& mt : kMethods) breakdown << "," << mt.prefix << "_qr_status";
    breakdown << "\n";
}

template <typename T>
static void write_results(const std::vector<RunResult<T>>& runs, std::ofstream& out, std::ofstream& breakdown) {
    for (const auto& r : runs) {
        out << r.m << "," << r.n << "," << r.run << "," << std::fixed << std::setprecision(1)
            << (double)r.m / (double)r.n << "," << std::scientific << std::setprecision(6) << r.cond_num << ","
            << std::fixed << std::setprecision(6) << r.density << std::scientific << std::setprecision(6);
        for (const auto& a : r.alg)
            out << "," << a.orth_error << "," << a.max_orth_cols << "," << (a.is_orthonormal ? 1 : 0) << "," << a.time;
        for (const auto& a : r.alg) out << "," << a.peak_rss_kb << "," << a.analytical_kb;
        for (const auto& a : r.alg) out << "," << a.qr_status;
        for (const auto& a : r.alg) out << "," << a.chol_retries << "," << a.chol_shift_abs << "," << a.chol_shift_rel;
        out << "\n";

        breakdown << r.m << "," << r.n << "," << r.run;
        for (size_t k = 0; k < kMethods.size(); ++k) {
            const auto& v = r.alg[k].breakdown;
            for (size_t i = 0; i < kMethods[k].slots.size(); ++i) breakdown << "," << (i < v.size() ? v[i] : -1L);
        }
        for (const auto& a : r.alg) breakdown << "," << a.peak_rss_kb << "," << a.analytical_kb;
        for (const auto& a : r.alg) breakdown << "," << a.qr_status;
        breakdown << "\n";
    }
    out.flush();
    breakdown.flush();
}

// The fastest successful run of each method.
template <typename T>
static void print_summary(const std::vector<RunResult<T>>& runs, int64_t n) {
    for (size_t k = 0; k < kMethods.size(); ++k) {
        const AlgResult<T>* best = nullptr;
        for (const auto& r : runs)
            if (r.alg[k].qr_status == 0 && (best == nullptr || r.alg[k].time < best->time)) best = &r.alg[k];
        std::cout << "  " << std::left << std::setw(12) << kMethods[k].prefix << std::right;
        if (best == nullptr) { std::cout << ": failed or skipped on every run\n"; continue; }
        std::cout << ": orth_err=" << std::scientific << std::setprecision(2) << best->orth_error
                  << ", max_orth=" << best->max_orth_cols << "/" << n << ", time=" << best->time
                  << " us, peak RSS / model = " << best->peak_rss_kb << " / " << best->analytical_kb << " KB\n";
    }
    std::cout << "\n";
}

static void prepend_runtime(const std::string& path, double seconds) {
    std::string content;
    {
        std::ifstream in(path);
        std::ostringstream ss;
        ss << in.rdbuf();
        content = ss.str();
    }
    std::ofstream out(path);
    out << "# Total benchmark runtime: " << std::fixed << std::setprecision(1) << seconds << " seconds\n" << content;
}

// ============================================================================
// Modes
// ============================================================================

template <typename T, typename RNG>
static std::vector<RunResult<T>> run_generated(int64_t m, int64_t n, T cond_num, T density, T d_factor,
                                               int64_t block_size, int64_t sketch_nnz, int64_t num_runs,
                                               RandBLAS::RNGState<RNG>& state) {
    auto run_state = state;
    auto A_coo = RandLAPACK::gen::gen_sparse_cond_coo<T>(m, n, cond_num, state, density);
    RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
    CSRLinOp<T> A(m, n, A_csr);
    return run_algorithms<T, RNG>(A, m, n, cond_num, (T)A_coo.nnz / ((T)m * n), d_factor, block_size,
                                  sketch_nnz, num_runs, run_state, false);
}

template <typename T>
static int run_benchmark(const Settings& s, const std::string& argv_line) {
    using RNG = r123::Philox4x32;
    const bool file_mode = !s.input_file.empty();
    const std::string stamp = bench::make_run_timestamp();
    auto benchmark_start = steady_clock::now();
    auto state = RandBLAS::RNGState<RNG>();

    // The output files are opened first, then file mode reads and checks its matrix, all before
    // the warmup.
    int64_t m = 0, n = 0, nnz = 0;
    std::unique_ptr<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_csr;
    std::vector<std::pair<int64_t, int64_t>> sizes;
    std::string results_file, breakdown_file;
    std::ostringstream extra;
    if (file_mode) {
        const std::string base = std::filesystem::path(s.input_file).stem().string();
        results_file   = s.out_dir + "/" + stamp + "_" + base + "_results.csv";
        breakdown_file = s.out_dir + "/" + stamp + "_" + base + "_breakdown.csv";
        extra << "# Input file: " << s.input_file << "\n";
    } else {
        for (int64_t i = 0; i < s.num_sizes; ++i) {
            int64_t mi = (s.num_sizes == 1) ? s.m_start : s.m_start + (s.m_end - s.m_start) * i / (s.num_sizes - 1);
            sizes.push_back({mi, std::max<int64_t>(1, (int64_t)(mi / s.aspect_ratio))});
        }
        results_file   = s.out_dir + "/" + stamp + "_scaling_results.csv";
        breakdown_file = s.out_dir + "/" + stamp + "_scaling_breakdown.csv";
        extra << "# Fixed aspect ratio: " << s.aspect_ratio << ":1\n# Condition number: " << s.cond_num
              << "\n# Target density: " << s.density << "\n";
    }
    std::ofstream out(results_file), breakdown(breakdown_file);
    if (!out || !breakdown) {
        std::cerr << "Error: cannot write " << results_file << " and " << breakdown_file << "\n";
        return 1;
    }
    if (file_mode) {
        A_csr = std::make_unique<RandBLAS::sparse_data::csr::CSRMatrix<T>>(
            bench::load_csr<T>(s.input_file, m, n, nnz));
        if (m < n || n < 1) {
            std::cerr << "Error: " << s.input_file << " is " << m << " x " << n << "; need m >= n >= 1\n";
            return 1;
        }
    }
    write_csv_headers(out, breakdown, s, stamp, extra.str(), argv_line);

    std::cout << "=== CQRRTO_linop vs CholQR vs sCholQR3 vs CQRRT_expl ("
              << (file_mode ? s.input_file : std::string("generated")) << ", " << s.precision
              << ", d_factor " << s.d_factor << ", sketch_nnz " << s.sketch_nnz << ", block_size "
              << s.block_size << ", " << s.num_runs << " runs, " << RandLAPACK::util::get_omp_threads()
              << " OpenMP threads) ===\n";

    // Untimed warmup so thread pools and allocators are not charged to the first measurement.
    {
        auto ws = state;
        int64_t wm = file_mode ? 1000 : sizes[0].first, wn = file_mode ? 50 : sizes[0].second;
        run_generated<T, RNG>(wm, wn, file_mode ? T(1e4) : (T)s.cond_num, file_mode ? T(0.1) : (T)s.density,
                              (T)s.d_factor, s.block_size, s.sketch_nnz, 1, ws);
    }

    if (file_mode) {
        CSRLinOp<T> A(m, n, *A_csr);
        std::cout << "Matrix " << m << " x " << n << ", nnz " << nnz << "\n";
        T cond = std::numeric_limits<T>::quiet_NaN();
        if (s.compute_cond) {
            std::vector<T> A_dense(m * n);
            RandLAPACK::materialize(A, m, n, A_dense.data(), m);
            auto sigma = RandLAPACK::testing::compute_singular_values<T>(A_dense.data(), m, n);
            cond = sigma[0] / sigma[n - 1];
            std::cout << "cond(A) = " << std::scientific << (double)cond << "\n";
        }
        auto runs = run_algorithms<T, RNG>(A, m, n, cond, (T)nnz / ((T)m * n), (T)s.d_factor, s.block_size,
                                           s.sketch_nnz, s.num_runs, state, s.skip_dense);
        print_summary(runs, n);
        write_results(runs, out, breakdown);
    } else {
        for (size_t i = 0; i < sizes.size(); ++i) {
            auto [mi, ni] = sizes[i];
            std::cout << "Testing " << mi << " x " << ni << " [" << i + 1 << "/" << sizes.size() << "]\n";
            auto runs = run_generated<T, RNG>(mi, ni, (T)s.cond_num, (T)s.density, (T)s.d_factor, s.block_size,
                                              s.sketch_nnz, s.num_runs, state);
            print_summary(runs, ni);
            write_results(runs, out, breakdown);
        }
    }
    out.close();
    breakdown.close();

    const double total_s = duration_cast<microseconds>(steady_clock::now() - benchmark_start).count() / 1e6;
    prepend_runtime(results_file, total_s);
    prepend_runtime(breakdown_file, total_s);
    bench::check_csv_arity(results_file);
    bench::check_csv_arity(breakdown_file);
    std::cout << "Done in " << std::fixed << std::setprecision(1) << total_s << " s\nResults: " << results_file
              << "\nBreakdown: " << breakdown_file << "\n";
    return 0;
}

int main(int argc, char* argv[]) {
    Settings s;
    try {
        bench::BenchArgs a(argc, argv);
        if (argc < 2 || a.has("help")) { print_usage(argv[0]); return argc < 2 ? 1 : 0; }
        s = parse_settings(a);
        if (s.precision == "double") check_in_precision<double>(s);
        else                         check_in_precision<float>(s);
        bench::qr_driver_options();   // reject invalid environment values before any work
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }
    const std::string argv_line = bench::quote_join_argv(argc, argv);
    try {
        return (s.precision == "double") ? run_benchmark<double>(s, argv_line) : run_benchmark<float>(s, argv_line);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
