// qfa_micro — QFA micro-benchmark: block vs scalar, adaptive vs fixed (LOCAL).
//
// Implements the FROZEN plan at
//   agent-workspace/randnla/project-plans/2026-09-02-qfa-micro-benchmark-plan.md
// (pass 10, 2026-09-02). Read that plan for the full rationale; this header only
// summarizes the mechanics needed to read the code and the CSV it produces.
//
// PURPOSE: at the ORACLE level (quadratic-form family only, no Nystrom phase, no
// estimator plumbing), compare block vs scalar Lanczos-QFA and adaptive
// (certified) vs fixed depth, on a synthetic A = Q diag(lambda) Q' with known
// closed-form ground truth tr(B' f(A) B).
//
// ARMS (2x2 + one control; see the plan's "The arms" table):
//   scalar-fixed         LanczosQFA,      adaptive=false, depth from the grid
//   scalar-certified     LanczosQFA,      adaptive=true,  per-column Gauss-Radau,
//                         cap = 1024 (fixed constant, NOT n; see plan "Cap asymmetry")
//   block-fixed           BlockLanczosQFA, fixed depth from the grid, reorth=1
//   block-certified        BlockLanczosQFA, stop_rule=Radau, adaptive_rtol=tol,
//                         cap = floor(n/s) (the structural s*d<=n wall)
//   block-fixed-reorth0    BlockLanczosQFA, fixed depth from the grid, reorth=0
//                         (matched-stability control; geo1e6 ONLY, both s, both f,
//                         full grid, all trials -- see plan Runtime item 3)
//
// GROUND TRUTH: A is formed once per (spectrum, kappa) as A = Q diag(lambda) Q',
// with Q a Haar-random orthogonal rotation from QR of a seeded n x n Gaussian
// WITH THE SIGN CORRECTION Q <- Q * sign(diag(R)) (an uncorrected QR-of-Gaussian
// is NOT Haar-distributed -- plan pass 8 F8.12; this correction is applied here
// explicitly because RandLAPACK::gen::gen_spd_from_eigvals does not apply it).
// Per-trial ground truth is exact: truth = sum_j (Q' b_j)' f(Lambda) (Q' b_j),
// computed directly from the stored (Q, lambda) -- never from the oracle output.
//
// RNG SUB-STREAMS (two independently-keyed roles, both documented here and in
// the CSV header comments -- plan "seeds" and pass 2 SS7a):
//   matrix_seed: key = derive_key(base_seed, ROLE_MATRIX, matrix_global_index)
//                matrix_global_index: geo1e3=0, geo1e6=1, logu1e6=2 (fixed,
//                independent of which matrices a given run mode actually visits).
//                Drawn ONCE per (spectrum, kappa) -- fixes the Q rotation AND the
//                spectrum draw for that matrix (the spectrum itself is a closed
//                form in kappa, not randomly drawn, but Q is).
//   probe_seed:  key = derive_key(base_seed, ROLE_PROBE, s_global_index, trial)
//                s_global_index: s=4 -> 0, s=16 -> 1. Drawn once per (s, trial),
//                and DELIBERATELY REUSED across matrix and f: the same trial
//                index always draws the same underlying probe block B (for a
//                given s), regardless of which matrix or f is being evaluated.
//                This is a driver design choice (the plan requires only that the
//                two roles be independently keyed and documented, not that probe
//                realizations differ per matrix/f) that maximizes pairing across
//                matrices and f at a fixed trial index. Within one (matrix, f, s,
//                trial) cell, this SAME probe block B is shared by all five
//                arm-variants (the plan's core paired-design requirement).
//
// CSV SCHEMA (17 columns; NaN literal where a field does not apply to that row --
// see the plan's "NaN convention" for the exact per-field population rules):
//   matrix,f,s,tol,arm,reorth,trial,d_or_cap,matvecs,rel_err,rel_err_midpoint,
//   certified,col_depth_min,col_depth_med,col_depth_max,wall_limited,cap_saturated
//
// METRIC 7 (certificate honesty check) is BOTH a pre-publish console gate (stderr,
// fires immediately on any violation) AND a summary-table input. The frozen CSV
// row schema above has no field for it (tr_L / per-column gauss/radau values are
// not row data), so violation counts are shipped as trailing `# metric7_violation
// ...` comment lines at the END of the CSV (after all data rows), one line per
// (matrix, f, s, tol, arm) cell that had at least one certified row. This is the
// one place this driver extends the plan's literal text with an implementation
// decision rather than a directly-specified mechanism; it is flagged as such in
// this session's report to the plan's owner.
//
// USAGE:
//   qfa_micro <out.csv> [seed] [--smoke|--calibrate]
//   (no flag)   full sweep: n=1500, all 3 matrices, both f, both s, both tol,
//               8 trials -- 3440 rows (see the plan's row-count arithmetic).
//               NOT run by this implementation session; the driver supports it.
//   --smoke     stage (a): n=200, matrix=geo1e6, f=log1p, s=4, tol=1e-2,
//               3 depths, 2 trials, all five arm-variants -- 22 rows, asserted.
//   --calibrate stage (b): n=1500, matrix=geo1e6, f=log1p, s=4, tol=1e-2,
//               2 trials, all five arm-variants, the REAL s=4 depth grid --
//               80 rows, asserted. Prints per-arm wall-clock timing to stderr
//               (total_sec, runs, sec_per_run) for the full-sweep runtime
//               extrapolation the plan's Runtime section describes.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace linops = RandLAPACK::linops;
using T   = double;
using RNG = r123::Philox4x32;

static constexpr T QNAN = std::numeric_limits<T>::quiet_NaN();

// =====================================================================
// [RNG sub-streams]
// =====================================================================
constexpr uint64_t ROLE_MATRIX = 1;
constexpr uint64_t ROLE_PROBE  = 2;

// Deterministic, non-sequential sub-stream key: distinct (role, a, b) always
// give distinct keys, and no key is derived by advancing another key's stream
// (each RNGState below is constructed fresh from its own key), matching the
// plan's "distinct sub-stream offsets, never sequential draws off one stream."
static uint64_t derive_key(uint64_t base_seed, uint64_t role, uint64_t a, uint64_t b = 0) {
    return base_seed * 1000003ull + role * 9176ull + a * 131ull + b;
}

static int matrix_global_index(const std::string& name) {
    if (name == "geo1e3")  return 0;
    if (name == "geo1e6")  return 1;
    if (name == "logu1e6") return 2;
    throw std::runtime_error("qfa_micro: unknown matrix name '" + name + "'");
}
static int s_global_index(int64_t s) {
    if (s == 4)  return 0;
    if (s == 16) return 1;
    throw std::runtime_error("qfa_micro: unknown s value");
}

// =====================================================================
// [Depth grid construction -- Parameter grid section of the plan]
// =====================================================================
// geomspace(lo, hi, npts), rounded to nearest integer, deduplicated (ascending).
static std::vector<int64_t> geomspace_round_unique(double lo, double hi, int npts) {
    std::vector<int64_t> out;
    if (npts <= 1) { out.push_back((int64_t)std::llround(lo)); return out; }
    const double lg_lo = std::log(lo), lg_hi = std::log(hi);
    for (int i = 0; i < npts; ++i) {
        const double t = (double)i / (double)(npts - 1);
        const double v = std::exp(lg_lo + t * (lg_hi - lg_lo));
        out.push_back((int64_t)std::llround(v));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
static std::vector<int64_t> merge_unique_sorted(std::vector<int64_t> a, const std::vector<int64_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    return a;
}
// block grid = unique(round(geomspace(4, floor(n/s), 10)))
static std::vector<int64_t> block_grid_for(int64_t n, int64_t s) {
    return geomspace_round_unique(4.0, (double)(n / s), 10);
}
// scalar grid = unique(round(geomspace(4, 1024, 10))) UNION block grid
static std::vector<int64_t> scalar_grid_for(int64_t n, int64_t s) {
    return merge_unique_sorted(geomspace_round_unique(4.0, 1024.0, 10), block_grid_for(n, s));
}

constexpr int64_t SCALAR_CERT_CAP = 1024;   // fixed constant, NOT a function of n (plan "Cap asymmetry")
static int64_t block_cert_cap(int64_t n, int64_t s) { return n / s; }   // floor(n/s)

// =====================================================================
// [Matrix generation: A = Q diag(lambda) Q', Q Haar-random with sign fix]
// =====================================================================
struct MatrixData {
    std::string name;
    int64_t n = 0;
    T* Q      = nullptr;   // n x n, Haar-random orthogonal (col-major)
    T* lambda = nullptr;   // n
    T* A      = nullptr;   // n x n, dense symmetric (col-major, both triangles valid)

    MatrixData() = default;
    MatrixData(const MatrixData&)            = delete;
    MatrixData& operator=(const MatrixData&) = delete;
    MatrixData(MatrixData&& o) noexcept
        : name(std::move(o.name)), n(o.n), Q(o.Q), lambda(o.lambda), A(o.A) {
        o.Q = nullptr; o.lambda = nullptr; o.A = nullptr; o.n = 0;
    }
    ~MatrixData() { delete[] Q; delete[] lambda; delete[] A; }
};

static void make_lambda(bool geometric, T kappa, int64_t n, T* lambda) {
    if (geometric) {
        // geo: lambda_i = kappa^(1 - i/n), i = 0..n-1 (RandLAPACK::gen::gen_geometric_singvals form).
        for (int64_t i = 0; i < n; ++i) lambda[i] = std::pow(kappa, (T)1 - (T)i / (T)n);
    } else {
        // logu: lambda_i = kappa^(-i/(n-1)), i = 0..n-1 (cond == kappa exactly).
        for (int64_t i = 0; i < n; ++i) lambda[i] = std::pow(kappa, -(T)i / (T)(n - 1));
    }
}

static MatrixData build_matrix(const std::string& name, int64_t n, T kappa, bool geometric, uint64_t seed_key) {
    MatrixData M;
    M.name   = name;
    M.n      = n;
    M.Q      = new T[n * n];
    M.lambda = new T[n];
    M.A      = new T[n * n];

    make_lambda(geometric, kappa, n, M.lambda);

    // Haar-random Q: QR of a seeded n x n Gaussian, THEN the sign correction
    // Q <- Q * sign(diag(R)) (Stewart 1980) so Q is actually Haar-distributed.
    RandBLAS::RNGState<RNG> state(seed_key);
    RandBLAS::DenseDist D(n, n);
    state = RandBLAS::fill_dense(D, M.Q, state);
    T* tau = new T[n];
    lapack::geqrf(n, n, M.Q, n, tau);
    // R's diagonal sits at M.Q[i + i*n] immediately after geqrf, BEFORE orgqr
    // overwrites the buffer with Q itself -- capture the signs now.
    T* signs = new T[n];
    for (int64_t i = 0; i < n; ++i) signs[i] = (M.Q[i + i * n] >= (T)0) ? (T)1 : (T)-1;
    lapack::orgqr(n, n, n, M.Q, n, tau);
    for (int64_t j = 0; j < n; ++j) {
        T* col = M.Q + j * n;
        const T sj = signs[j];
        if (sj < (T)0) for (int64_t i = 0; i < n; ++i) col[i] = -col[i];
    }
    delete[] tau;
    delete[] signs;

    // A = (Q * diag(lambda)) * Q'.
    T* Qs = new T[n * n];
    for (int64_t j = 0; j < n; ++j) {
        const T lj = M.lambda[j];
        const T* qcol = M.Q + j * n;
        T* qscol = Qs + j * n;
        for (int64_t i = 0; i < n; ++i) qscol[i] = qcol[i] * lj;
    }
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
               n, n, n, (T)1.0, Qs, n, M.Q, n, (T)0.0, M.A, n);
    delete[] Qs;
    return M;
}

// =====================================================================
// [Probe generation -- inlined 4-line sphere-Gaussian pattern from
//  rl_fun_nystrom_pp.hh:571-578 (fill_probe_block is private, plan SS5.1)]
// =====================================================================
static void gen_probe(int64_t n, int64_t s, T* B, uint64_t seed_key) {
    RandBLAS::RNGState<RNG> state(seed_key);
    RandBLAS::DenseDist D(n, s);
    state = RandBLAS::fill_dense(D, B, state);
    const T target = std::sqrt((T)n);
    for (int64_t j = 0; j < s; ++j) {
        T* col = B + j * n;
        T nrm = blas::nrm2(n, col, 1);
        if (nrm > (T)0) blas::scal(n, target / nrm, col, 1);
    }
}

// truth_j[j] = (Q' b_j)' f(Lambda) (Q' b_j); total_truth = sum_j truth_j[j].
// Y is n x s scratch (Q' * B); reused by the caller across s <= its allocated width.
static void compute_truth(const MatrixData& M, const T* B, int64_t s,
                          const std::function<T(T)>& f, T* Y, T* truth_j, T& total_truth) {
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
               M.n, s, M.n, (T)1.0, M.Q, M.n, B, M.n, (T)0.0, Y, M.n);
    total_truth = (T)0;
    for (int64_t j = 0; j < s; ++j) {
        const T* yj = Y + j * M.n;
        T acc = (T)0;
        for (int64_t i = 0; i < M.n; ++i) {
            const T fl = f(M.lambda[i]);
            acc += fl * yj[i] * yj[i];
        }
        truth_j[j] = acc;
        total_truth += acc;
    }
}

// =====================================================================
// [CSV row model]
// =====================================================================
struct Row {
    const char* matrix;
    const char* fname;
    int         s;
    T           tol;               // QNAN if not applicable (fixed arms)
    const char* arm;
    int         reorth;            // -1 => NaN
    int         trial;
    long long   d_or_cap;
    long long   matvecs;
    T           rel_err;
    T           rel_err_midpoint;  // QNAN unless block-certified & certified
    int         certified;         // -1 => NaN
    T           col_depth_min, col_depth_med, col_depth_max;   // QNAN unless scalar-certified
    int         wall_limited;      // -1 => NaN
    int         cap_saturated;     // -1 => NaN
};

static void write_num(FILE* fp, T v) {
    if (std::isnan(v)) std::fprintf(fp, "NaN");
    else                std::fprintf(fp, "%.17e", v);
}
static void write_bool(FILE* fp, int v) {
    if (v < 0) std::fprintf(fp, "NaN");
    else        std::fprintf(fp, "%d", v);
}
static void write_row(FILE* fp, const Row& r) {
    std::fprintf(fp, "%s,%s,%d,", r.matrix, r.fname, r.s);
    write_num(fp, r.tol);
    std::fprintf(fp, ",%s,", r.arm);
    write_bool(fp, r.reorth);
    std::fprintf(fp, ",%d,%lld,%lld,", r.trial, r.d_or_cap, r.matvecs);
    write_num(fp, r.rel_err);
    std::fprintf(fp, ",");
    write_num(fp, r.rel_err_midpoint);
    std::fprintf(fp, ",");
    write_bool(fp, r.certified);
    std::fprintf(fp, ",");
    write_num(fp, r.col_depth_min);
    std::fprintf(fp, ",");
    write_num(fp, r.col_depth_med);
    std::fprintf(fp, ",");
    write_num(fp, r.col_depth_max);
    std::fprintf(fp, ",");
    write_bool(fp, r.wall_limited);
    std::fprintf(fp, ",");
    write_bool(fp, r.cap_saturated);
    std::fprintf(fp, "\n");
}

// =====================================================================
// [Metric 7: certificate honesty check -- console gate + trailing CSV comments]
// =====================================================================
struct Metric7Key { std::string matrix, fname, arm; int64_t s; T tol; };
static std::map<std::string, std::pair<long long, long long>> g_metric7;   // key -> (violations, checked)

static std::string metric7_key_str(const Metric7Key& k) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s|%s|%lld|%.6e|%s",
                  k.matrix.c_str(), k.fname.c_str(), (long long)k.s, (double)k.tol, k.arm.c_str());
    return buf;
}
static long long g_metric7_total_violations = 0;

static void metric7_record(const Metric7Key& key, bool violated) {
    auto& e = g_metric7[metric7_key_str(key)];
    e.second += 1;
    if (violated) {
        e.first += 1;
        g_metric7_total_violations += 1;
    }
}

// =====================================================================
// [Timing accumulation, used for the --calibrate extrapolation]
// =====================================================================
struct ArmStat { double total_sec = 0.0; long long runs = 0; };
static std::map<std::string, ArmStat> g_timing;
static void record_time(const std::string& arm, double sec) {
    auto& a = g_timing[arm];
    a.total_sec += sec;
    a.runs += 1;
}

// =====================================================================
// [Sweep: shared by --calibrate and the full sweep -- both use the REAL
//  geomspace-derived depth grid. --smoke uses its own small hardcoded grid
//  (run_smoke, below) since it is not part of the "Parameter grid" the
//  real grid-construction rule governs.]
// =====================================================================
struct FuncSpec { std::string name; std::function<T(T)> fn; };
struct MatrixSpec { std::string name; T kappa; bool geometric; };

static long long run_sweep(
    const std::vector<MatrixSpec>& matrices,
    const std::vector<FuncSpec>&   funcs,
    const std::vector<int64_t>&    s_list,
    const std::vector<T>&          tol_list,
    int trials, int64_t n, uint64_t base_seed, FILE* fp
) {
    long long rows_written = 0;
    const int64_t max_s = *std::max_element(s_list.begin(), s_list.end());

    // Reused scratch, sized to the widest s this sweep visits.
    T* B       = new T[n * max_s];
    T* Y       = new T[n * max_s];
    T* truth_j = new T[max_s];
    T* out_s   = new T[max_s];        // scalar arms' length-s output
    T* out_ss  = new T[max_s * max_s]; // block arms' s x s output (only tr is used)

    RandLAPACK::LanczosQFA<T>      scalar_oracle;
    RandLAPACK::BlockLanczosQFA<T> block_oracle;

    for (const auto& mspec : matrices) {
        const uint64_t mkey = derive_key(base_seed, ROLE_MATRIX, (uint64_t)matrix_global_index(mspec.name));
        MatrixData M = build_matrix(mspec.name, n, mspec.kappa, mspec.geometric, mkey);
        linops::ExplicitSymLinOp<T> A_op(n, Uplo::Upper, M.A, n, Layout::ColMajor);
        const bool do_reorth0 = (mspec.name == "geo1e6");

        for (int64_t s : s_list) {
            const auto block_grid  = block_grid_for(n, s);
            const auto scalar_grid = scalar_grid_for(n, s);
            const int64_t block_cap = block_cert_cap(n, s);
            const uint64_t pkey_base = derive_key(base_seed, ROLE_PROBE, (uint64_t)s_global_index(s), 0);
            (void)pkey_base;

            for (int trial = 0; trial < trials; ++trial) {
                const uint64_t pkey = derive_key(base_seed, ROLE_PROBE, (uint64_t)s_global_index(s), (uint64_t)trial);
                gen_probe(n, s, B, pkey);

                for (const auto& fs : funcs) {
                    T total_truth;
                    compute_truth(M, B, s, fs.fn, Y, truth_j, total_truth);

                    // ---- scalar-fixed ----
                    for (int64_t d : scalar_grid) {
                        scalar_oracle.adaptive      = false;
                        scalar_oracle.adaptive_rtol = (T)1e-2;   // unused when !adaptive; set for hygiene
                        scalar_oracle.check_every   = 1;
                        auto t0 = std::chrono::steady_clock::now();
                        scalar_oracle.call(A_op, B, n, s, fs.fn, d, out_s);
                        auto t1 = std::chrono::steady_clock::now();
                        record_time("scalar-fixed", std::chrono::duration<double>(t1 - t0).count());
                        T est = 0; for (int64_t j = 0; j < s; ++j) est += out_s[j];
                        Row r{};
                        r.matrix = mspec.name.c_str(); r.fname = fs.name.c_str(); r.s = (int)s;
                        r.tol = QNAN; r.arm = "scalar-fixed"; r.reorth = -1; r.trial = trial;
                        r.d_or_cap = d; r.matvecs = scalar_oracle.matvecs;
                        r.rel_err = std::abs(est - total_truth) / std::abs(total_truth);
                        r.rel_err_midpoint = QNAN; r.certified = -1;
                        r.col_depth_min = r.col_depth_med = r.col_depth_max = QNAN;
                        r.wall_limited = -1; r.cap_saturated = -1;
                        write_row(fp, r); ++rows_written;
                    }

                    // ---- block-fixed (reorth=1) ----
                    for (int64_t d : block_grid) {
                        block_oracle.reorth        = true;
                        block_oracle.adaptive       = false;
                        block_oracle.stop_rule      = RandLAPACK::BlockQFAStop::Radau;
                        block_oracle.adaptive_rtol  = (T)1e-2;   // unused when !adaptive
                        block_oracle.check_every    = 1;
                        auto t0 = std::chrono::steady_clock::now();
                        block_oracle.call(A_op, B, n, s, fs.fn, d, out_ss);
                        auto t1 = std::chrono::steady_clock::now();
                        record_time("block-fixed", std::chrono::duration<double>(t1 - t0).count());
                        Row r{};
                        r.matrix = mspec.name.c_str(); r.fname = fs.name.c_str(); r.s = (int)s;
                        r.tol = QNAN; r.arm = "block-fixed"; r.reorth = 1; r.trial = trial;
                        r.d_or_cap = d; r.matvecs = block_oracle.matvecs;
                        r.rel_err = std::abs(block_oracle.tr_U - total_truth) / std::abs(total_truth);
                        r.rel_err_midpoint = QNAN; r.certified = -1;
                        r.col_depth_min = r.col_depth_med = r.col_depth_max = QNAN;
                        r.wall_limited = -1; r.cap_saturated = -1;
                        write_row(fp, r); ++rows_written;
                    }

                    // ---- block-fixed-reorth0 (geo1e6 only, matched-stability control) ----
                    if (do_reorth0) {
                        for (int64_t d : block_grid) {
                            block_oracle.reorth        = false;
                            block_oracle.adaptive       = false;
                            block_oracle.stop_rule      = RandLAPACK::BlockQFAStop::Radau;
                            block_oracle.adaptive_rtol  = (T)1e-2;
                            block_oracle.check_every    = 1;
                            auto t0 = std::chrono::steady_clock::now();
                            block_oracle.call(A_op, B, n, s, fs.fn, d, out_ss);
                            auto t1 = std::chrono::steady_clock::now();
                            record_time("block-fixed-reorth0", std::chrono::duration<double>(t1 - t0).count());
                            Row r{};
                            r.matrix = mspec.name.c_str(); r.fname = fs.name.c_str(); r.s = (int)s;
                            r.tol = QNAN; r.arm = "block-fixed-reorth0"; r.reorth = 0; r.trial = trial;
                            r.d_or_cap = d; r.matvecs = block_oracle.matvecs;
                            r.rel_err = std::abs(block_oracle.tr_U - total_truth) / std::abs(total_truth);
                            r.rel_err_midpoint = QNAN; r.certified = -1;
                            r.col_depth_min = r.col_depth_med = r.col_depth_max = QNAN;
                            r.wall_limited = -1; r.cap_saturated = -1;
                            write_row(fp, r); ++rows_written;
                        }
                    }

                    // ---- scalar-certified & block-certified ----
                    for (T tol : tol_list) {
                        // scalar-certified
                        {
                            scalar_oracle.adaptive      = true;
                            scalar_oracle.adaptive_rtol = tol;
                            scalar_oracle.check_every   = 1;
                            auto t0 = std::chrono::steady_clock::now();
                            scalar_oracle.call(A_op, B, n, s, fs.fn, SCALAR_CERT_CAP, out_s);
                            auto t1 = std::chrono::steady_clock::now();
                            record_time("scalar-certified", std::chrono::duration<double>(t1 - t0).count());
                            T est = 0; for (int64_t j = 0; j < s; ++j) est += out_s[j];
                            const bool certified = scalar_oracle.all_certified;
                            std::vector<int64_t> tused(scalar_oracle.t_used, scalar_oracle.t_used + s);
                            std::sort(tused.begin(), tused.end());
                            const T cd_min = (T)tused.front();
                            const T cd_max = (T)tused.back();
                            const T cd_med = (s % 2 == 1) ? (T)tused[s / 2]
                                             : (T)0.5 * (T)(tused[s / 2 - 1] + tused[s / 2]);
                            const bool cap_sat = (cd_max == (T)SCALAR_CERT_CAP) && !certified;
                            Row r{};
                            r.matrix = mspec.name.c_str(); r.fname = fs.name.c_str(); r.s = (int)s;
                            r.tol = tol; r.arm = "scalar-certified"; r.reorth = -1; r.trial = trial;
                            r.d_or_cap = (long long)cd_max; r.matvecs = scalar_oracle.matvecs;
                            r.rel_err = std::abs(est - total_truth) / std::abs(total_truth);
                            r.rel_err_midpoint = QNAN; r.certified = certified ? 1 : 0;
                            r.col_depth_min = cd_min; r.col_depth_med = cd_med; r.col_depth_max = cd_max;
                            r.wall_limited = -1; r.cap_saturated = cap_sat ? 1 : 0;
                            write_row(fp, r); ++rows_written;

                            // Metric 7 (scalar): per-column violation on CERTIFIED columns only.
                            bool row_violates = false;
                            for (int64_t j = 0; j < s; ++j) {
                                if (!scalar_oracle.certified[j]) continue;
                                const T U = scalar_oracle.gauss_val[j], L = scalar_oracle.radau_val[j];
                                const T scale = std::max({std::abs(U), std::abs(L), std::numeric_limits<T>::min()});
                                if (std::abs(U - truth_j[j]) > tol * scale) {
                                    row_violates = true;
                                    std::fprintf(stderr,
                                        "[metric7] VIOLATION arm=scalar-certified matrix=%s f=%s s=%lld tol=%g "
                                        "trial=%d col=%lld |U-truth|=%.6e > tol*scale=%.6e\n",
                                        mspec.name.c_str(), fs.name.c_str(), (long long)s, (double)tol, trial,
                                        (long long)j, (double)std::abs(U - truth_j[j]), (double)(tol * scale));
                                }
                            }
                            Metric7Key mk{mspec.name, fs.name, "scalar-certified", s, tol};
                            metric7_record(mk, row_violates);
                        }
                        // block-certified
                        {
                            block_oracle.reorth        = true;
                            block_oracle.adaptive       = true;
                            block_oracle.stop_rule      = RandLAPACK::BlockQFAStop::Radau;
                            block_oracle.adaptive_rtol  = tol;
                            block_oracle.check_every    = 1;
                            auto t0 = std::chrono::steady_clock::now();
                            block_oracle.call(A_op, B, n, s, fs.fn, block_cap, out_ss);
                            auto t1 = std::chrono::steady_clock::now();
                            record_time("block-certified", std::chrono::duration<double>(t1 - t0).count());
                            const bool certified = block_oracle.certified;
                            const bool wall_lim = (block_oracle.d_used == block_cap) && !certified;
                            Row r{};
                            r.matrix = mspec.name.c_str(); r.fname = fs.name.c_str(); r.s = (int)s;
                            r.tol = tol; r.arm = "block-certified"; r.reorth = 1; r.trial = trial;
                            r.d_or_cap = block_oracle.d_used; r.matvecs = block_oracle.matvecs;
                            r.rel_err = std::abs(block_oracle.tr_U - total_truth) / std::abs(total_truth);
                            // rel_err_midpoint: block-certified rows WITH certified==true only
                            // (certified==true is reachable only via a live Radau pair -- see the
                            // file header note and the plan's NaN convention).
                            if (certified) {
                                const T mid = (T)0.5 * (block_oracle.tr_U + block_oracle.tr_L);
                                r.rel_err_midpoint = std::abs(mid - total_truth) / std::abs(total_truth);
                            } else {
                                r.rel_err_midpoint = QNAN;
                            }
                            r.certified = certified ? 1 : 0;
                            r.col_depth_min = r.col_depth_med = r.col_depth_max = QNAN;
                            r.wall_limited = wall_lim ? 1 : 0; r.cap_saturated = -1;
                            write_row(fp, r); ++rows_written;

                            // Metric 7 (block): certificate's own scale, on CERTIFIED rows only.
                            if (certified) {
                                const T U = block_oracle.tr_U, L = block_oracle.tr_L;
                                const T scale = std::max({std::abs(U), std::abs(L), std::numeric_limits<T>::min()});
                                const bool violates = std::abs(U - total_truth) > tol * scale;
                                if (violates) {
                                    std::fprintf(stderr,
                                        "[metric7] VIOLATION arm=block-certified matrix=%s f=%s s=%lld tol=%g "
                                        "trial=%d |U-truth|=%.6e > tol*scale=%.6e\n",
                                        mspec.name.c_str(), fs.name.c_str(), (long long)s, (double)tol, trial,
                                        (double)std::abs(U - total_truth), (double)(tol * scale));
                                }
                                Metric7Key mk{mspec.name, fs.name, "block-certified", s, tol};
                                metric7_record(mk, violates);
                            }
                        }
                    }
                }
            }
        }
    }

    delete[] B; delete[] Y; delete[] truth_j; delete[] out_s; delete[] out_ss;
    return rows_written;
}

// =====================================================================
// [Smoke: stage (a), its own tiny hardcoded grid -- NOT the real grid rule]
// =====================================================================
static long long run_smoke(uint64_t base_seed, FILE* fp) {
    const int64_t n = 200, s = 4, trials = 2;
    const T tol = (T)1e-2;
    const std::vector<int64_t> depths = {4, 10, 20};   // "2-3 depths" per the plan
    const MatrixSpec mspec{"geo1e6", (T)1e6, true};
    const FuncSpec fs{"log1p", [](T x) { return std::log1p(std::max(x, (T)0)); }};

    long long rows_written = 0;
    const uint64_t mkey = derive_key(base_seed, ROLE_MATRIX, (uint64_t)matrix_global_index(mspec.name));
    MatrixData M = build_matrix(mspec.name, n, mspec.kappa, mspec.geometric, mkey);
    linops::ExplicitSymLinOp<T> A_op(n, Uplo::Upper, M.A, n, Layout::ColMajor);
    const int64_t block_cap = block_cert_cap(n, s);

    T* B = new T[n * s]; T* Y = new T[n * s]; T* truth_j = new T[s];
    T* out_s = new T[s]; T* out_ss = new T[s * s];
    RandLAPACK::LanczosQFA<T> scalar_oracle;
    RandLAPACK::BlockLanczosQFA<T> block_oracle;

    for (int trial = 0; trial < trials; ++trial) {
        const uint64_t pkey = derive_key(base_seed, ROLE_PROBE, (uint64_t)s_global_index(s), (uint64_t)trial);
        gen_probe(n, s, B, pkey);
        T total_truth;
        compute_truth(M, B, s, fs.fn, Y, truth_j, total_truth);

        for (int64_t d : depths) {
            scalar_oracle.adaptive = false; scalar_oracle.check_every = 1;
            scalar_oracle.call(A_op, B, n, s, fs.fn, d, out_s);
            T est = 0; for (int64_t j = 0; j < s; ++j) est += out_s[j];
            Row r{}; r.matrix = "geo1e6"; r.fname = "log1p"; r.s = (int)s; r.tol = QNAN;
            r.arm = "scalar-fixed"; r.reorth = -1; r.trial = trial; r.d_or_cap = d;
            r.matvecs = scalar_oracle.matvecs; r.rel_err = std::abs(est - total_truth) / std::abs(total_truth);
            r.rel_err_midpoint = QNAN; r.certified = -1;
            r.col_depth_min = r.col_depth_med = r.col_depth_max = QNAN;
            r.wall_limited = -1; r.cap_saturated = -1;
            write_row(fp, r); ++rows_written;
        }
        for (int64_t d : depths) {
            block_oracle.reorth = true; block_oracle.adaptive = false;
            block_oracle.stop_rule = RandLAPACK::BlockQFAStop::Radau; block_oracle.check_every = 1;
            block_oracle.call(A_op, B, n, s, fs.fn, d, out_ss);
            Row r{}; r.matrix = "geo1e6"; r.fname = "log1p"; r.s = (int)s; r.tol = QNAN;
            r.arm = "block-fixed"; r.reorth = 1; r.trial = trial; r.d_or_cap = d;
            r.matvecs = block_oracle.matvecs; r.rel_err = std::abs(block_oracle.tr_U - total_truth) / std::abs(total_truth);
            r.rel_err_midpoint = QNAN; r.certified = -1;
            r.col_depth_min = r.col_depth_med = r.col_depth_max = QNAN;
            r.wall_limited = -1; r.cap_saturated = -1;
            write_row(fp, r); ++rows_written;
        }
        for (int64_t d : depths) {
            block_oracle.reorth = false; block_oracle.adaptive = false;
            block_oracle.stop_rule = RandLAPACK::BlockQFAStop::Radau; block_oracle.check_every = 1;
            block_oracle.call(A_op, B, n, s, fs.fn, d, out_ss);
            Row r{}; r.matrix = "geo1e6"; r.fname = "log1p"; r.s = (int)s; r.tol = QNAN;
            r.arm = "block-fixed-reorth0"; r.reorth = 0; r.trial = trial; r.d_or_cap = d;
            r.matvecs = block_oracle.matvecs; r.rel_err = std::abs(block_oracle.tr_U - total_truth) / std::abs(total_truth);
            r.rel_err_midpoint = QNAN; r.certified = -1;
            r.col_depth_min = r.col_depth_med = r.col_depth_max = QNAN;
            r.wall_limited = -1; r.cap_saturated = -1;
            write_row(fp, r); ++rows_written;
        }
        // scalar-certified
        {
            scalar_oracle.adaptive = true; scalar_oracle.adaptive_rtol = tol; scalar_oracle.check_every = 1;
            scalar_oracle.call(A_op, B, n, s, fs.fn, SCALAR_CERT_CAP, out_s);
            T est = 0; for (int64_t j = 0; j < s; ++j) est += out_s[j];
            const bool certified = scalar_oracle.all_certified;
            std::vector<int64_t> tused(scalar_oracle.t_used, scalar_oracle.t_used + s);
            std::sort(tused.begin(), tused.end());
            const T cd_min = (T)tused.front(), cd_max = (T)tused.back();
            const T cd_med = (s % 2 == 1) ? (T)tused[s / 2] : (T)0.5 * (T)(tused[s / 2 - 1] + tused[s / 2]);
            const bool cap_sat = (cd_max == (T)SCALAR_CERT_CAP) && !certified;
            Row r{}; r.matrix = "geo1e6"; r.fname = "log1p"; r.s = (int)s; r.tol = tol;
            r.arm = "scalar-certified"; r.reorth = -1; r.trial = trial; r.d_or_cap = (long long)cd_max;
            r.matvecs = scalar_oracle.matvecs; r.rel_err = std::abs(est - total_truth) / std::abs(total_truth);
            r.rel_err_midpoint = QNAN; r.certified = certified ? 1 : 0;
            r.col_depth_min = cd_min; r.col_depth_med = cd_med; r.col_depth_max = cd_max;
            r.wall_limited = -1; r.cap_saturated = cap_sat ? 1 : 0;
            write_row(fp, r); ++rows_written;
        }
        // block-certified
        {
            block_oracle.reorth = true; block_oracle.adaptive = true;
            block_oracle.stop_rule = RandLAPACK::BlockQFAStop::Radau;
            block_oracle.adaptive_rtol = tol; block_oracle.check_every = 1;
            block_oracle.call(A_op, B, n, s, fs.fn, block_cap, out_ss);
            const bool certified = block_oracle.certified;
            const bool wall_lim = (block_oracle.d_used == block_cap) && !certified;
            Row r{}; r.matrix = "geo1e6"; r.fname = "log1p"; r.s = (int)s; r.tol = tol;
            r.arm = "block-certified"; r.reorth = 1; r.trial = trial; r.d_or_cap = block_oracle.d_used;
            r.matvecs = block_oracle.matvecs; r.rel_err = std::abs(block_oracle.tr_U - total_truth) / std::abs(total_truth);
            r.rel_err_midpoint = certified
                ? std::abs((T)0.5 * (block_oracle.tr_U + block_oracle.tr_L) - total_truth) / std::abs(total_truth)
                : QNAN;
            r.certified = certified ? 1 : 0;
            r.col_depth_min = r.col_depth_med = r.col_depth_max = QNAN;
            r.wall_limited = wall_lim ? 1 : 0; r.cap_saturated = -1;
            write_row(fp, r); ++rows_written;
        }
    }
    delete[] B; delete[] Y; delete[] truth_j; delete[] out_s; delete[] out_ss;
    return rows_written;
}

// =====================================================================
// [Header comment metadata]
// =====================================================================
static std::string git_revision() {
    std::string src_dir = __FILE__;
    size_t p = src_dir.find_last_of('/');
    src_dir = (p == std::string::npos) ? "." : src_dir.substr(0, p);
    std::string cmd = "git -C '" + src_dir + "' rev-parse --short HEAD 2>/dev/null";
    std::string rev;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buf[128];
        if (std::fgets(buf, sizeof(buf), pipe)) {
            rev = buf;
            while (!rev.empty() && (rev.back() == '\n' || rev.back() == '\r')) rev.pop_back();
        }
        pclose(pipe);
    }
    return rev.empty() ? "unknown" : rev;
}
static std::string hostname_str() {
#if defined(__unix__) || defined(__APPLE__)
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) return std::string(buf);
#endif
    return "unknown";
}
static std::string blas_backend() {
#if defined(BLAS_HAVE_MKL)
    return "MKL";
#elif defined(BLAS_HAVE_ACCELERATE)
    return "Accelerate";
#else
    return "unknown (see blaspp build config)";
#endif
}

static void write_header(FILE* fp, const std::string& mode, uint64_t base_seed, int64_t n) {
    std::fprintf(fp, "# qfa_micro benchmark output (plan: 2026-09-02-qfa-micro-benchmark-plan.md, pass 10 FROZEN)\n");
    std::fprintf(fp, "# mode=%s\n", mode.c_str());
    std::fprintf(fp, "# n=%lld\n", (long long)n);
    std::fprintf(fp, "# base_seed=%llu\n", (unsigned long long)base_seed);
    std::fprintf(fp, "# matrix_seed: key = derive_key(base_seed, ROLE_MATRIX=1, matrix_global_index); "
                      "matrix_global_index geo1e3=0 geo1e6=1 logu1e6=2; drawn once per (spectrum,kappa)\n");
    std::fprintf(fp, "# probe_seed: key = derive_key(base_seed, ROLE_PROBE=2, s_global_index, trial); "
                      "s_global_index s=4->0 s=16->1; drawn once per (s,trial), SHARED across matrix and f "
                      "(driver design choice, see file header comment in qfa_micro.cc)\n");
    std::fprintf(fp, "# derive_key(base,role,a,b) = base*1000003 + role*9176 + a*131 + b\n");
    std::fprintf(fp, "# scalar-certified cap=%lld (fixed constant, not a function of n); "
                      "block-certified cap=floor(n/s)\n", (long long)SCALAR_CERT_CAP);
    std::fprintf(fp, "# block-fixed-reorth0 scope: geo1e6 only, both s, both f, full grid (matched-stability control)\n");
    std::fprintf(fp, "# git_revision=%s\n", git_revision().c_str());
    std::fprintf(fp, "# hostname=%s\n", hostname_str().c_str());
    std::fprintf(fp, "# blas_backend=%s\n", blas_backend().c_str());
#ifdef _OPENMP
    std::fprintf(fp, "# omp_max_threads=%d\n", omp_get_max_threads());
#else
    std::fprintf(fp, "# omp_max_threads=1 (built without OpenMP)\n");
#endif
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    std::fprintf(fp, "# OMP_NUM_THREADS_env=%s\n", omp_env ? omp_env : "(unset)");
    std::fprintf(fp, "matrix,f,s,tol,arm,reorth,trial,d_or_cap,matvecs,rel_err,rel_err_midpoint,"
                      "certified,col_depth_min,col_depth_med,col_depth_max,wall_limited,cap_saturated\n");
}

static void write_metric7_trailer(FILE* fp) {
    for (const auto& kv : g_metric7) {
        // key was built as "matrix|f|s|tol|arm"
        std::string k = kv.first;
        std::vector<std::string> parts;
        size_t start = 0;
        for (size_t i = 0; i <= k.size(); ++i) {
            if (i == k.size() || k[i] == '|') { parts.push_back(k.substr(start, i - start)); start = i + 1; }
        }
        std::fprintf(fp, "# metric7_violation matrix=%s f=%s s=%s tol=%s arm=%s violations=%lld checked=%lld\n",
                     parts[0].c_str(), parts[1].c_str(), parts[2].c_str(), parts[3].c_str(), parts[4].c_str(),
                     kv.second.first, kv.second.second);
    }
    std::fprintf(fp, "# metric7_total_violations=%lld\n", g_metric7_total_violations);
}

static void print_timing_summary() {
    std::fprintf(stderr, "\n[timing summary]\n");
    for (const auto& kv : g_timing) {
        const double per_run = kv.second.runs > 0 ? kv.second.total_sec / (double)kv.second.runs : 0.0;
        std::fprintf(stderr, "  arm=%-22s runs=%6lld total_sec=%10.4f sec_per_run=%.6f\n",
                     kv.first.c_str(), kv.second.runs, kv.second.total_sec, per_run);
    }
}

// =====================================================================
// [main]
// =====================================================================
static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: %s <out.csv> [seed] [--smoke|--calibrate]\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) { print_usage(argv[0]); return 1; }
    const std::string out_path = argv[1];
    const uint64_t seed = (argc >= 3) ? std::strtoull(argv[2], nullptr, 10) : 42ull;
    std::string mode = "full";
    if (argc == 4) {
        std::string flag = argv[3];
        if (flag == "--smoke") mode = "smoke";
        else if (flag == "--calibrate") mode = "calibrate";
        else { std::fprintf(stderr, "unknown flag '%s'\n", flag.c_str()); print_usage(argv[0]); return 1; }
    }

    FILE* fp = std::fopen(out_path.c_str(), "w");
    if (!fp) { std::fprintf(stderr, "cannot open '%s' for writing\n", out_path.c_str()); return 2; }

    try {
        if (mode == "smoke") {
            const int64_t n = 200;
            write_header(fp, mode, seed, n);
            long long rows = run_smoke(seed, fp);
            write_metric7_trailer(fp);
            std::fclose(fp);
            print_timing_summary();
            constexpr long long EXPECTED_SMOKE_ROWS = 22;
            std::fprintf(stderr, "[smoke] rows_written=%lld (expected %lld)\n", rows, EXPECTED_SMOKE_ROWS);
            if (rows != EXPECTED_SMOKE_ROWS) {
                std::fprintf(stderr, "FATAL: smoke row count mismatch: got %lld, expected %lld\n",
                             rows, EXPECTED_SMOKE_ROWS);
                return 3;
            }
            std::fprintf(stderr, "[smoke] PASS\n");
        } else if (mode == "calibrate") {
            const int64_t n = 1500;
            write_header(fp, mode, seed, n);
            std::vector<MatrixSpec> matrices = {{"geo1e6", (T)1e6, true}};
            std::vector<FuncSpec> funcs;
            funcs.push_back({"log1p", [](T x) { return std::log1p(std::max(x, (T)0)); }});
            std::vector<int64_t> s_list = {4};
            std::vector<T> tol_list = {(T)1e-2};
            long long rows = run_sweep(matrices, funcs, s_list, tol_list, /*trials=*/2, n, seed, fp);
            write_metric7_trailer(fp);
            std::fclose(fp);
            print_timing_summary();
            constexpr long long EXPECTED_CALIBRATE_ROWS = 80;
            std::fprintf(stderr, "[calibrate] rows_written=%lld (expected %lld)\n", rows, EXPECTED_CALIBRATE_ROWS);
            if (rows != EXPECTED_CALIBRATE_ROWS) {
                std::fprintf(stderr, "FATAL: calibrate row count mismatch: got %lld, expected %lld\n",
                             rows, EXPECTED_CALIBRATE_ROWS);
                return 3;
            }
            std::fprintf(stderr, "[calibrate] PASS\n");
        } else {   // full sweep
            const int64_t n = 1500;
            write_header(fp, mode, seed, n);
            std::vector<MatrixSpec> matrices = {
                {"geo1e3", (T)1e3, true}, {"geo1e6", (T)1e6, true}, {"logu1e6", (T)1e6, false}
            };
            std::vector<FuncSpec> funcs;
            funcs.push_back({"sqrt",  [](T x) { return std::sqrt(std::max(x, (T)0)); }});
            funcs.push_back({"log1p", [](T x) { return std::log1p(std::max(x, (T)0)); }});
            std::vector<int64_t> s_list = {4, 16};
            std::vector<T> tol_list = {(T)1e-2, (T)1e-4};
            long long rows = run_sweep(matrices, funcs, s_list, tol_list, /*trials=*/8, n, seed, fp);
            write_metric7_trailer(fp);
            std::fclose(fp);
            print_timing_summary();
            constexpr long long EXPECTED_FULL_ROWS = 3440;
            std::fprintf(stderr, "[full] rows_written=%lld (expected %lld)\n", rows, EXPECTED_FULL_ROWS);
            if (rows != EXPECTED_FULL_ROWS) {
                std::fprintf(stderr, "FATAL: full-sweep row count mismatch: got %lld, expected %lld\n",
                             rows, EXPECTED_FULL_ROWS);
                return 3;
            }
            std::fprintf(stderr, "[full] PASS\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qfa_micro: exception: %s\n", e.what());
        return 4;
    }

    if (g_metric7_total_violations > 0) {
        std::fprintf(stderr, "\n*** METRIC 7 GATE: %lld certificate violation(s) detected. "
                              "DO NOT PUBLISH this run without investigating. ***\n",
                     g_metric7_total_violations);
    } else {
        std::fprintf(stderr, "[metric7] gate PASS: 0 violations.\n");
    }
    return 0;
}
