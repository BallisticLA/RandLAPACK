// qfa_timing_probe — where does BlockLanczosQFA (block-certified, Radau
// stopping rule) spend its wall time: inner-recurrence matvecs, block
// reorthogonalization, or the certificate-check / eigendecomposition bucket?
//
// Reuses qfa_micro.cc's synthetic-spectrum matrix construction (A = Q
// diag(lambda) Q', Q Haar-random via sign-corrected QR-of-Gaussian), its
// sphere-Gaussian probe-block construction, and its derive_key seeding
// convention (base_seed + role + index), so cells here are directly
// comparable in construction (not in seed VALUES -- this file uses its own
// role/index space) to the qfa_micro sweep.
//
// Instrumentation note (2026-09-02): BlockLanczosQFA previously exposed only
// `times` = {matvec_us, run_lanczos(net of cert)_us, apply_us, rest_us,
// total_us, reorth_us}, where `apply_us` FOLDED the certificate-check time
// together with the final compute_M reconstruction time -- there was no
// separately-readable certificate bucket and no counter of how many
// certificate checks fired. This probe required querying an isolated
// certificate cost, so RandLAPACK/comps/rl_lanczos_qfa_block.hh gained two
// new public members (documented there): `_t_cert_us` (long, cumulative
// certificate-check microseconds, mirroring the existing `_t_matvec_us`
// convention) and `cert_checks` (int64_t, count of radau_bracket_check()
// calls -- ladder-triggered plus the at-cap re-check). Both are set at the
// end of call(), unconditionally (cert_checks is meaningful whenever
// adaptive+Radau; _t_cert_us is meaningful whenever timing=true).
//
// Caveat carried into the analysis: the O(s^3)-per-step block-LDL^T pivot
// maintenance (update_pivot(), run every recurrence step regardless of
// whether that step is a check depth) is NOT part of _t_cert_us -- it happens
// inside the timed lanczos span, not inside the c0/c1 window around
// radau_bracket_check(). It lands in this probe's "other" bucket along with
// the final compute_M call (when the certificate never fired) and
// bookkeeping. This is called out explicitly rather than silently folded in.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace linops = RandLAPACK::linops;
using T   = double;
using RNG = r123::Philox4x32;

// =====================================================================
// [Seeding -- same derive_key form as qfa_micro.cc, own role/index space]
// =====================================================================
constexpr uint64_t ROLE_MATRIX = 1;
constexpr uint64_t ROLE_PROBE  = 2;
static uint64_t derive_key(uint64_t base_seed, uint64_t role, uint64_t a, uint64_t b = 0) {
    return base_seed * 1000003ull + role * 9176ull + a * 131ull + b;
}

// =====================================================================
// [Matrix generation -- geo1e6 spectrum, Haar-random Q with sign fix.
//  Copied from qfa_micro.cc's build_matrix/make_lambda (geometric branch
//  only -- this probe uses geo1e6 exclusively).]
// =====================================================================
struct MatrixData {
    int64_t n = 0;
    T* Q      = nullptr;
    T* lambda = nullptr;
    T* A      = nullptr;
    MatrixData() = default;
    MatrixData(const MatrixData&)            = delete;
    MatrixData& operator=(const MatrixData&) = delete;
    MatrixData(MatrixData&& o) noexcept
        : n(o.n), Q(o.Q), lambda(o.lambda), A(o.A) {
        o.Q = nullptr; o.lambda = nullptr; o.A = nullptr; o.n = 0;
    }
    ~MatrixData() { delete[] Q; delete[] lambda; delete[] A; }
};

static void make_lambda_geo(T kappa, int64_t n, T* lambda) {
    for (int64_t i = 0; i < n; ++i) lambda[i] = std::pow(kappa, (T)1 - (T)i / (T)n);
}

static MatrixData build_matrix_geo1e6(int64_t n, uint64_t seed_key) {
    const T kappa = (T)1e6;
    MatrixData M;
    M.n      = n;
    M.Q      = new T[n * n];
    M.lambda = new T[n];
    M.A      = new T[n * n];
    make_lambda_geo(kappa, n, M.lambda);

    RandBLAS::RNGState<RNG> state(seed_key);
    RandBLAS::DenseDist D(n, n);
    state = RandBLAS::fill_dense(D, M.Q, state);
    T* tau = new T[n];
    lapack::geqrf(n, n, M.Q, n, tau);
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
// [Probe generation -- sphere-Gaussian block, from qfa_micro.cc gen_probe]
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

// =====================================================================
// [Cell / run bookkeeping]
// =====================================================================
struct Cell {
    const char* label;
    int64_t n;
    const char* fname;
    T tol;
    int64_t s;
};

struct RunResult {
    long total_us, matvec_us, reorth_us, cert_us, other_us;
    int64_t d_used, s, n;
    int64_t cert_checks;
    bool certified;
};

static void print_run_row(const char* cell, int trial, const RunResult& r) {
    auto pct = [&](long part) { return 100.0 * (double)part / (double)std::max(1L, r.total_us); };
    std::printf("%-4s %2d  n=%5lld s=%3lld d_stop=%5lld  total_us=%10ld  "
                "matvec=%10ld(%5.1f%%)  reorth=%10ld(%5.1f%%)  cert=%10ld(%5.1f%%)  "
                "other=%10ld(%5.1f%%)  certified=%d checks=%lld\n",
                cell, trial, (long long)r.n, (long long)r.s, (long long)r.d_used, r.total_us,
                r.matvec_us, pct(r.matvec_us), r.reorth_us, pct(r.reorth_us),
                r.cert_us, pct(r.cert_us), r.other_us, pct(r.other_us),
                r.certified ? 1 : 0, (long long)r.cert_checks);
}

static RunResult run_once(MatrixData& M, T* B, int64_t n, int64_t s, T tol,
                          const std::function<T(T)>& f, T* out_ss) {
    linops::ExplicitSymLinOp<T> A_op(n, Uplo::Upper, M.A, n, Layout::ColMajor);
    RandLAPACK::BlockLanczosQFA<T> qfa;
    qfa.reorth        = true;
    qfa.timing        = true;
    qfa.adaptive       = true;
    qfa.stop_rule      = RandLAPACK::BlockQFAStop::Radau;
    qfa.return_mode    = RandLAPACK::BlockQFAReturn::Midpoint;
    qfa.stop_scale     = RandLAPACK::BlockQFAScale::MaxBoth;
    qfa.adaptive_rtol  = tol;
    qfa.check_every    = 1;
    const int64_t cap = n / s;   // structural s*d <= n wall, matching qfa_micro's block_cert_cap

    qfa.call(A_op, B, n, s, f, cap, out_ss);

    RunResult r{};
    r.total_us    = qfa.times[4];
    r.matvec_us   = qfa.times[0];
    r.reorth_us   = qfa.times[5];
    r.cert_us     = qfa._t_cert_us;
    r.other_us    = r.total_us - r.matvec_us - r.reorth_us - r.cert_us;
    r.d_used      = qfa.d_used;
    r.s           = s;
    r.n           = n;
    r.cert_checks = qfa.cert_checks;
    r.certified   = qfa.certified;
    return r;
}

int main() {
    using namespace std::chrono;
    const uint64_t base_seed = 20260902ull;
    const int NTRIALS_DEFAULT = 3;

    std::function<T(T)> f_sqrt   = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    std::function<T(T)> f_log1p  = [](T x) { return std::log1p(std::max(x, (T)0)); };

    std::vector<Cell> cells = {
        {"a", 1500, "sqrt",  (T)1e-4, 4},
        {"b", 1500, "log1p", (T)1e-4, 4},
        {"c", 1500, "sqrt",  (T)1e-4, 16},
        {"d", 4000, "sqrt",  (T)1e-4, 4},
        {"e", 4000, "sqrt",  (T)1e-4, 16},
    };
    // Time-budget notes are printed to stderr as they're decided; see the
    // per-cell trial counts below (deviations from 3 trials are flagged
    // explicitly at the point they are decided AND in the final summary).
    std::vector<int> ntrials_for_cell(cells.size(), NTRIALS_DEFAULT);

    std::printf("=== qfa_timing_probe: per-run timing table ===\n");
    std::printf("cell tr  n      s   d_stop  total_us    matvec(pct)              reorth(pct)              cert(pct)                other(pct)               certified checks\n");

    // Cache built matrices per n (geo1e6 only), so cells sharing n reuse A/Q.
    std::vector<MatrixData> mats_by_n;    // parallel to unique n values encountered
    std::vector<int64_t> mats_n;
    auto get_matrix = [&](int64_t n) -> MatrixData& {
        for (size_t i = 0; i < mats_n.size(); ++i) if (mats_n[i] == n) return mats_by_n[i];
        const uint64_t mkey = derive_key(base_seed, ROLE_MATRIX, (uint64_t)n);
        mats_by_n.push_back(build_matrix_geo1e6(n, mkey));
        mats_n.push_back(n);
        return mats_by_n.back();
    };

    struct AggRow {
        const char* cell; int64_t n, s; T tol; const char* fname;
        std::vector<RunResult> runs;
        int ntrials_planned;
        bool budget_cut;
    };
    std::vector<AggRow> agg;

    int64_t max_s = 16;
    T* B     = new T[4000 * max_s];
    T* out_ss = new T[max_s * max_s];

    for (size_t ci = 0; ci < cells.size(); ++ci) {
        const Cell& c = cells[ci];
        MatrixData& M = get_matrix(c.n);
        const std::function<T(T)>& f = (std::string(c.fname) == "sqrt") ? f_sqrt : f_log1p;

        AggRow row;
        row.cell = c.label; row.n = c.n; row.s = c.s; row.tol = c.tol; row.fname = c.fname;
        row.ntrials_planned = ntrials_for_cell[ci];
        row.budget_cut = false;

        int trial = 0;
        for (; trial < row.ntrials_planned; ++trial) {
            const uint64_t pkey = derive_key(base_seed, ROLE_PROBE, (uint64_t)ci, (uint64_t)trial);
            gen_probe(c.n, c.s, B, pkey);

            auto t0 = steady_clock::now();
            RunResult r = run_once(M, B, c.n, c.s, c.tol, f, out_ss);
            auto t1 = steady_clock::now();
            double wall_sec = duration<double>(t1 - t0).count();

            print_run_row(c.label, trial, r);
            row.runs.push_back(r);

            // Time-budget guard for the slow n=4000 cells (d, e): if a single
            // trial exceeds ~4 minutes, cut this cell to what's already run
            // and note it explicitly (task Step 2 instruction).
            if ((c.n == 4000) && wall_sec > 240.0) {
                std::fprintf(stderr,
                    "[qfa_timing_probe] BUDGET CUT: cell %s trial %d took %.1f s (> 240 s); "
                    "stopping this cell at %d trial(s) instead of the planned %d.\n",
                    c.label, trial, wall_sec, trial + 1, row.ntrials_planned);
                row.budget_cut = true;
                ++trial;
                break;
            }
        }
        agg.push_back(std::move(row));
    }

    std::printf("\n=== qfa_timing_probe: per-cell aggregate (avg over trials actually run) ===\n");
    std::printf("cell n     s   f      tol      trials  avg_total_us  matvec%%  reorth%%  cert%%  other%%  checks_performed(total/avg)  final_eig_size(d_stop*s, avg)\n");
    for (const auto& row : agg) {
        long sum_total = 0, sum_matvec = 0, sum_reorth = 0, sum_cert = 0, sum_other = 0;
        long long sum_checks = 0, sum_eig = 0;
        for (const auto& r : row.runs) {
            sum_total += r.total_us; sum_matvec += r.matvec_us; sum_reorth += r.reorth_us;
            sum_cert += r.cert_us; sum_other += r.other_us;
            sum_checks += r.cert_checks;
            sum_eig += r.d_used * r.s;
        }
        const int nr = (int)row.runs.size();
        const double avg_total = nr ? (double)sum_total / nr : 0.0;
        auto avg_pct = [&](long sum) { return nr && sum_total > 0 ? 100.0 * (double)sum / (double)sum_total : 0.0; };
        std::printf("%-4s %5lld %3lld %-6s %8.1e  %6d  %12.1f  %6.1f  %6.1f  %6.1f  %6.1f  %6lld (%.1f)  %10.1f%s\n",
                    row.cell, (long long)row.n, (long long)row.s, row.fname, (double)row.tol,
                    nr, avg_total, avg_pct(sum_matvec), avg_pct(sum_reorth), avg_pct(sum_cert), avg_pct(sum_other),
                    sum_checks, nr ? (double)sum_checks / nr : 0.0,
                    nr ? (double)sum_eig / nr : 0.0,
                    row.budget_cut ? "  [BUDGET-CUT: fewer trials than planned]" : "");
    }

    delete[] B; delete[] out_ss;
    for (auto& m : mats_by_n) { (void)m; }
    return 0;
}
