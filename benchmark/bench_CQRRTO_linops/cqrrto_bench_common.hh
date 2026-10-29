#pragma once

// Helpers shared by the least-squares benchmarks: CSV provenance, the QR method dispatch,
// solver records, the preconditioned orthogonality check, the sketched backward error, and
// named arguments.

#include "RandLAPACK.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "../rl_bench_utils.hh"
#include <RandBLAS.hh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace RandLAPACK {
namespace bench {

// ============================================================================
// Provenance
// ============================================================================

/// YYYYMMDD_HHMMSS, for output file names and CSV headers: timestamp_prefix() without its
/// trailing underscore.
inline std::string make_run_timestamp() {
    std::string t = timestamp_prefix();
    t.pop_back();
    return t;
}

/// argv, each argument double-quoted, for the CSV header.
inline std::string quote_join_argv(int argc, char* argv[]) {
    std::ostringstream oss;
    for (int i = 0; i < argc; ++i) oss << (i ? " " : "") << '"' << argv[i] << '"';
    return oss.str();
}

inline std::string get_hostname() {
#if defined(_WIN32)
    const char* h = std::getenv("COMPUTERNAME");
    return (h != nullptr && *h != '\0') ? std::string(h) : std::string("(unknown)");
#else
    char buf[256];
    if (gethostname(buf, sizeof(buf)) != 0 || buf[0] == '\0') return "(unknown)";
    buf[sizeof(buf) - 1] = '\0';
    return std::string(buf);
#endif
}

inline std::string env_or(const char* key, const char* fallback) {
    const char* s = std::getenv(key);
    return (s != nullptr && *s != '\0') ? std::string(s) : std::string(fallback);
}

// ============================================================================
// QR driver options
// ============================================================================

/// Driver options that job scripts set through the environment, read once:
///   RANDLAPACK_CHOL_MAX_RETRIES=k   max_retries (default -1, retry until positive definite)
///   RANDLAPACK_CHOL_SYMMETRIZE=0|1  symmetrize_gram (default 1)
///   RANDLAPACK_GRAM_LEFT=trsm|gemm  gram_left_gemm (default trsm)
///   RANDLAPACK_SCHOLQR3_SHIFT=theory|eps  sCholQR3's first-pass shift factor: 11 n eps
///                                   (default) or eps
/// They apply to every method of run_qr_method except sCholQR3_basic, which has none of them.
/// An unrecognized value throws std::runtime_error.
struct QRDriverOptions {
    int  max_retries        = -1;
    bool symmetrize_gram    = true;
    bool gram_left_gemm     = false;
    bool scholqr3_eps_shift = false;
};

inline const QRDriverOptions& qr_driver_options() {
    static const QRDriverOptions opts = [] {
        auto env = [](const char* key) { return env_or(key, ""); };
        auto bad = [](const char* key, const std::string& v) {
            return std::runtime_error(std::string(key) + "='" + v + "' is not a recognized value");
        };
        QRDriverOptions o;
        if (std::string v = env("RANDLAPACK_CHOL_MAX_RETRIES"); !v.empty()) {
            std::optional<int> k = thread_cap_from_env("RANDLAPACK_CHOL_MAX_RETRIES");
            if (!k) throw bad("RANDLAPACK_CHOL_MAX_RETRIES", v);
            o.max_retries = *k;
        }
        if (std::string v = env("RANDLAPACK_CHOL_SYMMETRIZE"); !v.empty()) {
            if (v != "0" && v != "1") throw bad("RANDLAPACK_CHOL_SYMMETRIZE", v);
            o.symmetrize_gram = (v == "1");
        }
        if (std::string v = env("RANDLAPACK_GRAM_LEFT"); !v.empty()) {
            if (v != "trsm" && v != "gemm") throw bad("RANDLAPACK_GRAM_LEFT", v);
            o.gram_left_gemm = (v == "gemm");
        }
        if (std::string v = env("RANDLAPACK_SCHOLQR3_SHIFT"); !v.empty()) {
            if (v != "theory" && v != "eps") throw bad("RANDLAPACK_SCHOLQR3_SHIFT", v);
            o.scholqr3_eps_shift = (v == "eps");
        }
        return o;
    }();
    return opts;
}

/// Provenance lines every results CSV carries: the host, the environment knobs the library
/// reads, the commit stamp job scripts export ("unknown" when unset), and the QR driver
/// options in effect.
inline void write_env_lines(std::ostream& out) {
    const QRDriverOptions& o = qr_driver_options();
    out << "# host=" << get_hostname() << "\n"
        << "# env RANDLAPACK_BLAS2_THREADS=" << env_or("RANDLAPACK_BLAS2_THREADS", "(unset)")
        << " RANDLAPACK_FFT_THREADS=" << env_or("RANDLAPACK_FFT_THREADS", "(unset)")
        << " RANDLAPACK_SOLVE_WIDTH_MATCH=" << env_or("RANDLAPACK_SOLVE_WIDTH_MATCH", "(unset)")
        << " RANDLAPACK_GIT_COMMIT=" << build_commit() << "\n"
        << "# qr driver options (not sCholQR3_basic): max_retries=" << o.max_retries
        << " symmetrize_gram=" << o.symmetrize_gram << " gram_left_gemm=" << o.gram_left_gemm
        << " scholqr3_shift_iter1=" << (o.scholqr3_eps_shift ? "eps" : "11*n*eps") << "\n";
}

/// Warns when a written CSV's header and first data row disagree on the field count, which
/// would shift every later column for a reader.
inline void check_csv_arity(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return;
    auto n_fields = [](const std::string& s) {
        return s.empty() ? 0 : (int)std::count(s.begin(), s.end(), ',') + 1;
    };
    std::string line, header;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (header.empty()) { header = line; continue; }
        int h = n_fields(header), r = n_fields(line);
        if (h != r)
            std::cerr << "ERROR: " << filename << ": CSV header has " << h
                      << " fields but the first data row has " << r << "\n";
        return;
    }
}

// ============================================================================
// QR method dispatch
// ============================================================================

/// Accounting of one QR run. The chol_* fields are -1 for a driver without shift retries
/// (sCholQR3_basic); the times and the model are -1 or empty on failure.
struct QRRun {
    int    status         = 0;    ///< call()'s return
    int    chol_status    = -1;   ///< the failed pass's cholqr_primitive code (0 on success)
    long   qr_time_us     = -1;
    int    chol_retries   = -1;
    double chol_shift_abs = -1;   ///< pass-1 absolute shift
    double chol_shift_rel = -1;   ///< largest shift / trace(G) over the passes
    long   analytical_kb  = -1;
    std::vector<long> breakdown;  ///< the driver's times
};

/// Runs `method` on A into R with the options of qr_driver_options(): one of the Q-less
/// CQRRTO_linop, CholQR, CholQR2 and sCholQR3, or sCholQR3_basic, which forms Q = A R_1^{-1}.
/// sketch_nnz, d_factor and state are used by CQRRTO_linop only.
template <typename T, typename RNG, RandLAPACK::linops::LinearOperator GLO>
QRRun run_qr_method(const std::string& method, GLO& A, T* R, int64_t ldr, int64_t block_size,
                    int64_t sketch_nnz, T d_factor, RandBLAS::RNGState<RNG> state) {
    const QRDriverOptions& o = qr_driver_options();
    const int64_t m = A.n_rows, n = A.n_cols;
    auto configure = [&](auto& qr) {
        qr.block_size      = block_size;
        qr.max_retries     = o.max_retries;
        qr.symmetrize_gram = o.symmetrize_gram;
    };
    auto finish = [](auto& qr, int status, auto&& model) {
        QRRun out;
        out.status = status;
        if constexpr (requires { qr.n_chol_retries; }) {
            out.chol_status    = qr.chol_status;
            out.chol_retries   = qr.n_chol_retries;
            out.chol_shift_abs = (double)qr.chol_applied_shifts[0];
            double rel = 0;
            for (size_t i = 0; i < std::size(qr.chol_gram_traces); ++i)
                if (qr.chol_gram_traces[i] > 0)
                    rel = std::max(rel, (double)(qr.chol_applied_shifts[i] / qr.chol_gram_traces[i]));
            out.chol_shift_rel = rel;
        }
        if (status == 0) {
            out.qr_time_us    = qr.total_us();
            out.breakdown     = qr.times;
            out.analytical_kb = model();
        }
        return out;
    };

    if (method == "CQRRTO_linop") {
        RandLAPACK::CQRRTO_linops<T, RNG> qr(true, T(0));
        configure(qr);
        qr.nnz = sketch_nnz;
        qr.gram_left_gemm = o.gram_left_gemm;
        int st = qr.call(A, R, ldr, d_factor, state);
        return finish(qr, st, [&] {
            return cqrrto_linops_analytical_kb<T>(m, n, (double)d_factor, block_size, o.gram_left_gemm,
                                                  qr.precond_method, qr.nnz); });
    }
    if (method == "CholQR") {
        RandLAPACK::CholQR_linops<T> qr(true, T(0));
        configure(qr);
        int st = qr.call(A, R, ldr);
        return finish(qr, st, [&] {
            return cholqr_linops_analytical_kb<T>(m, n, block_size, RandLAPACK::linops::HasMaterializeCols<GLO>); });
    }
    if (method == "CholQR2") {
        RandLAPACK::CholQR2_linops<T> qr(true, T(0));
        configure(qr);
        qr.gram_left_gemm = o.gram_left_gemm;
        int st = qr.call(A, R, ldr);
        return finish(qr, st, [&] {
            return cholqr2_linops_analytical_kb<T>(m, n, block_size, o.gram_left_gemm); });
    }
    if (method == "sCholQR3") {
        RandLAPACK::sCholQR3_linops<T> qr(true, T(0));
        configure(qr);
        qr.gram_left_gemm = o.gram_left_gemm;
        if (o.scholqr3_eps_shift) qr.shift_factor_iter1 = std::numeric_limits<T>::epsilon();
        int st = qr.call(A, R, ldr);
        return finish(qr, st, [&] {
            return scholqr3_linops_analytical_kb<T>(m, n, block_size, o.gram_left_gemm); });
    }
    if (method == "sCholQR3_basic") {
        RandLAPACK::sCholQR3_linops_basic<T> qr(true, T(0));
        int st = qr.call(A, R, ldr);
        return finish(qr, st, [&] { return scholqr3_linops_basic_analytical_kb<T>(m, n); });
    }
    throw std::invalid_argument("run_qr_method: unknown method '" + method + "'");
}

/// A method-mask bit and the CSV rows it selects.
struct MethodBit {
    int64_t bit;
    std::vector<std::string> rows;
};

/// Rows selected by `mask`, in table order. Throws on bits outside the table or an empty
/// selection.
inline std::vector<std::string> decode_method_mask(int64_t mask, const std::vector<MethodBit>& table) {
    int64_t known = 0;
    std::vector<std::string> rows;
    for (const auto& e : table) {
        known |= e.bit;
        if (mask & e.bit) rows.insert(rows.end(), e.rows.begin(), e.rows.end());
    }
    if ((mask & ~known) != 0)
        throw std::runtime_error("method mask " + std::to_string(mask) + " has bits outside "
                                 + std::to_string(known));
    if (rows.empty())
        throw std::runtime_error("method mask " + std::to_string(mask) + " selects no methods");
    return rows;
}

// ============================================================================
// Solver records
// ============================================================================

/// Short CSV name of a restarted_pcg_ne exit.
inline const char* pcg_stop_reason(RestartedPCGNETermination t) {
    switch (t) {
        case RestartedPCGNETermination::converged:        return "tol";
        case RestartedPCGNETermination::iteration_budget: return "budget";
        case RestartedPCGNETermination::breakdown:        return "breakdown";
        case RestartedPCGNETermination::round_budget:     return "rounds";
        case RestartedPCGNETermination::ls_floor:         return "floor";
        case RestartedPCGNETermination::oracle_met:       return "be";
    }
    return "unknown";
}

/// Short CSV name of an LSQR exit (stop_test as reported by lsqr).
inline const char* lsqr_stop_reason(bool converged, int stop_test) {
    if (!converged) return "budget";
    return (stop_test == 2) ? "ne_floor" : "tol";
}

/// The round of a run that ended worst: a breakdown, else a capped round, else a stagnated
/// one, else the round with the largest CG relres.
struct WorstRound {
    int    status      = -1;   ///< PCGInnerTermination as int; -1 when no round ran
    double relres      = -1;
    double best_relres = -1;
    int    best_iter   = -1;
};

template <typename T>
WorstRound worst_round(const RestartedPCGNEReport<T>& rep) {
    WorstRound w;
    if (rep.cg_status.empty()) return w;
    auto severity = [](PCGInnerTermination s) {
        switch (s) {
            case PCGInnerTermination::breakdown: return 3;
            case PCGInnerTermination::hit_cap:   return 2;
            case PCGInnerTermination::stagnated: return 1;
            default:                             return 0;
        }
    };
    size_t k = 0;
    for (size_t i = 1; i < rep.cg_status.size(); ++i)
        if (severity(rep.cg_status[i]) > severity(rep.cg_status[k])) k = i;
    if (severity(rep.cg_status[k]) == 0)
        for (size_t i = 0; i < rep.cg_relres.size(); ++i)
            if (rep.cg_relres[i] > rep.cg_relres[k]) k = i;
    w.status      = static_cast<int>(rep.cg_status[k]);
    w.relres      = (double)rep.cg_relres[k];
    w.best_relres = (double)rep.cg_best_relres[k];
    w.best_iter   = rep.cg_best_iter[k];
    return w;
}

/// Oracle value of the returned iterate: the last round's, else the warm start's; +inf when
/// the oracle was on but measured nothing, -1 when it was off.
template <typename T>
T final_oracle_value(const RestartedPCGNEReport<T>& rep, bool oracle_on) {
    if (!rep.be.empty()) return rep.be.back();
    if (rep.be_x0 >= T(0)) return rep.be_x0;
    return oracle_on ? std::numeric_limits<T>::infinity() : T(-1);
}

/// Header of the per-round sidecar CSV.
inline void write_rounds_header(std::ostream& out) {
    out << "# Per-round restarted_pcg_ne records. inner_status: 0 converged, 1 hit_cap, "
           "2 breakdown, 3 stagnated, 4 polled (an in-round oracle poll met the target).\n"
        << "# be_kw: sketched Karlson-Walden backward error after the round, relative to "
           "||A||_F; -1 when the oracle was off. be_polls: in-round oracle evaluations.\n"
        << "algorithm,run,round,inner_iters,inner_status,inner_relres,best_relres,best_iter,"
           "ls_relres,be_kw,be_polls\n";
}

/// One sidecar row per round of `rep` (rounds numbered from 1).
template <typename T>
void write_rounds(std::ostream& out, const std::string& alg, int64_t run,
                  const RestartedPCGNEReport<T>& rep) {
    for (size_t k = 0; k < rep.cg_iters.size(); ++k) {
        out << alg << "," << run << "," << k + 1 << "," << rep.cg_iters[k] << ","
            << static_cast<int>(rep.cg_status[k]) << ","
            << std::scientific << std::setprecision(6) << rep.cg_relres[k] << ","
            << rep.cg_best_relres[k] << "," << rep.cg_best_iter[k] << "," << rep.ls_relres[k] << ","
            << rep.be[k] << "," << rep.polls[k] << "\n";
    }
}

// ============================================================================
// Preconditioner quality
// ============================================================================

/// Largest n for which compute_orth_error_explicit forms cond(A R^{-1}).
inline constexpr int64_t kCondMaxN = 16384;

/// ||Q^T Q - I||_F / sqrt(n) for Q = A R^{-1}, with Q formed by form_q_from_r (A's columns from
/// materialize_cols, or A applied to identity column blocks of width block_size, so A itself is
/// never materialized beyond Q). When cond is non-null it receives cond_2(Q) from Q's singular
/// values (not from the eigenvalues of Q^T Q, which lose it above about u^{-1/2}): +inf when the
/// smallest singular value is 0, and -1 for n > kCondMaxN or a failed SVD.
template <typename T, typename GLO>
T compute_orth_error_explicit(GLO& A, const T* R, int64_t block_size, T* cond = nullptr) {
    const int64_t m = A.n_rows, n = A.n_cols;
    std::unique_ptr<T[]> Q(new T[m * n]);
    form_q_from_r<T>(A, R, n, block_size, Q.get());
    const T orth = RandLAPACK::testing::orthogonality_error<T>(Q.get(), m, n);
    if (cond != nullptr) {
        *cond = T(-1);
        if (n <= kCondMaxN) {
            try {
                std::vector<T> sv = RandLAPACK::testing::compute_singular_values<T>(Q.get(), m, n);   // destroys Q
                *cond = (sv[n - 1] > 0) ? sv[0] / sv[n - 1] : std::numeric_limits<T>::infinity();
            } catch (const std::exception&) {}
        }
    }
    return orth;
}

// ============================================================================
// Sketched Karlson-Walden backward error (Epperly, Meier and Nakatsukasa, arXiv:2406.03468)
// ============================================================================
//
// For x approximately solving min ||b - A x|| and weight theta, the backward error
// BE_theta(x) = min ||[dA, theta db]||_F over perturbations that make x optimal is estimated
// within sqrt(2) (their Fact 4.1) by
//   theta / sqrt(1 + theta^2 ||x||^2) * ||(A^T A + nu^2 I)^{-1/2} A^T (b - A x)||,
//   nu^2 = theta^2 ||b - A x||^2 / (1 + theta^2 ||x||^2),
// and with A^T A replaced by (SA)^T (SA) for a sparse sign sketch S of d >= 2n rows by
// ||(Sigma^2 + nu^2 I)^{-1/2} V^T A^T (b - A x)|| from the SVD SA = U Sigma V^T (their eq. 4.2).
// Two weights are reported, both relative to ||A||_F: theta = ||A||_F / ||b|| and theta = inf.

/// Nonzeros per column of the reference sketch (their zeta = 8), independent of the solver's,
/// and capped at the sketch's dimensions.
inline constexpr int64_t kKWSketchNNZ = 8;
/// Seed of the reference sketch.
inline constexpr uint32_t kKWSeed = 20240914;

/// The reference sketch's nonzeros per column for A m x n (d = 2n rows).
inline int64_t kw_sketch_nnz(int64_t m, int64_t n) { return std::min({kKWSketchNNZ, 2 * n, m}); }

template <typename T>
struct KWBackwardErrorRef {
    int64_t n = 0, d = 0, nnz = 0;   ///< nnz: the nonzeros per column actually used
    T A_fro = 0;              ///< ||A||_F, exact, from the operator's columns
    std::vector<T> sigma;     ///< singular values of SA, descending
    std::vector<T> VT;        ///< n x n, ColMajor; row i is the i-th right singular vector
};

/// Forms SA (d x n, sketch_nnz capped at min(d, m) nonzeros per column) one column block of A
/// at a time (from materialize_cols when A provides it, otherwise A applied to identity
/// columns through the product the LinearOperator concept requires), ||A||_F alongside, and
/// the SVD of SA. m and n must be A's dimensions.
template <typename T, typename RNG, typename GLO>
KWBackwardErrorRef<T> build_kw_reference(GLO& A, int64_t m, int64_t n, int64_t d,
                                         int64_t sketch_nnz, RandBLAS::RNGState<RNG> state,
                                         int64_t block_size) {
    randlapack_require(m == A.n_rows && n == A.n_cols) << "build_kw_reference: m=" << m << ", n=" << n
        << " must equal the operator's " << A.n_rows << " x " << A.n_cols;
    randlapack_require(d >= n) << "build_kw_reference: d=" << d << " must be >= n=" << n;
    KWBackwardErrorRef<T> ref;
    ref.n = n; ref.d = d; ref.nnz = std::min({sketch_nnz, d, m});
    RandBLAS::SparseDist D(d, m, ref.nnz, RandBLAS::Axis::Short);
    RandBLAS::SparseSkOp<T, RNG> S(D, state);
    RandBLAS::fill_sparse(S);

    const int64_t b = gram_block_width(block_size, n);
    constexpr bool direct = RandLAPACK::linops::HasMaterializeCols<GLO>;
    std::vector<T> SA((size_t)d * n), E(direct ? 0 : (size_t)n * b, T(0)), Y((size_t)m * b);
    T fro_sq = 0;
    for (int64_t j0 = 0; j0 < n; j0 += b) {
        const int64_t bk = std::min(b, n - j0);
        if constexpr (direct) {
            A.materialize_cols(j0, bk, Y.data(), m);
        } else {
            for (int64_t j = 0; j < bk; ++j) E[(size_t)(j0 + j) + (size_t)j * n] = T(1);
            A(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
              m, bk, n, T(1), E.data(), n, T(0), Y.data(), m);
            for (int64_t j = 0; j < bk; ++j) E[(size_t)(j0 + j) + (size_t)j * n] = T(0);
        }
        for (int64_t j = 0; j < bk; ++j) {
            T c = blas::nrm2(m, Y.data() + (size_t)j * m, 1);
            fro_sq += c * c;
        }
        // RandBLAS samples unscaled +/-1 entries; the paper's S carries zeta^{-1/2}.
        RandBLAS::sketch_general(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                 d, bk, m, (T)S.dist.isometry_scale, S, 0, 0, Y.data(), m,
                                 T(0), SA.data() + (size_t)j0 * d, d);
    }
    ref.A_fro = std::sqrt(fro_sq);
    ref.sigma.assign(n, T(0));
    ref.VT.assign((size_t)n * n, T(0));
    // gesdd, not gesvd: the estimate needs the small singular directions, and gesvd with
    // vectors is far slower at large n. OverwriteVec with d >= n writes U into SA.
    int64_t info = lapack::gesdd(lapack::Job::OverwriteVec, d, n, SA.data(), d,
                                 ref.sigma.data(), nullptr, 1, ref.VT.data(), n);
    randlapack_require(info == 0) << "build_kw_reference: gesdd returned info=" << info;
    return ref;
}

/// The estimate for one solution: ATr = A^T (b - A x), r_norm = ||b - A x||, x_norm = ||x||,
/// b_norm = ||b||. be_theta_rel and be_inf_rel are relative to ||A||_F: -1 when undefined
/// (||A||_F = 0, or x = 0 for be_inf), +inf when NaN; res_orth = ||A^T r|| / (||A||_F ||r||).
template <typename T>
void kw_backward_error(const KWBackwardErrorRef<T>& ref, const T* ATr,
                       T r_norm, T x_norm, T b_norm,
                       T& be_theta_rel, T& be_inf_rel, T& theta_out, T& res_orth) {
    const int64_t n = ref.n;
    std::vector<T> z((size_t)n);
    blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, n, n, T(1), ref.VT.data(), n,
               ATr, 1, T(0), z.data(), 1);
    auto estimate = [&](T nu_sq, T prefactor) {
        T acc = 0;
        for (int64_t i = 0; i < n; ++i) acc += z[i] * z[i] / (ref.sigma[i] * ref.sigma[i] + nu_sq);
        return prefactor * std::sqrt(acc);
    };
    const T theta = (b_norm > 0) ? ref.A_fro / b_norm : T(0);
    const T den   = T(1) + theta * theta * x_norm * x_norm;
    const T be_theta = estimate(theta * theta * r_norm * r_norm / den, theta / std::sqrt(den));
    // Negated tests, so that a NaN norm reaches the NaN-to-inf mapping below.
    const T be_inf   = !(x_norm <= 0) ? estimate(r_norm * r_norm / (x_norm * x_norm), T(1) / x_norm) : T(-1);
    be_theta_rel = (ref.A_fro > 0) ? be_theta / ref.A_fro : T(-1);
    be_inf_rel   = (ref.A_fro > 0 && !(be_inf < 0)) ? be_inf / ref.A_fro : T(-1);
    if (std::isnan(be_theta_rel)) be_theta_rel = std::numeric_limits<T>::infinity();
    if (std::isnan(be_inf_rel))   be_inf_rel   = std::numeric_limits<T>::infinity();
    theta_out = theta;
    res_orth  = (ref.A_fro > 0 && r_norm > 0) ? blas::nrm2(n, ATr, 1) / (ref.A_fro * r_norm) : T(-1);
}

/// The engine's oracle on the reference: be_theta relative to ||A||_F, the be_kw_theta of the
/// sidecar; a sentinel or NaN maps to +inf so it never meets a target. ref must outlive it.
template <typename T>
RandLAPACK::BackwardErrorOracle<T> make_kw_oracle(const KWBackwardErrorRef<T>& ref, int64_t m,
                                                  int64_t n, T b_norm) {
    return [&ref, m, n, b_norm](const T* x, const T* r, const T* ATr) -> T {
        T be_theta, be_inf, theta, res_orth;
        kw_backward_error<T>(ref, ATr, blas::nrm2(m, r, 1), blas::nrm2(n, x, 1), b_norm,
                             be_theta, be_inf, theta, res_orth);
        return (be_theta >= T(0)) ? be_theta : std::numeric_limits<T>::infinity();
    };
}

/// The oracle's target mult * sqrt(n) * u (u = eps / 2), or -1 (off) for mult <= 0. The
/// sqrt(n) follows the square-system rule of Epperly, Greenbaum and Nakatsukasa
/// (arXiv:2502.17767, Algorithm 5.1).
template <typename T>
T resolve_be_tol(double mult, int64_t n) {
    if (mult <= 0.0) return T(-1);
    return (T)(mult * std::sqrt((double)n) * 0.5 * (double)std::numeric_limits<T>::epsilon());
}

/// The backward-error reference of a benchmark problem A x ~ b (d = 2n, kKWSeed) and, when the
/// oracle is on, the oracle built on it.
template <typename T>
struct BackwardErrorSetup {
    T be_tol = T(-1);                             ///< the oracle's target; -1 = off
    std::unique_ptr<KWBackwardErrorRef<T>> ref;   ///< null until built
    double build_s = 0;
    RandLAPACK::BackwardErrorOracle<T> oracle;    ///< empty when the oracle is off
};

/// Builds the reference with a progress line.
template <typename T, typename RNG, typename GLO>
void build_kw_reference_logged(BackwardErrorSetup<T>& be, GLO& A, int64_t m, int64_t n,
                               int64_t block_size) {
    std::cout << "\nBackward-error reference: sketched Karlson-Walden, d=2n=" << 2 * n
              << ", nnz=" << kw_sketch_nnz(m, n) << " ... " << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    be.ref = std::make_unique<KWBackwardErrorRef<T>>(build_kw_reference<T, RNG>(
        A, m, n, 2 * n, kKWSketchNNZ, RandBLAS::RNGState<RNG>(kKWSeed), block_size));
    be.build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "done (" << std::fixed << std::setprecision(1) << be.build_s << " s, ||A||_F="
              << std::scientific << std::setprecision(6) << (double)be.ref->A_fro << ")"
              << std::defaultfloat << "\n";
}

/// With be_tol >= 0 the reference is built now, before the measured rows (so outside every
/// row's timer and RSS window), and the oracle on it; otherwise both wait for the post-pass.
/// Throws when the build fails.
template <typename T, typename RNG, typename GLO>
BackwardErrorSetup<T> setup_backward_error(GLO& A, int64_t m, int64_t n, T b_norm, T be_tol,
                                           int64_t block_size) {
    BackwardErrorSetup<T> be;
    be.be_tol = be_tol;
    if (be_tol >= T(0)) {
        build_kw_reference_logged<T, RNG>(be, A, m, n, block_size);
        be.oracle = make_kw_oracle<T>(*be.ref, m, n, b_norm);
    }
    return be;
}

/// A solution kept for the backward-error post-pass.
template <typename T>
struct Solution {
    std::string alg;
    int64_t run;
    std::vector<T> x;
};

/// Writes the backward-error sidecar: a provenance line, then the estimate for each solution of
/// A x ~ b, building the reference first when the oracle did not. A failed build, or a float
/// problem (whose single-precision SVD of SA cannot resolve the small singular directions the
/// estimate weights most), leaves a '#' line saying why instead.
template <typename T, typename RNG, typename GLO>
void write_backward_error_sidecar(const std::string& file, BackwardErrorSetup<T>& be, GLO& A,
                                  int64_t m, int64_t n, const T* b, T b_norm, int64_t block_size,
                                  const std::vector<Solution<T>>& solutions) {
    std::ofstream out(file);
    if constexpr (std::is_same_v<T, float>) {
        out << "# backward error not computed: a single-precision SVD of the reference sketch "
               "cannot resolve the small singular directions the estimate weights most\n";
        return;
    }
    try {
        if (!be.ref) build_kw_reference_logged<T, RNG>(be, A, m, n, block_size);
    } catch (const std::exception& e) {
        std::cerr << "Warning: backward-error reference failed (" << e.what() << ")\n";
        out << "# backward-error reference failed: " << e.what() << "\n";
        return;
    }
    const KWBackwardErrorRef<T>& ref = *be.ref;
    out << "# sketched Karlson-Walden backward error (Epperly, Meier and Nakatsukasa, eq. 4.2),"
        << " relative to ||A||_F; d=" << ref.d << " nnz=" << ref.nnz << " seed=" << kKWSeed
        << " ||A||_F=" << std::scientific << std::setprecision(6) << (double)ref.A_fro
        << " ||b||=" << (double)b_norm
        << " theta=||A||_F/||b||=" << (double)((b_norm > 0) ? ref.A_fro / b_norm : T(0))
        << " reference_build_s=" << std::fixed << std::setprecision(1) << be.build_s << "\n"
        << "algorithm,run,be_kw_theta,be_kw_inf,theta,r_norm,x_norm,res_orth\n";
    std::vector<T> r((size_t)m), ATr((size_t)n);
    for (const auto& sol : solutions) {
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          m, 1, n, T(1), sol.x.data(), n, T(0), r.data(), m);
        for (int64_t i = 0; i < m; ++i) r[i] = b[i] - r[i];
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, T(1), r.data(), m, T(0), ATr.data(), n);
        const T r_norm = blas::nrm2(m, r.data(), 1), x_norm = blas::nrm2(n, sol.x.data(), 1);
        T be_theta, be_inf, theta, res_orth;
        kw_backward_error<T>(ref, ATr.data(), r_norm, x_norm, b_norm, be_theta, be_inf, theta, res_orth);
        std::cout << "  [" << sol.alg << "] run " << sol.run << ": BE_theta=" << std::scientific
                  << std::setprecision(3) << (double)be_theta << " BE_inf=" << (double)be_inf
                  << " res_orth=" << (double)res_orth << std::defaultfloat << "\n";
        out << sol.alg << "," << sol.run << "," << std::scientific << std::setprecision(6)
            << be_theta << "," << be_inf << "," << theta << "," << r_norm << "," << x_norm << ","
            << res_orth << "\n";
    }
}

// ============================================================================
// Named command-line arguments
// ============================================================================

/// --name=value arguments (and a bare --name for booleans). An unknown, repeated, empty or
/// malformed argument is an error, so a value cannot land on the wrong parameter.
class BenchArgs {
public:
    BenchArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0)
                throw std::runtime_error("expected a --name=value argument, got '" + a + "'");
            a = a.substr(2);
            auto eq = a.find('=');
            std::string key = (eq == std::string::npos) ? a : a.substr(0, eq);
            if (kv_.count(key) != 0) throw std::runtime_error("argument --" + key + " given twice");
            kv_[key] = (eq == std::string::npos) ? std::string() : a.substr(eq + 1);
            bare_[key] = (eq == std::string::npos);
            order_.push_back(key);
        }
    }

    bool has(const std::string& k) const { return kv_.count(k) != 0; }

    std::string str(const std::string& k, const std::string& dflt) const {
        return has(k) ? value(k) : dflt;
    }
    std::string require_str(const std::string& k) const {
        if (!has(k)) throw std::runtime_error("missing required argument --" + k);
        return value(k);
    }
    /// A finite number.
    double dbl(const std::string& k, double dflt) const {
        if (!has(k)) return dflt;
        const std::string v = value(k);
        size_t pos = 0;
        double out = 0;
        try { out = std::stod(v, &pos); } catch (const std::exception&) { pos = 0; }
        if (pos == 0 || pos != v.size() || !std::isfinite(out))
            throw std::runtime_error("--" + k + " expects a finite number, got '" + v + "'");
        return out;
    }
    int64_t i64(const std::string& k, int64_t dflt) const {
        if (!has(k)) return dflt;
        const std::string v = value(k);
        size_t pos = 0;
        long long out = 0;
        try { out = std::stoll(v, &pos); } catch (const std::exception&) { pos = 0; }
        if (pos == 0 || pos != v.size())
            throw std::runtime_error("--" + k + " expects an integer in int64 range, got '" + v + "'");
        return (int64_t)out;
    }
    int i32(const std::string& k, int dflt) const {
        const int64_t v = i64(k, dflt);
        if (v < INT_MIN || v > INT_MAX)
            throw std::runtime_error("--" + k + " is out of int range: " + std::to_string(v));
        return (int)v;
    }
    /// A bare --name, or --name=1/true/yes; --name=0/false/no is false.
    bool boolean(const std::string& k, bool dflt = false) const {
        if (!has(k)) return dflt;
        if (bare_.at(k)) return true;
        const std::string& v = kv_.at(k);
        if (v == "1" || v == "true" || v == "yes") return true;
        if (v == "0" || v == "false" || v == "no") return false;
        throw std::runtime_error("--" + k + " expects a boolean, got '" + v + "'");
    }

    void reject_unknown(std::initializer_list<const char*> known) const {
        for (const auto& name : order_) {
            if (std::none_of(known.begin(), known.end(), [&](const char* k) { return name == k; })) {
                std::string msg = "unknown argument --" + name + "\nknown arguments:";
                for (const char* k : known) msg += std::string(" --") + k;
                throw std::runtime_error(msg);
            }
        }
    }

private:
    std::string value(const std::string& k) const {
        const std::string& v = kv_.at(k);
        if (v.empty()) throw std::runtime_error("--" + k + " needs a value (--" + k + "=...)");
        return v;
    }
    std::map<std::string, std::string> kv_;
    std::map<std::string, bool> bare_;
    std::vector<std::string> order_;
};

} // namespace bench
} // namespace RandLAPACK
