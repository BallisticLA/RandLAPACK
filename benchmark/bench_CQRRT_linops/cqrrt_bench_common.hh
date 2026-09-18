// cqrrt_bench_common.hh: shared utilities for CQRRT linop benchmarks
#pragma once

#include "RandLAPACK.hh"
#include "../../extras/misc/ext_util.hh"
#include <RandBLAS.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

// Load a Matrix Market file into a CSRMatrix. Sets m, n, nnz on exit.
template <typename T>
static RandBLAS::sparse_data::csr::CSRMatrix<T> load_csr(
    const std::string& path, int64_t& m, int64_t& n, int64_t& nnz)
{
    auto coo = RandLAPACK_extras::coo_from_matrix_market<T>(path);
    m = coo.n_rows; n = coo.n_cols; nnz = coo.nnz;
    RandBLAS::sparse_data::csr::CSRMatrix<T> csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
    return csr;
}

// Load a sparse matrix and emit the standard "Loading <label> from <path>... done (m x n, nnz=N)"
// progress messages used across all CQRRT benchmarks.
template <typename T>
static RandBLAS::sparse_data::csr::CSRMatrix<T> load_csr_verbose(
    const std::string& label, const std::string& path,
    int64_t& m, int64_t& n, int64_t& nnz)
{
    std::cout << "Loading " << label << " from " << path << "... " << std::flush;
    auto csr = load_csr<T>(path, m, n, nnz);
    std::cout << "done (" << m << " x " << n << ", nnz=" << nnz << ")\n";
    return csr;
}

// Return a YYYYMMDD_HHMMSS timestamp string suitable for naming output files
// or stamping a CSV provenance header.
inline std::string make_run_timestamp() {
    char buf[64];
    std::time_t now = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    return std::string(buf);
}

// ============================================================================
// Helpers shared by every least-squares benchmark in this suite
// (bench_CQRRT_linops and bench_toeplitz_ls). Consolidated here so a fix lands
// once instead of drifting into per-file copies (duplicated dispatch/helper
// blocks are this suite's standing failure mode, see refined_blendenpik.hh).
// ============================================================================
namespace RandLAPACK {
namespace bench {

// Campaign knob: retry cap applied to every MEASURED Q-less driver row.
// RANDLAPACK_CHOL_MAX_RETRIES=0 makes a Cholesky breakdown report as a failed
// row (qr_status != 0) instead of silently switching the row to the
// shift-rescued variant of its method; unset keeps the library default (-1,
// unbounded rescue). Warm-up constructions are exempt. Static cache: fixed for
// the process lifetime after the first read, so validate via benchmarks, not
// gtests.
inline int bench_chol_max_retries() {
    static const int v = []() {
        const char* s = std::getenv("RANDLAPACK_CHOL_MAX_RETRIES");
        return (s != nullptr && *s != '\0') ? std::atoi(s) : -1;
    }();
    return v;
}

// Verifies that a just-written CSV's header and its first data row agree on field count.
// The headers here are string literals and the rows are long chains of <<, so nothing
// otherwise ties them together: a writer that gains a column on one side and not the other
// silently shifts every later column, and every consumer then misreads the file. That has
// happened in this suite before. One reopen of a file we just closed is far cheaper than
// the benchmark that produced it.
inline void check_csv_arity(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return;
    auto n_fields = [](const std::string& s) {
        return s.empty() ? 0 : (int)std::count(s.begin(), s.end(), ',') + 1;
    };
    std::string line, header;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;   // provenance lines
        if (header.empty()) { header = line; continue; }
        int h = n_fields(header), r = n_fields(line);
        if (h != r) {
            std::cerr << "ERROR: " << filename << ": CSV header has " << h
                      << " fields but the first data row has " << r
                      << ". Every column after the mismatch will be misread.\n";
        }
        return;   // one data row is enough; they are all written by the same code path
    }
}

// Accounting harvested from one Q-less QR run.
struct QRRun {
    int  status        = 0;
    long qr_time_us    = 0;
    int  chol_retries  = 0;
    long analytical_kb = -1;
    std::vector<long> breakdown;
};

// Runs one CholQR-family driver (CholQR, CholQR2, sCholQR3, sCholQR3_basic all share
// call(A, R, ldr)) and harvests its accounting. Both benchmarks dispatch the same
// methods, and harvesting in one place is what keeps them from drifting: they had
// already diverged, one recording chol_retries unconditionally and the other only on
// success. chol_retries is meaningful whether or not the factorization succeeded, so
// it is recorded unconditionally here.
// The caller sets any driver knobs (block_size, nnz) before calling, and folds the
// shift records itself, since the two benchmarks store those differently.
template <typename QR, typename GLO, typename T, typename AnalyticalFn>
QRRun run_cholqr_family(QR& qr, GLO& A, T* R, int64_t ldr, AnalyticalFn&& analytical_kb) {
    QRRun out;
    qr.max_retries   = bench_chol_max_retries();
    out.status       = qr.call(A, R, ldr);
    out.chol_retries = qr.n_chol_retries;
    if (out.status == 0) {
        out.qr_time_us    = qr.total_us();
        out.breakdown     = qr.times;
        out.analytical_kb = analytical_kb();
    }
    return out;
}

// Named exit conditions for the CSV: distinguishes "hit the LS floor honestly"
// from "ran out of budget", which shared a flag value before.
inline const char* pcg_stop_reason(int status) {
    switch (status) {
        case 0: return "tol";       case 1: return "budget";
        case 2: return "breakdown"; case 3: return "rounds";
        case 4: return "floor";     case 5: return "be";
        default: return "unknown";
    }
}
inline const char* lsqr_stop_reason(bool converged, int stop_test) {
    if (!converged) return "budget";
    return (stop_test == 2) ? "ne_floor" : "tol";
}

// Env-knob provenance for every results CSV: these knobs change the
// algorithms without changing the row labels, so a CSV that does not echo
// them cannot identify its own campaign arm.
inline std::string env_or(const char* key) {
    const char* s = std::getenv(key);
    return (s != nullptr && *s != '\0') ? std::string(s) : std::string("(unset)");
}
inline void write_env_line(std::ostream& out) {
    out << "# env RANDLAPACK_GRAM_LEFT=" << env_or("RANDLAPACK_GRAM_LEFT")
        << " RANDLAPACK_CHOL_MAX_RETRIES=" << env_or("RANDLAPACK_CHOL_MAX_RETRIES")
        << " RANDLAPACK_CHOL_SYMMETRIZE=" << env_or("RANDLAPACK_CHOL_SYMMETRIZE")
        << " RANDLAPACK_SCHOLQR3_SHIFT=" << env_or("RANDLAPACK_SCHOLQR3_SHIFT")
        << " RANDLAPACK_BLAS2_THREADS=" << env_or("RANDLAPACK_BLAS2_THREADS")
        << " RANDLAPACK_FFT_THREADS=" << env_or("RANDLAPACK_FFT_THREADS")
        << " RANDLAPACK_SOLVE_FFT_MATCH=" << env_or("RANDLAPACK_SOLVE_FFT_MATCH")
        << " RANDLAPACK_GIT_COMMIT=" << env_or("RANDLAPACK_GIT_COMMIT") << "\n";
}

// Provenance helpers: a CSV must be traceable back to the exact invocation
// and machine that produced it without a side log. Shared across every
// benchmark in this suite so a fix lands once instead of drifting into
// per-file copies.
inline std::string quote_join_argv(int argc, char* argv[]) {
    std::ostringstream oss;
    for (int i = 0; i < argc; ++i) {
        if (i) oss << ' ';
        oss << '"' << argv[i] << '"';
    }
    return oss.str();
}
inline std::string get_hostname() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) != 0 || buf[0] == '\0') return "(unknown)";
    buf[sizeof(buf) - 1] = '\0';
    return std::string(buf);
}
inline void write_host_line(std::ostream& out) {
    out << "# host=" << get_hostname() << "\n";
}

// Schema for the per-round engine records sidecar (restarted_pcg_ne /
// IterRefineLSQ): one row per (algorithm, run, round).
// be_kw: sketched Karlson-Walden backward error of the iterate after the round,
// relative to ||A||_F; -1 when the oracle was off.
inline const char* kRoundsCsvHeader =
    "algorithm,run,round,inner_iters,inner_status,inner_relres,best_relres,best_iter,ls_relres,be_kw\n";

// Write one round record row (round_idx is 1-based, matching every caller).
template <typename T>
static void write_round_row(std::ostream& out, const std::string& alg, int64_t run_idx,
                            size_t round_idx, int iters, int status, T relres,
                            T best_relres, int best_iter, T ls_relres, T be_kw) {
    out << alg << "," << run_idx << "," << round_idx << ","
        << iters << "," << status << ","
        << std::scientific << std::setprecision(6) << relres << ","
        << best_relres << "," << best_iter << "," << ls_relres << "," << be_kw << "\n";
}

// Fold a driver's per-pass Cholesky shift record (chol_applied_shifts /
// chol_gram_traces, fixed-size arrays sized by the pass count) into
// (shift_abs, shift_rel): pass-1 absolute shift plus the worst relative shift
// across passes. npasses defaults to the full array N (every current call
// site sizes its array to exactly its own pass count).
template <typename T, typename TOut, size_t N>
static void fold_chol_shift(TOut& shift_abs, TOut& shift_rel,
                            const T (&shifts)[N], const T (&traces)[N],
                            size_t npasses = N) {
    shift_abs = (TOut)shifts[0];
    T rel = T(0);
    for (size_t i = 0; i < npasses; ++i)
        if (traces[i] > T(0)) rel = std::max(rel, shifts[i] / traces[i]);
    shift_rel = (TOut)rel;
}

// Estimate ||A||_2 via power iteration on A^T A. O(iters * (m+n)) memory, no
// materialization. Returns the operator 2-norm directly (||A v|| at the
// converged v), NOT the Gram eigenvalue (see power_lambda_max below for the
// related-but-distinct estimator used by the Toeplitz benchmark). The two
// compute mathematically related quantities (lambda_max = sigma^2) via
// different floating-point paths (Rayleigh quotient vs squared norm) and are
// NOT interchangeable without re-deriving bit-identical output, since
// power_lambda_max's result scales that benchmark's regularization operator.
template <typename T, typename GLO>
static T estimate_op_2norm(GLO& A_op, int64_t m, int64_t n, int iters = 10) {
    T* v  = new T[n];
    T* Av = new T[m];
    {
        std::mt19937 rng(7);
        std::normal_distribution<T> N01(0, 1);
        for (int64_t i = 0; i < n; ++i) v[i] = N01(rng);
    }
    T sigma = (T)0;
    for (int it = 0; it < iters; ++it) {
        T nv = blas::nrm2(n, v, 1);
        if (nv == 0) { delete[] v; delete[] Av; return (T)0; }
        blas::scal(n, (T)1.0 / nv, v, 1);
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, 1, n, (T)1.0, v, n, (T)0.0, Av, m);
        sigma = blas::nrm2(m, Av, 1);
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             n, 1, m, (T)1.0, Av, m, (T)0.0, v, n);
    }
    delete[] v;
    delete[] Av;
    return sigma;
}

// Power iteration for lambda_max(A^T A): x <- A'(A x), normalize; Rayleigh
// quotient. Double precision only (the Toeplitz benchmark this serves is
// double-only). See estimate_op_2norm above for why this is kept as a
// separate function rather than derived from it.
template <typename TOp>
static double power_lambda_max(TOp& A_op, int64_t m, int64_t n, int iters) {
    std::vector<double> x(n), Ax(m), Gx(n);
    std::mt19937 rng(12345); std::normal_distribution<double> nd(0, 1);
    for (auto& v : x) v = nd(rng);
    double nrm = blas::nrm2(n, x.data(), 1); blas::scal(n, 1.0 / nrm, x.data(), 1);
    double lam = 0;
    for (int it = 0; it < iters; ++it) {
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, 1, n, 1.0, x.data(), n, 0.0, Ax.data(), m);
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             n, 1, m, 1.0, Ax.data(), m, 0.0, Gx.data(), n);
        lam = blas::dot(n, x.data(), 1, Gx.data(), 1);   // x'(A'A)x with ||x||=1
        double gn = blas::nrm2(n, Gx.data(), 1);
        if (gn == 0) break;
        blas::scal(n, 1.0 / gn, Gx.data(), 1);
        std::copy(Gx.begin(), Gx.end(), x.begin());
    }
    return lam;
}

// orth_err: ||Q^T Q - I||_F / sqrt(n), with Q = A * R^{-1} materialized
// explicitly one column block at a time (peak extra memory beyond Q stays
// O(n^2 + m*b)). cond_out (optional): when non-null, also fills it with
// cond(A R^{-1}) = sqrt(lambda_max/lambda_min) of Q^T Q, reusing the Gram this
// routine already forms, whenever n <= cond_cap (cond_cap <= 0 = unlimited);
// otherwise *cond_out is left at no_cond_sentinel. The eig is O(n^3), which is
// why it is capped and why computing it at all is optional.
template <typename T, typename GLO>
static T compute_orth_error_explicit(GLO& A_op, const T* R, int64_t m, int64_t n,
                                     int64_t block_size, T* cond_out = nullptr,
                                     int64_t cond_cap = 16384,
                                     T no_cond_sentinel = (T)-1) {
    int64_t b = (block_size > 0 && block_size < n) ? block_size : n;
    // No value-init on Q: the loop below writes every column block with beta = 0, so all m*n
    // entries are overwritten. Zeroing first costs a full m*n pass for nothing, which at the
    // large FEM2 cell (m ~ 3.0e5, n ~ 3.3e4) is about 80 GB of stores per call.
    T* Q       = new T[m * n];
    T* E_block = new T[n * b]();   // identity column-block scratch

    // Materialize Q = A * R^{-1} one column block at a time:
    //   E_block = I[:, j:j+b];  Q[:, j:j+b] = A_op * E_block.
    for (int64_t j0 = 0; j0 < n; j0 += b) {
        int64_t bk = std::min(b, n - j0);
        // E_block is value-initialized to zero above, and each iteration sets only the bk
        // entries E_block[(j0+j) + j*n]. Clearing just those after the apply keeps the block
        // zero for the next iteration, where re-zeroing all n*b entries every time costs
        // n^2 stores per call (about 8.7 GB at n = 33024).
        for (int64_t j = 0; j < bk; ++j)
            E_block[(j0 + j) + j * n] = (T)1.0;
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, bk, n, (T)1.0, E_block, n, (T)0.0, Q + j0 * m, m);
        for (int64_t j = 0; j < bk; ++j)
            E_block[(j0 + j) + j * n] = (T)0.0;
    }
    delete[] E_block;

    // Q := A * R^{-1}  (TRSM amplifies error by kappa(R) only.)
    blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
               blas::Op::NoTrans, blas::Diag::NonUnit, m, n, (T)1.0, R, n, Q, m);

    // G = Q^T Q (upper triangle), formed once and reused for both orth and cond.
    T* G = new T[n * n]();
    blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, m, (T)1.0, Q, m, (T)0.0, G, n);
    delete[] Q;

    T orth;
    {
        // ||G - I||_F, computed by shifting G's diagonal in place rather than copying G.
        // The copy was an n x n temporary (about 8.7 GB at n = 33024) to change n entries.
        // Only the diagonal is saved and restored, so G is bit-identical afterwards for the
        // syevd below: the restore writes back the saved values rather than adding 1 back,
        // which would not round-trip exactly once a diagonal entry is far from 1.
        T* diag_save = new T[n];
        for (int64_t j = 0; j < n; ++j) {
            diag_save[j] = G[j + j * n];
            G[j + j * n] -= (T)1.0;
        }
        orth = lapack::lansy(lapack::Norm::Fro, blas::Uplo::Upper, n, G, n) / std::sqrt((T)n);
        for (int64_t j = 0; j < n; ++j) G[j + j * n] = diag_save[j];
        delete[] diag_save;
    }

    // cond(A R^{-1}) from the eigenvalues of the Gram we already formed above.
    if (cond_out) {
        *cond_out = no_cond_sentinel;
        if (cond_cap <= 0 || n <= cond_cap) {
            T* evals = new T[n];
            int64_t info = lapack::syevd(lapack::Job::NoVec, blas::Uplo::Upper, n, G, n, evals);
            if (info == 0 && evals[0] > 0)
                *cond_out = std::sqrt(evals[n - 1] / evals[0]);   // syevd returns ascending
            delete[] evals;
        }
    }
    delete[] G;
    return orth;
}

// ---------------------------------------------------------------------------
// Least-squares backward error: Karlson-Walden estimate, sketched form
// (Epperly, Meier, Nakatsukasa 2024, Fact 4.1 and eq. 4.2).
//
// For a computed solution x of min ||b - A y||, the backward error with
// perturbation weight theta in (0, inf] is
//
//   BE_theta(x) = min ||[dA, theta*db]||_F  s.t.  x = argmin ||(b+db) - (A+dA) y||,
//
// estimated within a factor sqrt(2) (their Fact 4.1) by
//
//   BEhat_theta(x) = theta / sqrt(1 + theta^2 ||x||^2)
//                    * || (A^T A + nu^2 I)^{-1/2} A^T (b - A x) ||,
//   nu^2 = theta^2 ||b - A x||^2 / (1 + theta^2 ||x||^2).
//
// With A^T A replaced by (SA)^T (SA) for a sparse sign sketch S with d rows
// this is the sketched estimate (their eq. 4.2): from the SVD SA = U Sigma V^T,
//   || (Sigma^2 + nu^2 I)^{-1/2} V^T A^T (b - A x) ||,
// accurate to a small constant factor for d >= 2n (their Prop. 4.2 and the
// remark after it). Two weights are reported, both relative to ||A||_F, the
// scale on which "backward stable" means "a small multiple of u":
//   theta = ||A||_F / ||b||   (what their Algorithm 4 tests at run time), and
//   theta = inf               (perturb A only; their Definition 1.1).
//
// The reference (sketch + SVD) is built once per problem, AFTER every timed
// row, so no row's timing or peak-RSS window contains it: n operator
// applications to form SA plus one d x n SVD. ||A||_F is accumulated exactly
// from the operator's columns while SA is formed. The estimate needs the small
// singular directions accurately (they carry the largest weights
// 1/(sigma_i^2 + nu^2)), which is why this is an SVD of the sketch and not an
// eigendecomposition of its Gram matrix.
// Nonzeros per column of the reference sketch. Pinned at the paper's zeta = 8
// (EMN24 Section 4.2) and deliberately NOT the benchmark's --sketch-nnz knob:
// the oracle's estimate must not move when a campaign sweeps the solver's
// sketch density.
inline constexpr int64_t kKWSketchNNZ = 8;

template <typename T>
struct KWBackwardErrorRef {
    int64_t n = 0, d = 0, nnz = 0;
    T A_fro = 0;              // ||A||_F, exact, from the operator's columns
    std::vector<T> sigma;     // singular values of SA, descending
    std::vector<T> VT;        // n x n col-major; row i = i-th right singular vector
};

template <typename T, typename RNG, typename GLO>
static KWBackwardErrorRef<T> build_kw_reference(GLO& A_op, int64_t m, int64_t n, int64_t d,
                                                int64_t sketch_nnz, RandBLAS::RNGState<RNG> state,
                                                int64_t block_size) {
    randlapack_require(d >= n) << "build_kw_reference: d=" << d << " must be >= n=" << n;
    KWBackwardErrorRef<T> ref;
    ref.n = n; ref.d = d; ref.nnz = sketch_nnz;
    RandBLAS::SparseDist D(d, m, sketch_nnz, RandBLAS::Axis::Short);
    RandBLAS::SparseSkOp<T> S(D, state);
    RandBLAS::fill_sparse(S);

    int64_t b = (block_size > 0 && block_size < n) ? block_size : n;
    std::vector<T> SA((size_t)d * n, (T)0);
    std::vector<T> E((size_t)n * b, (T)0), Y((size_t)m * b, (T)0);
    T fro_sq = 0;
    for (int64_t j0 = 0; j0 < n; j0 += b) {
        int64_t bk = std::min(b, n - j0);
        std::fill(E.begin(), E.end(), (T)0);
        for (int64_t j = 0; j < bk; ++j) E[(size_t)(j0 + j) + (size_t)j * n] = (T)1;
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, bk, n, (T)1, E.data(), n, (T)0, Y.data(), m);
        for (int64_t j = 0; j < bk; ++j) {
            T c = blas::nrm2(m, Y.data() + (size_t)j * m, 1);
            fro_sq += c * c;
        }
        // The paper's embedding is S = zeta^{-1/2} [s_1 ... s_m] with +/-1 entries.
        // RandBLAS samples the +/-1 entries unscaled and exposes the zeta^{-1/2}
        // factor as dist.isometry_scale; without it every singular value of SA is
        // sqrt(zeta) too large and the estimate is understated by up to that factor
        // exactly in the converged regime (nu^2 -> 0), i.e. at the stop test.
        RandBLAS::sketch_general(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                 d, bk, m, (T)S.dist.isometry_scale, S, 0, 0, Y.data(), m,
                                 (T)0, SA.data() + (size_t)j0 * d, d);
    }
    ref.A_fro = std::sqrt(fro_sq);
    ref.sigma.assign(n, (T)0);
    ref.VT.assign((size_t)n * n, (T)0);
    // Divide-and-conquer SVD (gesdd, not gesvd: the bidiagonal QR phase of
    // gesvd with vectors is essentially sequential and would take hours at
    // n = 33024). jobz = OverwriteVec with d >= n: the left vectors overwrite
    // SA in place (U is not referenced) and the n rows of V^T land in VT.
    int64_t info = lapack::gesdd(lapack::Job::OverwriteVec, d, n, SA.data(), d,
                                 ref.sigma.data(), nullptr, 1, ref.VT.data(), n);
    randlapack_require(info == 0) << "build_kw_reference: gesdd returned info=" << info;
    return ref;
}

// Evaluate the estimate for one solution. ATr = A^T (b - A x) (length n),
// r_norm = ||b - A x||, x_norm = ||x||, b_norm = ||b||. be_theta_rel and
// be_inf_rel are relative to ||A||_F; theta_out echoes ||A||_F/||b||; res_orth
// is the residual-orthogonality ratio ||A^T r|| / (||A||_F ||r||) (their
// Corollary 1.2), recorded alongside. -1 marks an undefined value.
template <typename T>
static void kw_backward_error(const KWBackwardErrorRef<T>& ref, const T* ATr,
                              T r_norm, T x_norm, T b_norm,
                              T& be_theta_rel, T& be_inf_rel, T& theta_out, T& res_orth) {
    const int64_t n = ref.n;
    std::vector<T> z((size_t)n, (T)0);
    blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, n, n, (T)1, ref.VT.data(), n,
               ATr, 1, (T)0, z.data(), 1);
    auto estimate = [&](T nu_sq, T prefactor) {
        T acc = 0;
        for (int64_t i = 0; i < n; ++i)
            acc += z[i] * z[i] / (ref.sigma[i] * ref.sigma[i] + nu_sq);
        return prefactor * std::sqrt(acc);
    };
    T theta = (b_norm > 0) ? ref.A_fro / b_norm : (T)0;
    T den   = (T)1 + theta * theta * x_norm * x_norm;
    T be_theta = estimate(theta * theta * r_norm * r_norm / den, theta / std::sqrt(den));
    T be_inf   = (x_norm > 0) ? estimate(r_norm * r_norm / (x_norm * x_norm), (T)1 / x_norm) : (T)-1;
    be_theta_rel = (ref.A_fro > 0) ? be_theta / ref.A_fro : (T)-1;
    be_inf_rel   = (ref.A_fro > 0 && be_inf >= 0) ? be_inf / ref.A_fro : (T)-1;
    // A NaN (0/0 from a zero singular value of a rank-deficient sketch meeting a
    // zero component, or a diverged iterate) is reported as +inf: unmistakable in
    // the sidecar and never below any target. -1 stays the "undefined" sentinel.
    if (std::isnan(be_theta_rel)) be_theta_rel = std::numeric_limits<T>::infinity();
    if (std::isnan(be_inf_rel))   be_inf_rel   = std::numeric_limits<T>::infinity();
    theta_out    = theta;
    T ATr_norm   = blas::nrm2(n, ATr, 1);
    res_orth     = (ref.A_fro > 0 && r_norm > 0) ? ATr_norm / (ref.A_fro * r_norm) : (T)-1;
}

// Sidecar CSV (one row per successful (algorithm, run)); the writer prepends
// a '#' provenance line naming d, nnz, seed, ||A||_F, ||b||, theta.
inline const char* kKWCsvHeader =
    "algorithm,run,be_kw_theta,be_kw_inf,theta,r_norm,x_norm,res_orth\n";

template <typename T>
static std::string kw_provenance_line(const KWBackwardErrorRef<T>& ref, T b_norm, double build_s) {
    std::ostringstream h;
    h << "# sketched Karlson-Walden backward error (Epperly-Meier-Nakatsukasa 2024, eq. 4.2),"
      << " relative to ||A||_F; d=" << ref.d << " nnz=" << ref.nnz << " seed=20240914"
      << " ||A||_F=" << std::scientific << std::setprecision(6) << (double)ref.A_fro
      << " ||b||=" << (double)b_norm
      << " theta=||A||_F/||b||=" << (double)((b_norm > 0) ? ref.A_fro / b_norm : (T)0)
      << " reference_build_s=" << std::fixed << std::setprecision(1) << build_s << "\n";
    return h.str();
}

// The engine's convergence oracle built on the sketched Karlson-Walden reference: the
// paper's step-two termination test. The reference must outlive the returned function
// (the drivers hold it in a unique_ptr for the whole run). Returns be_theta relative to
// ||A||_F, the same number the sidecar's be_kw_theta column reports for the final x.
template <typename T>
static RandLAPACK::BackwardErrorOracle<T>
make_kw_oracle(const KWBackwardErrorRef<T>& ref, int64_t m, int64_t n, T b_norm) {
    return [&ref, m, n, b_norm](const T* x, const T* r, const T* ATr) -> T {
        T r_norm = blas::nrm2(m, r, 1);
        T x_norm = blas::nrm2(n, x, 1);
        T be_theta, be_inf, theta, res_orth;
        kw_backward_error<T>(ref, ATr, r_norm, x_norm, b_norm, be_theta, be_inf, theta, res_orth);
        // The estimator's -1 sentinel (||A||_F = 0) and a NaN from a diverged iterate
        // must never satisfy `be <= be_tol`: map both to +inf so the run continues
        // and the CSV shows the failure instead of a spurious convergence.
        if (!(be_theta >= (T)0)) return std::numeric_limits<T>::infinity();
        return be_theta;
    };
}

// be-tol-mult knob: <= 0 turns the oracle off (-1), otherwise the tolerance is
// mult * sqrt(n) * u with u the unit roundoff (epsilon / 2: 1.1e-16 in double).
// The estimate is Epperly-Meier-Nakatsukasa 2024's sketched Karlson-Walden backward
// error (arXiv:2406.03468, eq. 4.2), relative to ||A||_F; their Algorithm 4 stops it at
// 1u, their code at MATLAB eps. The sqrt(n) factor follows the square-system rule of
// Epperly-Greenbaum-Nakatsukasa 2025 (arXiv:2502.17767, Algorithm 5.1 line 20,
// berr <= n^{1/2} u on the Rigal-Gaches error), transplanted here to the least-squares
// estimate as a dimension-aware constant; mult lets a campaign move it.
template <typename T>
static T resolve_be_tol(double mult, int64_t n) {
    if (mult <= 0.0) return (T)-1;
    const double u = 0.5 * (double)std::numeric_limits<T>::epsilon();
    return (T)(mult * std::sqrt((double)n) * u);
}

// ---------------------------------------------------------------------------
// Named command-line arguments.
//
// These benchmarks were driven by long positional argument lists (20 slots for the
// applications driver, 18 for the Toeplitz one). That interface lost a whole campaign: a value
// intended for one slot landed in the slot before it, the binary parsed it, echoed it, and ran
// seven cluster jobs that silently encoded a different experiment than the scripts meant. The
// mitigation at the time was an external script that re-checked argument order, which concedes
// the interface was the defect.
//
// Named flags remove that failure mode by construction: a value is bound to a name rather than
// a position, an unknown or malformed flag is a hard error instead of a silent shift, and an
// omitted flag takes a documented default. Accepted forms are --name=value and, for booleans,
// a bare --name.
// ---------------------------------------------------------------------------
class BenchArgs {
public:
    BenchArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) {
                throw std::runtime_error("expected a --name=value argument, got '" + a + "'");
            }
            a = a.substr(2);
            auto eq = a.find('=');
            if (eq == std::string::npos) kv_[a] = "";           // bare flag
            else                         kv_[a.substr(0, eq)] = a.substr(eq + 1);
            order_.push_back(eq == std::string::npos ? a : a.substr(0, eq));
        }
    }

    // True when the command line uses the named form at all. Lets a driver keep accepting the
    // old positional form for existing job scripts while steering new ones to flags.
    static bool looks_named(int argc, char** argv) {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]).rfind("--", 0) == 0) return true;
        return false;
    }

    bool has(const std::string& k) const { return kv_.count(k) != 0; }

    std::string str(const std::string& k, const std::string& dflt) const {
        auto it = kv_.find(k);
        return (it == kv_.end() || it->second.empty()) ? dflt : it->second;
    }
    std::string require_str(const std::string& k) const {
        auto it = kv_.find(k);
        if (it == kv_.end() || it->second.empty())
            throw std::runtime_error("missing required argument --" + k);
        return it->second;
    }
    double dbl(const std::string& k, double dflt) const {
        auto it = kv_.find(k);
        if (it == kv_.end() || it->second.empty()) return dflt;
        return parse<double>(k, it->second);
    }
    int64_t i64(const std::string& k, int64_t dflt) const {
        auto it = kv_.find(k);
        if (it == kv_.end() || it->second.empty()) return dflt;
        return (int64_t)parse<long long>(k, it->second);
    }
    // Bare --name, or --name=1/true/yes. Absent means false.
    bool boolean(const std::string& k, bool dflt = false) const {
        auto it = kv_.find(k);
        if (it == kv_.end()) return dflt;
        if (it->second.empty()) return true;
        const std::string& v = it->second;
        if (v == "1" || v == "true" || v == "yes") return true;
        if (v == "0" || v == "false" || v == "no") return false;
        throw std::runtime_error("--" + k + " expects a boolean, got '" + v + "'");
    }

    // A misspelled flag must not be silently ignored: that would reintroduce exactly the
    // "ran a different experiment than the script encodes" failure the named form exists to stop.
    void reject_unknown(std::initializer_list<const char*> known) const {
        for (const auto& name : order_) {
            bool ok = false;
            for (const char* k : known) if (name == k) { ok = true; break; }
            if (!ok) {
                std::string msg = "unknown argument --" + name + "\nknown arguments:";
                for (const char* k : known) msg += std::string(" --") + k;
                throw std::runtime_error(msg);
            }
        }
    }

private:
    template <typename T>
    static T parse(const std::string& k, const std::string& v) {
        try {
            size_t pos = 0;
            T out;
            if constexpr (std::is_same_v<T, double>) out = std::stod(v, &pos);
            else                                     out = (T)std::stoll(v, &pos);
            if (pos != v.size())
                throw std::invalid_argument("trailing characters");
            return out;
        } catch (const std::exception&) {
            throw std::runtime_error("--" + k + " expects a number, got '" + v + "'");
        }
    }
    std::map<std::string, std::string> kv_;
    std::vector<std::string> order_;
};

} // namespace bench
} // namespace RandLAPACK
