// cqrrt_bench_common.hh: shared utilities for CQRRT linop benchmarks
#pragma once

#include "RandLAPACK.hh"
#include "../../extras/misc/ext_util.hh"
#include <RandBLAS.hh>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
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

// Named exit conditions for the CSV: distinguishes "hit the LS floor honestly"
// from "ran out of budget", which shared a flag value before.
inline const char* pcg_stop_reason(int status) {
    switch (status) {
        case 0: return "tol";       case 1: return "budget";
        case 2: return "breakdown"; case 3: return "rounds";
        case 4: return "floor";     default: return "unknown";
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
inline const char* kRoundsCsvHeader =
    "algorithm,run,round,inner_iters,inner_status,inner_relres,best_relres,best_iter,ls_relres\n";

// Write one round record row (round_idx is 1-based, matching every caller).
template <typename T>
static void write_round_row(std::ostream& out, const std::string& alg, int64_t run_idx,
                            size_t round_idx, int iters, int status, T relres,
                            T best_relres, int best_iter, T ls_relres) {
    out << alg << "," << run_idx << "," << round_idx << ","
        << iters << "," << status << ","
        << std::scientific << std::setprecision(6) << relres << ","
        << best_relres << "," << best_iter << "," << ls_relres << "\n";
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
    T* Q       = new T[m * n]();
    T* E_block = new T[n * b]();   // identity column-block scratch

    // Materialize Q = A * R^{-1} one column block at a time:
    //   E_block = I[:, j:j+b];  Q[:, j:j+b] = A_op * E_block.
    for (int64_t j0 = 0; j0 < n; j0 += b) {
        int64_t bk = std::min(b, n - j0);
        std::fill(E_block, E_block + n * b, (T)0.0);
        for (int64_t j = 0; j < bk; ++j)
            E_block[(j0 + j) + j * n] = (T)1.0;
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, bk, n, (T)1.0, E_block, n, (T)0.0, Q + j0 * m, m);
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
        T* GmI = new T[n * n];
        std::copy(G, G + n * n, GmI);
        for (int64_t j = 0; j < n; ++j) GmI[j + j * n] -= (T)1.0;
        orth = lapack::lansy(lapack::Norm::Fro, blas::Uplo::Upper, n, GmI, n) / std::sqrt((T)n);
        delete[] GmI;
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
        RandBLAS::sketch_general(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                 d, bk, m, (T)1, S, 0, 0, Y.data(), m,
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

} // namespace bench
} // namespace RandLAPACK
