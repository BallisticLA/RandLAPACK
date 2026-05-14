// CLI driver for the v2 funNyström++ comparison harness. Phase 1 (Phases 0–2
// of the v2 plan) was the exact-dense f(A) oracle only; Phase 5 adds the
// scalar and block Lanczos-FA options so the benchmark can mirror the
// production Phase 2 setup (block LFA at d=200) for cross-validation.
//
// Loads A.bin, Omega1.bin, Omega2.bin (column-major doubles, format per
// RandLAPACK::util::save_dense_bin / matlab `save_dense_bin.m`). Runs the
// v2 driver end-to-end with the requested Phase-2 oracle.
//
// Usage:
//   FunNystromPP_v2_benchmark A.bin Omega1.bin Omega2.bin func q [poly_lambda] [lfa_type] [d] [sketch_type] [vec_nnz] [seed] [timing] [force_fallback]
//
//   func           sqrt | log | poly | square | identity   (scalar f)
//   q              subspace-iteration count (e.g. 2)
//   poly_lambda    λ in f(x) = x(x + λ) when func=poly (default 10)
//   lfa_type       exact | scalar | block        (default exact)
//                    exact: dense V·diag(f(λ))·Vᵀ via syevd
//                    scalar: per-column scalar Lanczos-FA at depth d
//                    block:  block Lanczos-FA at depth d (Chen 2024 §9)
//   d              Lanczos depth (default 200 for scalar, 20 for block)
//   sketch_type    gaussian | saso              (default gaussian — use loaded Ω₁.bin)
//                    saso: replace loaded Ω₁ with a SparseDist SASO sketch of
//                          the same shape, materialized to dense; loaded Ω₁ ignored.
//                          Ω₂ is unaffected (always loaded from disk).
//   vec_nnz        non-zeros per column for SASO (default 8); ignored if sketch_type=gaussian
//   seed           RNG seed for the SASO sketch; only used when sketch_type=saso
//                    (default 42)
//   timing         0 | 1   (default 0). When 1, suppress the syevd true-trace
//                    oracle (which would dominate wall-clock at n=2000) and report
//                    Phase 1 + Phase 2 driver wall-clock in ms.
//   force_fallback 0 | 1   (default 0). When 1, sets driver.force_fallback so
//                    NystromEVD_v2 skips Cholesky-fast and takes the SVD-pinv path.
//                    A/B benchmarks of Phase 7a's HMT §5.1 trick.
//
// Stdout: one CSV row:
//   t1,t2,est,true_tr,err,lfa_type,d,sketch_type,vec_nnz,n,k,t_driver_ms,t_phase1_ms,t_phase2_ms,t_specrec_ms,force_fallback
// where true_tr is the dense-syevd oracle (suppressed in timing mode; reported as NaN).
// t_specrec_ms is the wall-clock of just the dual-path spectral-recovery
// block inside NystromEVD_v2 — the tight A/B measurement for Phase 7a.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace linops = RandLAPACK::linops;

static void print_usage(const char *prog) {
    std::fprintf(stderr,
        "Usage: %s A.bin Omega1.bin Omega2.bin func q "
        "[poly_lambda] [lfa_type] [d] [sketch_type] [vec_nnz] [seed] [timing] [force_fallback]\n"
        "  func           sqrt | log | poly | square | identity\n"
        "  q              subspace-iter count (e.g. 2)\n"
        "  poly_lambda    λ in x(x+λ) when func=poly (default 10)\n"
        "  lfa_type       exact | scalar | block  (default exact)\n"
        "  d              Lanczos depth (default 200 scalar, 20 block)\n"
        "  sketch_type    gaussian | saso  (default gaussian — use loaded Ω₁)\n"
        "  vec_nnz        non-zeros per column for SASO (default 8)\n"
        "  seed           RNG seed for SASO generation (default 42)\n"
        "  timing         0 | 1 (default 0). 1 = skip syevd oracle, report driver ms\n"
        "  force_fallback 0 | 1 (default 0). 1 = skip Cholesky-fast in NystromEVD_v2\n"
        "Output (stdout): t1,t2,est,true_tr,err,lfa_type,d,sketch_type,vec_nnz,n,k,t_driver_ms,t_phase1_ms,t_phase2_ms,t_specrec_ms,force_fallback\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 14) { print_usage(argv[0]); return 1; }
    using T = double;

    const std::string A_path     = argv[1];
    const std::string O1_path    = argv[2];
    const std::string O2_path    = argv[3];
    const std::string fstr       = argv[4];
    const int64_t     q          = std::strtoll(argv[5], nullptr, 10);
    const T poly_lambda          = (argc >= 7)  ? std::strtod(argv[6],  nullptr) : (T)10;
    const std::string lfa_str    = (argc >= 8)  ? argv[7] : "exact";
    const int64_t d_default      = (lfa_str == "block") ? 20 : 200;
    const int64_t d              = (argc >= 9)  ? std::strtoll(argv[8],  nullptr, 10) : d_default;
    const std::string sketch_str = (argc >= 10) ? argv[9] : "gaussian";
    const int64_t vec_nnz        = (argc >= 11) ? std::strtoll(argv[10], nullptr, 10) : 8;
    const uint64_t saso_seed     = (argc >= 12) ? std::strtoull(argv[11], nullptr, 10) : 42;
    const bool     timing_mode   = (argc >= 13) ? (std::strtoll(argv[12], nullptr, 10) != 0) : false;
    const bool     force_fallback= (argc >= 14) ? (std::strtoll(argv[13], nullptr, 10) != 0) : false;

    int64_t n_A = 0, n2_A = 0, n_O1 = 0, k = 0, n_O2 = 0, s = 0;

    constexpr int64_t CAP = (int64_t)1 << 26;   // 64M doubles per buffer ≈ 512 MB total
    std::vector<T> A_buf(CAP), O1_buf(CAP), O2_buf(CAP);

    try {
        RandLAPACK::util::load_dense_bin<T>(A_path,  n_A,  n2_A, A_buf.data(),  CAP);
        RandLAPACK::util::load_dense_bin<T>(O1_path, n_O1, k,    O1_buf.data(), CAP);
        RandLAPACK::util::load_dense_bin<T>(O2_path, n_O2, s,    O2_buf.data(), CAP);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "load error: %s\n", e.what());
        return 2;
    }
    if (n_A != n2_A || n_O1 != n_A || n_O2 != n_A) {
        std::fprintf(stderr, "dimension mismatch: A=%ldx%ld O1=%ldx%ld O2=%ldx%ld\n",
                     (long)n_A, (long)n2_A, (long)n_O1, (long)k, (long)n_O2, (long)s);
        return 3;
    }
    const int64_t n = n_A;
    A_buf.resize(n * n);
    O1_buf.resize(n * k);
    O2_buf.resize(n * s);

    // Phase 6 + Gap 5: SASO sketch handling is now driver-resident. This
    // block only handles benchmark-side bookkeeping:
    //   - For SASO + q >= 2: symmetrize A in place (right_spmm doesn't
    //     exploit symmetry); the SparseSkOp itself is sampled and passed
    //     directly to the FunNystromPP_v2::call SkOp overload at the
    //     dispatch site below.
    //   - For SASO + q == 1: densify the sketch into O1_buf and fall back
    //     to the dense overload (v2 always does an initial QR + final
    //     matvec at q == 1, so the SkOp path can't amortize).
    //   - For "gaussian": O1_buf has already been loaded from disk.
    //
    // The SkOp path is ~2× off optimal until upstream RandBLAS gains a
    // `sparse_symm_spmm` — `right_spmm` reads both triangles independently.
    if (sketch_str == "saso") {
        if (q >= 2) {
            // Mirror upper triangle into lower so right_spmm sees a full
            // symmetric matrix (it reads A as generic dense).
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = j + 1; i < n; ++i)
                    A_buf[i + j * n] = A_buf[j + i * n];
        } else {
            // q == 1: densify SASO into O1_buf so the dense overload handles it.
            using RNG = r123::Philox4x32;
            RandBLAS::RNGState<RNG> state((uint32_t)saso_seed);
            auto S = RandBLAS::SparseDist(n, k, vec_nnz).sample<T, RNG, int64_t>(state);
            RandBLAS::fill_sparse(S);
            auto Scoo = RandBLAS::coo_view_of_skop(S);
            std::fill(O1_buf.begin(), O1_buf.end(), (T)0);
            RandLAPACK::util::sparse_to_dense(Scoo, Layout::ColMajor, O1_buf.data());
        }
    } else if (sketch_str != "gaussian") {
        std::fprintf(stderr, "unknown sketch_type '%s'\n", sketch_str.c_str());
        return 6;
    }

    // Scalar function f.
    std::function<T(T)> fscalar;
    if      (fstr == "sqrt")     fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    else if (fstr == "log")      fscalar = [](T x) { return std::log(x); };
    else if (fstr == "poly")     fscalar = [poly_lambda](T x) { return x * (x + poly_lambda); };
    else if (fstr == "square")   fscalar = [](T x) { return x * x; };
    else if (fstr == "identity") fscalar = [](T x) { return x; };
    else { std::fprintf(stderr, "unknown func '%s'\n", fstr.c_str()); return 4; }

    // True trace via syevd of a copy.
    // In timing mode we skip syevd (dominates wall-clock at n=2000) — the
    // estimate vs ground-truth is verified separately by cross-validation.
    // The exact-LFA path also needs the eigendecomp, so it is forced off
    // in timing mode (use scalar or block LFA instead).
    T true_tr = std::nan("0");
    std::vector<T> A_cpy;
    std::vector<T> ev;
    if (!timing_mode || lfa_str == "exact") {
        A_cpy = A_buf;
        ev.assign(n, (T)0);
        lapack::syevd(lapack::Job::Vec, lapack::Uplo::Upper, n,
                      A_cpy.data(), n, ev.data());
        true_tr = 0;
        for (int64_t i = 0; i < n; ++i) true_tr += fscalar(ev[i]);
    }

    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A_buf.data(), n, Layout::ColMajor);

    // Three oracle types share a single std::function signature so the
    // driver doesn't need to be retemplated. The exact path captures the
    // eigenvectors V and f(λ); the Lanczos-FA paths construct a long-lived
    // LFA instance and dispatch through its .call().
    using FAFun = std::function<void(int64_t, int64_t, const T*, T*)>;
    FAFun fAfun;
    RandLAPACK::LanczosFA<T>      scalar_lfa;
    RandLAPACK::BlockLanczosFA<T> block_lfa;

    if (lfa_str == "exact") {
        // V·diag(f(λ))·Vᵀ via the same eigendecomp we already computed.
        std::vector<T> V = std::move(A_cpy);
        std::vector<T> f_lambda(n);
        for (int64_t i = 0; i < n; ++i) f_lambda[i] = fscalar(ev[i]);
        fAfun = [n, V = std::move(V), f_lambda = std::move(f_lambda)]
                (int64_t m_, int64_t s_, const T *B, T *Y) {
            std::vector<T> tmp1((int64_t)n * s_);
            blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                       n, s_, m_, (T)1, V.data(), n, B, m_, (T)0, tmp1.data(), n);
            for (int64_t j = 0; j < s_; ++j)
                for (int64_t i = 0; i < n; ++i)
                    tmp1[i + j * n] *= f_lambda[i];
            blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                       m_, s_, n, (T)1, V.data(), m_, tmp1.data(), n, (T)0, Y, m_);
        };
    } else if (lfa_str == "scalar") {
        // Capture references by stable handle. fscalar / A_op live for the
        // remainder of main, so capturing by reference is safe.
        fAfun = [&scalar_lfa, &A_op, &fscalar, d](int64_t m_, int64_t s_, const T *B, T *Y) {
            scalar_lfa.call(A_op, B, m_, s_, fscalar, d, Y);
        };
    } else if (lfa_str == "block") {
        fAfun = [&block_lfa, &A_op, &fscalar, d](int64_t m_, int64_t s_, const T *B, T *Y) {
            block_lfa.call(A_op, B, m_, s_, fscalar, d, Y);
        };
    } else {
        std::fprintf(stderr, "unknown lfa_type '%s'\n", lfa_str.c_str());
        return 5;
    }

    RandLAPACK::FunNystromPP_v2<T> driver;
    driver.force_fallback = force_fallback;
    T t1 = 0, t2 = 0;
    auto t_start = std::chrono::steady_clock::now();
    T est;
    if (sketch_str == "saso" && q >= 2) {
        // Sparse-sketch path: hand the SparseSkOp directly to the driver's
        // SkOp overload, which routes the first matvec through right_spmm.
        using RNG = r123::Philox4x32;
        RandBLAS::RNGState<RNG> state((uint32_t)saso_seed);
        auto S = RandBLAS::SparseDist(n, k, vec_nnz).sample<T, RNG, int64_t>(state);
        RandBLAS::fill_sparse(S);
        est = driver.call(A_op, fAfun, fscalar, k, s, q,
                          S, O2_buf.data(), t1, t2);
    } else {
        // Dense path: O1_buf holds Gaussian Ω₁ (loaded) or densified SASO (q < 2).
        est = driver.call(A_op, fAfun, fscalar, k, s, q,
                          O1_buf.data(), O2_buf.data(), t1, t2);
    }
    auto t_end = std::chrono::steady_clock::now();
    double t_driver_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    T err = std::isnan(true_tr) ? std::nan("0") : std::abs(est - true_tr) / std::abs(true_tr);

    std::printf("%.17e,%.17e,%.17e,%.17e,%.6e,%s,%ld,%s,%ld,%ld,%ld,%.3f,%.3f,%.3f,%.3f,%d\n",
                t1, t2, est, true_tr, err, lfa_str.c_str(), (long)d,
                sketch_str.c_str(), (long)vec_nnz,
                (long)n, (long)k, t_driver_ms,
                driver.t_phase1_ms, driver.t_phase2_ms, driver.t_specrec_ms,
                force_fallback ? 1 : 0);
    return 0;
}
