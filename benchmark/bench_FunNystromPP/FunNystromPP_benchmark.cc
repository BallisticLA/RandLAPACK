// CLI driver for the v2 funNyström++ comparison harness. Phase 1 (Phases 0–2
// of the v2 plan) was the exact-dense f(A) oracle only; Phase 5 adds the
// scalar and block Lanczos-FA options so the benchmark can mirror the
// production Phase 2 setup (block LFA at d=200) for cross-validation.
//
// Loads A.bin, Omega1.bin, Omega2.bin (column-major doubles, format per
// RandLAPACK::testing::save_dense_bin / matlab `save_dense_bin.m`). Runs the
// driver end-to-end with the requested Phase-2 oracle.
//
// Usage:
//   FunNystromPP_benchmark A.bin Omega1.bin Omega2.bin func q [poly_lambda] [lfa_type] [d] [sketch_type] [vec_nnz] [seed] [timing]
//
//   func           sqrt | log | poly | effdim | square | identity   (scalar f)
//   q              subspace-iteration count (e.g. 2)
//   poly_lambda    λ in f(x) = x(x + λ) (func=poly) or x/(x + λ) (func=effdim); default 10
//   lfa_type       exact | scalar | block        (default exact)
//                    exact: dense V·diag(f(λ))·Vᵀ via syevd
//                    scalar: per-column scalar Lanczos-FA at depth d
//                    block:  block Lanczos-FA at depth d (Chen 2024 §9)
//   d              Lanczos depth (default 200 for scalar, 20 for block)
//   sketch_type    IGNORED (accepted for CLI compatibility). The Phase-1
//                    sketch is now always a kernel-internal SASO drawn inside
//                    NystromEVD; Omega1.bin is read only for its (n, k) header
//                    (its data is never loaded). Ω₂ is unaffected (always
//                    loaded from disk). CSV reports sketch_type=saso.
//   vec_nnz        non-zeros per ROW of the n×k SASO (default 8; 0 = auto,
//                    resolved to ~log(k) inside NystromEVD)
//   seed           RNG seed for the Phase-1 sketch (default 42)
//   timing         0 | 1   (default 0). When 1, suppress the syevd true-trace
//                    oracle (which would dominate wall-clock at n=2000) and report
//                    Phase 1 + Phase 2 driver wall-clock in ms; true_tr and err
//                    are then NaN — EXCEPT with lfa_type=exact, whose oracle
//                    needs the eigendecomposition anyway, so true_tr and err
//                    are computed and reported even in timing mode.
//
// Stdout: one CSV row:
//   t1,t2,est,true_tr,err,lfa_type,d,sketch_type,vec_nnz,n,k,t_driver_ms,t_phase1_ms,t_phase2_ms,t_specrec_ms
// where true_tr is the dense-syevd oracle (NaN in timing mode unless
// lfa_type=exact; see `timing` above).
// t_specrec_ms is the wall-clock of just the shifted spectral-recovery block
// inside NystromEVD (Alg. 2 lines 3-8); the sketch, subspace-iteration, and
// matvec costs that precede it contribute to t_phase1_ms instead.

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
        "[poly_lambda] [lfa_type] [d] [sketch_type] [vec_nnz] [seed] [timing]\n"
        "  func           sqrt | log | poly | effdim | square | identity\n"
        "  q              subspace-iter count (e.g. 2)\n"
        "  poly_lambda    λ in x(x+λ) when func=poly (default 10)\n"
        "  lfa_type       exact | scalar | block  (default exact)\n"
        "  d              Lanczos depth (default 200 scalar, 20 block)\n"
        "  sketch_type    IGNORED (Phase-1 sketch is always a kernel-internal SASO;\n"
        "                 Omega1.bin supplies only the (n, k) header)\n"
        "  vec_nnz        non-zeros per row of the SASO (default 8; 0 = auto ~log(k))\n"
        "  seed           RNG seed for the Phase-1 sketch (default 42)\n"
        "  timing         0 | 1 (default 0). 1 = skip syevd oracle, report driver ms\n"
        "                 (true_tr/err NaN, except lfa_type=exact which reports them)\n"
        "Output (stdout): t1,t2,est,true_tr,err,lfa_type,d,sketch_type,vec_nnz,n,k,t_driver_ms,t_phase1_ms,t_phase2_ms,t_specrec_ms\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 13) { print_usage(argv[0]); return 1; }
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

    int64_t n_A = 0, n2_A = 0, n_O1 = 0, k = 0, n_O2 = 0, s = 0;

    // Size each buffer from its file header (no fixed over-allocation): peek
    // the (rows, cols) header first, validate, then allocate exactly and load.
    try {
        RandLAPACK::testing::peek_dense_bin_dims(A_path,  n_A,  n2_A);
        RandLAPACK::testing::peek_dense_bin_dims(O1_path, n_O1, k);
        RandLAPACK::testing::peek_dense_bin_dims(O2_path, n_O2, s);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "header read error: %s\n", e.what());
        return 2;
    }
    if (n_A != n2_A || n_O1 != n_A || n_O2 != n_A) {
        std::fprintf(stderr, "dimension mismatch: A=%ldx%ld O1=%ldx%ld O2=%ldx%ld\n",
                     (long)n_A, (long)n2_A, (long)n_O1, (long)k, (long)n_O2, (long)s);
        return 3;
    }
    const int64_t n = n_A;

    // Raw buffers per house rule (no std::vector for matrix/vector data).
    T *A_buf  = new T[n * n];
    T *O2_buf = new T[n * s];
    try {
        // Omega1.bin is peeked above for (n, k) only; its data is unused —
        // the Phase-1 sketch is drawn inside NystromEVD.
        RandLAPACK::testing::load_dense_bin<T>(A_path,  n_A,  n2_A, A_buf,  n * n);
        RandLAPACK::testing::load_dense_bin<T>(O2_path, n_O2, s,    O2_buf, n * s);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "load error: %s\n", e.what());
        return 2;
    }

    if (argc >= 10 && sketch_str != "saso")
        std::fprintf(stderr,
            "note: sketch_type '%s' ignored — the Phase-1 sketch is always a "
            "kernel-internal SASO now.\n", sketch_str.c_str());

    // Mirror upper triangle into lower unconditionally: the kernel's sparse
    // first A-application goes through right_spmm, which reads A as generic
    // dense (symmetry not exploited; ~2× off optimal until upstream RandBLAS
    // gains a `sparse_symm_spmm`).
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            A_buf[i + j * n] = A_buf[j + i * n];

    // Scalar function f.
    std::function<T(T)> fscalar;
    if      (fstr == "sqrt")     fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    else if (fstr == "log")      fscalar = [](T x) { return std::log(x + (T)1); }; // log(x+1) = tr(log(A+I))
    else if (fstr == "poly")     fscalar = [poly_lambda](T x) { return x * (x + poly_lambda); };
    else if (fstr == "effdim")   fscalar = [poly_lambda](T x) { return x / (x + poly_lambda); };
    else if (fstr == "square")   fscalar = [](T x) { return x * x; };
    else if (fstr == "identity") fscalar = [](T x) { return x; };
    else { std::fprintf(stderr, "unknown func '%s'\n", fstr.c_str()); return 4; }

    // True trace via syevd of a copy.
    // In timing mode we skip syevd (dominates wall-clock at n=2000) — the
    // estimate vs ground-truth is verified separately by cross-validation.
    // The exact-LFA path also needs the eigendecomp, so it is forced off
    // in timing mode (use scalar or block LFA instead).
    T true_tr = std::nan("0");
    T *A_cpy = nullptr;   // eigenvectors of A after syevd (raw; owned here)
    T *ev    = nullptr;   // eigenvalues
    if (!timing_mode || lfa_str == "exact") {
        A_cpy = new T[n * n];
        ev    = new T[n];
        std::copy(A_buf, A_buf + n * n, A_cpy);
        lapack::syevd(lapack::Job::Vec, lapack::Uplo::Upper, n, A_cpy, n, ev);
        true_tr = 0;
        for (int64_t i = 0; i < n; ++i) true_tr += fscalar(ev[i]);
    }

    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A_buf, n, Layout::ColMajor);

    // Three oracle types share a single std::function signature so the
    // driver doesn't need to be retemplated. The exact path captures the
    // eigenvectors V and f(λ); the Lanczos-FA paths construct a long-lived
    // LFA instance and dispatch through its .call().
    using FAFun = std::function<void(int64_t, int64_t, const T*, T*)>;
    FAFun fAfun;
    RandLAPACK::LanczosFA<T>      scalar_lfa;
    RandLAPACK::BlockLanczosFA<T> block_lfa;

    if (lfa_str == "exact") {
        // V·diag(f(λ))·Vᵀ via the eigendecomp we already computed for true_tr.
        // Shared with the test + MEX through RandLAPACK::testing. The testing
        // util takes std::vector by value (it moves ownership into the
        // closure); construct them from the raw buffers at the boundary.
        std::vector<T> V(A_cpy, A_cpy + n * n);
        std::vector<T> f_lambda(n);
        for (int64_t i = 0; i < n; ++i) f_lambda[i] = fscalar(ev[i]);
        fAfun = RandLAPACK::testing::make_exact_fa_oracle_from_eig<T>(
                    n, std::move(V), std::move(f_lambda));
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

    RandLAPACK::FunNystromPP<T> driver;
    driver.vec_nnz = vec_nnz;
    T t1 = 0, t2 = 0;
    auto t_start = std::chrono::steady_clock::now();
    using RNG = r123::Philox4x32;
    RandBLAS::RNGState<RNG> state((uint32_t)saso_seed);
    T est = driver.call(A_op, fAfun, fscalar, k, s, q,
                        state, O2_buf, t1, t2);
    auto t_end = std::chrono::steady_clock::now();
    double t_driver_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    T err = std::isnan(true_tr) ? std::nan("0") : std::abs(est - true_tr) / std::abs(true_tr);

    std::printf("%.17e,%.17e,%.17e,%.17e,%.6e,%s,%ld,%s,%ld,%ld,%ld,%.3f,%.3f,%.3f,%.3f\n",
                t1, t2, est, true_tr, err, lfa_str.c_str(), (long)d,
                "saso", (long)vec_nnz,
                (long)n, (long)k, t_driver_ms,
                driver.t_phase1_ms, driver.t_phase2_ms, driver.t_specrec_ms);

    delete[] A_buf;
    delete[] O2_buf;
    delete[] A_cpy;
    delete[] ev;
    return 0;
}
