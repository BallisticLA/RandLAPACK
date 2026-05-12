// Phase 1 driver for the cross-validation harness.
//
// Loads A.bin, Omega1.bin, Omega2.bin (column-major doubles, format per
// RandLAPACK::util::save_dense_bin / matlab `save_dense_bin.m`). Runs
// the v2 funNyström++ end-to-end. Phase 2's f(A) oracle is an exact
// dense sqrtm-style operator built from a syevd of A, so that any
// discrepancy vs the MATLAB reference at the same RNG is attributable
// to the driver itself, not to Krylov truncation.
//
// Usage:
//   FunNystromPP_v2_benchmark A.bin Omega1.bin Omega2.bin func q [poly_lambda]
// where func is one of {sqrt, log, poly, square, identity} and q is the
// subspace-iteration count (e.g. 2). For func=poly, the optional 6th arg
// sets λ in f(x)=x(x+λ) (default 10, matching setup_matrix.m).
//
// Stdout: a single CSV row
//   t1,t2,est,true_tr,err
// where true_tr is computed from a separate syevd of A and err is the
// relative error of the estimate.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
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
        "Usage: %s A.bin Omega1.bin Omega2.bin func q\n"
        "  A.bin       — n×n SPD matrix (column-major doubles)\n"
        "  Omega1.bin  — n×k Phase 1 sketch\n"
        "  Omega2.bin  — n×s Phase 2 Hutchinson Ω\n"
        "  func        — sqrt | log | square | identity\n"
        "  q           — subspace-iteration count (e.g. 2)\n"
        "Output (stdout): t1,t2,est,true_tr,err\n", prog);
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) { print_usage(argv[0]); return 1; }
    using T = double;

    const std::string A_path  = argv[1];
    const std::string O1_path = argv[2];
    const std::string O2_path = argv[3];
    const std::string fstr    = argv[4];
    const int64_t     q       = std::strtoll(argv[5], nullptr, 10);
    const T poly_lambda       = (argc == 7) ? std::strtod(argv[6], nullptr) : (T)10;

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

    // Pick the scalar function.
    std::function<T(T)> fscalar;
    if      (fstr == "sqrt")     fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    else if (fstr == "log")      fscalar = [](T x) { return std::log(x); };
    else if (fstr == "poly")     fscalar = [poly_lambda](T x) { return x * (x + poly_lambda); };
    else if (fstr == "square")   fscalar = [](T x) { return x * x; };
    else if (fstr == "identity") fscalar = [](T x) { return x; };
    else {
        std::fprintf(stderr, "unknown func '%s'\n", fstr.c_str());
        return 4;
    }

    // True trace via syevd of a copy.
    std::vector<T> ev(n);
    {
        std::vector<T> A_cpy = A_buf;
        lapack::syevd(lapack::Job::Vec, lapack::Uplo::Upper, n,
                      A_cpy.data(), n, ev.data());
        // For exact f(A)·B oracle below, keep eigenvectors in A_cpy.
        // For true_tr, just sum f(λᵢ).

        // Build the exact fAfun oracle. We capture A_cpy (eigenvectors) and
        // ev by value via std::function.
        std::vector<T> V = std::move(A_cpy);
        std::vector<T> f_lambda(n);
        for (int64_t i = 0; i < n; ++i) f_lambda[i] = fscalar(ev[i]);

        auto fAfun = [n, V = std::move(V), f_lambda = std::move(f_lambda)]
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

        T true_tr = 0;
        for (int64_t i = 0; i < n; ++i) true_tr += fscalar(ev[i]);

        // Run the driver.
        linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A_buf.data(), n, Layout::ColMajor);
        RandLAPACK::FunNystromPP_v2<T> driver;
        T t1 = 0, t2 = 0;
        T est = driver.call(A_op, fAfun, fscalar, k, s, q,
                            O1_buf.data(), O2_buf.data(), t1, t2);
        T err = std::abs(est - true_tr) / std::abs(true_tr);

        // CSV row.
        std::printf("%.17e,%.17e,%.17e,%.17e,%.6e\n", t1, t2, est, true_tr, err);
    }
    return 0;
}
