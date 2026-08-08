#pragma once

// Blas2ThreadGuard -- caps the thread count of small dense LEVEL-2 BLAS calls
// (the triangular solves against an n x n preconditioner) for the duration of a
// scope, restoring the caller's setting on exit.
//
// WHY THIS EXISTS (measured 2026-08-07).
//
// A triangular solve with a single right-hand side is O(n^2) work on O(n^2) data:
// memory-bound, and with a sequential dependency chain. Threaded implementations
// pay a barrier per column (or per block), so barrier cost grows linearly in n
// while the useful work between barriers stays flat. Measured on a dense n = 2000
// upper-triangular factor:
//
//     threads                1          4         16
//     MKL dtrsv          0.448 ms   0.162 ms   31.3 ms
//
// Three findings pin the diagnosis:
//
//   * It is NOT about triangularity, so blocking does not fix it. A plain dgemv of
//     the same size degrades the same way (0.234 ms -> 7.04 ms at 16 threads), and a
//     hand-blocked solve (sequential diagonal block + threaded gemv update) still
//     measured 15.1 ms at 16 threads, ~100x the 4-thread cost.
//   * It does NOT improve with size, so there is no crossover to wait for. At 16
//     threads dtrsv costs 46 / 84 / 162 ms for n = 2000 / 4000 / 8000, against
//     0.16 / 1.15 / 4.16 ms at 4 threads. The 16-thread cost grows linearly in n,
//     which is the signature of a per-column barrier rather than of arithmetic.
//   * At 4 threads all variants sit at the memory-bandwidth floor (~16 MB of the
//     factor read per solve), so there is nothing left for an algorithm to win.
//
// CONSEQUENCE IF LEFT UNGUARDED. The preconditioned least-squares solvers apply the
// preconditioner twice per inner iteration; the unpreconditioned baseline applies it
// zero times. The overhead therefore taxes exactly the methods that converge in few
// iterations, and inverts the wall-clock ranking: on a 16000 x 2000 prolate-Toeplitz
// case the CQRRT solve measured 1531 ms against the unpreconditioned 106 ms, and
// re-running the identical binary with OMP_NUM_THREADS=4 moved CQRRT to 10.3 ms
// (149x) while the unpreconditioned baseline barely moved (106 -> 85 ms).
//
// SCOPE. Wrap ONLY the level-2 solves. The operator applies around them (FFTs,
// sparse solves, GEMMs) are level-3-like and genuinely want every thread, so a guard
// spanning a whole apply would trade one pathology for another.
//
// PORTABILITY. MKL is the only BLAS we can cap through a documented per-thread API,
// so the guard compiles to a no-op elsewhere. That is deliberate: other vendors are
// not known to thread trsv this way, and a global fallback (omp_set_num_threads)
// would leak the cap into concurrently running regions.
//
// TUNING. The cap defaults to kDefaultBlas2Threads and is overridable at runtime via
// the RANDLAPACK_BLAS2_THREADS environment variable (read once), so a new machine can
// be calibrated without a rebuild. A value <= 0 disables the guard entirely.

#include <cstdlib>

#if defined(RandBLAS_HAS_MKL)
// mkl_service.h ONLY, never the umbrella <mkl.h>: the latter redeclares the LAPACK
// entry points with MKL's own integer width, which conflicts with the declarations
// LAPACK++ already provides (measured: dozens of "conflicting declaration of C
// function" errors on ILP64 builds). The service header carries the threading
// controls and no BLAS/LAPACK prototypes.
#include <mkl_service.h>
#endif


namespace RandLAPACK {


/// Size-dependent cap, calibrated on the benchmark hardware (Xeon Gold 6430, 64
/// cores, 2026-08-07; dtrsv, milliseconds):
///
///     threads          1        4        8       16       32       64
///     n =  2000     0.562    0.408   *0.392*   0.774    0.909    0.887
///     n =  8256    18.147    6.232    4.915  * 4.735*   5.828    7.130
///     n = 20000   108.209   45.121   27.110  *19.276*  19.564   24.354
///
/// Threading genuinely helps up to 8-16 threads and degrades past that, so the
/// cap is a peak-seeker, not a "run it serially" switch. The optimum moves with
/// n because the parallel work per barrier grows with n while barrier cost does
/// not, hence the two-tier rule below.
///
/// NOTE ON MAGNITUDE. On this hardware the penalty for leaving it unguarded (64
/// threads) is a moderate 1.5-2.3x. A 16-thread WSL2 desktop showed a 100x
/// collapse instead, because 16 OpenMP threads there oversubscribe 8 physical
/// cores; do not quote desktop numbers as if they were cluster numbers.
constexpr int kBlas2ThreadsSmall = 8;    ///< n <= kBlas2SmallDim
constexpr int kBlas2ThreadsLarge = 16;   ///< n >  kBlas2SmallDim
constexpr int64_t kBlas2SmallDim = 4000;


/// Thread cap for a level-2 solve on an n x n factor. RANDLAPACK_BLAS2_THREADS
/// overrides the calibrated values (read once); a value <= 0 disables the guard.
inline int blas2_thread_cap(int64_t n) {
    static const int override_cap = []() -> int {
        const char* s = std::getenv("RANDLAPACK_BLAS2_THREADS");
        if (s == nullptr || *s == '\0') return -1;   // -1 = "no override"
        return std::atoi(s);
    }();
    if (override_cap >= 0) return override_cap;
    return (n <= kBlas2SmallDim) ? kBlas2ThreadsSmall : kBlas2ThreadsLarge;
}


/// RAII cap on the calling thread's BLAS thread count. Construct in the narrowest
/// scope containing a level-2 solve; the previous setting is restored on destruction
/// (including on an exception unwinding through the scope).
class Blas2ThreadGuard {
    public:
        /// @param n  dimension of the triangular factor being solved against;
        ///           selects the calibrated cap (see blas2_thread_cap).
        explicit Blas2ThreadGuard(int64_t n) {
        #if defined(RandBLAS_HAS_MKL)
            int cap = blas2_thread_cap(n);
            if (cap > 0) {
                // mkl_set_num_threads_local returns the PREVIOUS thread-local value,
                // where 0 means "no local setting, follow the global one". Restoring
                // that value in the destructor therefore also restores the
                // follow-the-global state, rather than pinning the global count.
                prev_ = mkl_set_num_threads_local(cap);
                active_ = true;
            }
        #endif
        }

        ~Blas2ThreadGuard() {
        #if defined(RandBLAS_HAS_MKL)
            if (active_) mkl_set_num_threads_local(prev_);
        #endif
        }

        Blas2ThreadGuard(const Blas2ThreadGuard&) = delete;
        Blas2ThreadGuard& operator=(const Blas2ThreadGuard&) = delete;
        Blas2ThreadGuard(Blas2ThreadGuard&&) = delete;
        Blas2ThreadGuard& operator=(Blas2ThreadGuard&&) = delete;

    private:
    #if defined(RandBLAS_HAS_MKL)
        int  prev_   = 0;
    #endif
        bool active_ = false;
};


} // namespace RandLAPACK
