#pragma once

// Blas2ThreadGuard: caps the thread count of small dense LEVEL-2 BLAS calls
// (the triangular solves against an n x n preconditioner) for the duration of a
// scope, restoring the caller's setting on exit.
//
// WHY THIS EXISTS.
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
// This is not specific to triangularity (a plain dgemv of the same size degrades
// the same way) and does not improve with problem size (there is no crossover to
// wait for), so it is a fixed cost of threading a memory-bound, sequentially
// dependent kernel, not something blocking or a larger problem can fix.
//
// CONSEQUENCE IF LEFT UNGUARDED. The preconditioned least-squares solvers apply the
// preconditioner every inner iteration; the unpreconditioned baseline does not. The
// overhead therefore taxes exactly the methods that converge in few iterations, and
// can invert the wall-clock ranking against the unpreconditioned baseline by two
// orders of magnitude.
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
/// cores; dtrsv, milliseconds):
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


/// Cap for MKL's threaded FFT (DFTI). Measured on the benchmark node (Xeon Gold
/// 6430, 64 cores) for one forward+backward apply of the Toeplitz operator,
/// milliseconds:
///
///     threads          1      8     16     32     64
///     L =  32768    0.274  0.355  0.267  0.297  0.319
///     L = 131072    3.29   1.24   1.00   0.93   0.97
///     L = 524288   15.52   4.95   3.66   3.05   3.13
///
/// The mean barely improves past 16 threads, and at 64 the call becomes
/// INTERMITTENTLY unstable (individual transforms occasionally take 100x
/// their typical cost), which a solver that converges in a handful of
/// iterations cannot average out. Capping the transform is the
/// vendor-recommended remedy for single small transforms (Intel advises
/// reducing the thread count rather than expecting a lone FFT to scale).
/// Batching via DFTI_NUMBER_OF_TRANSFORMS is the remedy for the multi-column
/// build path (sketch and Gram applies); the CG loop still produces one
/// right-hand side at a time and keeps this cap, further narrowed by
/// SolveWidthScope while a solver is running.
///
/// MKL decides an FFT descriptor's threading at commit time in general, but a
/// cap applied only around DftiComputeForward/Backward (via Blas2ThreadGuard
/// below) was measured to still bind FFT width on this hardware even for a
/// descriptor committed at ambient width. ext_toeplitz_linop.hh also sets
/// DFTI_THREAD_LIMIT at commit time; the two caps compose, and the narrower
/// compute-time cap wins.
constexpr int kDefaultFFTThreads = 16;


/// Thread cap for an FFT apply. RANDLAPACK_FFT_THREADS overrides (read once);
/// <= 0 disables the guard.
inline int fft_thread_cap() {
    static const int cap = []() -> int {
        const char* s = std::getenv("RANDLAPACK_FFT_THREADS");
        if (s == nullptr || *s == '\0') return kDefaultFFTThreads;
        return std::atoi(s);
    }();
    return cap;
}


/// Solve-scoped width matching. Alternating OpenMP team widths cost the wider
/// region ~300 us per re-formation on the benchmark node (libgomp, dual-socket
/// Gold 6430). Inside an iterative solve the trsv runs at blas2_thread_cap(n)
/// ACTUAL width, and widths can only be equalized DOWNWARD: MKL does not form
/// wide teams for small trsv, so raising the trsv request does not remove the
/// alternation. The scope below therefore narrows every width-capped kernel
/// that consults it (currently the Toeplitz FFT apply) to the trsv width for
/// the duration of a solver call, so the inner loop runs at ONE width.
/// Build-phase applies see no active scope and keep their own calibrated caps.
/// RANDLAPACK_SOLVE_FFT_MATCH=0 disables the matching (read once; for A/B probes).
inline bool solve_width_match_enabled() {
    static const bool on = []() {
        const char* s = std::getenv("RANDLAPACK_SOLVE_FFT_MATCH");
        return !(s != nullptr && s[0] == '0' && s[1] == '\0');
    }();
    return on;
}

/// The active solve-context width for the calling thread; 0 = no active scope.
inline int& solve_context_width_ref() {
    thread_local int width = 0;
    return width;
}
inline int solve_context_width() { return solve_context_width_ref(); }

/// RAII solve-width context. Instantiated by the iterative solvers for their
/// whole duration; width-capped kernels take min(own cap, context width) while
/// one is active. Nesting restores the enclosing context on destruction.
class SolveWidthScope {
    public:
        /// @param n  preconditioner dimension; the context width is the trsv cap
        ///           blas2_thread_cap(n), the narrow width the loop already pays.
        explicit SolveWidthScope(int64_t n) {
            const int cap = blas2_thread_cap(n);
            if (solve_width_match_enabled() && cap > 0) {
                prev_ = solve_context_width_ref();
                solve_context_width_ref() = cap;
                active_ = true;
            }
        }
        ~SolveWidthScope() {
            if (active_) solve_context_width_ref() = prev_;
        }
        SolveWidthScope(const SolveWidthScope&) = delete;
        SolveWidthScope& operator=(const SolveWidthScope&) = delete;
        SolveWidthScope(SolveWidthScope&&) = delete;
        SolveWidthScope& operator=(SolveWidthScope&&) = delete;
    private:
        int  prev_   = 0;
        bool active_ = false;
};


/// RAII cap on the calling thread's MKL thread count. Construct in the narrowest
/// scope containing the guarded call; the previous setting is restored on
/// destruction (including when an exception unwinds through the scope).
class Blas2ThreadGuard {
    public:
        /// @param n  dimension of the triangular factor being solved against;
        ///           selects the calibrated cap (see blas2_thread_cap).
        explicit Blas2ThreadGuard(int64_t n) : Blas2ThreadGuard(blas2_thread_cap(n), 0) {}

        /// Explicit-cap form, for callers with their own calibration (e.g. the FFT
        /// operator). The dummy second parameter disambiguates from the int64_t
        /// dimension overload.
        Blas2ThreadGuard(int cap, int /*tag*/) {
        #if defined(RandBLAS_HAS_MKL)
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
