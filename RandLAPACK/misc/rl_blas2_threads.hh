#pragma once

// Blas2ThreadGuard: caps the thread count of small dense level-2 BLAS calls (the
// triangular solves against an n x n preconditioner) for the duration of a scope,
// restoring the caller's setting on exit.
//
// A single-right-hand-side triangular solve is memory-bound with a sequential dependency
// chain, so threading it pays a barrier per column while the work between barriers stays
// flat. Left unguarded it taxes only the preconditioned solvers, which apply the
// preconditioner every inner iteration, and can therefore invert the wall-clock ranking
// against the unpreconditioned baseline.
//
// Wrap only the level-2 solves: the operator applies around them (FFTs, sparse solves,
// GEMMs) are level-3-like and want every thread.
//
// MKL only. Other backends compile to a no-op deliberately, because the alternative
// (omp_set_num_threads) is global and would leak the cap into concurrent regions.

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


/// Size-dependent cap. Threading helps up to 8-16 threads and degrades past that, so this
/// seeks the peak rather than forcing serial execution; the optimum grows with n because
/// work per barrier does while barrier cost does not. Values come from a dtrsv sweep on the
/// benchmark node, recorded in the dev log rather than here since they are hardware-specific.
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


/// Cap for MKL's threaded FFT (DFTI). A single small transform stops improving past about 16
/// threads and becomes intermittently unstable above that (occasional transforms cost ~100x
/// their typical time), which a solver converging in a handful of iterations cannot average
/// out. Batching via DFTI_NUMBER_OF_TRANSFORMS is the remedy for the multi-column build path;
/// the CG loop produces one right-hand side at a time and keeps this cap, narrowed further by
/// SolveWidthScope while a solver runs. Sweep data is in the dev log, not here.
///
/// ext_toeplitz_linop.hh also sets DFTI_THREAD_LIMIT at commit time and the two compose, but
/// do not rely on either alone to make a run reproducible: with the compute-time cap above the
/// ambient width, FFT width has been observed to vary between otherwise identical runs. Pin
/// RANDLAPACK_FFT_THREADS explicitly for bit-comparable results.
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
                // Never widen: mkl_set_num_threads_local sets the thread-local count
                // outright, so passing the calibrated cap when the caller has fewer threads
                // available RAISES the width instead of capping it. That oversubscribes a
                // restricted allocation and makes OMP_NUM_THREADS=1 not mean 1, which breaks
                // reproducibility runs. Taking the minimum keeps this a cap in both directions.
                const int avail = mkl_get_max_threads();
                const int want  = (avail > 0 && avail < cap) ? avail : cap;
                // Returns the PREVIOUS thread-local value, where 0 means "no local setting,
                // follow the global one"; restoring it also restores that follow-global state
                // rather than pinning the global count.
                prev_ = mkl_set_num_threads_local(want);
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
