#pragma once

// Thread-count caps for the memory-bound single-vector triangular solves inside iterative
// solvers, which MKL would otherwise spread over every core (the operator applies around them
// keep every thread). No-ops without MKL: OpenBLAS runs these solves on one thread, and
// Accelerate has no per-call thread control.

#include <RandBLAS/config.h>

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <optional>

#if defined(RandBLAS_HAS_MKL)
#include <blas.hh>
// Match MKL's integer width to BLAS++ before any MKL header fixes it (as RandBLAS does).
#if defined(BLAS_ILP64) && !defined(MKL_ILP64)
#define MKL_ILP64
#endif
// Not <mkl.h>: its LAPACK prototypes conflict with LAPACK++'s on ILP64 builds; the service
// header has the threading controls and no BLAS/LAPACK prototypes.
#include <mkl_service.h>
#endif


namespace RandLAPACK {


/// Calibrated level-2 caps (MKL, 64-core node): threading a dtrsv helps up to 8-16 threads and
/// degrades past that, with the optimum growing with n.
inline constexpr int kBlas2ThreadsSmall = 8;    ///< n <= kBlas2SmallDim
inline constexpr int kBlas2ThreadsLarge = 16;   ///< n >  kBlas2SmallDim
inline constexpr int64_t kBlas2SmallDim = 4000;


/// Integer value of environment variable `name` (not cached), or nullopt when it is
/// absent, empty, not an integer, or out of int range.
inline std::optional<int> thread_cap_from_env(const char* name) {
#if defined(_MSC_VER)
#pragma warning(suppress: 4996)   // getenv is safe here: the result is read immediately
#endif
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') return std::nullopt;
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || *end != '\0' || v < INT_MIN || v > INT_MAX) return std::nullopt;
    return static_cast<int>(v);
}


/// Thread cap for a level-2 solve on an n x n factor. RANDLAPACK_BLAS2_THREADS overrides
/// the calibrated values (read once); a value <= 0 disables the cap.
inline int blas2_thread_cap(int64_t n) {
    static const std::optional<int> override_cap = thread_cap_from_env("RANDLAPACK_BLAS2_THREADS");
    if (override_cap) return *override_cap;
    return (n <= kBlas2SmallDim) ? kBlas2ThreadsSmall : kBlas2ThreadsLarge;
}


namespace detail {
/// Whether SolveWidthScope takes effect; RANDLAPACK_SOLVE_WIDTH_MATCH=0 turns it off (read once).
inline bool solve_width_match_enabled() {
    static const bool on = thread_cap_from_env("RANDLAPACK_SOLVE_WIDTH_MATCH") != 0;
    return on;
}

inline int& solve_context_width_ref() {
    thread_local int width = 0;
    return width;
}
} // namespace detail

/// The calling thread's active solve-context width; 0 when no SolveWidthScope is active.
inline int solve_context_width() { return detail::solve_context_width_ref(); }


/// Publishes blas2_thread_cap(n) as the calling thread's solve width while a solver runs, so
/// width-aware kernels (such as an FFT-based operator) avoid a thread-team re-formation per
/// width switch. Apply it through ThreadCapGuard (never widens); nested scopes restore.
class SolveWidthScope {
    public:
        explicit SolveWidthScope(int64_t n) {
            const int cap = blas2_thread_cap(n);
            if (detail::solve_width_match_enabled() && cap > 0) {
                prev_ = detail::solve_context_width_ref();
                detail::solve_context_width_ref() = cap;
                active_ = true;
            }
        }
        ~SolveWidthScope() {
            if (active_) detail::solve_context_width_ref() = prev_;
        }
        SolveWidthScope(const SolveWidthScope&) = delete;
        SolveWidthScope& operator=(const SolveWidthScope&) = delete;
    private:
        int  prev_   = 0;
        bool active_ = false;
};


/// RAII cap on the calling thread's MKL thread count, restored on destruction (including
/// during exception unwinding). Never widens; a cap <= 0 is a no-op.
class ThreadCapGuard {
    public:
        explicit ThreadCapGuard(int cap) {
        #if defined(RandBLAS_HAS_MKL)
            if (cap > 0 && mkl_get_max_threads() > cap) {
                // Returns the previous thread-local value; 0 means "follow the global
                // setting", and restoring it restores that state.
                prev_ = mkl_set_num_threads_local(cap);
                active_ = true;
            }
        #else
            (void) cap;
        #endif
        }

        ~ThreadCapGuard() {
        #if defined(RandBLAS_HAS_MKL)
            if (active_) mkl_set_num_threads_local(prev_);
        #endif
        }

        ThreadCapGuard(const ThreadCapGuard&) = delete;
        ThreadCapGuard& operator=(const ThreadCapGuard&) = delete;

    private:
    #if defined(RandBLAS_HAS_MKL)
        int  prev_   = 0;
        bool active_ = false;
    #endif
};


/// ThreadCapGuard at the calibrated level-2 cap for a solve against an n x n factor.
class Blas2ThreadGuard : public ThreadCapGuard {
    public:
        explicit Blas2ThreadGuard(int64_t n) : ThreadCapGuard(blas2_thread_cap(n)) {}
};


} // namespace RandLAPACK
