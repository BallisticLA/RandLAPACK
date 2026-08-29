#pragma once

// Memory tracking utilities for benchmarking:
// - Peak RSS sampling via background thread
// - Analytical peak working memory computation for each algorithm

#include "rl_exceptions.hh"

#include <fstream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <algorithm>
#if defined(__GLIBC__) || defined(__linux__)
#include <malloc.h>   // malloc_trim
#endif

namespace RandLAPACK {

// Read current Resident Set Size (RSS) in KB.
// Uses /proc/self/status on Linux; returns -1 on unsupported platforms.
static inline long get_rss_kb() {
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            long rss = 0;
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &rss);
            return rss;
        }
    }
#endif
    return -1;
}

// Tracks peak RSS during an algorithm execution via a sampling background thread.
// Usage:
//   PeakRSSTracker tracker;
//   tracker.start();
//   algorithm.call(...);
//   long peak_increase_kb = tracker.stop();
class PeakRSSTracker {
public:
    // Joins a still-running sampler thread instead of letting a joinable
    // std::thread destruct (which calls std::terminate). Reached when start()
    // was called but stop() was skipped, e.g. an exception thrown in between.
    ~PeakRSSTracker() {
        if (sampler_.joinable()) {
            running_.store(false, std::memory_order_relaxed);
            sampler_.join();
        }
    }

    void start() {
        // A second start() without an intervening stop() would assign into an
        // already-joinable sampler_ and call std::terminate; require stop() first.
        randlapack_require(!sampler_.joinable())
            << "PeakRSSTracker::start: sampler already running; call stop() first";

        // Release freed heap back to the OS before taking the baseline.
        // RSS is process-cumulative: glibc keeps freed arenas
        // mapped, so without this the FIRST tracked algorithm in a benchmark
        // absorbs the whole process ramp-up (its delta over-reports) while
        // every later one reuses already-faulted pages (delta ~0, the
        // "peak_rss_kb=4" effect). Trimming resets
        // the floor to live memory only, making per-method deltas comparable
        // regardless of execution order. Frees only unused arena space; no
        // effect on correctness or on MKL's internal buffers (a warmup pass
        // should absorb those).
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        baseline_kb_ = get_rss_kb();
        peak_kb_.store(baseline_kb_, std::memory_order_relaxed);
        running_.store(true, std::memory_order_relaxed);
        sampler_ = std::thread([this]() {
            while (running_.load(std::memory_order_relaxed)) {
                long current = get_rss_kb();
                long prev = peak_kb_.load(std::memory_order_relaxed);
                while (current > prev) {
                    if (peak_kb_.compare_exchange_weak(prev, current, std::memory_order_relaxed))
                        break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    // Stops the sampler thread and returns the peak RSS increase in KB.
    long stop() {
        running_.store(false, std::memory_order_relaxed);
        sampler_.join();
        // One final sample after join
        long final_rss = get_rss_kb();
        long prev = peak_kb_.load(std::memory_order_relaxed);
        if (final_rss > prev)
            peak_kb_.store(final_rss, std::memory_order_relaxed);
        return peak_kb_.load(std::memory_order_relaxed) - baseline_kb_;
    }

private:
    std::atomic<bool> running_{false};
    std::atomic<long> peak_kb_{0};
    long baseline_kb_{0};
    std::thread sampler_;
};

// ---------------------------------------------------------------------------
// Analytical peak working memory functions.
// These compute the peak memory from known buffer sizes in each algorithm,
// excluding test-mode Q-factor allocation (which is only for verification).
// All return memory in KB.
//
// SCOPE. These model the DRIVER's workspace only. They do NOT model:
//   * the operator's per-apply temporaries (CompositeOperator allocates a
//     fresh scratch buffer on every apply, once per nesting level; negligible
//     at small block sizes, but can dominate at b_eff = n).
//   * the operator's SKETCH path (VStackOp's sketch overload allocates
//     scratch proportional to (m + n) * b_blk; every sketching method pays
//     it, CholQR/CholQR2 do not sketch). This is deliberately NOT folded into
//     the formulas below: it is a property of the operator passed in, not of
//     the driver, and would be wrong for an operator whose sketch path
//     allocates differently.
//   * operator state allocated once and reused (e.g. the Toeplitz FFT plans
//     and batch buffer), which the benchmark warmup faults in before the
//     tracker baseline is taken, so it cancels out of the per-row delta.
// Consequence: peak-vs-predicted is a meaningful check for blocked methods on
// any operator, and for non-blocked methods only on operators whose applies
// allocate nothing (e.g. the matrix-free Toeplitz operator).
//
// PARAMETER CONVENTION: `m` is the ROW COUNT OF THE OPERATOR THE QR RUNS ON.
// For an augmented operator A_hat = [A; mu I] that is m + n, not m.
//
// `d` matches the drivers' truncating cast, not ceil.
// ---------------------------------------------------------------------------

// CQRRT_linops (TRSM_IDENTITY / GEQP3), PHASED allocation.
// The sketch phase and the Gram/Cholesky phase have disjoint working sets, and the
// driver now allocates each only while it is needed, so the peak is the MAX of the
// two moments rather than their sum:
//   sketch moment : A_hat(d*n) + tau(n) + P(n*n)
//   Gram moment   : P + R_pre + G (3*n*n) + A_temp(m*b_eff) + Z_buf(n*b_eff)
//                   + diag_backup(n), allocated inside cholqr_primitive whenever
//                   retries are enabled (the driver default: max_retries = -1).
// cholqr_primitive's shift-retry no longer keeps a full n x n Gram backup: a failed
// potrf attempt is undone from the Gram's own untouched strict lower triangle plus
// an O(n) diagonal snapshot (see rl_cholqr.hh), so the retry scratch is +n, not +n*n.
// For any d <= 3n the Gram moment dominates, which puts CQRRT at exactly the
// CholQR2 / sCholQR3 peak. Phasing (splitting the sketch and Gram allocations so
// they don't coexist) is still a real cut versus their sum, d*n + n + 3*n*n +
// (m+n)*b_eff + n, for any d > n.
template <typename T>
static inline long cqrrt_linops_analytical_kb(int64_t m, int64_t n, double d_factor, int64_t block_size) {
    int64_t d = static_cast<int64_t>(d_factor * n); if (d < n) d = n;   // matches the drivers
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long sketch_moment = static_cast<long>(sizeof(T)) * ((long)d * n + n + (long)n * n);
    long gram_moment   = static_cast<long>(sizeof(T)) * (3L * n * n + (long)(m + n) * b_eff + n);
    return std::max(sketch_moment, gram_moment) / 1024;
}

// CholQR_linops (adaptive-shift retries enabled by default):
//   cholqr_primitive owns G(n*n) + A_temp(m*b_eff) + diag_backup(n) (the O(n)
//   retry snapshot; allocated whenever retries are enabled, the driver default)
//   plus blocked_preconditioned_gram's I_block(n*b_eff) inside the Gram loop.
// Peak = n*n + (m+n)*b_eff + n.
template <typename T>
static inline long cholqr_linops_analytical_kb(int64_t m, int64_t n, int64_t block_size = 0) {
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bytes = static_cast<long>(sizeof(T)) * (1L * n * n + (long)(m + n) * b_eff + n);
    return bytes / 1024;
}

// CholQR2_linops: call() is a thin wrapper around cholqr_iterate(num_iters=2)
// (comps/rl_cholqr.hh), which owns the scratch directly (no per-class member
// buffers). Iter 1 allocates and frees its own G(n*n) + A_temp(m*b_eff) inside
// cholqr_primitive's unpreconditioned overload BEFORE the iter-2 scratch below
// is allocated, so the two moments are sequential, not coexistent; iter 1 is
// bounded by the iter-2 peak. Peak (iter 2, preconditioned):
//   cholqr_iterate's persistent G, R_pre, P_prev (3 n^2) + A_temp (m * b_eff)
//   + Z_buf (n * b_eff) + cholqr_primitive's diag_backup(n), the O(n) retry
//   snapshot allocated whenever retries are enabled (the driver default).
// See the SCOPE note at the top of this section.
template <typename T>
static inline long cholqr2_linops_analytical_kb(int64_t m, int64_t n, int64_t block_size) {
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bytes = static_cast<long>(sizeof(T)) *
        ( 3 * n * n        // G + R_pre + P_prev (iter-2 persistent scratch)
        + m * b_eff        // A_temp
        + n * b_eff        // Z_buf
        + n                // diag_backup (retry snapshot)
        );
    return bytes / 1024;
}

// sCholQR3_linops: call() is cholqr_iterate(num_iters=3) (comps/rl_cholqr.hh);
// same sequencing argument as CholQR2_linops above (iter 1's primitive frees its
// own G/A_temp BEFORE the iters-2/3 scratch is allocated, so iter 1 is bounded
// by the iters-2/3 peak).
// Peak moment (iters 2/3): driver persistent G(n*n) + R_pre(n*n) + P_prev(n*n)
//   + A_temp(m*b_eff) + Z_buf(n*b_eff)
//   + primitive diag_backup(n), the O(n) retry snapshot (rl_cholqr.hh: a failed
//     potrf attempt is restored from G's own untouched strict lower triangle
//     plus this diagonal snapshot, not a full n x n Gram backup).
// = 3*n*n + (m+n)*b_eff + n, which also bounds the iter-1 moment.
template <typename T>
static inline long scholqr3_linops_analytical_kb(int64_t m, int64_t n, int64_t block_size = 0) {
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bytes = static_cast<long>(sizeof(T)) * (3L * n * n + (long)(m + n) * b_eff + n);
    return bytes / 1024;
}

// sCholQR3_linops_basic: same sequencing argument as the blocked variant; b_eff = n
// collapse of 3*n*n + (m+n)*b_eff + n gives 3*n*n + (m+n)*n + n = 4*n*n + m*n + n.
// Validated against the formula above on the matrix-free Toeplitz operator:
// ratio 0.987 to 1.028 (n = 1000 and n = 4000), superseding the earlier
// validation against the pre-diag-backup 5*n*n + m*n formula. On an operator whose
// applies allocate (FEM2's nested CompositeOperator) this driver-only figure is
// NOT the process peak: b_eff = n makes each per-apply temporary inner_dim * n,
// and FEM2 large measured 255 GB against a 126 GB driver workspace. See the
// SCOPE note at the top of this section before quoting this number.
template <typename T>
static inline long scholqr3_linops_basic_analytical_kb(int64_t m, int64_t n) {
    long bytes = static_cast<long>(sizeof(T)) * (4L * n * n + (long)m * n + n);
    return bytes / 1024;
}

// Blendenpik_linops (sketch + Householder QR + LSQR): R is handed to the caller
// as R_out (ownership transfer, rl_blendenpik.hh: `R_out = R; R = nullptr;`) with
// no copy, so R_out never coexists with a separate R buffer: R_out just IS R.
// Buffers, all live simultaneously at the peak because call() holds the sketch
// (Ask, tau) allocated until cleanup() at the very end, well after R is done
// with it:
//   Ask(d*n) + tau(n) + R(n*n)
//   ws (warm_start || init_only) only : Sb(d) + x0(n) + r0(m)
//   LSQR workspace (with_lsqr only)   : u(m) + av(m) + v(n) + w(n) + atu(n)
//                                       + sc(n) = 2m + 4n
// init_only (with_lsqr = false, the refined rows' mode) forces ws = true inside
// call() regardless of the warm_start member: init_only implies ws by
// construction, so the x0-build buffers are always live in that mode. The
// `warm_start` parameter here therefore only gates the vectors when with_lsqr is
// true (the published Blendenpik/Blendenpik_cold rows, where ws genuinely
// depends on the member); when with_lsqr is false it is ignored and treated as
// true, so a caller that mislabels a cold init_only row still gets a faithful
// prediction. init_only rows then continue into restarted_pcg_ne, whose own
// workspace (9n + m) is smaller than the sketch term for any d >= 1, so the
// Blendenpik moment remains the peak.
template <typename T>
static inline long blendenpik_linops_analytical_kb(int64_t m, int64_t n, double d_factor,
                                                   bool warm_start = true, bool with_lsqr = true) {
    int64_t d = static_cast<int64_t>(d_factor * n); if (d < n) d = n;   // matches the driver
    bool ws = with_lsqr ? warm_start : true;   // init_only always forces ws = true
    long vecs = ws ? ((long)d + n + m) : 0L;   // Sb, x0, r0
    if (with_lsqr) vecs += 2L * m + 4L * n;    // u, av | v, w, atu, sc
    long bytes = static_cast<long>(sizeof(T)) * ((long)d * n + n + (long)n * n + vecs);
    return bytes / 1024;
}

// Dense CQRRT (materialize + rl_cqrrt):
// Peak = A_materialized(m*n) + A_hat(d*n) + tau(n) + gram_backup(n*n)
// (I_mat freed before rl_cqrrt allocates A_hat, so they don't overlap; A_hat
// stays live through the whole call, so it coexists with gram_backup).
// gram_backup is allocated whenever rl_cqrrt's max_retries != 0 (the driver
// default is -1, unbounded retries, so it is live in the common case);
// retries_enabled defaults to true to match that default and the current
// CQRRT_linop_basic.cc caller, which never overrides max_retries.
template <typename T>
static inline long dense_cqrrt_analytical_kb(int64_t m, int64_t n, double d_factor,
                                             bool retries_enabled = true) {
    int64_t d = static_cast<int64_t>(d_factor * n); if (d < n) d = n;   // matches the drivers
    long gram_backup = retries_enabled ? (long)n * n : 0L;
    long bytes = static_cast<long>(sizeof(T)) * ((long)m * n + (long)d * n + n + gram_backup);
    return bytes / 1024;
}

} // namespace RandLAPACK
