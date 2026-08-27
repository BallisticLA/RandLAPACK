#pragma once

// Memory tracking utilities for benchmarking:
// - Peak RSS sampling via background thread
// - Analytical peak working memory computation for each algorithm

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
    void start() {
        // Release freed heap back to the OS before taking the baseline
        // (2026-08-05). RSS is process-cumulative: glibc keeps freed arenas
        // mapped, so without this the FIRST tracked algorithm in a benchmark
        // absorbs the whole process ramp-up (its delta over-reports) while
        // every later one reuses already-faulted pages (delta ~0 -- the
        // "peak_rss_kb=4" effect diagnosed 2026-07-29, and the inflated
        // CQRRT_linop storage bars in the Toeplitz figures). Trimming resets
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
// ---------------------------------------------------------------------------

// CQRRT_linops (TRSM_IDENTITY / GEQP3).
// Peak during the blocked Gram + Cholesky moment (recounted 2026-08-27 against
// the post-2a2577e allocation lifetimes; the old formula understated by
// n*n + n*b_eff, about 545 MB at n = 8256 double):
//   A_hat(d*n) + tau(n) + P(n*n) + R_pre(n*n) + G(n*n)
//   + G_backup(n*n, allocated inside cholqr_primitive whenever retries are
//     enabled, which is the driver default) + A_temp(m*b_eff) + Z_buf(n*b_eff)
// = d*n + n + 4*n*n + (m + n)*b_eff.
template <typename T>
static inline long cqrrt_linops_analytical_kb(int64_t m, int64_t n, double d_factor, int64_t block_size) {
    int64_t d = static_cast<int64_t>(std::ceil(d_factor * n));
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bytes = static_cast<long>(sizeof(T)) * (d * n + n + 4L * n * n + (long)(m + n) * b_eff);
    return bytes / 1024;
}

// CQRRT_linops (BQRRP): the execution has two distinct peak-memory moments:
//   (1) BQRRP-preconditioner moment (the column-pivoted-QR inversion inside
//       cholqr_primitive, rl_cholqr.hh): A_hat(d*n) + R_sk_inv(n*n) + G(n*n)
//       + P_copy(n*n) + R_buf(n*n) = d*n + 4*n*n.  A_pre is NOT allocated yet here.
//       (W was eliminated by transposing Q^T in place.)
//   (2) Gram-loop moment (same as the non-BQRRP path):
//         A_hat(d*n) + R_sk_inv(n*n) + tau(n) + A_pre(m*b_eff)
//       = d*n + n + n*n + m*b_eff.  P_copy/R_buf have been freed by now.
// The true analytical peak is the max of (1) and (2).  Roughly: moment (1) wins for
// short-and-wide matrices (m <~ 3n); moment (2) wins for tall matrices.
template <typename T>
static inline long cqrrt_linops_bqrrp_analytical_kb(int64_t m, int64_t n, double d_factor, int64_t block_size) {
    int64_t d = static_cast<int64_t>(std::ceil(d_factor * n));
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bqrrp_peak = static_cast<long>(sizeof(T)) * ((long)d * n + 4L * n * n);
    long gram_peak  = static_cast<long>(sizeof(T)) * ((long)d * n + n + (long)n * n + (long)m * b_eff);
    return std::max(bqrrp_peak, gram_peak) / 1024;
}

// CholQR_linops (post-2026-06-05 rework with adaptive-shift retries enabled by default):
//   cholqr_primitive owns G(n*n) + G_backup(n*n) + A_temp(m*b_eff)
//   plus blocked_preconditioned_gram's I_block(n*b_eff) inside the Gram loop.
// Peak = 2*n*n + (m+n)*b_eff.
template <typename T>
static inline long cholqr_linops_analytical_kb(int64_t m, int64_t n, int64_t block_size = 0) {
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bytes = static_cast<long>(sizeof(T)) * (2L * n * n + (long)(m + n) * b_eff);
    return bytes / 1024;
}

// sCholQR3_linops (recounted 2026-08-27 against the post-2a2577e lifetimes; the
// old formula overstated by n*n + m*b_eff by assuming iter-1's primitive
// transients coexist with the iters-2/3 scratch, which the refactor sequenced:
// iter 1's primitive frees its own G/G_backup/A_temp BEFORE the iters-2/3
// scratch is allocated).
// Peak moment (iters 2/3): driver persistent G(n*n) + R_pre(n*n) + P_prev(n*n)
//   + primitive G_backup(n*n) + A_temp(m*b_eff) + Z_buf(n*b_eff)
// = 4*n*n + (m+n)*b_eff, which also bounds the iter-1 moment.
template <typename T>
static inline long scholqr3_linops_analytical_kb(int64_t m, int64_t n, int64_t block_size = 0) {
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bytes = static_cast<long>(sizeof(T)) * (4L * n * n + (long)(m + n) * b_eff);
    return bytes / 1024;
}

// sCholQR3_linops_basic (recounted 2026-08-27; same sequencing argument as the
// blocked variant): b_eff = n collapse of 4*n*n + (m+n)*b_eff gives
//   4*n*n + (m + n)*n = 5*n*n + m*n.
template <typename T>
static inline long scholqr3_linops_basic_analytical_kb(int64_t m, int64_t n) {
    long bytes = static_cast<long>(sizeof(T)) * (5L * n * n + (long)m * n);
    return bytes / 1024;
}

// Dense CQRRT (materialize + rl_cqrrt):
// Peak = A_materialized(m*n) + A_hat(d*n) + tau(n)
// (I_mat freed before rl_cqrrt allocates A_hat, so they don't overlap)
template <typename T>
static inline long dense_cqrrt_analytical_kb(int64_t m, int64_t n, double d_factor) {
    int64_t d = static_cast<int64_t>(std::ceil(d_factor * n));
    long bytes = static_cast<long>(sizeof(T)) * ((long)m * n + (long)d * n + n);
    return bytes / 1024;
}

} // namespace RandLAPACK
