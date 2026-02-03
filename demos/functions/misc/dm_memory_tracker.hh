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

namespace RandLAPACK_demos {

// Read current Resident Set Size (RSS) in KB from /proc/self/status.
// Returns -1 on failure (e.g., non-Linux platforms).
static inline long get_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            long rss = 0;
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &rss);
            return rss;
        }
    }
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

// CQRRT_linops: A_hat(d*n) + tau(n) + R_sk_inv(n*n) + A_pre(m*b_eff)
template <typename T>
static inline long cqrrt_linops_analytical_kb(int64_t m, int64_t n, double d_factor, int64_t block_size) {
    int64_t d = static_cast<int64_t>(std::ceil(d_factor * n));
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bytes = static_cast<long>(sizeof(T)) * (d * n + n + (long)n * n + (long)m * b_eff);
    return bytes / 1024;
}

// CholQR_linops: I_mat(n*n) + A_temp(m*b_eff)
template <typename T>
static inline long cholqr_linops_analytical_kb(int64_t m, int64_t n, int64_t block_size = 0) {
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    long bytes = static_cast<long>(sizeof(T)) * ((long)n * n + (long)m * b_eff);
    return bytes / 1024;
}

// sCholQR3_linops: I_mat(n*n) + Q_buf(m*n) + G(n*n) + R_temp(n*n)
// NOTE: block_size parameter accepted for API consistency, but does NOT reduce
// peak memory because Q_buf(m*n) is required for iterations 2 and 3.
// Blocking only reduces memory during iteration 1's Gram computation phase.
template <typename T>
static inline long scholqr3_linops_analytical_kb(int64_t m, int64_t n, int64_t block_size = 0) {
    (void)block_size;  // Unused - peak memory unchanged by blocking
    long bytes = static_cast<long>(sizeof(T)) * ((long)m * n + 3L * n * n);
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

} // namespace RandLAPACK_demos
