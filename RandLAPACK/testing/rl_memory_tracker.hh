#pragma once

// Memory tracking utilities for benchmarking:
// - Peak RSS over a window, from the kernel's peak mark (a sampling thread where it is unavailable)
// - Analytical peak working memory computation for each algorithm

#include "rl_exceptions.hh"
#include "rl_cholqr.hh"

#include <algorithm>
#include <fstream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim
#endif

namespace RandLAPACK {

// The value in KB of a "key:" line of /proc/self/status (VmRSS:, VmHWM:, ...), or -1 when it
// cannot be read (not Linux, or no such line).
static inline long read_status_kb(const char* key) {
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    const size_t len = std::char_traits<char>::length(key);
    while (std::getline(status, line)) {
        if (line.compare(0, len, key) == 0) {
            long kb = -1;
            std::sscanf(line.c_str() + len, "%ld", &kb);
            return kb;
        }
    }
#else
    (void) key;
#endif
    return -1;
}

// Read current Resident Set Size (RSS) in KB.
// Uses /proc/self/status on Linux; returns -1 on unsupported platforms.
static inline long get_rss_kb() { return read_status_kb("VmRSS:"); }

// Resets the kernel's peak-RSS mark (VmHWM) to the current RSS (Linux >= 4.0); false where
// /proc/self/clear_refs is missing or refuses the write.
static inline bool reset_peak_rss_mark() {
#ifdef __linux__
    std::ofstream f("/proc/self/clear_refs");
    if (!f) return false;
    f << '5';
    f.close();
    return !f.fail();
#else
    return false;
#endif
}

// Tracks the peak RSS increase over a window of execution. Where the kernel's peak mark can be
// reset (Linux), stop() reads it: exact, and no thread competes with the timed work. Elsewhere
// a background thread samples the RSS every 100 us. The mark is process-wide, so one tracker at
// a time holds it; a tracker started while another holds it samples instead, which keeps nested
// and overlapping windows correct.
// Usage:
//   PeakRSSTracker tracker;
//   tracker.start();
//   algorithm.call(...);
//   long peak_increase_kb = tracker.stop();
class PeakRSSTracker {
public:
    // Releases the peak mark and joins a sampler that stop() never reached (for example when an
    // exception skipped stop()); destroying a joinable std::thread calls std::terminate.
    ~PeakRSSTracker() {
        if (active_ && use_mark_) mark_users_.fetch_sub(1);
        if (sampler_.joinable()) {
            running_.store(false, std::memory_order_relaxed);
            sampler_.join();
        }
    }

    void start() {
        randlapack_require(!active_)
            << "PeakRSSTracker::start: already started; call stop() first";
        // glibc keeps freed heap pages resident, so without a trim the first tracked call absorbs
        // the process ramp-up while later calls reuse faulted pages, and per-call deltas
        // depend on execution order. malloc_trim releases only unused arena space.
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        // A throw below (bad_alloc, no thread) leaves the tracker idle and the mark free.
        baseline_kb_ = get_rss_kb();
        peak_kb_.store(baseline_kb_, std::memory_order_relaxed);
        use_mark_ = false;
        int free_mark = 0;
        if (mark_users_.compare_exchange_strong(free_mark, 1)) {
            try {
                use_mark_ = reset_peak_rss_mark();
            } catch (...) {
                mark_users_.fetch_sub(1);
                throw;
            }
            if (!use_mark_) mark_users_.fetch_sub(1);
        }
        if (!use_mark_) {
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
        active_ = true;
    }

    // Ends the window and returns the peak RSS increase in KB.
    long stop() {
        randlapack_require(active_)
            << "PeakRSSTracker::stop: not started; call start() first";
        active_ = false;
        long peak = get_rss_kb();
        if (use_mark_) {
            long hwm = read_status_kb("VmHWM:");
            if (hwm > peak) peak = hwm;
            mark_users_.fetch_sub(1);
        } else {
            running_.store(false, std::memory_order_relaxed);
            sampler_.join();
            long sampled = peak_kb_.load(std::memory_order_relaxed);
            if (sampled > peak) peak = sampled;
        }
        return peak - baseline_kb_;
    }

    /// Whether the current (or last) window reads the kernel's peak mark rather than sampling.
    bool uses_peak_mark() const { return use_mark_; }

private:
    static inline std::atomic<int> mark_users_{0};   // trackers holding the peak mark: 0 or 1
    bool active_   = false;
    bool use_mark_ = false;
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

// CholQR_linops: G(n*n) + A_temp(m*b), with the Gram's identity block I_block(n*b) during the
// Gram loop (none for an operator with materialize_cols, materializes_cols = true) and the
// retry copy of diag(G) (n) after it.
template <typename T>
static inline long cholqr_linops_analytical_kb(int64_t m, int64_t n,
                                               int64_t block_size = kDefaultGramBlockSize,
                                               bool materializes_cols = false) {
    int64_t b = gram_block_width(block_size, n);
    long elems = (long)n * n + (long)m * b + (materializes_cols ? (long)n : std::max<long>((long)n * b, n));
    return static_cast<long>(sizeof(T)) * elems / 1024;
}

// CholQR2_linops: pass 1 as CholQR_linops. Pass 2 reads pass 1's R as its preconditioner and
// solves each column panel of its inverse in the Gram loop (n*b, then the retry copy of diag(G),
// n), so its peak is pass 1's with an identity block, whatever the operator; with
// gram_left_gemm it forms R_pre (n*n) and Z_buf (n*b) instead, alive with the diag(G) copy.
template <typename T>
static inline long cholqr2_linops_analytical_kb(int64_t m, int64_t n,
                                                int64_t block_size = kDefaultGramBlockSize,
                                                bool gram_left_gemm = false) {
    int64_t b = gram_block_width(block_size, n);
    long common = (long)n * n + (long)m * b;
    long pass1  = common + std::max<long>((long)n * b, n);
    bool explicit_inverse = gram_left_by_gemm(CholQRPrecondMethod::trsm_identity, gram_left_gemm);
    long pass2  = explicit_inverse ? common + (long)n * n + (long)n * b + n : pass1;
    return static_cast<long>(sizeof(T)) * std::max(pass1, pass2) / 1024;
}

// sCholQR3_linops: passes 2 and 3 reuse CholQR2_linops's buffers.
template <typename T>
static inline long scholqr3_linops_analytical_kb(int64_t m, int64_t n,
                                                 int64_t block_size = kDefaultGramBlockSize,
                                                 bool gram_left_gemm = false) {
    return cholqr2_linops_analytical_kb<T>(m, n, block_size, gram_left_gemm);
}

// sCholQR3_linops_basic: Q_buf(m*n) + G(n*n) + R_temp(n*n) + M(n*n)
// Materializes Q = A * R1^{-1} after iteration 1, then uses dense syrk for iterations 2-3.
// No blocking: always O(m*n + n^2) peak.
template <typename T>
static inline long scholqr3_linops_basic_analytical_kb(int64_t m, int64_t n) {
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

} // namespace RandLAPACK
