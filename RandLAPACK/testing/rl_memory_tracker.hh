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
// All return memory in KB. They count the driver's own buffers: an operator's per-apply
// temporaries (including those of its sketch overload) are not included. m is the row count
// of the operator the driver runs on ([A; mu I] has m + n rows).
// ---------------------------------------------------------------------------

// CQRRTO_linops: one max(d, 2n) x n buffer, holding S A and then P and G, plus the larger of
// its two phases' other buffers: the sketch's tau (n) and the sparse sketch itself (values and
// int64_t row and column indices, vec_nnz * max(d, m) of each, vec_nnz = min(nnz, d, m) as the
// driver caps it), or the Gram's A_temp (m*b), R_pre (n*n) unless trsm_identity applies the
// left factor by TRSM, Z_buf (n*b) when gram_left_by_gemm, and the largest of that default
// route's panel of P^{-1} (n*b) in the Gram loop, the retry copy of diag(G) (n) and, for geqp3
// and bqrrp, the pivoted QR of P held while P is inverted (n*n + n, and n int64_t pivots).
// Not counted: BQRRP's own workspace.
template <typename T>
static inline long cqrrto_linops_analytical_kb(int64_t m, int64_t n, double d_factor,
                                               int64_t block_size = kDefaultGramBlockSize,
                                               bool gram_left_gemm = false,
                                               CholQRPrecondMethod precond_method = CholQRPrecondMethod::trsm_identity,
                                               int64_t nnz = 4) {
    const int64_t t = sizeof(T);
    int64_t d = cqrrt_sketch_rows(static_cast<T>(d_factor), n);
    int64_t b = gram_block_width(block_size, n);
    bool pivoted   = precond_method == CholQRPrecondMethod::geqp3 || precond_method == CholQRPrecondMethod::bqrrp;
    bool left_gemm = gram_left_by_gemm(precond_method, gram_left_gemm);
    bool panel     = !forms_explicit_inverse(precond_method, gram_left_gemm);
    int64_t buf    = t * std::max(d, 2 * n) * n;
    int64_t sketch = t * n + (t + 2 * (int64_t)sizeof(int64_t)) * std::min({nnz, d, m}) * std::max(d, m);
    int64_t gram   = t * (m * b + (panel ? 0 : n * n) + (left_gemm ? n * b : 0))
                   + std::max<int64_t>(t * (panel ? n * b : n),
                                       pivoted ? t * (n * n + n) + (int64_t)sizeof(int64_t) * n : 0);
    return static_cast<long>((buf + std::max(sketch, gram)) / 1024);
}

// CholQR_linops: G(n*n) + A_temp(m*b), with the Gram's identity block I_block(n*b) during the
// Gram loop (none for an operator with materialize_cols, materializes_cols = true) and the
// retry copy of diag(G) (n) after it.
template <typename T>
static inline long cholqr_linops_analytical_kb(int64_t m, int64_t n,
                                               int64_t block_size = kDefaultGramBlockSize,
                                               bool materializes_cols = false) {
    int64_t b = gram_block_width(block_size, n);
    int64_t elems = n * n + m * b + (materializes_cols ? n : std::max<int64_t>(n * b, n));
    return static_cast<long>((int64_t)sizeof(T) * elems / 1024);
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
    bool explicit_inverse = forms_explicit_inverse(CholQRPrecondMethod::trsm_identity, gram_left_gemm);
    int64_t elems = n * n + m * b + (explicit_inverse ? n * n + n * b + n : std::max<int64_t>(n * b, n));
    return static_cast<long>((int64_t)sizeof(T) * elems / 1024);
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
    return static_cast<long>((int64_t)sizeof(T) * (m * n + 3 * n * n) / 1024);
}

// Dense CQRRT on the materialized operator: A(m*n) + A_hat(d*n) + tau(n).
template <typename T>
static inline long dense_cqrrt_analytical_kb(int64_t m, int64_t n, double d_factor) {
    int64_t d = cqrrt_sketch_rows(static_cast<T>(d_factor), n);
    return static_cast<long>((int64_t)sizeof(T) * (m * n + d * n + n) / 1024);
}

// Blendenpik_linops: Ask(d*n) + tau(n) + R(n*n) and the sparse sketch (values and int64_t row
// and column indices, vec_nnz * max(d, m) of each, vec_nnz = min(nnz, d, m) as the driver caps
// it), all live until call() returns, plus Sb(d) + x0(n) + r0(m) when the sketch-and-solve x0
// is built (warm_start or init_only) and LSQR's m + 4*n when LSQR runs.
template <typename T>
static inline long blendenpik_linops_analytical_kb(int64_t m, int64_t n, double d_factor, int64_t nnz,
                                                   bool with_x0, bool with_lsqr) {
    int64_t d = cqrrt_sketch_rows(static_cast<T>(d_factor), n);
    int64_t sketch_nnz = std::min({nnz, d, m}) * std::max(d, m);
    int64_t elems = d * n + n + n * n + sketch_nnz;
    if (with_x0)   elems += d + n + m;
    if (with_lsqr) elems += m + 4 * n;
    return static_cast<long>(((int64_t)sizeof(T) * elems + 2 * (int64_t)sizeof(int64_t) * sketch_nnz) / 1024);
}

} // namespace RandLAPACK
