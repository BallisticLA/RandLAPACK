#include "RandLAPACK/testing/rl_memory_tracker.hh"

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

class TestMemoryTracker : public ::testing::Test
{
    protected:
    virtual void SetUp() {};
    virtual void TearDown() {};
};

#ifdef __linux__

// get_rss_kb should return a positive value on Linux.
TEST_F(TestMemoryTracker, GetRSSReturnsPositive) {
    long rss = RandLAPACK::get_rss_kb();
    EXPECT_GT(rss, 0) << "get_rss_kb() should return positive on Linux";
}

// Allocate a known amount of memory, touch it to ensure RSS increase,
// then verify the tracker reports a value in a statistically plausible range.
TEST_F(TestMemoryTracker, PeakRSSDetectsAllocation) {
    const size_t alloc_bytes = 100 * 1024 * 1024; // 100 MB
    const long alloc_kb = static_cast<long>(alloc_bytes / 1024);

    RandLAPACK::PeakRSSTracker tracker;
    tracker.start();

    char* volatile buf = static_cast<char*>(malloc(alloc_bytes));   // volatile: kept, not elided
    ASSERT_NE(buf, nullptr);
    memset(buf, 1, alloc_bytes);

    // Hold the allocation briefly so the sampling fallback can observe it.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    long increase = tracker.stop();
    free(buf);

    EXPECT_GT(increase, alloc_kb / 2)
        << "RSS increase (" << increase << " KB) is less than 50% of "
        << alloc_kb << " KB allocation";
    EXPECT_LT(increase, alloc_kb * 2)
        << "RSS increase (" << increase << " KB) is more than 200% of "
        << alloc_kb << " KB allocation";
}

// Verify that stop() returns a small value when no large allocation happens.
TEST_F(TestMemoryTracker, NoAllocationReportsSmall) {
    RandLAPACK::PeakRSSTracker tracker;
    tracker.start();
    // Do trivial work; no large allocations.
    volatile int x = 0;
    for (int i = 0; i < 1000; ++i)
        x = x + i;
    long increase = tracker.stop();
    EXPECT_LT(increase, 4096)
        << "Expected small RSS increase with no allocation, got " << increase << " KB";
}

// A peak released again before stop() still counts: the kernel's peak mark keeps it, and the
// sampling fallback sees it too (touching 64 MB takes far longer than one 100 us period).
TEST_F(TestMemoryTracker, PeakFreedBeforeStopIsCounted) {
    const size_t alloc_bytes = 64 * 1024 * 1024;
    const long alloc_kb = static_cast<long>(alloc_bytes / 1024);
    RandLAPACK::PeakRSSTracker tracker;
    tracker.start();
    char* volatile buf = static_cast<char*>(malloc(alloc_bytes));   // volatile: kept, not elided
    ASSERT_NE(buf, nullptr);
    memset(buf, 1, alloc_bytes);
    free(buf);
    long increase = tracker.stop();
    EXPECT_GT(increase, alloc_kb / 2) << increase << " KB";
    EXPECT_LT(increase, alloc_kb * 2) << increase << " KB";
}

// A tracker started inside another must not reset the outer one's peak: the outer tracker
// still reports a peak reached and released before the inner one started.
TEST_F(TestMemoryTracker, NestedTrackerKeepsTheOuterPeak) {
    const size_t alloc_bytes = 64 * 1024 * 1024;
    const long alloc_kb = static_cast<long>(alloc_bytes / 1024);
    RandLAPACK::PeakRSSTracker outer, inner;
    outer.start();
    char* volatile buf = static_cast<char*>(malloc(alloc_bytes));
    ASSERT_NE(buf, nullptr);
    memset(buf, 1, alloc_bytes);
    free(buf);
    inner.start();
    EXPECT_FALSE(inner.uses_peak_mark());
    long inner_increase = inner.stop();
    long outer_increase = outer.stop();
    EXPECT_GT(outer_increase, alloc_kb / 2) << outer_increase << " KB";
    EXPECT_LT(inner_increase, alloc_kb / 2) << inner_increase << " KB";
}

#endif

// One tracker at a time holds the peak mark, in any order of starts and stops: a tracker started
// while another holds it samples, and the mark is free again after its holder stops or is
// destroyed.
TEST_F(TestMemoryTracker, OverlappingTrackersHoldTheMarkInTurn) {
    RandLAPACK::PeakRSSTracker a, b, c;
    a.start();
    b.start();
    EXPECT_FALSE(b.uses_peak_mark());
    EXPECT_GE(a.stop(), 0);
    c.start();
    EXPECT_EQ(c.uses_peak_mark(), a.uses_peak_mark());
    EXPECT_GE(b.stop(), 0);
    EXPECT_GE(c.stop(), 0);
    bool had_mark = false;
    {
        RandLAPACK::PeakRSSTracker d;
        d.start();
        had_mark = d.uses_peak_mark();
    }
    RandLAPACK::PeakRSSTracker e;
    e.start();
    EXPECT_EQ(e.uses_peak_mark(), had_mark);
    EXPECT_GE(e.stop(), 0);
}

// A second start() without stop() must throw, not assign a running sampler thread
// (std::terminate), and leave the tracker usable.
TEST_F(TestMemoryTracker, DoubleStartThrows) {
    RandLAPACK::PeakRSSTracker tracker;
    tracker.start();
    EXPECT_THROW(tracker.start(), RandLAPACK::Error);
    EXPECT_GE(tracker.stop(), 0);
}

TEST_F(TestMemoryTracker, StopWithoutStartThrows) {
    RandLAPACK::PeakRSSTracker tracker;
    EXPECT_THROW(tracker.stop(), RandLAPACK::Error);
}

// Destroying a started tracker must not terminate: a running sampler is joined.
TEST_F(TestMemoryTracker, DestructionWithoutStopJoinsSampler) {
    RandLAPACK::PeakRSSTracker tracker;
    tracker.start();
}

TEST_F(TestMemoryTracker, RestartAfterStop) {
    RandLAPACK::PeakRSSTracker tracker;
    tracker.start();
    EXPECT_GE(tracker.stop(), 0);
    tracker.start();
    EXPECT_GE(tracker.stop(), 0);
    EXPECT_THROW(tracker.stop(), RandLAPACK::Error);
}

// The Blendenpik model counts the sparse sketch (values and two int64_t index arrays of
// min(nnz, d, m) * max(d, m) entries), adds the x0 and LSQR buffers only when asked, and clamps
// d to n.
TEST_F(TestMemoryTracker, BlendenpikModelCountsSketchAndOptionalBuffers) {
    const int64_t m = 1000, n = 100, d = 200;
    auto kb = [](int64_t elems, int64_t sketch) {
        return (long)(((int64_t)sizeof(double) * (elems + sketch) + 2 * 8 * sketch) / 1024);
    };
    const int64_t base   = d * n + n + n * n;
    const int64_t sketch = 4 * m;                 // nnz 4, max(d, m) = m
    const int64_t x0     = d + n + m;
    const int64_t lsqr   = m + 4 * n;
    using RandLAPACK::blendenpik_linops_analytical_kb;
    EXPECT_EQ(blendenpik_linops_analytical_kb<double>(m, n, 2.0, 4, false, false), kb(base, sketch));
    EXPECT_EQ(blendenpik_linops_analytical_kb<double>(m, n, 2.0, 4, true,  false), kb(base + x0, sketch));
    EXPECT_EQ(blendenpik_linops_analytical_kb<double>(m, n, 2.0, 4, false, true),  kb(base + lsqr, sketch));
    EXPECT_EQ(blendenpik_linops_analytical_kb<double>(m, n, 2.0, 4, true,  true),  kb(base + x0 + lsqr, sketch));
    // nnz is capped at min(d, m), as the driver caps it.
    EXPECT_EQ(blendenpik_linops_analytical_kb<double>(m, n, 2.0, 5000, false, false), kb(base, d * m));
    // d_factor < 1 still sketches to n rows, as the driver does.
    EXPECT_EQ(blendenpik_linops_analytical_kb<double>(m, n, 0.5, 4, false, false),
              kb(2 * n * n + n, 4 * m));
    // Byte counts past 2^31 do not wrap (long is 32-bit on MSVC).
    const int64_t M = 1 << 20, N = 1 << 14;
    EXPECT_EQ(blendenpik_linops_analytical_kb<double>(M, N, 2.0, 4, false, false),
              kb(2 * N * N + N + N * N, 4 * M));
}

#ifndef __linux__

TEST_F(TestMemoryTracker, NonLinuxReturnsNegative) {
    // get_rss_kb should return -1 on unsupported platforms.
    EXPECT_EQ(RandLAPACK::get_rss_kb(), -1);
}

#endif

// The CQRRT analytical models in plain byte arithmetic (double, so 8 bytes per entry; every
// total is a whole number of KB). n = 128, b = 64, so n^2 = 16384, and d = d_factor * n. The
// max(d, 2n) x n buffer (S A, then P and G) is held throughout.
TEST(TestAnalyticalModels, cqrrto_linops_phases_and_routes) {
    using M = RandLAPACK::CholQRPrecondMethod;
    const long t = 8, n = 128, nn = n * n, b = 64;

    // Gram phase dominates (m = 65536, d = 256 = 2n): A_temp and trsm_identity's panel of P^{-1}.
    const long m = 65536, buf = t * 2 * nn;
    long gram = t * (m * b) + t * n * b;
    ASSERT_GT(gram, t * n + 24 * 2 * m);
    EXPECT_EQ(RandLAPACK::cqrrto_linops_analytical_kb<double>(m, n, 2.0, b), (buf + gram) / 1024);
    // d < 2n (d_factor = 1.25) still holds 2n x n, for P and G.
    EXPECT_EQ(RandLAPACK::cqrrto_linops_analytical_kb<double>(m, n, 1.25, b), (buf + gram) / 1024);

    // gram_left_gemm forms R_pre (n^2) and Z_buf (n x b), with the diag(G) copy in place of
    // the panel.
    long gram_z = t * (m * b + nn + n * b) + t * n;
    EXPECT_EQ(RandLAPACK::cqrrto_linops_analytical_kb<double>(m, n, 2.0, b, true), (buf + gram_z) / 1024);

    // geqp3 and bqrrp always hold R_pre and Z_buf, and while P is inverted its pivoted QR
    // (n^2 + n entries and n int64_t pivots) instead of the diag(G) copy.
    long gram_piv = gram_z - t * n + t * (nn + n) + 8 * n;
    for (M method : {M::geqp3, M::bqrrp})
        for (bool left_gemm : {false, true})
            EXPECT_EQ(RandLAPACK::cqrrto_linops_analytical_kb<double>(m, n, 2.0, b, left_gemm, method),
                      (buf + gram_piv) / 1024) << (int)method << " " << left_gemm;
    EXPECT_EQ(RandLAPACK::cqrrto_linops_analytical_kb<double>(m, n, 2.0, b, false, M::trtri),
              (buf + t * (m * b + nn) + t * n) / 1024);

    // Sketch phase dominates (m = n, d = 512, b = 8): the d x n buffer + tau + the sparse
    // sketch's values and two int64_t index arrays, min(nnz, d, m) * max(d, m) = 4 * 512 entries
    // each (the default nnz).
    long sketch = t * n + (t + 16) * 4 * 512;
    ASSERT_GT(sketch, t * (n * 8) + t * n * 8);
    EXPECT_EQ(RandLAPACK::cqrrto_linops_analytical_kb<double>(n, n, 4.0, 8), (t * 512 * n + sketch) / 1024);
}

// Where the sparse sketch outweighs the rest (the scaling study's smallest problem, 1000 x 10,
// d = 20, nnz = 4): its 4 * 1000 entries of value, row and column dominate the sketch phase.
TEST(TestAnalyticalModels, cqrrto_linops_counts_the_sparse_sketch) {
    using M = RandLAPACK::CholQRPrecondMethod;
    const int64_t t = 8, m = 1000, n = 10, d = 20, b = 10;
    const int64_t buf    = t * d * n;
    const int64_t sketch = t * n + (t + 16) * 4 * m;
    const int64_t gram   = t * (m * b) + t * n * b;
    ASSERT_GT(sketch, gram);
    EXPECT_EQ(RandLAPACK::cqrrto_linops_analytical_kb<double>(m, n, 2.0, b, false, M::trsm_identity, 4),
              (long)((buf + sketch) / 1024));
    // nnz is capped at min(d, m), as the driver caps it.
    EXPECT_EQ(RandLAPACK::cqrrto_linops_analytical_kb<double>(m, n, 2.0, b, false, M::trsm_identity, 500),
              (long)((buf + t * n + (t + 16) * d * m) / 1024));
}

// The CholQR-family models (double, m = 4096, n = 128, b = 64): pass 1 holds G, A_temp and the
// identity block; later passes need no more by default (the previous R is the preconditioner
// and its inverse is solved panel by panel), and with gram_left_gemm add R_pre (n^2), Z_buf
// (n x b) and the diag(G) copy.
TEST(TestAnalyticalModels, cholqr_family) {
    const int64_t t = 8, m = 4096, n = 128, b = 64;
    const long pass1 = (long)(t * (n * n + m * b + n * b) / 1024);
    EXPECT_EQ(RandLAPACK::cholqr_linops_analytical_kb<double>(m, n, b), pass1);
    // An operator with materialize_cols needs no identity block: the diag(G) copy is the third term.
    EXPECT_EQ(RandLAPACK::cholqr_linops_analytical_kb<double>(m, n, b, true), (long)(t * (n * n + m * b + n) / 1024));
    EXPECT_EQ(RandLAPACK::cholqr2_linops_analytical_kb<double>(m, n, b), pass1);
    EXPECT_EQ(RandLAPACK::scholqr3_linops_analytical_kb<double>(m, n, b), pass1);
    const long pass2_gemm = (long)(t * (2 * n * n + m * b + n * b + n) / 1024);
    EXPECT_EQ(RandLAPACK::cholqr2_linops_analytical_kb<double>(m, n, b, true), pass2_gemm);
    EXPECT_EQ(RandLAPACK::scholqr3_linops_analytical_kb<double>(m, n, b, true), pass2_gemm);
}

TEST(TestAnalyticalModels, dense_cqrrt) {
    // The materialized A (m x n) + A_hat (d x n) + tau (n), m = 65536, n = 128, d = 256.
    const long t = 8, m = 65536, n = 128;
    EXPECT_EQ(RandLAPACK::dense_cqrrt_analytical_kb<double>(m, n, 2.0), t * (m * n + 256 * n + n) / 1024);
}
