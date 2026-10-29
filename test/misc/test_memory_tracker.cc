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

#ifndef __linux__

TEST_F(TestMemoryTracker, NonLinuxReturnsNegative) {
    // get_rss_kb should return -1 on unsupported platforms.
    EXPECT_EQ(RandLAPACK::get_rss_kb(), -1);
}

#endif
