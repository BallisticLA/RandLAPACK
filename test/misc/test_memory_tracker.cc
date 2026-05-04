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

    char* buf = static_cast<char*>(malloc(alloc_bytes));
    ASSERT_NE(buf, nullptr);
    memset(buf, 1, alloc_bytes);

    // Hold the allocation briefly so the sampler can observe it.
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
    // Do trivial work — no large allocations.
    volatile int x = 0;
    for (int i = 0; i < 1000; ++i)
        x += i;
    long increase = tracker.stop();
    EXPECT_LT(increase, 4096)
        << "Expected small RSS increase with no allocation, got " << increase << " KB";
}

#else // non-Linux

TEST_F(TestMemoryTracker, NonLinuxReturnsNegative) {
    // get_rss_kb should return -1 on unsupported platforms.
    EXPECT_EQ(RandLAPACK::get_rss_kb(), -1);
}

#endif
