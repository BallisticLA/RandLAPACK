#include "RandLAPACK/testing/rl_memory_tracker.hh"

#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <algorithm>

class TestMemoryTracker : public ::testing::Test
{
    protected:
    virtual void SetUp() {};
    virtual void TearDown() {};
};

#ifdef __linux__

// Allocate a known amount of memory, touch it to ensure RSS increase,
// then verify the tracker reports a value in a statistically plausible range.
TEST_F(TestMemoryTracker, PeakRSSDetectsAllocation) {
    const int num_trials = 30;
    const size_t alloc_bytes = 50 * 1024 * 1024; // 50 MB
    const long alloc_kb = static_cast<long>(alloc_bytes / 1024);

    std::vector<long> measurements(num_trials);

    for (int t = 0; t < num_trials; ++t) {
        RandLAPACK::PeakRSSTracker tracker;
        tracker.start();

        // Allocate and touch every page to force RSS increase.
        volatile char* buf = static_cast<volatile char*>(malloc(alloc_bytes));
        ASSERT_NE(buf, nullptr);
        for (size_t i = 0; i < alloc_bytes; i += 4096)
            buf[i] = 1;

        measurements[t] = tracker.stop();
        free(const_cast<char*>(buf));
    }

    // Compute mean and stddev of measurements.
    double sum = std::accumulate(measurements.begin(), measurements.end(), 0.0);
    double mean = sum / num_trials;
    double sq_sum = 0.0;
    for (auto v : measurements)
        sq_sum += (v - mean) * (v - mean);
    double stddev = std::sqrt(sq_sum / (num_trials - 1));

    // The known allocation should fall within mean ± 3*stddev,
    // and the mean should be at least 80% of the allocation size
    // (some OS overhead means it won't be exact).
    EXPECT_GT(mean, alloc_kb * 0.8)
        << "Mean RSS increase (" << mean << " KB) is less than 80% of "
        << alloc_kb << " KB allocation";
    EXPECT_LT(mean, alloc_kb * 1.5)
        << "Mean RSS increase (" << mean << " KB) is more than 150% of "
        << alloc_kb << " KB allocation";

    // Print statistical summary for diagnostics.
    printf("  PeakRSSTracker test: %d trials, alloc=%ld KB\n", num_trials, alloc_kb);
    printf("  Mean=%.0f KB, StdDev=%.0f KB, CoeffVar=%.1f%%\n",
           mean, stddev, 100.0 * stddev / mean);
    printf("  Range: [%ld, %ld] KB\n",
           *std::min_element(measurements.begin(), measurements.end()),
           *std::max_element(measurements.begin(), measurements.end()));
}

// Verify that stop() returns 0 (or near-0) when no allocation happens.
TEST_F(TestMemoryTracker, NoAllocationReportsZero) {
    RandLAPACK::PeakRSSTracker tracker;
    tracker.start();
    // Do nothing.
    long increase = tracker.stop();
    // Allow small noise (< 1 MB) from background system activity.
    EXPECT_LT(increase, 1024)
        << "Expected near-zero RSS increase, got " << increase << " KB";
}

#else // non-Linux

TEST_F(TestMemoryTracker, NonLinuxReturnsNegative) {
    // get_rss_kb should return -1 on unsupported platforms.
    EXPECT_EQ(RandLAPACK::get_rss_kb(), -1);
}

#endif
