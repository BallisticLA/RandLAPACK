#include "rl_test_utils.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace linops = RandLAPACK::linops;

// ===== Binary I/O round trip (kept from Phase 0) =============================
TEST(TestBinaryFixtureIO, BinaryIoRoundTrip) {
    using T = double;
    int64_t m = 5, n = 3;
    T *orig = new T[m * n];
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            orig[i + j * m] = (T)(100 * j + i + 1);

    // Portable temp path: gtest's per-run TempDir works on Windows too, where
    // POSIX mkstemps/close do not exist. save_dense_bin creates the file.
    std::string tmpname = ::testing::TempDir() + "rl_v2_bin_roundtrip.bin";

    RandLAPACK::testing::save_dense_bin<T>(tmpname, m, n, orig);
    T *back = new T[m * n];
    std::fill(back, back + m * n, (T)-1.0);
    int64_t m_b = 0, n_b = 0;
    RandLAPACK::testing::load_dense_bin<T>(tmpname, m_b, n_b, back, m * n);
    EXPECT_EQ(m_b, m);
    EXPECT_EQ(n_b, n);
    for (int64_t i = 0; i < m * n; ++i) EXPECT_DOUBLE_EQ(back[i], orig[i]);
    std::remove(tmpname.c_str());
    delete[] orig;
    delete[] back;
}

} // namespace
