// The test suite defines RandLAPACK_POISON_WORKSPACE, so memory handed out uninitialized is poisoned.

#include "rl_memory.hh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <utility>
#include <gtest/gtest.h>

using RandLAPACK::util::GrowBuffer;
using RandLAPACK::util::MallocBuffer;

// AddressSanitizer aborts on an oversized realloc unless told to fail it like the default allocator,
// which the allocation-failure test needs; this hook sets that for the whole test program. A failed
// throwing new is reported as an error under AddressSanitizer either way.
#if defined(__SANITIZE_ADDRESS__)
#define RL_TEST_UNDER_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RL_TEST_UNDER_ASAN 1
#endif
#endif
#ifdef RL_TEST_UNDER_ASAN
extern "C" const char* __asan_default_options() { return "allocator_may_return_null=1"; }
#endif

namespace {
constexpr int64_t too_many = std::numeric_limits<int64_t>::max() / 16;
}

TEST(TestMemory, checked_mul) {
    EXPECT_EQ(RandLAPACK::util::checked_mul(3, 4), 12);
    EXPECT_EQ(RandLAPACK::util::checked_mul(0, std::numeric_limits<int64_t>::max()), 0);
    EXPECT_THROW(RandLAPACK::util::checked_mul(-1, 4), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::util::checked_mul(int64_t(1) << 32, int64_t(1) << 32), RandLAPACK::Error);
}

TEST(TestMemory, make_buffer_poisons_and_zeroed_buffer_zeroes) {
    auto d = RandLAPACK::util::make_buffer<double>(5);
    for (int i = 0; i < 5; ++i)
        EXPECT_TRUE(std::isnan(d[i]));
    auto p = RandLAPACK::util::make_buffer<int64_t>(5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(p[i], -1);
        // LP64 LAPACK++ narrows 64-bit pivots to 32 bits; the poison must stay nonzero there.
        EXPECT_NE(static_cast<int32_t>(p[i]), 0);
    }
    auto u = RandLAPACK::util::make_buffer<uint32_t>(5);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(u[i], std::numeric_limits<uint32_t>::max());
    auto z = RandLAPACK::util::make_zeroed_buffer<float>(5);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(z[i], 0.0f);
    EXPECT_NO_THROW(RandLAPACK::util::make_buffer<double>(0));
    EXPECT_THROW(RandLAPACK::util::make_buffer<double>(-1), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::util::make_zeroed_buffer<double>(-1), RandLAPACK::Error);
}

TEST(TestMemory, grow_buffer_ensure_reallocates_only_to_grow) {
    GrowBuffer<double> buf;
    EXPECT_EQ(buf.capacity(), 0);
    double* p = buf.ensure(10);
    EXPECT_EQ(buf.capacity(), 10);
    for (int i = 0; i < 10; ++i)
        p[i] = 1.0;
    EXPECT_EQ(buf.ensure(4), p);
    EXPECT_EQ(buf.capacity(), 10);
    // Scratch is dead between requests, so a request poisons what it returns.
    for (int i = 0; i < 4; ++i)
        EXPECT_TRUE(std::isnan(p[i]));
    EXPECT_EQ(p[4], 1.0);
    buf.ensure(20);
    EXPECT_EQ(buf.capacity(), 20);
    EXPECT_THROW(buf.ensure(-1), RandLAPACK::Error);
    buf.release();
    EXPECT_EQ(buf.capacity(), 0);
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(TestMemory, grow_buffer_grow_keeps_the_prefix) {
    GrowBuffer<double> buf;
    double* p = buf.grow(4, 0);
    for (int i = 0; i < 4; ++i)
        p[i] = i;
    // Reallocating: the prefix moves, the tail is poisoned.
    p = buf.grow(8, 3);
    EXPECT_EQ(buf.capacity(), 8);
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(p[i], i);
    for (int i = 3; i < 8; ++i)
        EXPECT_TRUE(std::isnan(p[i]));
    // In place: the prefix stays, the rest of the requested range is poisoned.
    p[5] = 5.0;
    EXPECT_EQ(buf.grow(6, 2), p);
    EXPECT_EQ(p[0], 0.0);
    EXPECT_EQ(p[1], 1.0);
    EXPECT_TRUE(std::isnan(p[5]));
    EXPECT_THROW(buf.grow(10, 9), RandLAPACK::Error);
    EXPECT_THROW(buf.grow(2, 3), RandLAPACK::Error);
}

TEST(TestMemory, grow_buffer_moves) {
    GrowBuffer<int64_t> a;
    int64_t* p = a.ensure(3);
    GrowBuffer<int64_t> b(std::move(a));
    EXPECT_EQ(b.data(), p);
    EXPECT_EQ(b.capacity(), 3);
    // The source is empty and usable again.
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.capacity(), 0);
    EXPECT_NE(a.ensure(2), nullptr);
    GrowBuffer<int64_t> c;
    c = std::move(b);
    EXPECT_EQ(c.data(), p);
    EXPECT_EQ(b.capacity(), 0);
    EXPECT_NE(b.grow(4, 0), nullptr);
}

TEST(TestMemory, malloc_buffer_grows_and_hands_over) {
    MallocBuffer<double> buf;
    double* p = buf.grow(3, 0);
    for (int i = 0; i < 3; ++i)
        p[i] = i;
    p = buf.grow(6, 3);
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(p[i], i);
    for (int i = 3; i < 6; ++i)
        EXPECT_TRUE(std::isnan(p[i]));
    double* owned = buf.release();
    EXPECT_EQ(owned, p);
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.capacity(), 0);
    std::free(owned);
}

TEST(TestMemory, allocation_failure_keeps_the_old_block) {
    MallocBuffer<double> m;
    double* p = m.grow(4, 0);
    p[0] = 7.0;
    EXPECT_THROW(m.grow(too_many, 4), std::bad_alloc);
    EXPECT_EQ(m.data(), p);
    EXPECT_EQ(m.capacity(), 4);
    EXPECT_EQ(p[0], 7.0);

#ifndef RL_TEST_UNDER_ASAN
    GrowBuffer<double> g;
    g.ensure(4);
    EXPECT_THROW(g.ensure(too_many), std::bad_alloc);
    EXPECT_EQ(g.capacity(), 0);
    EXPECT_EQ(g.data(), nullptr);
#endif
}
