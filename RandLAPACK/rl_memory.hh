#pragma once

#include "rl_exceptions.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

// Memory ownership for library code (docs/CONVENTIONS.md, "Memory"). Library code owns memory through
// RAII types such as these, so every return and every exception releases it.
//
// Test builds define RandLAPACK_POISON_WORKSPACE: memory these helpers hand out uninitialized is
// filled with NaN (floating types) or -1 (integers; the largest value for unsigned ones), so code
// that reads it before writing it fails a test. Define it for every translation unit of a program
// or for none.

namespace RandLAPACK::util {

/// a * b for sizes; throws RandLAPACK::Error if either is negative or the product overflows int64_t.
inline int64_t checked_mul(int64_t a, int64_t b) {
    randlapack_require(a >= 0 && b >= 0) << "sizes must be nonnegative, got " << a << " and " << b;
    randlapack_require(a == 0 || b <= std::numeric_limits<int64_t>::max() / a)
        << "size " << a << " x " << b << " overflows int64_t";
    return a * b;
}

namespace internal {

template <typename T>
void poison(T* p, int64_t n) {
#ifdef RandLAPACK_POISON_WORKSPACE
    // -1 survives the narrowing of 64-bit pivot arrays to 32-bit LAPACK integers.
    if constexpr (std::is_floating_point_v<T>)
        std::fill(p, p + n, std::numeric_limits<T>::quiet_NaN());
    else if constexpr (std::is_integral_v<T>)
        std::fill(p, p + n, static_cast<T>(-1));
#else
    (void) p;
    (void) n;
#endif
}

inline std::size_t checked_count(int64_t n) {
    randlapack_require(n >= 0) << "buffer size must be nonnegative, got " << n;
    return static_cast<std::size_t>(n);
}

} // namespace internal

/// n elements, not initialized (poisoned in test builds).
template <typename T>
std::unique_ptr<T[]> make_buffer(int64_t n) {
    std::unique_ptr<T[]> buf(new T[internal::checked_count(n)]);
    internal::poison(buf.get(), n);
    return buf;
}

/// n elements, all zero.
template <typename T>
std::unique_ptr<T[]> make_zeroed_buffer(int64_t n) {
    return std::unique_ptr<T[]>(new T[internal::checked_count(n)]());
}

/// A grow-only buffer that a driver object keeps across calls: it reallocates only when a request
/// exceeds its capacity. Move-only. A pointer it returns stays valid until an ensure or grow that
/// reallocates, a release, or the buffer is moved or destroyed.
template <typename T>
class GrowBuffer {
public:
    GrowBuffer() = default;
    GrowBuffer(GrowBuffer&& other) noexcept
        : buf_(std::move(other.buf_)), cap_(std::exchange(other.cap_, 0)) {}
    GrowBuffer& operator=(GrowBuffer&& other) noexcept {
        buf_ = std::move(other.buf_);
        cap_ = std::exchange(other.cap_, 0);
        return *this;
    }
    GrowBuffer(const GrowBuffer&) = delete;
    GrowBuffer& operator=(const GrowBuffer&) = delete;

    /// Scratch of at least n elements; earlier contents are not preserved.
    T* ensure(int64_t n) {
        if (n > cap_) {
            buf_.reset();  // the old contents are dead, so do not hold both blocks at once
            cap_ = 0;
            buf_ = make_buffer<T>(n);
            cap_ = n;
        } else {
            internal::checked_count(n);
            internal::poison(buf_.get(), n);
        }
        return buf_.get();
    }

    /// At least n elements, the first keep of which are preserved; the rest are not initialized.
    T* grow(int64_t n, int64_t keep) {
        randlapack_require(keep >= 0 && keep <= cap_ && keep <= n)
            << "keep=" << keep << " must lie in [0, min(capacity=" << cap_ << ", n=" << n << ")]";
        if (n > cap_) {
            std::unique_ptr<T[]> fresh(new T[internal::checked_count(n)]);
            std::copy(buf_.get(), buf_.get() + keep, fresh.get());
            buf_ = std::move(fresh);
            cap_ = n;
        }
        internal::poison(buf_.get() + keep, n - keep);
        return buf_.get();
    }

    T* data() noexcept { return buf_.get(); }
    const T* data() const noexcept { return buf_.get(); }
    int64_t capacity() const noexcept { return cap_; }
    void release() noexcept { buf_.reset(); cap_ = 0; }

private:
    std::unique_ptr<T[]> buf_;
    int64_t cap_ = 0;
};

/// Owns a std::malloc'd array destined for a caller that frees it with std::free (the older
/// `T*&` output contract). Transitional: it goes once no routine has such an output.
template <typename T>
class MallocBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "realloc moves the elements bytewise");
public:
    MallocBuffer() = default;
    ~MallocBuffer() { std::free(ptr_); }
    MallocBuffer(MallocBuffer&& other) noexcept : ptr_(other.ptr_), cap_(other.cap_) {
        other.ptr_ = nullptr;
        other.cap_ = 0;
    }
    MallocBuffer& operator=(MallocBuffer&& other) noexcept {
        if (this != &other) {
            std::free(ptr_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            cap_ = std::exchange(other.cap_, 0);
        }
        return *this;
    }
    MallocBuffer(const MallocBuffer&) = delete;
    MallocBuffer& operator=(const MallocBuffer&) = delete;

    /// At least n elements, the first keep of which are preserved. On allocation failure the old
    /// block stays owned (and is freed by the destructor) and std::bad_alloc is thrown.
    T* grow(int64_t n, int64_t keep) {
        randlapack_require(keep >= 0 && keep <= cap_ && keep <= n)
            << "keep=" << keep << " must lie in [0, min(capacity=" << cap_ << ", n=" << n << ")]";
        if (n > cap_) {
            std::size_t count = internal::checked_count(n);
            randlapack_require(count <= std::numeric_limits<std::size_t>::max() / sizeof(T))
                << "buffer of " << n << " elements overflows size_t";
            T* resized = static_cast<T*>(std::realloc(ptr_, count * sizeof(T)));
            if (resized == nullptr)
                throw std::bad_alloc();
            ptr_ = resized;
            cap_ = n;
        }
        internal::poison(ptr_ + keep, n - keep);
        return ptr_;
    }

    /// Hands the array to the caller, who must std::free it.
    T* release() noexcept {
        T* p = ptr_;
        ptr_ = nullptr;
        cap_ = 0;
        return p;
    }

    T* data() noexcept { return ptr_; }
    int64_t capacity() const noexcept { return cap_; }

private:
    T* ptr_ = nullptr;
    int64_t cap_ = 0;
};

} // namespace RandLAPACK::util

namespace RandLAPACK {

/// Bytes a call needs: host and device workspace, plus the bound on rank-adaptive results.
struct WorkspaceSize {
    std::size_t host_bytes = 0;
    std::size_t device_bytes = 0;
    std::size_t result_bound_bytes = 0;
};

/// A column-major view of storage that a driver object owns (rows x cols, leading dimension ld).
template <typename T>
struct MatrixView {
    const T* data = nullptr;
    int64_t ld = 0;
    int64_t rows = 0;
    int64_t cols = 0;
};

} // namespace RandLAPACK
