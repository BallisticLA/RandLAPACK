#pragma once

#include "rl_blaspp.hh"
#include "rl_exceptions.hh"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace RandLAPACK::gen {

namespace matrix_io_detail {

[[noreturn]] inline void fail(const char* filename, const char* reason) {
    throw RandLAPACK::Error(std::string(reason) + ": " + (filename ? filename : "(null)"));
}

template <typename T>
size_t checked_size(int64_t m, int64_t n, const char* filename) {
    if (m <= 0 || n <= 0)
        fail(filename, "Matrix dimensions must be positive");
    // Bound both signed indexing arithmetic and the size of the output array.
    const uint64_t rows = static_cast<uint64_t>(m);
    const uint64_t cols = static_cast<uint64_t>(n);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / cols ||
        rows > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T)) / cols ||
        rows > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max() / sizeof(T)) / cols)
        fail(filename, "Matrix dimensions overflow the output size");
    return static_cast<size_t>(rows * cols);
}

template <typename T>
void check_output_size(Layout layout, int64_t m, int64_t n, int64_t lda, const char* filename) {
    checked_size<T>(m, n, filename);
    const int64_t inner = layout == Layout::ColMajor ? m : n;
    const int64_t outer = layout == Layout::ColMajor ? n : m;
    if (lda < inner)
        fail(filename, "Leading dimension is too small");
    const uint64_t max_elements = std::min({
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T)),
        static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max() / sizeof(T))});
    // The last stored entry has index (outer - 1)*lda + inner - 1.
    if (static_cast<uint64_t>(outer - 1) >
        (max_elements - static_cast<uint64_t>(inner)) / static_cast<uint64_t>(lda))
        fail(filename, "Leading dimension overflows the output size");
    return;
}

} // namespace matrix_io_detail

/// Read a nonempty, rectangular matrix with one row per text line.
/// Values are whitespace-delimited numbers parsed as double and converted to T.
/// Blank rows, malformed numbers, and inconsistent row lengths throw RandLAPACK::Error.
/// A final newline is optional; CRLF line endings are accepted.
///
/// With query=true, validate the file and replace m and n with its dimensions;
/// A may be nullptr and lda is ignored. Otherwise, m and n must match the file.
/// For Layout::ColMajor, lda >= m and entries are stored as A[row + lda*col];
/// for Layout::RowMajor, lda >= n and entries are stored as A[row*lda + col].
/// A must hold at least (n-1)*lda + m or (m-1)*lda + n elements, respectively.
/// Padding and dimensions are unchanged during a data read. Invalid layout,
/// dimensions, stride, or file input throw RandLAPACK::Error. If reading fails,
/// A may be partially filled. The file is opened separately on each call.
template <typename T>
void read_txt_matrix(Layout layout, int64_t& m, int64_t& n, T* A, int64_t lda,
                     const char* filename, bool query) {
    using namespace matrix_io_detail;
    if (!filename)
        fail(filename, "Missing filename");
    if (layout != Layout::ColMajor && layout != Layout::RowMajor)
        fail(filename, "Invalid matrix layout");
    if (!query) {
        check_output_size<T>(layout, m, n, lda, filename);
        if (!A)
            fail(filename, "Missing output buffer");
    }
    // Preserve every byte on Windows too; CRLF whitespace is parsed below.
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        fail(filename, "Cannot open file");

    int64_t rows = 0, cols = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (rows == std::numeric_limits<int64_t>::max() || (!query && rows >= m))
            fail(filename, "Too many matrix rows");
        int64_t row_cols = 0;
        const char* ptr = line.c_str();
        const char* limit = ptr + line.size();
        while (ptr < limit) {
            while (ptr < limit && std::isspace(static_cast<unsigned char>(*ptr)))
                ++ptr;
            if (ptr == limit)
                break;
            char* end;
            errno = 0;
            const double value = std::strtod(ptr, &end);
            if (end == ptr || errno == ERANGE ||
                (end < limit && !std::isspace(static_cast<unsigned char>(*end))))
                fail(filename, "Invalid matrix entry");
            if (row_cols == std::numeric_limits<int64_t>::max() || (!query && row_cols >= n))
                fail(filename, "Too many matrix columns");
            if (!query)
                A[layout == Layout::ColMajor ? rows + lda * row_cols : rows * lda + row_cols] =
                    static_cast<T>(value);
            ++row_cols;
            ptr = end;
        }
        if (row_cols == 0)
            fail(filename, "Empty matrix row");
        if (rows == 0)
            cols = row_cols;
        if (row_cols != cols || (!query && row_cols != n))
            fail(filename, "Inconsistent matrix row length");
        ++rows;
    }
    if (file.bad())
        fail(filename, "Failed to read matrix");
    checked_size<T>(rows, cols, filename);
    if (query) {
        m = rows;
        n = cols;
    } else if (rows != m) {
        fail(filename, "Matrix dimensions do not match output buffer");
    }
    return;
}

/// Read a nonempty dense matrix from a native-endian binary file.
/// The file contains two int64_t dimensions (m, n), followed by exactly m*n
/// IEEE-754 64-bit doubles in row-major order. Files must use the reader's byte
/// order; this format has no byte-order marker or automatic byte swapping.
///
/// With query=true, validate dimensions and payload size, then replace m and n;
/// A may be nullptr and lda is ignored. Otherwise, m and n must match the header.
/// Values are converted to T. For Layout::ColMajor, lda >= m and entries are
/// stored as A[row + lda*col]; for Layout::RowMajor, lda >= n and entries are
/// stored as A[row*lda + col]. A must hold at least (n-1)*lda + m or
/// (m-1)*lda + n elements, respectively. Padding and dimensions are unchanged
/// during a data read. Invalid layout, dimensions, stride, payload size, or
/// file input throw RandLAPACK::Error.
template <typename T>
void read_bin_matrix(Layout layout, int64_t& m, int64_t& n, T* A, int64_t lda,
                     const char* filename, bool query) {
    using namespace matrix_io_detail;
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                  "Binary matrix input requires IEEE-754 64-bit doubles");
    if (!filename)
        fail(filename, "Missing filename");
    if (layout != Layout::ColMajor && layout != Layout::RowMajor)
        fail(filename, "Invalid matrix layout");
    if (!query) {
        check_output_size<T>(layout, m, n, lda, filename);
        if (!A)
            fail(filename, "Missing output buffer");
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        fail(filename, "Cannot open file");
    int64_t dims[2];
    if (!file.read(reinterpret_cast<char*>(dims), sizeof(dims)))
        fail(filename, "Failed to read matrix header");
    const size_t count = checked_size<T>(dims[0], dims[1], filename);
    checked_size<double>(dims[0], dims[1], filename);
    if (count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(double))
        fail(filename, "Matrix payload is too large to read");
    const auto bytes = static_cast<std::streamsize>(count * sizeof(double));
    if (!query && (m != dims[0] || n != dims[1]))
        fail(filename, "Matrix dimensions do not match output buffer");

    const auto payload_start = file.tellg();
    file.seekg(0, std::ios::end);
    const auto payload_end = file.tellg();
    if (payload_start == std::streampos(-1) || payload_end == std::streampos(-1) ||
        payload_end - payload_start != bytes)
        fail(filename, "Matrix payload size does not match header");
    if (query) {
        m = dims[0];
        n = dims[1];
        return;
    }

    file.seekg(payload_start);
    std::vector<double> values(count);
    if (!file.read(reinterpret_cast<char*>(values.data()), bytes))
        fail(filename, "Failed to read matrix payload");
    for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < n; ++col)
            A[layout == Layout::ColMajor ? row + lda * col : row * lda + col] =
                static_cast<T>(values[row * n + col]);
    return;
}

} // namespace RandLAPACK::gen
